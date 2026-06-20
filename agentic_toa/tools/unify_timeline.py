#!/usr/bin/env python3
"""Reduce per-chunk extraction outputs into a single unified timeline.

Inputs:
    runs/{pid}/chunks/chunk_*.json   (from extract_chunk.py)

Outputs:
    runs/{pid}/timeline_objects.jsonl       (one event per line, sorted by date)
    runs/{pid}/background.json              (unified pre-cancer baseline)
    runs/{pid}/unification_report.json      (stats + flagged conflicts)

This is the **deterministic v1** unifier. It applies these rules:

1. **Dedup events by (source_event_refs ∩):** any two events that share at least
   one [E<n>] reference are merged into one — they're describing the same
   underlying graph node. The merge prefers richer description / non-null
   modality+site / specific date.
2. **Demote before_chunk_mentions when corresponding in-chunk events exist.**
   A `before_chunk_mention` from chunk N is dropped if any chunk M has an
   `events` entry sharing one of its `source_event_refs`. If no in-chunk
   event is found, the before_chunk_mention is promoted to a regular event
   and flagged in the report (lower-confidence row).
3. **Background union.** Per-key merge across chunks. Conflicts (e.g., one
   chunk says "Former, 30 pack-years", another says "Never") are recorded in
   the report and the first non-null value wins (deterministic).
4. **Conflict surfacing.** Same (date, type, modality, site) cluster but
   meaningfully different descriptions → recorded as `status="needs_review"`
   in the report so the agent can intervene with a graph search.

Future v2 will replace this deterministic merge with an LLM-led unifier
(prompts/timeline_unification_v2.txt) that consumes the deterministic output
+ raw chunks + graph context.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def normalize_description(desc: str) -> str:
    """Lowercase, strip punctuation/whitespace for dedup signature."""
    if not desc:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", desc.lower()).strip()


def merge_two_events(a: dict, b: dict) -> dict:
    """Combine two events that refer to the same underlying graph node."""
    out = dict(a)
    # Prefer richer description
    if len((b.get("description") or "")) > len(out.get("description") or ""):
        out["description"] = b["description"]
    # Merge source_event_refs (set union, preserve order)
    refs = list(dict.fromkeys((a.get("source_event_refs") or []) + (b.get("source_event_refs") or [])))
    out["source_event_refs"] = refs
    # Prefer non-null modality / site / subtype / evidence_date
    for k in ("modality", "site", "subtype", "evidence_date"):
        if not out.get(k) and b.get(k):
            out[k] = b[k]
    # Prefer the earlier evidence_date if both present
    if a.get("evidence_date") and b.get("evidence_date"):
        out["evidence_date"] = min(a["evidence_date"], b["evidence_date"])
    # Merge values dicts
    if a.get("values") or b.get("values"):
        merged = dict(a.get("values") or {})
        merged.update(b.get("values") or {})
        out["values"] = merged
    # Priority: MAJOR wins
    if (b.get("priority") == "MAJOR") or (a.get("priority") == "MAJOR"):
        out["priority"] = "MAJOR"
    # Track that this row is a merger
    chunks = sorted(set((a.get("_from_chunks") or [a.get("_from_chunk")]) +
                        (b.get("_from_chunks") or [b.get("_from_chunk")])))
    out["_from_chunks"] = [c for c in chunks if c is not None]
    out.pop("_from_chunk", None)
    return out


def unify(chunks: list[dict]) -> tuple[list[dict], dict, dict]:
    """Apply the deterministic unification rules.

    Returns (timeline_events, unified_background, report).
    """
    # 1. Collect all in-chunk events tagged with their chunk index
    in_chunk_events = []
    for chunk in chunks:
        chunk_idx = chunk.get("_chunk_meta", {}).get("chunk")
        for ev in chunk.get("events", []):
            ev = dict(ev)
            ev["_from_chunk"] = chunk_idx
            in_chunk_events.append(ev)

    # 2. Index by source_event_refs membership for dedup
    by_ref: dict[str, list[int]] = defaultdict(list)
    for i, ev in enumerate(in_chunk_events):
        for r in ev.get("source_event_refs") or []:
            by_ref[r].append(i)

    # 3. Merge clusters sharing any ref. Union-find over indices.
    parent = list(range(len(in_chunk_events)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for indices in by_ref.values():
        for j in range(1, len(indices)):
            union(indices[0], indices[j])

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(in_chunk_events)):
        clusters[find(i)].append(i)

    merged_events = []
    n_merges = 0
    for root, members in clusters.items():
        acc = in_chunk_events[members[0]]
        for j in members[1:]:
            acc = merge_two_events(acc, in_chunk_events[j])
            n_merges += 1
        # Normalize from_chunk -> from_chunks list
        if "_from_chunk" in acc:
            acc["_from_chunks"] = [acc.pop("_from_chunk")]
        merged_events.append(acc)

    # 4. Handle before_chunk_mentions
    all_in_chunk_refs = set()
    for ev in merged_events:
        all_in_chunk_refs.update(ev.get("source_event_refs") or [])

    promoted_before = []
    dropped_before = 0
    for chunk in chunks:
        chunk_idx = chunk.get("_chunk_meta", {}).get("chunk")
        for bcm in chunk.get("before_chunk_mentions", []):
            refs = set(bcm.get("source_event_refs") or [])
            if refs & all_in_chunk_refs:
                dropped_before += 1
                continue
            ev = dict(bcm)
            ev["_from_chunk"] = chunk_idx
            ev["_promoted_from_before_chunk"] = True
            promoted_before.append(ev)

    # 5. Dedup promoted before mentions among themselves (same union-find by ref)
    if promoted_before:
        p_by_ref: dict[str, list[int]] = defaultdict(list)
        for i, ev in enumerate(promoted_before):
            for r in ev.get("source_event_refs") or []:
                p_by_ref[r].append(i)
        pparent = list(range(len(promoted_before)))

        def pfind(x):
            while pparent[x] != x:
                pparent[x] = pparent[pparent[x]]
                x = pparent[x]
            return x

        def punion(a, b):
            ra, rb = pfind(a), pfind(b)
            if ra != rb:
                pparent[ra] = rb

        for idxs in p_by_ref.values():
            for j in range(1, len(idxs)):
                punion(idxs[0], idxs[j])
        pclusters: dict[int, list[int]] = defaultdict(list)
        for i in range(len(promoted_before)):
            pclusters[pfind(i)].append(i)
        merged_before = []
        for root, members in pclusters.items():
            acc = promoted_before[members[0]]
            for j in members[1:]:
                acc = merge_two_events(acc, promoted_before[j])
            if "_from_chunk" in acc:
                acc["_from_chunks"] = [acc.pop("_from_chunk")]
            acc["_promoted_from_before_chunk"] = True
            merged_before.append(acc)
        merged_events.extend(merged_before)

    # 6. Conflict scan: events with same (date, type, modality, site) but very different normalized desc
    conflicts = []
    sig_groups: dict[tuple, list[int]] = defaultdict(list)
    for i, ev in enumerate(merged_events):
        sig = (ev.get("date"), ev.get("type"), ev.get("modality"), ev.get("site"))
        sig_groups[sig].append(i)
    for sig, members in sig_groups.items():
        if len(members) < 2:
            continue
        descs = [normalize_description(merged_events[m].get("description") or "") for m in members]
        if len({d for d in descs if d}) > 1:
            conflicts.append({
                "signature": {"date": sig[0], "type": sig[1], "modality": sig[2], "site": sig[3]},
                "members": [
                    {
                        "from_chunks": merged_events[m].get("_from_chunks") or [],
                        "description": merged_events[m].get("description"),
                        "source_event_refs": merged_events[m].get("source_event_refs"),
                    }
                    for m in members
                ],
                "status": "needs_review",
            })

    # 7. Sort by date for stability
    merged_events.sort(key=lambda e: (e.get("date") or "0000-00-00", e.get("type") or ""))

    # 8. Background union
    bg = {
        "smoking": None,
        "comorbidities": [],
        "allergies": [],
        "family_history": [],
        "prior_cancers": [],
        "social_history": [],
        "prior_surgeries": [],
    }
    bg_conflicts = []
    for chunk in chunks:
        cbg = chunk.get("background") or {}
        # smoking: first non-null wins; conflict if mismatched
        if cbg.get("smoking"):
            if bg["smoking"] is None:
                bg["smoking"] = cbg["smoking"]
            elif normalize_description(cbg["smoking"]) != normalize_description(bg["smoking"]):
                bg_conflicts.append({
                    "field": "smoking",
                    "existing": bg["smoking"],
                    "from_chunk": chunk.get("_chunk_meta", {}).get("chunk"),
                    "new": cbg["smoking"],
                })
        # plain list fields: union, preserve order
        for key in ("comorbidities", "allergies", "family_history", "prior_cancers", "social_history"):
            for item in (cbg.get(key) or []):
                if item and item not in bg[key]:
                    bg[key].append(item)
        # prior_surgeries: list of dicts {procedure, date, context} — dedup on normalized procedure+date
        for surg in (cbg.get("prior_surgeries") or []):
            if not isinstance(surg, dict):
                # Some models may emit bare strings — coerce
                surg = {"procedure": str(surg), "date": None, "context": None}
            proc = (surg.get("procedure") or "").strip()
            if not proc:
                continue
            key = (normalize_description(proc), (surg.get("date") or "").strip())
            existing_keys = {(normalize_description(s.get("procedure") or ""), (s.get("date") or "").strip()) for s in bg["prior_surgeries"]}
            if key not in existing_keys:
                bg["prior_surgeries"].append({
                    "procedure": proc,
                    "date": surg.get("date") or None,
                    "context": surg.get("context") or None,
                })

    report = {
        "n_chunks": len(chunks),
        "n_in_chunk_events_raw": len(in_chunk_events),
        "n_in_chunk_events_after_merge": len(clusters),
        "n_in_chunk_merges": n_merges,
        "n_before_chunk_mentions_dropped_as_duplicates": dropped_before,
        "n_before_chunk_mentions_promoted": len(promoted_before),
        "final_event_count": len(merged_events),
        "conflicts": conflicts,
        "background_conflicts": bg_conflicts,
    }
    return merged_events, bg, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--method", default="deterministic", choices=["deterministic"])
    ap.add_argument("--runs-root", default=str(ROOT / "agentic_toa" / "runs"))
    args = ap.parse_args()

    runs_dir = Path(args.runs_root) / args.pid
    chunks_dir = runs_dir / "chunks"
    chunk_paths = sorted(chunks_dir.glob("chunk_*.json"),
                         key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if not chunk_paths:
        print(f"ERROR: no chunk_*.json files in {chunks_dir}", file=sys.stderr)
        return 1
    chunks = [json.loads(p.read_text()) for p in chunk_paths]

    events, background, report = unify(chunks)

    # Write outputs
    timeline_path = runs_dir / "timeline_objects.jsonl"
    with timeline_path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    (runs_dir / "background.json").write_text(json.dumps(background, indent=2))
    (runs_dir / "unification_report.json").write_text(json.dumps(report, indent=2))

    print(f"OK: unified pid={args.pid} method={args.method}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
