#!/usr/bin/env python3
"""Radiology alignment — verify TOA imaging events against the OMOP radiology vector.

Runs after unify_timeline (before display generation). For each entry in the
deterministic radiology procedure vector, check whether the unified timeline has
a corresponding imaging event. For each unmatched vector entry, ask an LLM to
read the relevant source chunks and decide whether it was a real imaging study
that should be added to the timeline.

The LLM is given a stripped-down, schema-free question (matching how the judge
succeeds): "is this date a real imaging study? what modality? cite the chart."

Outputs:
  runs/{pid}/radiology_alignment.json — full report
  runs/{pid}/timeline_objects.jsonl   — updated in place with any imaging events
                                         the LLM confirmed (with provenance refs
                                         from the source chunks)

Usage:
    python agentic_toa/tools/radiology_alignment.py --pid 136022342
    python agentic_toa/tools/radiology_alignment.py --pid 136022342 --dry-run  # report only
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import gsgpt
from toa.graph_store import CohortGraphStore
from toa.deterministic_retrieval import DeterministicRetriever

RUNS = ROOT / "agentic_toa" / "runs"

# Structured visit_detail codes that represent radiology visits but carry no
# explicit imaging modality — the OMOP ct_date_vector usually misses these because
# they're visit_details, not procedure codes. Yet they often ARE real imaging
# studies and the agentic chunk extractor skips them due to ambiguous typing.
RADIO_VISIT_CODES = re.compile(
    r'code="NUCC/(?:261QR0200X|2085R0202X|261QD0000X|2085D0003X)"',
    re.IGNORECASE,
)


def collect_xml_radio_visits(pid: str) -> list[str]:
    """Return all dates (YYYY-MM-DD) where the XML has an Ambulatory Radiology /
    Diagnostic Radiology visit_detail code. These are visit dates that often
    correspond to real imaging studies but aren't always captured as such."""
    xml_paths = list((ROOT / "patient_records" / "cohorts").glob(f"eval*{pid}*.xml"))
    if not xml_paths:
        return []
    xml = xml_paths[0].read_text()
    dates = set()
    for part in xml.split('<entry timestamp="')[1:]:
        body_end = part.find('</entry>')
        if body_end < 0: continue
        body = part[:body_end]
        if RADIO_VISIT_CODES.search(body):
            dates.add(part[:10])
    return sorted(dates, reverse=True)


def load_timeline(runs_dir: Path) -> list[dict]:
    p = runs_dir / "timeline_objects.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def imaging_events(timeline: list[dict]) -> list[dict]:
    out = []
    for ev in timeline:
        mod = (ev.get("modality") or "").lower()
        desc = (ev.get("description") or "").lower()
        is_imaging = (
            ev.get("type") == "imaging"
            or mod in ("ct", "pet-ct", "pet/ct", "mri", "xray", "x-ray", "ultrasound", "us", "nm")
            or any(k in desc for k in ("ct chest", "ct abdom", "pet/ct", "pet-ct", "mri", "x-ray", "cta "))
        )
        if is_imaging:
            out.append(ev)
    return out


def date_within(a: str, b: str, tol_days: int) -> bool:
    try:
        da = datetime.strptime(a[:10], "%Y-%m-%d")
        db = datetime.strptime(b[:10], "%Y-%m-%d")
        return abs((da - db).days) <= tol_days
    except Exception:
        return False


def find_chunk_window(pid: str, target_date: str, runs_dir: Path) -> str:
    """Return ~5000 chars of source text around the target date for the verifier.

    The serialized_text.txt uses "=== YYYY-MM-DD ===" date headers. We find the
    nearest such header and return a window around it.
    """
    p = runs_dir / "serialized_text.txt"
    if not p.exists():
        return ""
    text = p.read_text()
    # Try exact match first
    marker = f"=== {target_date} ==="
    idx = text.find(marker)
    if idx >= 0:
        start = max(0, idx - 1500)
        end = min(len(text), idx + 4500)
        return text[start:end]
    # Try nearby dates (±14 days)
    try:
        td = datetime.strptime(target_date, "%Y-%m-%d")
    except Exception:
        return ""
    for delta in range(1, 15):
        for sign in (-1, 1):
            d = (td + timedelta(days=sign*delta)).strftime("%Y-%m-%d")
            marker = f"=== {d} ==="
            idx = text.find(marker)
            if idx >= 0:
                start = max(0, idx - 1500)
                end = min(len(text), idx + 4500)
                return text[start:end]
    return ""


VERIFIER_PROMPT = """A patient's structured medical record has an imaging-procedure code logged for {target_date} — but our agentic timeline extractor didn't capture it as an imaging event. Read the chart snippet around that date and decide:

1. Was an actual imaging study performed on or near {target_date}?
2. If yes: what modality (CT, PET-CT, MRI, X-ray, ultrasound, other)? What anatomic site? Was it diagnostic, or e.g. a radiation-planning CT (excluded), or just a clinic check-in (not imaging)?
3. Cite the specific chart phrase that supports your answer.

Be honest about ambiguity. Schema-free reasoning: don't restrict yourself to entries with explicit modality codes — clinical context (oncology workup, radiology visit, report mentions) is sufficient.

CHART SNIPPET AROUND {target_date}:
---
{chunk_window}
---

Return JSON only:
{{
  "is_real_imaging": true | false,
  "modality": "ct" | "pet-ct" | "mri" | "xray" | "ultrasound" | "other" | null,
  "site": "<anatomic site or null>",
  "description": "<one-line description that would go in a timeline event>",
  "confidence": "high" | "med" | "low",
  "evidence": "<the chart phrase>",
  "exclude_reason": "<if is_real_imaging=false: why (e.g. radiation-planning CT, generic clinic visit, billing artifact)>"
}}
"""


def verify_one(pid: str, target_date: str, chunk_window: str, model: str) -> dict:
    if not chunk_window:
        return {"date": target_date, "is_real_imaging": None, "error": "no chart context found"}
    prompt = VERIFIER_PROMPT.format(target_date=target_date, chunk_window=chunk_window)
    try:
        kwargs = {"max_tokens": 1024}
        if "gemini" in model.lower() and "flash" in model.lower():
            kwargs["thinking_budget"] = 1024
        resp = gsgpt.chat(prompt, model=model, **kwargs)
        text = resp.strip()
        if "```json" in text: text = text.split("```json", 1)[1].split("```", 1)[0]
        elif text.startswith("```"): text = text.split("```", 1)[1].split("```", 1)[0]
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {"date": target_date, "is_real_imaging": None, "error": "no JSON"}
        j = json.loads(m.group(0))
        j["date"] = target_date
        j["error"] = None
        return j
    except Exception as e:
        return {"date": target_date, "is_real_imaging": None, "error": str(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--model", default="claude-opus-4-6")
    ap.add_argument("--parallel", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report alignment only, do not update timeline_objects.jsonl")
    ap.add_argument("--confidence-floor", default="med",
                    choices=["high", "med"],
                    help="Only ADD missing events at this confidence level or higher")
    args = ap.parse_args()

    runs_dir = RUNS / args.pid
    timeline = load_timeline(runs_dir)
    img_events = imaging_events(timeline)
    img_dates_in_timeline = {(ev.get("date") or "")[:10] for ev in img_events if ev.get("date")}

    graph = CohortGraphStore("graph_store").load_patient_graph(args.pid)
    vector = DeterministicRetriever(graph).get_ct_date_vector()
    vector_dates = [d[:10] for d in vector if d]
    xml_radio_visits = collect_xml_radio_visits(args.pid)

    # Pool both date sources: OMOP ct_date_vector + structured XML radiology
    # visit_details (Ambulatory/Diagnostic Radiology visits). The latter often
    # represents real imaging studies that the OMOP vector misses (because they
    # were coded as visit_details rather than procedures) and that the chunk
    # extractor skips (no explicit modality).
    candidate_dates = sorted(set(vector_dates) | set(xml_radio_visits), reverse=True)

    # Find candidates with no timeline match within ±3 days
    unmatched = []
    for cd in candidate_dates:
        matched = any(date_within(cd, td, tol_days=3) for td in img_dates_in_timeline)
        if not matched:
            unmatched.append(cd)

    # Find timeline imaging events with no vector match (informational only — these
    # are probably outside studies or modalities the vector doesn't track)
    timeline_only = []
    for td in sorted(img_dates_in_timeline, reverse=True):
        matched = any(date_within(td, vd, tol_days=3) for vd in vector_dates)
        if not matched:
            timeline_only.append(td)

    print(f"PID {args.pid}: timeline has {len(img_events)} imaging events; "
          f"candidate dates = {len(candidate_dates)} (OMOP vector {len(vector_dates)} + XML radio-visits {len(xml_radio_visits)}, dedup)",
          file=sys.stderr)
    print(f"  candidate dates WITHOUT timeline match: {len(unmatched)}", file=sys.stderr)
    print(f"  timeline imaging events WITHOUT candidate match: {len(timeline_only)} (likely outside/other-modality, not a problem)", file=sys.stderr)

    # Verify each unmatched vector entry
    verifications = []
    if unmatched:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(verify_one, args.pid, d, find_chunk_window(args.pid, d, runs_dir), args.model): d
                for d in unmatched
            }
            for fut in as_completed(futures):
                verifications.append(fut.result())
        verifications.sort(key=lambda v: v["date"], reverse=True)

    # Decide what to add
    to_add = []
    floor_levels = {"high": ["high"], "med": ["high", "med"]}
    accepted_conf = floor_levels[args.confidence_floor]
    for v in verifications:
        if v.get("is_real_imaging") is True and v.get("confidence") in accepted_conf:
            to_add.append(v)

    # Build new timeline events for confirmed additions
    new_events = []
    next_event_id = max([ev.get("event_id", -1) for ev in timeline] + [0]) + 1
    for v in to_add:
        new_events.append({
            "event_id": next_event_id,
            "date": v["date"],
            "type": "imaging",
            "subtype": None,
            "modality": v.get("modality"),
            "site": v.get("site"),
            "description": (v.get("description") or "")[:200] + " [recovered via radiology_alignment]",
            "priority": "MAJOR",
            "source_event_refs": [],
            "_from_chunks": [],
            "_alignment_confidence": v.get("confidence"),
            "_alignment_evidence": v.get("evidence", "")[:300],
        })
        next_event_id += 1

    # Write report
    report = {
        "pid": args.pid,
        "n_timeline_imaging": len(img_events),
        "n_vector_entries": len(vector_dates),
        "unmatched_vector_dates": unmatched,
        "timeline_only_dates": timeline_only,
        "verifications": verifications,
        "to_add": to_add,
        "dry_run": args.dry_run,
    }
    (runs_dir / "radiology_alignment.json").write_text(json.dumps(report, indent=2))

    # Update timeline if not dry-run
    if to_add and not args.dry_run:
        # Backup, then write augmented timeline
        backup = runs_dir / "timeline_objects_prealign.jsonl"
        if not backup.exists():
            backup.write_text((runs_dir / "timeline_objects.jsonl").read_text())
        merged = timeline + new_events
        merged.sort(key=lambda e: (e.get("date") or "0000-00-00", e.get("type") or ""))
        out = runs_dir / "timeline_objects.jsonl"
        out.write_text("\n".join(json.dumps(e) for e in merged) + "\n")

    print(f"\n=== Verifications ({len(verifications)} unmatched vector dates) ===")
    for v in verifications:
        mark = "✓" if v.get("is_real_imaging") and v.get("confidence") in accepted_conf else "○"
        print(f"  {mark} {v['date']}  is_real={v.get('is_real_imaging')}  conf={v.get('confidence')}  modality={v.get('modality')}  {(v.get('evidence','') or '')[:80]}")
    print(f"\n{'ADDED' if not args.dry_run else 'WOULD ADD'} {len(to_add)} imaging events to timeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
