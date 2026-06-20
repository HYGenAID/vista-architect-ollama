#!/usr/bin/env python3
"""Generate display items (patient_info.json + summary.json) from the unified timeline.

Reads:
    runs/{pid}/timeline_objects.jsonl
    runs/{pid}/background.json
    runs/{pid}/episodes.json
    (plus DeterministicRetriever context + CT date vector from the sandboxed graph)

Writes:
    runs/{pid}/patient_info.json
    runs/{pid}/summary.json
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gsgpt
from toa.graph import TOAGraph
from toa.deterministic_retrieval import DeterministicRetriever
from toa.backend import _format_deterministic_guidance

PROMPT_PATH = ROOT / "agentic_toa" / "prompts" / "display_items_v2.txt"


def extract_json_block(text: str) -> dict:
    """Pull JSON from a ```json ... ``` fence or fall back to balanced braces."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object in response")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("Unbalanced JSON braces in response")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--thinking", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--tumor-type", default="Thoracic Oncology")
    ap.add_argument("--include-source-text", action="store_true", default=True,
                    help="Pass the last N chars of serialized text to the display LLM")
    ap.add_argument("--no-source-text", dest="include_source_text", action="store_false",
                    help="Structured-only mode (A/B comparison; skip source_text)")
    ap.add_argument("--source-text-tail-chars", type=int, default=150_000,
                    help="When source_text is included, take this many chars from the END of serialized_text.txt. "
                         "Default 150k is suitable for million-token-context models "
                         "(e.g. Gemini-3.5-flash, Opus 4.6). Reduce for smaller-context models.")
    ap.add_argument("--output-suffix", default="",
                    help="Append to output filenames (e.g. '_nosrc' to produce patient_info_nosrc.json)")
    ap.add_argument("--runs-root", default=str(ROOT / "agentic_toa" / "runs"))
    ap.add_argument("--graph-store-root", default=str(ROOT / "agentic_toa" / "temp_graphs"))
    args = ap.parse_args()

    runs_dir = Path(args.runs_root) / args.pid
    timeline_path = runs_dir / "timeline_objects.jsonl"
    background_path = runs_dir / "background.json"
    episodes_path = runs_dir / "episodes.json"
    for p in (timeline_path, background_path, episodes_path):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    events = []
    for i, line in enumerate(timeline_path.read_text().splitlines()):
        line = line.strip()
        if line:
            ev = json.loads(line)
            ev["event_id"] = i
            events.append(ev)
    background = json.loads(background_path.read_text())
    episodes = json.loads(episodes_path.read_text())

    # Graph context + CT date vector + Person demographics
    graphml_path = Path(args.graph_store_root) / f"{args.pid}.graphml"
    graph = TOAGraph.load(str(graphml_path))
    retriever = DeterministicRetriever(graph)
    det_context = _format_deterministic_guidance(retriever.get_comprehensive_context())
    # NOTE: ct_date_vector is retained as a sanity-check input in the prompt,
    # but the display LLM uses the TIMELINE as the single source of truth for
    # date_of_last_ct. Proper imaging-event extraction is a TOA-pipeline
    # responsibility; the vector is NOT used as a patching mechanism here.
    ct_dates = retriever.get_ct_date_vector()

    # Deterministic demographics. Preferred source: runs/{pid}/demographics.json
    # written at setup time from the source XML. Fallback: try the Person node
    # in the graph (most Lumia graphs don't have one — copy_to_temp_graphs.py
    # is the canonical extraction).
    demographics = {}
    demo_path = runs_dir / "demographics.json"
    if demo_path.exists():
        demographics = json.loads(demo_path.read_text())
    if not demographics.get("date_of_birth"):
        for _, data in graph.G.nodes(data=True):
            if data.get("node_type") == "Person":
                demographics = {
                    "name": data.get("name"),
                    "date_of_birth": data.get("birth_datetime") or data.get("dob"),
                    "sex": data.get("gender") or data.get("sex"),
                    "race": data.get("race"),
                    "ethnicity": data.get("ethnicity"),
                }
                break

    # Optional source_text — last N chars of the serialized graph text
    source_text = ""
    if args.include_source_text:
        st_path = runs_dir / "serialized_text.txt"
        if st_path.exists():
            full = st_path.read_text()
            source_text = full[-args.source_text_tail_chars:] if len(full) > args.source_text_tail_chars else full
    if not source_text:
        source_text = "[source_text omitted — structured-only A/B variant]"

    prompt = (
        PROMPT_PATH.read_text()
        .replace("{TUMOR_TYPE}", args.tumor_type)
        .replace("{demographics_block}", json.dumps(demographics, indent=2))
        .replace("{timeline_context}", json.dumps(events, indent=2))
        .replace("{episodes_context}", json.dumps(episodes.get("episodes", []), indent=2))
        .replace("{background_block}", json.dumps(background, indent=2))
        .replace("{deterministic_context}", det_context)
        .replace("{ct_date_vector}", json.dumps(ct_dates))
        .replace("{source_text}", source_text)
    )

    t0 = time.time()
    response = gsgpt.chat(
        prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        thinking_budget=args.thinking,
    )
    elapsed = time.time() - t0

    if not response:
        print(f"ERROR: empty response from {args.model}", file=sys.stderr)
        return 2

    try:
        obj = extract_json_block(response)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: parse failed: {e}\n--- raw (1000) ---\n{response[:1000]}", file=sys.stderr)
        return 3

    if "patient_info" not in obj or "summary" not in obj:
        print(f"ERROR: response missing keys; got {list(obj.keys())}", file=sys.stderr)
        return 4

    patient_info = obj["patient_info"]
    summary_text = obj["summary"]

    # Write outputs in the schema quick_eval expects; optional suffix for A/B runs
    sfx = args.output_suffix
    pi_name = f"patient_info{sfx}.json"
    sm_name = f"summary{sfx}.json"
    (runs_dir / pi_name).write_text(json.dumps(patient_info, indent=2))
    (runs_dir / sm_name).write_text(json.dumps({"summary": summary_text}, indent=2))

    print(f"OK: pid={args.pid} elapsed={elapsed:.1f}s source_text={'yes' if args.include_source_text else 'no'} "
          f"-> {pi_name} + {sm_name}", file=sys.stderr)
    print(f"  TB note ({len(summary_text)} chars):", file=sys.stderr)
    print("  " + summary_text.replace("\n", "\n  ")[:500], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
