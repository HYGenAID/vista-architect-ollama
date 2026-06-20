#!/usr/bin/env python3
"""Evaluate an agentic-pipeline patient's display items using the display-quality judge.

This is an *alternative* to quick_eval.py's strict judge. It reuses quick_eval's
machinery to extract the 16 MTB-salient variables from the snapshot and to
assemble XML ground-truth evidence, but swaps in the prompt at
agentic_toa/prompts/display_quality_judge.txt — which frames the task as
"reviewing tumor-board dashboard items" rather than "grading test answers".

Usage:
    python agentic_toa/evaluate/judge_display_quality.py --pid 136009359 \
        --snapshot-dir agentic_toa/runs \
        --judge-model claude-opus-4-6 \
        --output /tmp/eval_display_quality.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gsgpt
import quick_eval as qe
from eval_variables_v2 import EVAL_VARIABLES_V2

PROMPT_PATH = ROOT / "agentic_toa" / "prompts" / "display_quality_judge.txt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--judge-model", default="claude-opus-4-6")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    snapshot_root = Path(args.snapshot_dir)
    json_dir = snapshot_root / args.pid

    # 1. Pull the 16 extracted values from the snapshot
    extracted = qe.extract_variables_from_snapshot(args.pid, json_dir, no_graph_fallbacks=True)

    # 2. Build XML ground-truth evidence the same way quick_eval does
    xml_path = qe._resolve_truncated_xml(args.pid, cohort_name=None)
    if xml_path is None or not Path(xml_path).exists():
        print(f"ERROR: could not resolve XML for pid={args.pid}", file=sys.stderr)
        return 2

    evidence_parts = []
    import xml.etree.ElementTree as ET
    person_elem = ET.parse(str(xml_path)).getroot().find(".//person")
    if person_elem is not None:
        evidence_parts.append("=== DEMOGRAPHICS ===\n" + ET.tostring(person_elem, encoding="unicode"))

    # Surgery + final-120K chunk (replicates quick_eval evidence assembly)
    full_xml = Path(xml_path).read_text()
    evidence_parts.append("=== XML (final 120K chars, where late-record content lives) ===\n" + full_xml[-120_000:])

    # 3. Add deterministic retrieval context for the judge (same as strict judge)
    try:
        from toa.graph_store import CohortGraphStore
        from toa.deterministic_retrieval import DeterministicRetriever
        graph = CohortGraphStore("graph_store").load_patient_graph(args.pid)
        det = DeterministicRetriever(graph).get_comprehensive_context()
        from toa.backend import _format_deterministic_guidance
        evidence_parts.append("=== DETERMINISTIC RETRIEVAL (graph search) ===\n" + _format_deterministic_guidance(det))
    except Exception as e:
        print(f"WARN: graph context unavailable ({e})", file=sys.stderr)

    xml_evidence = "\n\n".join(evidence_parts)

    # 4. Fill the display-quality judge prompt
    var_list = "\n".join([f"{i+1}. {v['name']} (expected format: {v['format']})"
                          for i, v in enumerate(EVAL_VARIABLES_V2)])
    prompt = (
        PROMPT_PATH.read_text()
        .replace("{n_variables}", str(len(EVAL_VARIABLES_V2)))
        .replace("{var_list}", var_list)
        .replace("{extracted_values}", json.dumps(extracted, indent=2))
        .replace("{xml_evidence}", xml_evidence)
    )

    # 5. Call the judge
    response = gsgpt.chat(
        prompt,
        model=args.judge_model,
        max_tokens=args.max_tokens,
    )

    # 6. Parse JSON (tolerant of code fences)
    text = response.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    judgments = json.loads(text)

    # 7. Format results consistently with quick_eval output
    results = []
    for v in EVAL_VARIABLES_V2:
        name = v["name"]
        j = judgments.get(name, {})
        results.append({
            "patient_id": args.pid,
            "variable": name,
            "category": v["category"],
            "primary": v["primary"],
            "correctness": j.get("correctness", "N/A"),
            "score": j.get("score", 0),
            "explanation": j.get("explanation", ""),
            "xml_value": j.get("xml_value", ""),
            "extracted_value": extracted.get(name, ""),
            "eval_mode": "agentic_display_quality_judge",
        })

    out = {
        "eval_version": "agentic_display_quality_v1",
        "judge_model": args.judge_model,
        "judge_prompt": "agentic_toa/prompts/display_quality_judge.txt",
        "snapshot_dir": str(snapshot_root),
        "num_variables": len(results),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    # Summary
    mean = sum(r["score"] for r in results) / len(results)
    ncorr = sum(1 for r in results if r["correctness"] == "Correct")
    print(f"OK: pid={args.pid} judge={args.judge_model} mean={mean:.2f}/10 correct={ncorr}/{len(results)} -> {args.output}",
          file=sys.stderr)
    for r in results:
        if r["score"] < 10:
            print(f"  {r['variable']:<32s} {r['score']:>2d}/10  {r['explanation'][:140]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
