#!/usr/bin/env python3
"""Verify the agentic-pipeline outputs for a patient are well-formed.

Checks:
- runs/{pid}/patient_info.json parses
- quick_eval.extract_variables_from_snapshot can read it
- runs/{pid}/summary.json parses and is reasonable length

Exits non-zero on any issue. Prints a short summary.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Import the existing evaluator's snapshot extractor for schema verification
from quick_eval import extract_variables_from_snapshot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--runs-root", default=str(ROOT / "agentic_toa" / "runs"))
    args = ap.parse_args()

    runs_dir = Path(args.runs_root) / args.pid
    issues = []
    summary = {"pid": args.pid}

    # 1. patient_info.json
    pi_path = runs_dir / "patient_info.json"
    if not pi_path.exists():
        issues.append(f"missing {pi_path}")
    else:
        try:
            pi = json.loads(pi_path.read_text())
            for key in ("PATIENT DEMOGRAPHICS", "TUMOR INFORMATION", "TREATMENTS"):
                if key not in pi:
                    issues.append(f"patient_info.json missing top-level key {key!r}")
            summary["patient_info_keys"] = list(pi.keys())
        except json.JSONDecodeError as e:
            issues.append(f"patient_info.json parse error: {e}")

    # 2. quick_eval snapshot extraction
    try:
        vals = extract_variables_from_snapshot(args.pid, runs_dir, no_graph_fallbacks=True)
        summary["extracted_vars_n"] = len(vals)
        summary["extracted_preview"] = {k: (v[:60] if isinstance(v, str) else v) for k, v in list(vals.items())[:8]}
    except Exception as e:
        issues.append(f"quick_eval.extract_variables_from_snapshot raised: {type(e).__name__}: {e}")

    # 3. summary.json
    sj_path = runs_dir / "summary.json"
    if not sj_path.exists():
        issues.append(f"missing {sj_path}")
    else:
        try:
            sj = json.loads(sj_path.read_text())
            note = sj.get("summary", "")
            summary["summary_chars"] = len(note)
            if len(note) < 200:
                issues.append(f"summary too short: {len(note)} chars (expected ~750-999)")
        except json.JSONDecodeError as e:
            issues.append(f"summary.json parse error: {e}")

    # 4. timeline / episodes integrity
    timeline_path = runs_dir / "timeline_objects.jsonl"
    episodes_path = runs_dir / "episodes.json"
    if timeline_path.exists():
        summary["n_timeline_events"] = sum(1 for line in timeline_path.read_text().splitlines() if line.strip())
    if episodes_path.exists():
        try:
            ep = json.loads(episodes_path.read_text())
            summary["n_episodes"] = len(ep.get("episodes", []))
        except json.JSONDecodeError as e:
            issues.append(f"episodes.json parse error: {e}")

    # 5. unification report
    report_path = runs_dir / "unification_report.json"
    if report_path.exists():
        rep = json.loads(report_path.read_text())
        summary["unification"] = {
            "n_chunks": rep.get("n_chunks"),
            "final_events": rep.get("final_event_count"),
            "conflicts": len(rep.get("conflicts", [])),
            "background_conflicts": len(rep.get("background_conflicts", [])),
        }

    summary["issues"] = issues
    summary["status"] = "OK" if not issues else "FAIL"
    print(json.dumps(summary, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
