"""
Aggregate RAG baseline vs VISTA results on the 30 clinician-validation patients.

Inputs:
  - clinician_validation/_pid_mapping.json (the 30 pids)
  - outputs/evals/eval_v2_1180_merged_results.json (VISTA scores, filtered to the 30 pids)
  - outputs/rag/rag_eval_gpt41.json, outputs/rag/rag_eval_gpt5.json (quick_eval outputs against RAG snapshots)
  - temp_jsons_rag_gpt41/_run_summary.json, temp_jsons_rag_gpt5/_run_summary.json
    (per-query latency from the RAG runs)

Outputs (printed + written to docs/rag_vs_vista_report.md):
  - Per-variable mean score + % correct, 3 columns (VISTA / RAG-gpt4.1 / RAG-gpt5)
  - Overall numbers
  - Per-patient paired deltas
  - Latency summary
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load_pids() -> list[str]:
    return list(json.load(open("clinician_validation/_pid_mapping.json")).values())


def load_results(path: str) -> list[dict]:
    data = json.load(open(path))
    return data.get("results", data) if isinstance(data, dict) else data


def aggregate(results: list[dict], pids_set: set[str]) -> dict:
    """Group by variable; compute mean score and % correct within the pid set."""
    per_var = defaultdict(list)
    for r in results:
        if r["patient_id"] not in pids_set:
            continue
        per_var[r["variable"]].append(r)
    summary = {}
    for var, rs in per_var.items():
        summary[var] = {
            "n": len(rs),
            "mean_score": mean(r["score"] for r in rs) if rs else 0.0,
            "pct_correct": 100.0 * sum(1 for r in rs if r["correctness"] == "Correct") / len(rs) if rs else 0.0,
        }
    return summary


def overall(results: list[dict], pids_set: set[str]) -> dict:
    rs = [r for r in results if r["patient_id"] in pids_set]
    return {
        "n": len(rs),
        "mean_score": mean(r["score"] for r in rs) if rs else 0.0,
        "pct_correct": 100.0 * sum(1 for r in rs if r["correctness"] == "Correct") / len(rs) if rs else 0.0,
    }


def per_patient(results: list[dict], pids_set: set[str]) -> dict:
    per_pid = defaultdict(list)
    for r in results:
        if r["patient_id"] in pids_set:
            per_pid[r["patient_id"]].append(r["score"])
    return {pid: mean(scores) for pid, scores in per_pid.items()}


def latency_summary(snapshot_dir: str) -> dict | None:
    """Aggregate per-query and per-patient latency from all {pid}/rag_meta.json
    files in the snapshot dir. Robust to multiple retry runs (run_summary.json
    only reflects the last run)."""
    root = Path(snapshot_dir)
    if not root.exists():
        return None
    per_patient = []
    per_query = []
    model = None
    for pid_dir in root.iterdir():
        meta_path = pid_dir / "rag_meta.json"
        if not meta_path.exists():
            continue
        m = json.load(open(meta_path))
        model = model or m.get("answer_model")
        if "total_latency_sec" in m:
            per_patient.append(m["total_latency_sec"])
        if "wall_time_sec" in m:
            pass
        for v in (m.get("per_query_latency") or {}).values():
            per_query.append(v)
    return {
        "answer_model": model,
        "n_patients": len(per_patient),
        "mean_patient_sec": mean(per_patient) if per_patient else 0.0,
        "mean_query_sec": mean(per_query) if per_query else 0.0,
    }


VARIABLE_ORDER = [
    "Date of Birth", "Sex", "Smoking Status",
    "Diagnosis", "Histology", "Metastasis", "Lymph Node Involvement", "Genetic Testing Panel",
    "ECOG Performance Status", "Therapy Toxicity / Comorbidities",
    "Previous Surgery", "Current Medical Therapy", "Radiation Therapy",
    "Date of Last CT",
    "Allergies", "DNR",
]


def format_row(var: str, vista: dict, rag41: dict, rag5: dict) -> str:
    def cell(s: dict | None) -> str:
        if not s or not s.get("n"):
            return f"{'—':>12}"
        return f"{s['mean_score']:5.2f} ({s['pct_correct']:5.1f}%)"
    return f"  {var:34s}  {cell(vista):>14s}  {cell(rag41):>14s}  {cell(rag5):>14s}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vista-results", default="outputs/evals/eval_v2_1180_merged_results.json")
    parser.add_argument("--rag-gpt41-results", default="outputs/rag/rag_eval_gpt41.json")
    parser.add_argument("--rag-gpt5-results", default="outputs/rag/rag_eval_gpt5.json")
    parser.add_argument("--rag-gpt41-dir", default="temp_jsons_rag_gpt41")
    parser.add_argument("--rag-gpt5-dir", default="temp_jsons_rag_gpt5")
    parser.add_argument("--output", default="docs/rag_vs_vista_report.md")
    args = parser.parse_args()

    pids = load_pids()
    pids_set = set(pids)

    vista = load_results(args.vista_results)
    vista_var = aggregate(vista, pids_set)
    vista_overall = overall(vista, pids_set)
    vista_per_pid = per_patient(vista, pids_set)

    rag41 = load_results(args.rag_gpt41_results) if Path(args.rag_gpt41_results).exists() else []
    rag41_var = aggregate(rag41, pids_set) if rag41 else {}
    rag41_overall = overall(rag41, pids_set) if rag41 else {}
    rag41_per_pid = per_patient(rag41, pids_set) if rag41 else {}
    lat41 = latency_summary(args.rag_gpt41_dir)

    rag5 = load_results(args.rag_gpt5_results) if Path(args.rag_gpt5_results).exists() else []
    rag5_var = aggregate(rag5, pids_set) if rag5 else {}
    rag5_overall = overall(rag5, pids_set) if rag5 else {}
    rag5_per_pid = per_patient(rag5, pids_set) if rag5 else {}
    lat5 = latency_summary(args.rag_gpt5_dir)

    lines = []
    def p(s=""):
        print(s); lines.append(s)

    p(f"# RAG Baseline vs VISTA Architect — 30 clinician-validation patients")
    p()
    p(f"Pids: {len(pids)} | Variables: 16 | Judge: gpt-5 | Ground truth: TB-truncated XML")
    p()
    p("## Per-variable (mean score 0–10, % correct)")
    p()
    p("```")
    p(f"  {'Variable':34s}  {'VISTA':>14s}  {'RAG gpt-4.1':>14s}  {'RAG gpt-5':>14s}")
    p(f"  {'-'*34}  {'-'*14}  {'-'*14}  {'-'*14}")
    for v in VARIABLE_ORDER:
        p(format_row(v, vista_var.get(v), rag41_var.get(v), rag5_var.get(v)))
    p(f"  {'-'*34}  {'-'*14}  {'-'*14}  {'-'*14}")
    def fmt_overall(s):
        if not s.get("n"):
            return f"{'—':>14}"
        return f"{s['mean_score']:5.2f} ({s['pct_correct']:5.1f}%)"
    p(f"  {'OVERALL':34s}  {fmt_overall(vista_overall):>14s}  {fmt_overall(rag41_overall):>14s}  {fmt_overall(rag5_overall):>14s}")
    p("```")
    p()

    p("## Per-patient mean score")
    p()
    p("```")
    p(f"  {'PID':12s}  {'VISTA':>7s}  {'RAG4.1':>7s}  {'RAG5':>7s}   Δ(V-R4.1)  Δ(V-R5)")
    for pid in pids:
        v = vista_per_pid.get(pid, 0.0)
        r41 = rag41_per_pid.get(pid, 0.0)
        r5 = rag5_per_pid.get(pid, 0.0)
        d41 = v - r41 if r41 else None
        d5 = v - r5 if r5 else None
        p(f"  {pid:12s}  {v:7.2f}  {r41:7.2f}  {r5:7.2f}   {(d41 if d41 is not None else 0):+7.2f}    {(d5 if d5 is not None else 0):+7.2f}")
    p("```")
    p()

    p("## Latency")
    p()
    p("```")
    for tag, lat in (("RAG gpt-4.1", lat41), ("RAG gpt-5", lat5)):
        if lat:
            p(f"  {tag}: {lat['mean_query_sec']:.2f}s/query  |  {lat['mean_patient_sec']:.1f}s/patient  "
              f"(aggregated over {lat['n_patients']} per-patient rag_meta.json files)")
        else:
            p(f"  {tag}: (no meta files)")
    p("```")
    p()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    p(f"Wrote {out}")


if __name__ == "__main__":
    main()
