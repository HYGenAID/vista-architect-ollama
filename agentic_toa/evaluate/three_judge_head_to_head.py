#!/usr/bin/env python3
"""3-judge × 2-pipeline head-to-head on the test30 (clinician_validation_v2) cohort.

For each (case, variable):
  - manuscript_extracted: VISTA-extracted value from the case_NN.html
  - agentic_extracted:    same-PID extraction from agentic_toa/runs/<pid>/patient_info.json
                          (collected via quick_eval.extract_variables_from_snapshot)
  - ground_truth:          one-line GT from the case_NN.html (originally GPT-5-derived)

Each of the two extractions is then scored by each of N judges using the SAME prompt,
giving a clean 3-judge × 2-pipeline matrix. The GPT-5 manuscript scores are also
retained so we can anchor everything to them.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import gsgpt
import quick_eval as qe

HTML_DIR = ROOT / "clinician_validation_v2"
RUNS = ROOT / "agentic_toa" / "runs"

PROMPT = """You are scoring a single tumor-board variable extraction against EHR ground truth.

VARIABLE: {variable}

EXTRACTED VALUE:
{extracted}

EHR GROUND TRUTH (one-line summary from the chart):
{ground_truth}

Rate the extraction on a 1-10 scale where:
- 10 = factually correct, matches the ground truth (style/format differences are NOT deductions; "NKDA" = "No Known Allergies" = "None"; YYYY-MM and YYYY-MM-DD are equivalent when day is genuinely uncertain; honest "Unknown" when not documented = 10; verbose accurate = same as terse accurate; for `previous_treatments_list:` style values containing drugs+surgeries+radiation, score Previous Surgery only by whether an oncologic resection for the current diagnosis is correctly present or absent)
- 5 = partial / wrong direction (e.g., "No" when ground truth shows "Yes")
- 1 = wrong / fabricated

Judge ONLY the truth value. Do NOT deduct for stylistic preferences, list ordering, verbosity, or completeness beyond what the ground truth requires.

Return JSON only:
{{"score": <1-10>, "rationale": "<one sentence>"}}
"""


def parse_case(html_path: Path) -> dict[str, dict]:
    """Return {variable: {extracted, ground_truth, gpt5_score}} for one case."""
    soup = BeautifulSoup(html_path.read_text(), "html.parser")
    out = {}
    for card in soup.find_all("div", class_="variable-card"):
        vn = card.find("span", class_="var-name")
        badge = card.find("span", class_="score-badge")
        if not vn or not badge: continue
        m = re.match(r"(\d+)/10", badge.get_text(strip=True))
        if not m: continue
        var = vn.get_text(strip=True)
        extracted = ground_truth = ""
        for fr in card.find_all("div", class_="field-row"):
            l = fr.find("span", class_="field-label")
            v = fr.find("span", class_="field-value")
            if not l or not v: continue
            lt = l.get_text(strip=True)
            vt = v.get_text(" ", strip=True)
            if "VISTA Extracted" in lt: extracted = vt
            elif "Ground Truth" in lt: ground_truth = vt
        out[var] = {
            "extracted": extracted,
            "ground_truth": ground_truth,
            "gpt5_score": int(m.group(1)),
        }
    return out


def score_one(pid: str, var: str, source: str, extracted: str, ground_truth: str,
              model: str, thinking: int | None) -> dict:
    prompt = PROMPT.format(
        variable=var, extracted=extracted or "(empty)",
        ground_truth=ground_truth or "(empty)",
    )
    try:
        kwargs = {"max_tokens": 1024}
        if thinking is not None: kwargs["thinking_budget"] = thinking
        resp = gsgpt.chat(prompt, model=model, **kwargs)
        text = resp.strip()
        if "```json" in text: text = text.split("```json",1)[1].split("```",1)[0].strip()
        elif text.startswith("```"): text = text.split("```",1)[1].split("```",1)[0].strip()
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            j = json.loads(m.group(0))
            return {"pid": pid, "variable": var, "source": source, "model": model,
                    "score": int(j.get("score", 0)),
                    "rationale": str(j.get("rationale", ""))[:200], "error": None}
        return {"pid": pid, "variable": var, "source": source, "model": model,
                "score": None, "rationale": "", "error": "no JSON"}
    except Exception as e:
        return {"pid": pid, "variable": var, "source": source, "model": model,
                "score": None, "rationale": "", "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+",
                    default=["claude-opus-4-6", "gemini-3.5-flash"],
                    help="judge models to run")
    ap.add_argument("--thinking", type=int, default=1024,
                    help="thinking_budget for Gemini Flash variants")
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--output", default="outputs/calibration/three_judge_h2h.json")
    args = ap.parse_args()

    # Build {case_id -> pid} mapping
    pid_map = json.load(open(ROOT / "clinician_validation_v2" / "_pid_mapping.json"))

    tasks = []  # list of (pid, var, source, extracted, ground_truth)
    for case_path in sorted(HTML_DIR.glob("case_*.html")):
        case_id = "Case " + case_path.stem.split("_")[1]
        pid = pid_map.get(case_id)
        if not pid: continue
        manu = parse_case(case_path)

        # Agentic extractions for this pid
        json_dir = RUNS / pid
        if not (json_dir / "patient_info.json").exists():
            print(f"WARN: no agentic patient_info for {pid}", file=sys.stderr)
            agentic_vals = {}
        else:
            try:
                extracted_v2 = qe.extract_variables_from_snapshot(pid, json_dir, no_graph_fallbacks=True)
                agentic_vals = extracted_v2
            except Exception as e:
                print(f"WARN: agentic extract failed for {pid}: {e}", file=sys.stderr)
                agentic_vals = {}

        for var, info in manu.items():
            tasks.append((pid, var, "manuscript", info["extracted"], info["ground_truth"], info["gpt5_score"]))
            ag_val = agentic_vals.get(var, "")
            tasks.append((pid, var, "agentic", str(ag_val), info["ground_truth"], None))

    print(f"Built {len(tasks)} (pid, variable, source) tasks for {len(args.judges)} judges = {len(tasks)*len(args.judges)} judge calls", file=sys.stderr)

    # Launch
    rows = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = []
        for pid, var, source, ext, gt, gpt5 in tasks:
            for model in args.judges:
                thinking = args.thinking if ("gemini" in model.lower() and "flash" in model.lower()) else None
                futures.append((pool.submit(score_one, pid, var, source, ext, gt, model, thinking), gpt5))
        for i, (fut, gpt5) in enumerate(as_completed([f[0] for f in futures]) and futures, 1):
            fut_obj, gpt5_val = fut, gpt5
            res = fut_obj.result()
            res["gpt5_anchor"] = gpt5_val
            rows.append(res)
            if i % 100 == 0:
                print(f"  {i}/{len(futures)} done", file=sys.stderr)

    out_path = ROOT / args.output
    out_path.write_text(json.dumps(rows, indent=2))

    # Aggregate
    from collections import defaultdict
    agg = defaultdict(list)  # (source, model) -> [scores]
    per_var = defaultdict(list)  # (source, model, var) -> [scores]
    gpt5_manu_scores = []  # for the manuscript-GPT-5 anchor
    for r in rows:
        if r["score"] is None: continue
        agg[(r["source"], r["model"])].append(r["score"])
        per_var[(r["source"], r["model"], r["variable"])].append(r["score"])
        if r["source"] == "manuscript" and r.get("gpt5_anchor") is not None:
            gpt5_manu_scores.append(r["gpt5_anchor"])

    # Deduplicate gpt5_manu (because each manuscript row is sent to multiple judges)
    seen = set()
    uniq_gpt5 = []
    for r in rows:
        if r["source"] == "manuscript" and r.get("gpt5_anchor") is not None:
            k = (r["pid"], r["variable"])
            if k not in seen:
                seen.add(k); uniq_gpt5.append(r["gpt5_anchor"])
    gpt5_manu_mean = sum(uniq_gpt5)/len(uniq_gpt5) if uniq_gpt5 else 0

    print()
    print("="*70)
    print("3-JUDGE × 2-PIPELINE HEAD-TO-HEAD (test30 / clinician_validation_v2)")
    print("Same prompt, same ground truths, same patients.")
    print("="*70)
    print()
    judges_used = args.judges
    print(f"{'Judge':<22} {'Manuscript':>11} {'Agentic':>10} {'Agentic−Manu':>14}")
    print("-"*60)
    # GPT-5 anchor
    print(f"{'GPT-5 (manuscript)':<22} {gpt5_manu_mean:>11.2f} {'—':>10} {'—':>14}")
    for model in judges_used:
        manu = sum(agg[('manuscript', model)])/len(agg[('manuscript', model)]) if agg[('manuscript', model)] else 0
        ag   = sum(agg[('agentic', model)])/len(agg[('agentic', model)]) if agg[('agentic', model)] else 0
        n_m = len(agg[('manuscript', model)]); n_a = len(agg[('agentic', model)])
        print(f"{model:<22} {manu:>11.2f} {ag:>10.2f} {(ag-manu):>+14.2f}   (n_manu={n_m}, n_ag={n_a})")

    print()
    print(f"=== Per-variable, agentic−manuscript Δ under each judge ===")
    print()
    all_vars = sorted({k[2] for k in per_var})
    cols = " ".join(f"{m[:10]:>10}" for m in judges_used)
    print(f"{'Variable':<32}  {cols}")
    print("-" * (32 + 2 + 11 * len(judges_used)))
    for var in all_vars:
        deltas = []
        for model in judges_used:
            manu_s = per_var.get(('manuscript', model, var), [])
            ag_s   = per_var.get(('agentic', model, var), [])
            if manu_s and ag_s:
                d = sum(ag_s)/len(ag_s) - sum(manu_s)/len(manu_s)
                deltas.append(f"{d:>+10.2f}")
            else:
                deltas.append(f"{'(n/a)':>10}")
        print(f"{var:<32}  " + " ".join(deltas))

    print(f"\nFull data: {out_path}")


if __name__ == "__main__":
    main()
