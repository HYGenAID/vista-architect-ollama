#!/usr/bin/env python3
"""Opus 4.6 vs GPT-5 judge calibration.

Parses the manuscript's clinician_validation_v2 HTML files (which carry the
GPT-5 judge scores + reasoning + VISTA extracted + XML ground truth for each
variable) and asks Opus 4.6 to score the same (extracted, ground_truth) pair
independently. The aggregate Opus-vs-GPT-5 per-variable delta tells us whether
the 0.14-point gap we see on test30 (Opus 9.62 vs manuscript GPT-5 9.76) is
real pipeline quality or just judge-strictness on the same outputs.

Per item, Opus is asked to produce a 1-10 score and a one-sentence rationale.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import gsgpt

HTML_DIR = ROOT / "clinician_validation_v2"


def parse_case(html_path: Path) -> list[dict]:
    soup = BeautifulSoup(html_path.read_text(), "html.parser")
    rows = []
    for card in soup.find_all("div", class_="variable-card"):
        var = card.find("span", class_="var-name")
        badge = card.find("span", class_="score-badge")
        if not var or not badge:
            continue
        score_text = badge.get_text(strip=True)
        # Format: "10/10 — Correct"
        m = re.match(r"(\d+)/10", score_text)
        if not m:
            continue
        gpt5_score = int(m.group(1))
        gpt5_label = score_text.split("—")[-1].strip() if "—" in score_text else ""

        # VISTA Extracted and Ground Truth are field-rows
        extracted, ground_truth = "", ""
        for fr in card.find_all("div", class_="field-row"):
            label = fr.find("span", class_="field-label")
            value = fr.find("span", class_="field-value")
            if not label or not value:
                continue
            ltxt = label.get_text(strip=True)
            vtxt = value.get_text(" ", strip=True)
            if "VISTA Extracted" in ltxt:
                extracted = vtxt
            elif "Ground Truth" in ltxt:
                ground_truth = vtxt

        # GPT-5 reasoning text
        reasoning = ""
        rsec = card.find("div", class_="reasoning-section")
        if rsec:
            full = rsec.get_text(" ", strip=True)
            reasoning = full.replace("LLM Judge Reasoning:", "").strip()

        rows.append({
            "case": html_path.stem,
            "variable": var.get_text(strip=True),
            "gpt5_score": gpt5_score,
            "gpt5_label": gpt5_label,
            "gpt5_reasoning": reasoning,
            "extracted": extracted,
            "ground_truth": ground_truth,
        })
    return rows


OPUS_JUDGE_PROMPT = """You are scoring a single tumor-board variable extraction against EHR ground truth.

VARIABLE: {variable}

VISTA EXTRACTED VALUE:
{extracted}

EHR GROUND TRUTH (one-line summary from the chart):
{ground_truth}

Rate the extraction on a 1-10 scale where:
- 10 = factually correct, matches the ground truth (style/format differences are NOT deductions; "NKDA" = "No Known Allergies" = "None"; YYYY-MM and YYYY-MM-DD are equivalent when day is genuinely uncertain; honest "Unknown" when not documented = 10; verbose accurate = same as terse accurate)
- 5 = partial / wrong direction (e.g., "No" when ground truth shows "Yes")
- 1 = wrong / fabricated

Judge ONLY the truth value. Do NOT deduct for stylistic preferences, list ordering, verbosity, or completeness beyond what the ground truth requires.

Return JSON only:
{{"score": <1-10>, "rationale": "<one sentence>"}}
"""


def score_with_judge(row: dict, model: str, thinking: int | None = None) -> dict:
    prompt = OPUS_JUDGE_PROMPT.format(
        variable=row["variable"],
        extracted=row["extracted"] or "(empty)",
        ground_truth=row["ground_truth"] or "(empty)",
    )
    try:
        kwargs = {"max_tokens": 1024}
        # Gemini Flash variants need explicit thinking_budget for reasoning
        if thinking is not None:
            kwargs["thinking_budget"] = thinking
        resp = gsgpt.chat(prompt, model=model, **kwargs)
        text = resp.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif text.startswith("```"):
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            j = json.loads(m.group(0))
            return {
                **row,
                "opus_score": int(j.get("score", 0)),
                "opus_rationale": str(j.get("rationale", "")),
                "error": None,
            }
        return {**row, "opus_score": None, "opus_rationale": "", "error": "no JSON in response"}
    except Exception as e:
        return {**row, "opus_score": None, "opus_rationale": "", "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/calibration/opus_vs_gpt5_calibration.json")
    ap.add_argument("--max-cases", type=int, default=30)
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--judge-model", default="claude-opus-4-6", help="Judge model to compare against GPT-5")
    ap.add_argument("--thinking", type=int, default=None, help="Thinking budget (Gemini Flash variants need explicit nonzero)")
    args = ap.parse_args()

    cases = sorted(HTML_DIR.glob("case_*.html"))[: args.max_cases]
    all_rows = []
    for c in cases:
        all_rows.extend(parse_case(c))
    print(f"Parsed {len(all_rows)} GPT-5 judgments from {len(cases)} cases", file=sys.stderr)

    scored = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(score_with_judge, r, args.judge_model, args.thinking) for r in all_rows]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            scored.append(res)
            if i % 25 == 0:
                print(f"  {i}/{len(futures)} done", file=sys.stderr)

    # Aggregate
    from collections import defaultdict
    per_var_gpt5 = defaultdict(list)
    per_var_opus = defaultdict(list)
    deltas = []
    for r in scored:
        if r["opus_score"] is None:
            continue
        per_var_gpt5[r["variable"]].append(r["gpt5_score"])
        per_var_opus[r["variable"]].append(r["opus_score"])
        deltas.append(r["opus_score"] - r["gpt5_score"])

    summary = {
        "n_rows": len(scored),
        "n_failed": sum(1 for r in scored if r["opus_score"] is None),
        "overall": {
            "gpt5_mean": sum(r["gpt5_score"] for r in scored if r["opus_score"] is not None) / max(1, len([r for r in scored if r["opus_score"] is not None])),
            "opus_mean": sum(r["opus_score"] for r in scored if r["opus_score"] is not None) / max(1, len([r for r in scored if r["opus_score"] is not None])),
            "mean_delta": sum(deltas) / max(1, len(deltas)),
            "pct_agree": sum(1 for d in deltas if d == 0) / max(1, len(deltas)) * 100,
            "pct_opus_lower": sum(1 for d in deltas if d < 0) / max(1, len(deltas)) * 100,
            "pct_opus_higher": sum(1 for d in deltas if d > 0) / max(1, len(deltas)) * 100,
        },
        "per_variable": {
            var: {
                "n": len(per_var_gpt5[var]),
                "gpt5_mean": sum(per_var_gpt5[var]) / len(per_var_gpt5[var]),
                "opus_mean": sum(per_var_opus[var]) / len(per_var_opus[var]),
                "delta": (sum(per_var_opus[var]) - sum(per_var_gpt5[var])) / len(per_var_gpt5[var]),
            }
            for var in per_var_gpt5
        },
        "rows": scored,
    }
    out = ROOT / args.output
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}", file=sys.stderr)

    # Print summary
    print("\n=== OPUS vs GPT-5 JUDGE CALIBRATION (on manuscript outputs) ===\n")
    print(f"n = {summary['n_rows']} items ({summary['n_failed']} parse failures)\n")
    o = summary["overall"]
    print(f"  GPT-5 mean : {o['gpt5_mean']:.2f}")
    print(f"  Opus  mean : {o['opus_mean']:.2f}")
    print(f"  Δ          : {o['mean_delta']:+.2f}   ({'Opus stricter' if o['mean_delta']<0 else 'Opus more lenient' if o['mean_delta']>0 else 'tied'})")
    print(f"  Agreement  : {o['pct_agree']:.0f}%   |   Opus lower: {o['pct_opus_lower']:.0f}%   |   Opus higher: {o['pct_opus_higher']:.0f}%")

    print(f"\n=== Per-variable delta ===\n")
    print(f"{'Variable':<32} {'n':>3} {'GPT-5':>6} {'Opus':>6} {'Δ':>6}")
    print("-" * 60)
    for var, v in sorted(summary["per_variable"].items(), key=lambda kv: kv[1]["delta"]):
        print(f"{var:<32} {v['n']:>3} {v['gpt5_mean']:>6.2f} {v['opus_mean']:>6.2f} {v['delta']:>+6.2f}")


if __name__ == "__main__":
    main()
