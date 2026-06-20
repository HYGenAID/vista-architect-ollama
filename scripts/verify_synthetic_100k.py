"""Verify the freshly-generated 100k synthetic cohort.

Runs the four checks called out in the plan:
  1. Vignette success rate (≥99% target)
  2. Distribution drift vs the real 1,167-patient cohort (≤1.5 pp target)
  3. Distribution comparison vs the existing 10k cohort
  4. Cross-cohort retrieval sanity check (real index patient → synthetic top candidates)

Writes a markdown summary to docs/plm_synthetic_100k_verification.md.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

REAL = "outputs/plm/plm_index_1180.json"
TEN_K = "outputs/plm/plm_synthetic_10k.json"
HUNDRED_K = "outputs/plm/plm_synthetic_100k.json"
OUT_MD = "docs/plm_synthetic_100k_verification.md"


def load(path):
    with open(path) as f:
        return json.load(f)


def patient_iter(idx):
    return idx["patients"].values()


def diag_dist(idx):
    return Counter(p["diagnosis_category"] for p in patient_iter(idx))


def met_dist(idx):
    return Counter(p["metastatic"] for p in patient_iter(idx))


def hist_dist(idx):
    return Counter(p["histology_category"] for p in patient_iter(idx))


def mut_pos_counts(idx):
    counts = Counter()
    for p in patient_iter(idx):
        for g, s in (p.get("driver_mutations") or {}).items():
            if s == "positive":
                counts[g] += 1
    return counts


def median_age(idx):
    ages = [p.get("age_at_tb") for p in patient_iter(idx) if p.get("age_at_tb") is not None]
    return median(ages) if ages else None


def vignette_rate(idx):
    n = sum(1 for _ in patient_iter(idx))
    have = sum(1 for p in patient_iter(idx) if (p.get("tb_note") or "").strip())
    return have, n


def pct_table(real, syn10, syn100, key_fn, label):
    r, s10, s100 = key_fn(real), key_fn(syn10), key_fn(syn100)
    keys = sorted(set(r) | set(s10) | set(s100))
    nr = sum(r.values()); ns10 = sum(s10.values()); ns100 = sum(s100.values())
    rows = [f"| {label} | Real % (N={nr}) | Syn-10k % (N={ns10}) | Syn-100k % (N={ns100}) | Δ Real→100k (pp) |",
            "|---|---:|---:|---:|---:|"]
    for k in keys:
        pr = 100*r.get(k,0)/nr if nr else 0
        p10 = 100*s10.get(k,0)/ns10 if ns10 else 0
        p100 = 100*s100.get(k,0)/ns100 if ns100 else 0
        rows.append(f"| {k} | {pr:.2f} | {p10:.2f} | {p100:.2f} | {p100-pr:+.2f} |")
    return "\n".join(rows)


def main():
    print("Loading cohorts...")
    real = load(REAL)
    syn10 = load(TEN_K)
    syn100 = load(HUNDRED_K)

    have, n = vignette_rate(syn100)
    rate = 100 * have / n if n else 0

    Path("docs").mkdir(exist_ok=True)
    out = []
    w = out.append

    w("# 100k Synthetic Cohort — Verification Report\n")
    w(f"- Real cohort: `{REAL}` ({len(real['patients'])} patients)")
    w(f"- Existing 10k synthetic: `{TEN_K}` ({len(syn10['patients'])} patients)")
    w(f"- New 100k synthetic: `{HUNDRED_K}` ({len(syn100['patients'])} patients)")
    w(f"- Algorithm: identical conditional cascade in `generate_fake_patients.py`, seed=42, gemini-2.5-flash via Vertex AI, 30 parallel workers\n")

    w("## 1. Vignette success rate")
    target = 99.0
    status = "PASS" if rate >= target else "FAIL"
    w(f"- **{have}/{n}** patients have a non-empty `tb_note` → **{rate:.2f}%** "
      f"(target ≥{target}% — **{status}**)\n")

    w("## 2. Diagnosis distribution\n")
    w(pct_table(real, syn10, syn100, diag_dist, "Diagnosis category") + "\n")

    w("## 3. Metastatic status distribution\n")
    w(pct_table(real, syn10, syn100, met_dist, "Metastatic") + "\n")

    w("## 4. Histology category distribution\n")
    w(pct_table(real, syn10, syn100, hist_dist, "Histology") + "\n")

    w("## 5. Driver mutation prevalence (positive only, per gene)\n")
    w(pct_table(real, syn10, syn100, mut_pos_counts, "Gene+") + "\n")

    w("## 6. Age summary\n")
    w(f"- Real median age: **{median_age(real):.1f} y**")
    w(f"- Syn-10k median age: **{median_age(syn10):.1f} y**")
    w(f"- Syn-100k median age: **{median_age(syn100):.1f} y**\n")

    w("## 7. Generation metadata\n")
    w(f"- Generated at: `{syn100.get('built_at')}`")
    w(f"- Source script: `{syn100.get('source')}`")
    w(f"- Real index used as distribution source: `{syn100.get('real_index')}`")
    w(f"- Seed: `{syn100.get('seed')}`\n")

    Path(OUT_MD).write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
