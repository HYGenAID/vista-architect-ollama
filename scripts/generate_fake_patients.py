#!/usr/bin/env python3
"""
Generate synthetic patients for PLM pipeline development and evaluation.

Samples patient structures from conditional distributions learned from
the real PLM index, then generates clinical vignettes via LLM.

Phase II MVP of the PLM ML roadmap (docs/plm/plm.tex).

Usage:
    python generate_fake_patients.py --real-index outputs/plm/plm_index_1180.json --n 400
    python generate_fake_patients.py --real-index outputs/plm/plm_index_1180.json --n 400 --skip-vignettes
    python generate_fake_patients.py --real-index outputs/plm/plm_index_1180.json --n 400 --seed 42
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import gsgpt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIONABLE_GENES = [
    "EGFR", "ALK", "KRAS", "ROS1", "BRAF", "MET", "RET", "NTRK",
    "ERBB2 (HER2)", "NRG1",
]

# Mutually exclusive driver mutations in NSCLC
EXCLUSIVE_DRIVERS = {"EGFR", "ALK", "KRAS", "ROS1", "BRAF", "RET", "NTRK", "NRG1"}

DRUG_CLASSES_BY_MUTATION = {
    "EGFR": ["egfr_tki", "platinum_doublet", "immunotherapy", "pemetrexed"],
    "ALK": ["alk_tki", "platinum_doublet", "immunotherapy", "pemetrexed"],
    "KRAS": ["kras_inhibitor", "immunotherapy", "platinum_doublet", "taxane"],
    "ROS1": ["ros1_inhibitor", "platinum_doublet", "immunotherapy"],
    "BRAF": ["braf_targeted", "immunotherapy", "platinum_doublet"],
    "MET": ["met_inhibitor", "immunotherapy", "platinum_doublet"],
    "RET": ["ret_inhibitor", "platinum_doublet", "immunotherapy"],
    "NTRK": ["platinum_doublet", "immunotherapy"],
    "ERBB2 (HER2)": ["her2_targeted", "platinum_doublet", "immunotherapy"],
    "none": ["immunotherapy", "platinum_doublet", "taxane", "pemetrexed", "antiangiogenic"],
}

ALL_DRUG_CLASSES = [
    "platinum_doublet", "immunotherapy", "egfr_tki", "alk_tki",
    "kras_inhibitor", "taxane", "antiangiogenic", "pemetrexed",
    "ret_inhibitor", "ros1_inhibitor", "met_inhibitor",
    "her2_targeted", "braf_targeted", "etoposide", "gemcitabine",
]

INTENT_OPTIONS = ["curative", "neoadjuvant", "adjuvant", "palliative", "definitive"]
TERMINATION_REASONS = ["progression", "toxicity", "completed", "patient_choice", "unknown"]


# ---------------------------------------------------------------------------
# Extract conditional distributions from real cohort
# ---------------------------------------------------------------------------

def extract_distributions(plm_index: dict) -> dict:
    """Extract all conditional frequency tables from the real PLM index."""
    patients = list(plm_index["patients"].values())
    dist = {}

    # 1. Diagnosis category (marginal)
    dist["diagnosis"] = dict(Counter(p["diagnosis_category"] for p in patients))

    # 2. Histology | diagnosis
    dist["histology_given_diagnosis"] = {}
    for diag in set(p["diagnosis_category"] for p in patients):
        subset = [p for p in patients if p["diagnosis_category"] == diag]
        dist["histology_given_diagnosis"][diag] = dict(
            Counter(p["histology_category"] for p in subset)
        )

    # 3. Metastatic | diagnosis
    dist["metastatic_given_diagnosis"] = {}
    for diag in set(p["diagnosis_category"] for p in patients):
        subset = [p for p in patients if p["diagnosis_category"] == diag]
        dist["metastatic_given_diagnosis"][diag] = dict(
            Counter(p["metastatic"] for p in subset)
        )

    # 4. Lymph node | diagnosis, metastatic
    dist["ln_given_diag_met"] = {}
    for p in patients:
        key = f"{p['diagnosis_category']}_{p['metastatic']}"
        dist["ln_given_diag_met"].setdefault(key, []).append(p.get("lymph_node", "Unknown"))
    dist["ln_given_diag_met"] = {k: dict(Counter(v)) for k, v in dist["ln_given_diag_met"].items()}

    # 5. Mutation rates | diagnosis (only NSCLC matters for most)
    dist["mutation_rates"] = {}
    for diag in set(p["diagnosis_category"] for p in patients):
        subset = [p for p in patients if p["diagnosis_category"] == diag]
        rates = {}
        for gene in ACTIONABLE_GENES:
            positive = sum(1 for p in subset if p.get("driver_mutations", {}).get(gene) == "positive")
            tested = sum(1 for p in subset if p.get("driver_mutations", {}).get(gene) in ("positive", "negative"))
            rates[gene] = {
                "positive_rate": positive / max(tested, 1),
                "tested_rate": tested / max(len(subset), 1),
                "n_positive": positive,
                "n_tested": tested,
            }
        dist["mutation_rates"][diag] = rates

    # 6. PD-L1 | diagnosis
    dist["pdl1_given_diagnosis"] = {}
    for diag in set(p["diagnosis_category"] for p in patients):
        subset = [p for p in patients if p["diagnosis_category"] == diag]
        dist["pdl1_given_diagnosis"][diag] = dict(
            Counter(p.get("pdl1_status", "unknown") for p in subset)
        )

    # 7. Sex | diagnosis
    dist["sex_given_diagnosis"] = {}
    for diag in set(p["diagnosis_category"] for p in patients):
        subset = [p for p in patients if p["diagnosis_category"] == diag]
        dist["sex_given_diagnosis"][diag] = dict(
            Counter(p.get("sex", "Unknown") for p in subset)
        )

    # 8. Age stats | diagnosis
    dist["age_stats"] = {}
    for diag in set(p["diagnosis_category"] for p in patients):
        subset = [p for p in patients if p["diagnosis_category"] == diag]
        ages = [p["age_at_tb"] for p in subset if p.get("age_at_tb")]
        if ages:
            dist["age_stats"][diag] = {
                "mean": float(np.mean(ages)),
                "std": float(np.std(ages)),
                "min": float(np.min(ages)),
                "max": float(np.max(ages)),
            }

    # 9. Smoking | mutation profile
    # EGFR+ patients are mostly never-smokers
    egfr_pos = [p for p in patients if p.get("driver_mutations", {}).get("EGFR") == "positive"]
    egfr_neg = [p for p in patients if p.get("driver_mutations", {}).get("EGFR") != "positive"]
    dist["smoking_given_egfr"] = {
        "egfr_positive": dict(Counter(p.get("smoking_category", "Unknown") for p in egfr_pos)),
        "egfr_other": dict(Counter(p.get("smoking_category", "Unknown") for p in egfr_neg)),
    }

    # 10. ECOG | metastatic
    dist["ecog_given_met"] = {}
    for met in ["Yes", "No", "Unknown"]:
        subset = [p for p in patients if p["metastatic"] == met]
        dist["ecog_given_met"][met] = dict(
            Counter(str(p.get("ecog", "Unknown")) for p in subset)
        )

    # 11. Treatment line | metastatic
    dist["treatment_line_given_met"] = {}
    for met in ["Yes", "No", "Unknown"]:
        subset = [p for p in patients if p["metastatic"] == met]
        lines = [p.get("current_line_number", 0) for p in subset]
        dist["treatment_line_given_met"][met] = dict(Counter(lines))

    # 12. Intent | line, metastatic
    dist["intent_given_line_met"] = {}
    for p in patients:
        line = min(p.get("current_line_number", 0), 3)  # cap at 3+
        met = p["metastatic"]
        key = f"line{line}_{met}"
        dist["intent_given_line_met"].setdefault(key, []).append(
            p.get("current_intent", "unknown")
        )
    dist["intent_given_line_met"] = {k: dict(Counter(v)) for k, v in dist["intent_given_line_met"].items()}

    # 13. Surgery/Radiation | metastatic
    for met in ["Yes", "No", "Unknown"]:
        subset = [p for p in patients if p["metastatic"] == met]
        if subset:
            dist.setdefault("surgery_rate", {})[met] = sum(1 for p in subset if p.get("had_surgery")) / len(subset)
            dist.setdefault("radiation_rate", {})[met] = sum(1 for p in subset if p.get("had_radiation")) / len(subset)

    # 14. Regimen classes | mutation, line
    dist["regimen_classes"] = defaultdict(list)
    for p in patients:
        primary_mut = "none"
        for gene in ["EGFR", "ALK", "KRAS", "ROS1", "BRAF", "MET", "RET"]:
            if p.get("driver_mutations", {}).get(gene) == "positive":
                primary_mut = gene
                break
        classes = p.get("regimen_classes", [])
        if classes:
            dist["regimen_classes"][primary_mut].append(classes)
    dist["regimen_classes"] = dict(dist["regimen_classes"])

    return dist


# ---------------------------------------------------------------------------
# Patient sampling
# ---------------------------------------------------------------------------

def _weighted_choice(freq_dict: dict, rng) -> str:
    """Sample from a frequency dictionary."""
    if not freq_dict:
        return "Unknown"
    keys = list(freq_dict.keys())
    weights = np.array([freq_dict[k] for k in keys], dtype=float)
    weights /= weights.sum()
    return keys[rng.choice(len(keys), p=weights)]


def sample_patient(dist: dict, patient_idx: int, rng, id_pad: int = 4) -> dict:
    """Sample a single synthetic patient from conditional distributions.

    `id_pad` controls zero-padding width of the SYN_ id (4 for the original
    10k cohort; use 6 for 100k so SYN_000001..SYN_100000 stays uniform).
    """
    p = {"patient_id": f"SYN_{patient_idx:0{id_pad}d}"}

    # 1. Diagnosis
    p["diagnosis_category"] = _weighted_choice(dist["diagnosis"], rng)
    diag = p["diagnosis_category"]

    # 2. Histology | diagnosis
    hist_dist = dist["histology_given_diagnosis"].get(diag, {"Other/NOS": 1})
    p["histology_category"] = _weighted_choice(hist_dist, rng)

    # 3. Metastatic | diagnosis
    met_dist = dist["metastatic_given_diagnosis"].get(diag, {"Unknown": 1})
    p["metastatic"] = _weighted_choice(met_dist, rng)

    # 4. Lymph node | diagnosis, metastatic
    ln_key = f"{diag}_{p['metastatic']}"
    ln_dist = dist["ln_given_diag_met"].get(ln_key, {"Unknown": 1})
    p["lymph_node"] = _weighted_choice(ln_dist, rng)

    # 5. Driver mutations (with mutual exclusivity)
    p["driver_mutations"] = {}
    p["positive_mutations"] = []

    if diag in dist["mutation_rates"]:
        rates = dist["mutation_rates"][diag]
        already_has_exclusive = False

        for gene in ACTIONABLE_GENES:
            gene_info = rates.get(gene, {"positive_rate": 0, "tested_rate": 0})

            # Is this gene tested?
            if rng.random() < gene_info["tested_rate"]:
                # Enforce mutual exclusivity for main drivers
                if gene in EXCLUSIVE_DRIVERS and already_has_exclusive:
                    p["driver_mutations"][gene] = "negative"
                elif rng.random() < gene_info["positive_rate"]:
                    p["driver_mutations"][gene] = "positive"
                    p["positive_mutations"].append(gene)
                    if gene in EXCLUSIVE_DRIVERS:
                        already_has_exclusive = True
                else:
                    p["driver_mutations"][gene] = "negative"
            else:
                p["driver_mutations"][gene] = "not_tested"
    else:
        for gene in ACTIONABLE_GENES:
            p["driver_mutations"][gene] = "not_tested"

    # 6. PD-L1
    pdl1_dist = dist["pdl1_given_diagnosis"].get(diag, {"unknown": 1})
    p["pdl1_status"] = _weighted_choice(pdl1_dist, rng)

    # 7. Sex
    sex_dist = dist["sex_given_diagnosis"].get(diag, {"M": 1, "F": 1})
    p["sex"] = _weighted_choice(sex_dist, rng)

    # 8. Age
    age_stats = dist["age_stats"].get(diag, {"mean": 70, "std": 12})
    age = rng.normal(age_stats["mean"], age_stats["std"])
    p["age_at_tb"] = round(float(np.clip(age, 20, 95)), 1)

    # 9. Smoking (conditioned on EGFR status)
    if "EGFR" in p["positive_mutations"]:
        smoke_dist = dist["smoking_given_egfr"].get("egfr_positive", {"Never": 1})
    else:
        smoke_dist = dist["smoking_given_egfr"].get("egfr_other", {"Former": 1})
    p["smoking_category"] = _weighted_choice(smoke_dist, rng)

    # 10. ECOG
    ecog_dist = dist["ecog_given_met"].get(p["metastatic"], {"1": 1})
    p["ecog"] = _weighted_choice(ecog_dist, rng)

    # 11. Treatment line | metastatic
    line_dist = dist["treatment_line_given_met"].get(p["metastatic"], {0: 1})
    # Convert keys to int for sampling
    line_keys = {int(k): v for k, v in line_dist.items()}
    p["current_line_number"] = int(_weighted_choice(line_keys, rng))
    p["treatment_line_count"] = max(1, p["current_line_number"])

    # 12. Intent | line, metastatic
    line_capped = min(p["current_line_number"], 3)
    intent_key = f"line{line_capped}_{p['metastatic']}"
    intent_dist = dist["intent_given_line_met"].get(intent_key, {"unknown": 1})
    p["current_intent"] = _weighted_choice(intent_dist, rng)

    # 13. Surgery and radiation | metastatic
    surgery_rate = dist.get("surgery_rate", {}).get(p["metastatic"], 0.5)
    radiation_rate = dist.get("radiation_rate", {}).get(p["metastatic"], 0.3)
    p["had_surgery"] = bool(rng.random() < surgery_rate)
    p["had_radiation"] = bool(rng.random() < radiation_rate)

    # 14. Regimen classes
    primary_mut = "none"
    for gene in ["EGFR", "ALK", "KRAS", "ROS1", "BRAF", "MET", "RET"]:
        if gene in p["positive_mutations"]:
            primary_mut = gene
            break

    # Sample from real regimen class combinations or use mutation-conditioned defaults
    real_regimens = dist.get("regimen_classes", {}).get(primary_mut, [])
    if real_regimens and p["current_line_number"] > 0:
        idx = rng.integers(0, len(real_regimens))
        p["regimen_classes"] = list(real_regimens[idx])
    elif p["current_line_number"] > 0:
        available = DRUG_CLASSES_BY_MUTATION.get(primary_mut, DRUG_CLASSES_BY_MUTATION["none"])
        n_classes = min(rng.integers(1, 4), len(available))
        p["regimen_classes"] = list(rng.choice(available, size=n_classes, replace=False))
    else:
        p["regimen_classes"] = []

    # 15. Termination reason
    if p["current_line_number"] > 0:
        term_weights = {"progression": 5, "toxicity": 2, "completed": 1, "unknown": 3}
        p["last_termination_reason"] = _weighted_choice(term_weights, rng)
    else:
        p["last_termination_reason"] = "unknown"

    # 16. Synthetic metadata
    p["tb_date"] = f"2025-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}"
    p["pack_years"] = 0 if p["smoking_category"] == "Never" else int(rng.integers(5, 60))
    p["event_count"] = int(rng.integers(100, 5000))
    p["node_count"] = int(rng.integers(500, 50000))
    p["record_span_years"] = round(float(rng.uniform(0.5, 10)), 1)

    # Placeholder for vignette (filled by LLM later)
    p["tb_note"] = ""
    p["tb_note_tokens"] = []

    return p


# ---------------------------------------------------------------------------
# LLM vignette generation
# ---------------------------------------------------------------------------

def _format_patient_for_vignette(p: dict) -> str:
    """Format a synthetic patient record for the vignette prompt."""
    mut_str = ", ".join(
        f"{g}+" if s == "positive" else f"{g}-"
        for g, s in p["driver_mutations"].items()
        if s != "not_tested"
    ) or "Not tested"

    regimens = ", ".join(p.get("regimen_classes", [])) or "None"

    lines = [
        f"  Diagnosis: {p['diagnosis_category']} {p['histology_category']}"
        f"{', Stage IV (metastatic)' if p['metastatic'] == 'Yes' else ', Non-metastatic' if p['metastatic'] == 'No' else ''}",
        f"  Mutations: {mut_str} | PD-L1: {p['pdl1_status']}",
        f"  Age: {p['age_at_tb']:.0f}{p['sex']} | ECOG: {p['ecog']} | Smoking: {p['smoking_category']}"
        + (f" ({p['pack_years']} pack-years)" if p['pack_years'] > 0 else ""),
        f"  Treatment: Line {p['current_line_number']} ({p['current_intent']}) | Regimens: {regimens}",
        f"  Surgery: {'Yes' if p['had_surgery'] else 'No'} | Radiation: {'Yes' if p['had_radiation'] else 'No'}",
    ]
    if p.get("last_termination_reason") and p["last_termination_reason"] != "unknown":
        lines.append(f"  Last line ended: {p['last_termination_reason']}")

    return "\n".join(lines)


def generate_vignettes_batch(
    patients: list[dict],
    model: str = "gemini-2.5-flash",
    batch_size: int = 20,
    max_workers: int = 10,
) -> dict:
    """Generate clinical vignettes for a batch of synthetic patients via LLM.

    Returns dict mapping patient_id -> tb_note.
    """
    # Build batches
    batches = []
    for i in range(0, len(patients), batch_size):
        batch = patients[i:i+batch_size]
        batches.append(batch)

    print(f"\nGenerating vignettes: {len(patients)} patients in {len(batches)} batches ({max_workers} workers)...")

    system_prompt = (
        "You are a clinical data generator creating realistic tumor board notes. "
        "Given structured patient data, write concise clinical vignettes in the style "
        "of a tumor board presentation note. Each vignette should be 3-5 sentences. "
        "Include: diagnosis, mutations, staging, treatment history, current status, "
        "and the clinical question for tumor board discussion. "
        "Do NOT include real names or dates. Use 'Patient' as a generic term. "
        "Output ONLY a JSON array, no markdown fences."
    )

    results = {}

    def _generate_batch(batch):
        patient_blocks = []
        for j, p in enumerate(batch, 1):
            block = f"Patient {j} (ID: {p['patient_id']}):\n{_format_patient_for_vignette(p)}"
            patient_blocks.append(block)

        user_prompt = (
            "Generate clinical vignettes for these patients:\n\n"
            + "\n\n".join(patient_blocks)
            + "\n\nOutput JSON array:\n"
            '[{"patient_id": "SYN_XXXX", "tb_note": "Patient is a 65-year-old..."}, ...]'
        )

        response = gsgpt.chat(user_prompt, system=system_prompt, model=model)

        # Parse response
        try:
            # Find JSON array
            start = response.find('[')
            if start == -1:
                return {}
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(response)):
                c = response[i]
                if esc:
                    esc = False
                    continue
                if c == '\\' and in_str:
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == '[':
                    depth += 1
                elif c == ']':
                    depth -= 1
                    if depth == 0:
                        json_str = response[start:i+1]
                        break
            else:
                return {}

            items = json.loads(json_str)
            batch_results = {}
            for item in items:
                pid = item.get("patient_id", "")
                note = item.get("tb_note", "")
                if pid and note:
                    batch_results[pid] = note
            return batch_results
        except (json.JSONDecodeError, ValueError):
            return {}

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_generate_batch, batch): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch_results = future.result()
                results.update(batch_results)
                print(f"  Batch {batch_idx+1}/{len(batches)}: {len(batch_results)} vignettes")
            except Exception as e:
                print(f"  Batch {batch_idx+1}/{len(batches)}: FAILED - {e}")

    elapsed = time.perf_counter() - t0
    print(f"  Vignettes complete: {len(results)}/{len(patients)} in {elapsed:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Distribution comparison plots
# ---------------------------------------------------------------------------

def plot_distribution_comparison(
    real_patients: list[dict],
    synthetic_patients: list[dict],
    output_path: str = "docs/plm/real_vs_synthetic_distributions.png",
):
    """Multi-panel comparison of real vs synthetic distributions."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Diagnosis categories
    ax = axes[0, 0]
    real_diag = Counter(p["diagnosis_category"] for p in real_patients)
    syn_diag = Counter(p["diagnosis_category"] for p in synthetic_patients)
    categories = sorted(set(real_diag.keys()) | set(syn_diag.keys()))
    x = np.arange(len(categories))
    w = 0.35
    real_pct = [100 * real_diag.get(c, 0) / len(real_patients) for c in categories]
    syn_pct = [100 * syn_diag.get(c, 0) / len(synthetic_patients) for c in categories]
    ax.bar(x - w/2, real_pct, w, label='Real (N=1,167)', color='#2196F3', alpha=0.8)
    ax.bar(x + w/2, syn_pct, w, label=f'Synthetic (N={len(synthetic_patients)})', color='#FF9800', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('% of Cohort')
    ax.set_title('Diagnosis Category')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 2. Mutation rates
    ax = axes[0, 1]
    genes = ["EGFR", "ALK", "KRAS", "BRAF", "MET", "RET", "ROS1"]
    real_rates = []
    syn_rates = []
    for g in genes:
        r_pos = sum(1 for p in real_patients if p.get("driver_mutations", {}).get(g) == "positive")
        r_total = len(real_patients)
        real_rates.append(100 * r_pos / r_total)
        s_pos = sum(1 for p in synthetic_patients if p.get("driver_mutations", {}).get(g) == "positive")
        s_total = len(synthetic_patients)
        syn_rates.append(100 * s_pos / s_total)
    x = np.arange(len(genes))
    ax.bar(x - w/2, real_rates, w, label='Real', color='#2196F3', alpha=0.8)
    ax.bar(x + w/2, syn_rates, w, label='Synthetic', color='#FF9800', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(genes, fontsize=10)
    ax.set_ylabel('% Positive')
    ax.set_title('Driver Mutation Rates')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 3. Age distribution
    ax = axes[1, 0]
    real_ages = [p["age_at_tb"] for p in real_patients if p.get("age_at_tb")]
    syn_ages = [p["age_at_tb"] for p in synthetic_patients if p.get("age_at_tb")]
    ax.hist(real_ages, bins=25, alpha=0.6, label='Real', color='#2196F3', density=True)
    ax.hist(syn_ages, bins=25, alpha=0.6, label='Synthetic', color='#FF9800', density=True)
    ax.set_xlabel('Age at Tumor Board')
    ax.set_ylabel('Density')
    ax.set_title('Age Distribution')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 4. Treatment lines
    ax = axes[1, 1]
    max_line = 6
    real_lines = Counter(min(p.get("current_line_number", 0), max_line) for p in real_patients)
    syn_lines = Counter(min(p.get("current_line_number", 0), max_line) for p in synthetic_patients)
    line_labels = [str(i) for i in range(max_line)] + [f"{max_line}+"]
    x = np.arange(len(line_labels))
    real_line_pct = [100 * real_lines.get(i, 0) / len(real_patients) for i in range(max_line + 1)]
    syn_line_pct = [100 * syn_lines.get(i, 0) / len(synthetic_patients) for i in range(max_line + 1)]
    ax.bar(x - w/2, real_line_pct, w, label='Real', color='#2196F3', alpha=0.8)
    ax.bar(x + w/2, syn_line_pct, w, label='Synthetic', color='#FF9800', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(line_labels)
    ax.set_xlabel('Treatment Line')
    ax.set_ylabel('% of Cohort')
    ax.set_title('Treatment Line Distribution')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Real vs Synthetic Patient Distributions',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic patients for PLM development",
    )
    parser.add_argument("--real-index", default="outputs/plm/plm_index_1180.json",
                        help="Real PLM index to learn distributions from")
    parser.add_argument("--n", type=int, default=400,
                        help="Number of synthetic patients to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="LLM model for vignette generation")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Parallel LLM workers")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Patients per LLM batch")
    parser.add_argument("--skip-vignettes", action="store_true",
                        help="Skip LLM vignette generation")
    parser.add_argument("--output", default="outputs/plm/plm_synthetic_index.json",
                        help="Output synthetic index file")
    parser.add_argument("--id-pad", type=int, default=4,
                        help="Zero-padding width of SYN_ patient IDs (default 4 for 10k; "
                             "use 6 for 100k so SYN_000001..SYN_100000 stays uniform)")
    args = parser.parse_args()

    print("=" * 70)
    print("VISTA PLM: Synthetic Patient Generation (Phase II MVP)")
    print("=" * 70)

    # Load real index
    print(f"\nLoading real PLM index: {args.real_index}")
    with open(args.real_index) as f:
        plm_index = json.load(f)
    real_patients = list(plm_index["patients"].values())
    print(f"  {len(real_patients)} real patients")

    # Extract distributions
    print("\nExtracting conditional distributions...")
    dist = extract_distributions(plm_index)

    # Sample synthetic patients
    print(f"\nSampling {args.n} synthetic patients (seed={args.seed})...")
    rng = np.random.default_rng(args.seed)
    synthetic_patients = []
    for i in range(args.n):
        p = sample_patient(dist, i + 1, rng, id_pad=args.id_pad)
        synthetic_patients.append(p)

    # Quick distribution check
    syn_diag = Counter(p["diagnosis_category"] for p in synthetic_patients)
    print(f"  Diagnosis: {dict(syn_diag)}")
    syn_met = Counter(p["metastatic"] for p in synthetic_patients)
    print(f"  Metastatic: {dict(syn_met)}")
    syn_muts = Counter()
    for p in synthetic_patients:
        for g, s in p.get("driver_mutations", {}).items():
            if s == "positive":
                syn_muts[g] += 1
    print(f"  Mutations: {dict(syn_muts)}")

    # Generate vignettes
    if not args.skip_vignettes:
        vignettes = generate_vignettes_batch(
            synthetic_patients,
            model=args.model,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
        )

        # Attach vignettes to patients
        for p in synthetic_patients:
            note = vignettes.get(p["patient_id"], "")
            p["tb_note"] = note
            p["tb_note_tokens"] = re.findall(r'[a-z0-9]+', note.lower()) if note else []

        n_with_notes = sum(1 for p in synthetic_patients if p["tb_note"])
        print(f"  Patients with vignettes: {n_with_notes}/{len(synthetic_patients)}")
    else:
        print("\n  Skipping vignette generation (--skip-vignettes)")

    # Build synthetic index
    synthetic_index = {
        "version": "1.0-synthetic",
        "source": "generate_fake_patients.py",
        "real_index": args.real_index,
        "n_patients": len(synthetic_patients),
        "seed": args.seed,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patients": {p["patient_id"]: p for p in synthetic_patients},
    }

    # Save
    with open(args.output, "w") as f:
        json.dump(synthetic_index, f, indent=2)
    print(f"\n  Synthetic index saved: {args.output}")

    # Distribution comparison plot
    print("\nGenerating distribution comparison plots...")
    plot_distribution_comparison(real_patients, synthetic_patients)

    # Summary table
    print("\n" + "=" * 70)
    print("DISTRIBUTION COMPARISON")
    print("=" * 70)
    print(f"\n{'Category':<25} {'Real':>10} {'Synthetic':>10} {'Real %':>10} {'Syn %':>10}")
    print("-" * 65)

    real_diag = Counter(p["diagnosis_category"] for p in real_patients)
    for cat in sorted(set(real_diag.keys()) | set(syn_diag.keys())):
        r = real_diag.get(cat, 0)
        s = syn_diag.get(cat, 0)
        print(f"{cat:<25} {r:>10} {s:>10} {100*r/len(real_patients):>9.1f}% {100*s/len(synthetic_patients):>9.1f}%")

    print(f"\n{'Mutation':<25} {'Real pos':>10} {'Syn pos':>10} {'Real %':>10} {'Syn %':>10}")
    print("-" * 65)
    real_muts = Counter()
    for p in real_patients:
        for g, s in p.get("driver_mutations", {}).items():
            if s == "positive":
                real_muts[g] += 1
    for gene in ["EGFR", "ALK", "KRAS", "BRAF", "MET", "RET", "ROS1"]:
        r = real_muts.get(gene, 0)
        s = syn_muts.get(gene, 0)
        print(f"{gene:<25} {r:>10} {s:>10} {100*r/len(real_patients):>9.1f}% {100*s/len(synthetic_patients):>9.1f}%")

    print(f"\n  Total synthetic patients: {len(synthetic_patients)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
