#!/usr/bin/env python3
"""
Build Patient Like Me (PLM) Index.

Pre-computes a searchable index from existing patient artifacts
(patient_info.json, summary.json, episodes.json, graph_store index)
for fast clinical similarity retrieval.

Usage:
    python build_plm_index.py --manifest patient_records/cohorts/eval_v2_600_manifest.json
    python build_plm_index.py --manifest patient_records/cohorts/eval_v2_1180_manifest.json --output outputs/plm/plm_index_1180.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

from cohort_table import (
    compute_age,
    categorize_diagnosis,
    categorize_histology,
    categorize_smoking,
    categorize_ecog,
)


# ---------------------------------------------------------------------------
# Drug class classification for trajectory matching
# ---------------------------------------------------------------------------

DRUG_CLASSES = {
    "platinum_doublet": ["carboplatin", "cisplatin", "oxaliplatin"],
    "immunotherapy": [
        "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab",
        "ipilimumab", "tremelimumab", "cemiplimab",
    ],
    "egfr_tki": [
        "osimertinib", "erlotinib", "gefitinib", "afatinib",
        "dacomitinib", "amivantamab", "lazertinib", "mobocertinib",
    ],
    "alk_tki": [
        "alectinib", "crizotinib", "ceritinib", "lorlatinib", "brigatinib",
    ],
    "kras_inhibitor": ["sotorasib", "adagrasib"],
    "taxane": ["docetaxel", "paclitaxel", "nab-paclitaxel", "abraxane"],
    "antiangiogenic": ["bevacizumab", "ramucirumab", "nintedanib"],
    "pemetrexed": ["pemetrexed"],
    "ret_inhibitor": ["selpercatinib", "pralsetinib"],
    "ros1_inhibitor": ["entrectinib"],  # crizotinib already in alk_tki
    "met_inhibitor": ["capmatinib", "tepotinib"],
    "her2_targeted": ["trastuzumab", "t-dxd", "trastuzumab deruxtecan", "fam-trastuzumab"],
    "braf_targeted": ["dabrafenib", "trametinib", "encorafenib", "binimetinib"],
    "etoposide": ["etoposide"],
    "gemcitabine": ["gemcitabine"],
}

# Reverse lookup: drug name → class
_DRUG_TO_CLASS = {}
for cls, drugs in DRUG_CLASSES.items():
    for drug in drugs:
        _DRUG_TO_CLASS[drug.lower()] = cls

# Actionable driver genes (excluding biomarkers like PD-L1, TMB)
ACTIONABLE_GENES = [
    "EGFR", "ALK", "KRAS", "ROS1", "BRAF", "MET", "RET", "NTRK",
    "ERBB2 (HER2)", "NRG1",
]

# Keywords indicating negative/pending/untested results
NEGATIVE_KEYWORDS = [
    "NOT DETECTED", "NEGATIVE", "WILD TYPE", "WILD-TYPE",
    "NO MUTATION", "NOT FOUND", "NONE DETECTED",
    "NOT REPORTED", "NOT DOCUMENTED", "PENDING",
    "INSUFFICIENT", "INCONCLUSIVE", "INDETERMINATE",
]


# ---------------------------------------------------------------------------
# Normalization functions
# ---------------------------------------------------------------------------

def normalize_mutation_status(value: str) -> str:
    """Map free-text driver_mutations value to positive/negative/not_tested."""
    if not value or not isinstance(value, str):
        return "not_tested"
    upper = value.upper().strip()
    if "NOT TESTED" in upper or upper == "":
        return "not_tested"
    if any(kw in upper for kw in NEGATIVE_KEYWORDS):
        return "negative"
    return "positive"


def classify_pdl1(value: str) -> str:
    """Classify PD-L1 into high/low/negative/unknown."""
    if not value or not isinstance(value, str):
        return "unknown"
    upper = value.upper().strip()
    if "NOT TESTED" in upper:
        return "unknown"
    # Check for <1% first (before percentage extraction)
    if "<1" in upper or "< 1" in upper:
        return "negative"
    # Try to extract percentage
    m = re.search(r'(\d+)\s*%', value)
    if m:
        pct = int(m.group(1))
        if pct >= 50:
            return "high"
        elif pct >= 1:
            return "low"
        else:
            return "negative"
    if any(kw in upper for kw in ["NEGATIVE", "NOT DETECTED"]):
        return "negative"
    if "POSITIVE" in upper:
        return "low"  # positive but unknown percentage
    return "unknown"


def classify_regimen(treatment_str: str) -> list[str]:
    """Map treatment text to drug class tags."""
    if not treatment_str:
        return []
    lower = treatment_str.lower()
    classes = set()
    for drug, cls in _DRUG_TO_CLASS.items():
        if drug in lower:
            classes.add(cls)
    # Check for chemoradiation patterns
    if "chemorad" in lower or ("concurrent" in lower and "rad" in lower):
        classes.add("chemoradiation")
    return sorted(classes)


def normalize_metastatic(met_str: str) -> str:
    """Normalize metastatic status to Yes/No/Unknown."""
    if not met_str or not isinstance(met_str, str):
        return "Unknown"
    upper = met_str.upper().strip()
    if upper.startswith("Y"):
        return "Yes"
    if upper.startswith("N"):
        return "No"
    if "SUSPECT" in upper or "POSSIBLE" in upper:
        return "Unknown"
    return "Unknown"


def normalize_lymph_node(ln_str: str) -> str:
    """Normalize lymph node involvement to Yes/No/Unknown."""
    if not ln_str or not isinstance(ln_str, str):
        return "Unknown"
    upper = ln_str.upper().strip()
    if upper.startswith("Y"):
        return "Yes"
    if upper.startswith("N"):
        return "No"
    return "Unknown"


def tokenize_text(text: str) -> list[str]:
    """Simple tokenization for BM25 — lowercase, split on non-alphanumeric."""
    if not text:
        return []
    return re.findall(r'[a-z0-9]+', text.lower())


# ---------------------------------------------------------------------------
# Treatment trajectory extraction
# ---------------------------------------------------------------------------

def extract_treatment_trajectory(episodes: list[dict]) -> dict:
    """Extract treatment trajectory from episodes.json."""
    treatment_lines = [
        ep for ep in episodes if ep.get("kind") == "treatment_line"
    ]

    if not treatment_lines:
        return {
            "treatment_line_count": 0,
            "current_line_number": 0,
            "current_intent": "unknown",
            "regimen_classes": [],
            "last_termination_reason": "none",
        }

    # Sort by line number
    treatment_lines.sort(key=lambda ep: ep.get("line_number", 0))

    # Collect all regimen classes across all lines
    all_regimen_classes = set()
    for line in treatment_lines:
        treatment = line.get("treatment", "")
        all_regimen_classes.update(classify_regimen(treatment))

    # Current line is the last one
    current = treatment_lines[-1]

    return {
        "treatment_line_count": len(treatment_lines),
        "current_line_number": current.get("line_number", len(treatment_lines)),
        "current_intent": current.get("intent", "unknown"),
        "regimen_classes": sorted(all_regimen_classes),
        "last_termination_reason": current.get("termination_reason", "unknown"),
        "current_treatment": current.get("treatment", ""),
    }


# ---------------------------------------------------------------------------
# Build one PLM record
# ---------------------------------------------------------------------------

def build_plm_record(
    pid: str,
    patient_info: dict,
    summary: dict,
    episodes: list[dict],
    index_entry: dict,
) -> dict:
    """Build one PLM record from existing patient artifacts."""
    demo = patient_info.get("PATIENT DEMOGRAPHICS", {})
    tumor = patient_info.get("TUMOR INFORMATION", {})
    treat = patient_info.get("TREATMENTS", {})

    tb_date = index_entry.get("tb_date", "")

    # Diagnosis and histology
    diagnosis = tumor.get("diagnosis", "")
    histology = tumor.get("histology", "")

    # Normalize mutations
    raw_mutations = tumor.get("driver_mutations", {})
    driver_mutations = {}
    positive_mutations = []
    for gene in ACTIONABLE_GENES:
        val = raw_mutations.get(gene, "")
        status = normalize_mutation_status(val)
        driver_mutations[gene] = status
        if status == "positive":
            positive_mutations.append(gene)

    # PD-L1
    pdl1_raw = raw_mutations.get("PD-L1", "")
    pdl1_status = classify_pdl1(pdl1_raw)

    # Treatment trajectory
    trajectory = extract_treatment_trajectory(episodes)

    # Surgery and radiation from patient_info
    surg_info = treat.get("surgical_candidate", {})
    had_surgery = False
    if isinstance(surg_info, dict):
        # Check if they had prior surgery (not just if they're a candidate)
        desc = str(surg_info.get("description", "")).upper()
        had_surgery = any(kw in desc for kw in [
            "RESECT", "LOBECTOM", "PNEUMONECTOM", "WEDGE",
            "SEGMENTECTOM", "PRIOR SURG", "POST-OP",
        ])
    # Also check previous treatments for surgical keywords
    prev_tx = treat.get("previous", [])
    if isinstance(prev_tx, list):
        for tx in prev_tx:
            tx_upper = str(tx).upper()
            if any(kw in tx_upper for kw in [
                "RESECT", "LOBECTOM", "PNEUMONECTOM", "WEDGE",
                "SEGMENTECTOM", "VATS",
            ]):
                had_surgery = True
                break

    had_radiation = str(treat.get("radiation_therapy", "")).upper().startswith("Y")

    # Summary / tb_note
    tb_note = ""
    if isinstance(summary, dict):
        tb_note = summary.get("summary", "")
    elif isinstance(summary, str):
        tb_note = summary

    # Demographics
    smoking_cat, pack_years = categorize_smoking(demo.get("smoking_history", ""))
    age = compute_age(demo.get("date_of_birth", ""), tb_date)

    # Record characteristics from graph store index
    date_range = index_entry.get("date_range", {})
    record_span_years = None
    if date_range.get("start") and date_range.get("end"):
        try:
            start = datetime.strptime(date_range["start"], "%Y-%m-%d")
            end = datetime.strptime(date_range["end"], "%Y-%m-%d")
            record_span_years = round((end - start).days / 365.25, 1)
        except (ValueError, TypeError):
            pass

    return {
        "patient_id": pid,
        "tb_date": tb_date,
        # Hard gate fields
        "diagnosis_category": categorize_diagnosis(diagnosis),
        "metastatic": normalize_metastatic(tumor.get("metastasis_status", "")),
        # Biology fields
        "histology_category": categorize_histology(histology),
        "driver_mutations": driver_mutations,
        "positive_mutations": positive_mutations,
        "pdl1_status": pdl1_status,
        "lymph_node": normalize_lymph_node(tumor.get("lymph_node_involvement", "")),
        # Trajectory fields
        "treatment_line_count": trajectory["treatment_line_count"],
        "current_line_number": trajectory["current_line_number"],
        "current_intent": trajectory["current_intent"],
        "regimen_classes": trajectory["regimen_classes"],
        "had_surgery": had_surgery,
        "had_radiation": had_radiation,
        "last_termination_reason": trajectory["last_termination_reason"],
        # Narrative
        "tb_note": tb_note,
        "tb_note_tokens": tokenize_text(tb_note),
        # Demographics
        "age_at_tb": age,
        "sex": demo.get("sex", "Unknown"),
        "smoking_category": smoking_cat,
        "ecog": categorize_ecog(demo.get("ecog_performance_status", "")),
        "pack_years": pack_years,
        # Record characteristics
        "event_count": index_entry.get("event_count", 0),
        "node_count": index_entry.get("node_count", 0),
        "record_span_years": record_span_years,
    }


# ---------------------------------------------------------------------------
# Build full PLM index
# ---------------------------------------------------------------------------

def build_plm_index(manifest_path: str, store_path: str = "graph_store") -> dict:
    """Build PLM index for all patients in manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    patients_list = manifest["patients"]

    # Load graph store index
    index_path = Path(store_path) / "index.json"
    gs_index = {}
    if index_path.exists():
        with open(index_path) as f:
            gs_data = json.load(f)
            gs_index = gs_data.get("patients", {})

    plm_patients = {}
    skipped = 0
    errors = 0

    for entry in patients_list:
        pid = entry["patient_id"]
        base = Path(f"temp_jsons/{pid}")

        # Load patient_info.json
        info_path = base / "patient_info.json"
        if not info_path.exists():
            skipped += 1
            continue
        try:
            with open(info_path) as f:
                patient_info = json.load(f)
        except (json.JSONDecodeError, IOError):
            errors += 1
            continue

        # Load summary.json
        summary = {}
        summary_path = base / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Load episodes.json
        episodes = []
        episodes_path = base / "episodes.json"
        if episodes_path.exists():
            try:
                with open(episodes_path) as f:
                    episodes = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Get graph store metadata, merge with manifest tb_date
        gs_meta = gs_index.get(pid, {})
        if not gs_meta.get("tb_date") and entry.get("tb_date"):
            gs_meta["tb_date"] = entry["tb_date"]

        try:
            record = build_plm_record(pid, patient_info, summary, episodes, gs_meta)
            plm_patients[pid] = record
        except Exception as e:
            print(f"  Error processing {pid}: {e}")
            errors += 1

    print(f"  Built PLM index: {len(plm_patients)} patients")
    if skipped:
        print(f"  Skipped (no patient_info.json): {skipped}")
    if errors:
        print(f"  Errors: {errors}")

    # Compute cohort statistics
    stats = _compute_index_stats(plm_patients)

    return {
        "version": "1.0",
        "manifest": str(manifest_path),
        "n_patients": len(plm_patients),
        "built_at": datetime.now().isoformat(),
        "stats": stats,
        "patients": plm_patients,
    }


def _compute_index_stats(patients: dict) -> dict:
    """Compute summary statistics for the PLM index."""
    from collections import Counter

    n = len(patients)
    if n == 0:
        return {}

    records = list(patients.values())

    diag_counts = Counter(r["diagnosis_category"] for r in records)
    hist_counts = Counter(r["histology_category"] for r in records)
    met_counts = Counter(r["metastatic"] for r in records)
    intent_counts = Counter(r["current_intent"] for r in records)

    mutation_counts = Counter()
    for r in records:
        for gene in r["positive_mutations"]:
            mutation_counts[gene] += 1

    line_counts = Counter(r["current_line_number"] for r in records)

    ages = [r["age_at_tb"] for r in records if r["age_at_tb"] is not None]

    return {
        "diagnosis_distribution": dict(diag_counts.most_common()),
        "histology_distribution": dict(hist_counts.most_common()),
        "metastatic_distribution": dict(met_counts),
        "intent_distribution": dict(intent_counts.most_common()),
        "positive_mutation_counts": dict(mutation_counts.most_common()),
        "treatment_line_distribution": {str(k): v for k, v in sorted(line_counts.items())},
        "age_median": round(sorted(ages)[len(ages) // 2], 0) if ages else None,
        "has_tb_note": sum(1 for r in records if r["tb_note"]),
        "has_episodes": sum(1 for r in records if r["treatment_line_count"] > 0),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Patient Like Me index from existing patient artifacts",
    )
    parser.add_argument("--manifest", required=True, help="Cohort manifest JSON")
    parser.add_argument("--store-path", default="graph_store", help="Graph store path")
    parser.add_argument("--output", default="plm_index.json", help="Output file")
    args = parser.parse_args()

    print(f"Building PLM index from: {args.manifest}")
    plm_index = build_plm_index(args.manifest, args.store_path)

    with open(args.output, "w") as f:
        json.dump(plm_index, f, indent=2)
    print(f"\nPLM index saved to: {args.output}")

    # Print summary
    stats = plm_index["stats"]
    print(f"\n--- PLM Index Summary ---")
    print(f"Total patients: {plm_index['n_patients']}")
    print(f"Diagnosis: {stats.get('diagnosis_distribution', {})}")
    print(f"Histology: {stats.get('histology_distribution', {})}")
    print(f"Metastatic: {stats.get('metastatic_distribution', {})}")
    print(f"Positive mutations: {stats.get('positive_mutation_counts', {})}")
    print(f"Treatment lines: {stats.get('treatment_line_distribution', {})}")
    print(f"Has TB note: {stats.get('has_tb_note', 0)}")
    print(f"Has treatment episodes: {stats.get('has_episodes', 0)}")


if __name__ == "__main__":
    main()
