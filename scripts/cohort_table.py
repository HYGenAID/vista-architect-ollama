#!/usr/bin/env python3
"""
Cohort Description Table Generator for VISTA Architect.

Generates a standard epidemiological "Table 1" from extracted patient_info.json files.
Outputs in markdown and/or LaTeX format for manuscript inclusion.

Usage:
    python cohort_table.py --manifest patient_records/cohorts/eval_v2_600_manifest.json
    python cohort_table.py --manifest ... --format both --output cohort_table
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import statistics


def compute_age(dob_str: str, tb_date_str: str) -> float | None:
    """Calculate age in years from DOB to TB date."""
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        tb = datetime.strptime(tb_date_str, "%Y-%m-%d")
        age = (tb - dob).days / 365.25
        if 0 < age < 120:
            return round(age, 1)
    except (ValueError, TypeError):
        pass
    return None


def categorize_diagnosis(diagnosis_str: str) -> str:
    """Map free-text diagnosis to standard categories."""
    if not diagnosis_str:
        return "Unknown"
    d = diagnosis_str.upper()
    # Check SCLC first (before NSCLC, since "non-small cell" also contains "small cell")
    if "MESOTHELIOMA" in d:
        return "Mesothelioma"
    if "THYMOMA" in d or "THYMIC" in d:
        return "Thymoma/Thymic"
    if any(kw in d for kw in ["NON-SMALL CELL", "NON SMALL CELL", "NSCLC"]):
        return "NSCLC"
    if any(kw in d for kw in ["ADENOCARCINOMA", "SQUAMOUS CELL", "LARGE CELL"]):
        return "NSCLC"
    if any(kw in d for kw in ["SMALL CELL", "SCLC"]):
        return "SCLC"
    if any(kw in d for kw in ["LUNG", "PULMONARY", "BRONCH"]):
        return "NSCLC"  # Default lung cancer to NSCLC
    if any(kw in d for kw in ["PANCREA", "COLON", "BREAST", "RENAL", "KIDNEY",
                               "ESOPHAG", "GASTRIC", "HEPATO", "LIVER"]):
        return "Other Primary"
    return "Other"


def categorize_histology(histology_str: str) -> str:
    """Map free-text histology to standard categories."""
    if not histology_str:
        return "Unknown/NOS"
    h = histology_str.upper()
    if "ADENOCARCINOMA" in h or "ADENO" in h:
        return "Adenocarcinoma"
    if "SQUAMOUS" in h:
        return "Squamous Cell"
    if "LARGE CELL" in h:
        return "Large Cell"
    if "SMALL CELL" in h:
        return "Small Cell"
    if "SARCOMATOID" in h:
        return "Sarcomatoid"
    if "CARCINOID" in h:
        return "Carcinoid"
    if "MESOTHELIOMA" in h:
        return "Mesothelioma"
    return "Other/NOS"


def categorize_smoking(smoking_str: str) -> tuple[str, float | None]:
    """Map free-text smoking to category + pack-years."""
    if not smoking_str:
        return "Unknown", None
    s = smoking_str.upper()
    pack_years = None
    # Try to extract pack-years
    import re
    py_match = re.search(r'(\d+\.?\d*)\s*(?:PACK[- ]?YEAR|PPY|PY)', s)
    if py_match:
        try:
            pack_years = float(py_match.group(1))
        except ValueError:
            pass

    if any(kw in s for kw in ["NEVER", "NON-SMOKER", "NONSMOKER", "NON SMOKER", "NO SMOKING"]):
        return "Never", None
    if any(kw in s for kw in ["CURRENT", "ACTIVE SMOKER", "ACTIVELY SMOKING"]):
        return "Current", pack_years
    if any(kw in s for kw in ["FORMER", "QUIT", "EX-SMOKER", "EX SMOKER", "STOPPED",
                               "PREVIOUSLY", "PAST SMOKER"]):
        return "Former", pack_years
    if any(kw in s for kw in ["UNKNOWN", "NOT DOCUMENTED", "NOT RECORDED"]):
        return "Unknown", None
    # If pack-years mentioned but no clear status
    if pack_years is not None:
        return "Former", pack_years  # Assume former if pack-years given
    return "Unknown", None


def categorize_ecog(ecog_str: str) -> str:
    """Normalize ECOG to category."""
    if not ecog_str:
        return "Unknown"
    e = str(ecog_str).strip()
    if e in ("0", "1", "2", "3", "4"):
        return e
    if "-" in e:  # ranges like "0-1", "1-2"
        return e
    if "UNKNOWN" in e.upper() or "NOT" in e.upper():
        return "Unknown"
    # Try to extract number
    import re
    m = re.search(r'(\d)', e)
    if m:
        return m.group(1)
    return "Unknown"


def analyze_mutations(driver_mutations: dict) -> dict:
    """Analyze driver mutation panel."""
    if not driver_mutations or not isinstance(driver_mutations, dict):
        return {"tested": False, "positive_genes": [], "genes_tested": 0}

    tested = False
    positive_genes = []
    genes_tested = 0
    skip_keys = {"TMB", "MSI", "MMR", "PD-L1", "Germline", "HPV/p16"}

    for gene, result in driver_mutations.items():
        if gene in skip_keys:
            continue
        if not result or not isinstance(result, str):
            continue
        result_upper = result.upper()
        if "NOT TESTED" in result_upper:
            continue
        genes_tested += 1
        tested = True
        # Check for negative/unknown results
        negative_keywords = ["NOT DETECTED", "NEGATIVE", "WILD TYPE", "WILD-TYPE",
                            "NO MUTATION", "NOT FOUND", "NONE DETECTED",
                            "NOT REPORTED", "NOT DOCUMENTED", "PENDING",
                            "INSUFFICIENT", "INCONCLUSIVE", "INDETERMINATE"]
        is_negative = any(kw in result_upper for kw in negative_keywords)
        if not is_negative:
            positive_genes.append(gene)

    # Check PD-L1 separately
    pdl1 = driver_mutations.get("PD-L1", "")
    pdl1_tested = pdl1 and isinstance(pdl1, str) and "NOT TESTED" not in pdl1.upper()

    return {
        "tested": tested,
        "positive_genes": positive_genes,
        "genes_tested": genes_tested,
        "pdl1_tested": pdl1_tested,
    }


def load_patient_data(manifest_path: str, store_path: str = "graph_store") -> list[dict]:
    """Load patient data from patient_info.json files + graph store index."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    patients_list = manifest["patients"]

    # Load graph store index for metadata
    index_path = Path(store_path) / "index.json"
    gs_index = {}
    if index_path.exists():
        with open(index_path) as f:
            gs_data = json.load(f)
            gs_index = gs_data.get("patients", {})

    data = []
    skipped = 0
    for entry in patients_list:
        pid = entry["patient_id"]
        info_path = Path(f"temp_jsons/{pid}/patient_info.json")
        if not info_path.exists():
            skipped += 1
            continue

        try:
            with open(info_path) as f:
                info = json.load(f)
        except (json.JSONDecodeError, IOError):
            skipped += 1
            continue

        demo = info.get("PATIENT DEMOGRAPHICS", {})
        tumor = info.get("TUMOR INFORMATION", {})
        treat = info.get("TREATMENTS", {})

        # Get tb_date from manifest or graph store
        tb_date = entry.get("tb_date", "")
        if not tb_date and pid in gs_index:
            tb_date = gs_index[pid].get("tb_date", "")

        # Get graph store metadata
        gs_meta = gs_index.get(pid, {})

        record = {
            "pid": pid,
            "dob": demo.get("date_of_birth", ""),
            "tb_date": tb_date,
            "sex": demo.get("sex", ""),
            "ecog": demo.get("ecog_performance_status", ""),
            "smoking": demo.get("smoking_history", ""),
            "allergies": demo.get("allergies", []),
            "dnr": demo.get("dnr", ""),
            "diagnosis": tumor.get("diagnosis", ""),
            "histology": tumor.get("histology", ""),
            "metastasis": tumor.get("metastasis_status", ""),
            "lymph_node": tumor.get("lymph_node_involvement", ""),
            "mutations": tumor.get("driver_mutations", {}),
            "current_therapy": treat.get("current", []),
            "radiation": treat.get("radiation_therapy", ""),
            "surgical_candidate": treat.get("surgical_candidate", {}),
            "date_of_last_ct": treat.get("date_of_last_ct", ""),
            # Graph store metadata
            "node_count": gs_meta.get("node_count", 0),
            "event_count": gs_meta.get("event_count", 0),
            "date_range": gs_meta.get("date_range", {}),
        }
        data.append(record)

    if skipped:
        print(f"  Warning: {skipped} patients skipped (no patient_info.json)")
    print(f"  Loaded {len(data)} patients")
    return data


def generate_table(patient_data: list[dict]) -> dict:
    """Compute all descriptive statistics."""
    n = len(patient_data)
    stats = {"n": n, "sections": []}

    # --- DEMOGRAPHICS ---
    ages = [a for a in (compute_age(p["dob"], p["tb_date"]) for p in patient_data) if a is not None]
    sex_counts = Counter(p["sex"] for p in patient_data)

    demo_rows = []
    if ages:
        q25, q50, q75 = (
            sorted(ages)[int(len(ages) * 0.25)],
            statistics.median(ages),
            sorted(ages)[int(len(ages) * 0.75)],
        )
        demo_rows.append(("Age at TB, median (IQR)", f"{q50:.0f} ({q25:.0f}–{q75:.0f})", None))
    for sex in ["Male", "Female"]:
        c = sex_counts.get(sex, 0)
        demo_rows.append((f"  {sex}", f"{c} ({100*c/n:.1f}%)", None))

    stats["sections"].append(("Patient Demographics", demo_rows))

    # --- CLINICAL ---
    smoking_cats = []
    pack_years_list = []
    for p in patient_data:
        cat, py = categorize_smoking(p["smoking"])
        smoking_cats.append(cat)
        if py is not None:
            pack_years_list.append(py)

    ecog_cats = [categorize_ecog(p["ecog"]) for p in patient_data]

    clinical_rows = []
    clinical_rows.append(("Smoking Status, n (%)", "", None))
    for cat in ["Current", "Former", "Never", "Unknown"]:
        c = smoking_cats.count(cat)
        clinical_rows.append((f"  {cat}", f"{c} ({100*c/n:.1f}%)", None))
    if pack_years_list:
        py_sorted = sorted(pack_years_list)
        q25 = py_sorted[int(len(py_sorted) * 0.25)]
        q50 = statistics.median(py_sorted)
        q75 = py_sorted[int(len(py_sorted) * 0.75)]
        clinical_rows.append((f"  Pack-years (smokers), median (IQR)", f"{q50:.0f} ({q25:.0f}–{q75:.0f})", None))

    clinical_rows.append(("ECOG Performance Status, n (%)", "", None))
    ecog_counter = Counter(ecog_cats)
    for val in ["0", "1", "2", "3", "4"]:
        c = ecog_counter.get(val, 0)
        if c > 0:
            clinical_rows.append((f"  {val}", f"{c} ({100*c/n:.1f}%)", None))
    # Ranges
    for val, c in ecog_counter.items():
        if "-" in val:
            clinical_rows.append((f"  {val}", f"{c} ({100*c/n:.1f}%)", None))
    c_unk = ecog_counter.get("Unknown", 0)
    if c_unk > 0:
        clinical_rows.append((f"  Unknown", f"{c_unk} ({100*c_unk/n:.1f}%)", None))

    stats["sections"].append(("Clinical Characteristics", clinical_rows))

    # --- TUMOR ---
    diag_cats = [categorize_diagnosis(p["diagnosis"]) for p in patient_data]
    hist_cats = [categorize_histology(p["histology"]) for p in patient_data]

    tumor_rows = []
    tumor_rows.append(("Primary Diagnosis, n (%)", "", None))
    diag_counter = Counter(diag_cats)
    for cat in ["NSCLC", "SCLC", "Mesothelioma", "Thymoma/Thymic", "Other Primary", "Other", "Unknown"]:
        c = diag_counter.get(cat, 0)
        if c > 0:
            tumor_rows.append((f"  {cat}", f"{c} ({100*c/n:.1f}%)", None))

    tumor_rows.append(("Histology, n (%)", "", None))
    hist_counter = Counter(hist_cats)
    for cat in ["Adenocarcinoma", "Squamous Cell", "Large Cell", "Small Cell",
                "Sarcomatoid", "Carcinoid", "Mesothelioma", "Other/NOS", "Unknown/NOS"]:
        c = hist_counter.get(cat, 0)
        if c > 0:
            tumor_rows.append((f"  {cat}", f"{c} ({100*c/n:.1f}%)", None))

    # Metastasis
    met_yes = sum(1 for p in patient_data if str(p["metastasis"]).upper().startswith("Y"))
    met_no = sum(1 for p in patient_data if str(p["metastasis"]).upper().startswith("N"))
    met_unk = n - met_yes - met_no
    tumor_rows.append(("Metastatic Disease, n (%)", "", None))
    tumor_rows.append(("  Yes", f"{met_yes} ({100*met_yes/n:.1f}%)", None))
    tumor_rows.append(("  No", f"{met_no} ({100*met_no/n:.1f}%)", None))
    if met_unk > 0:
        tumor_rows.append(("  Unknown/Suspected", f"{met_unk} ({100*met_unk/n:.1f}%)", None))

    # Lymph node
    ln_yes = sum(1 for p in patient_data if str(p["lymph_node"]).upper().startswith("Y"))
    ln_no = sum(1 for p in patient_data if str(p["lymph_node"]).upper().startswith("N"))
    ln_unk = n - ln_yes - ln_no
    tumor_rows.append(("Lymph Node Involvement, n (%)", "", None))
    tumor_rows.append(("  Yes", f"{ln_yes} ({100*ln_yes/n:.1f}%)", None))
    tumor_rows.append(("  No", f"{ln_no} ({100*ln_no/n:.1f}%)", None))
    if ln_unk > 0:
        tumor_rows.append(("  Unknown/Suspected", f"{ln_unk} ({100*ln_unk/n:.1f}%)", None))

    # Driver mutations
    mut_analyses = [analyze_mutations(p["mutations"]) for p in patient_data]
    tested_count = sum(1 for m in mut_analyses if m["tested"])
    positive_count = sum(1 for m in mut_analyses if m["positive_genes"])
    gene_counter = Counter()
    for m in mut_analyses:
        for g in m["positive_genes"]:
            gene_counter[g] += 1
    pdl1_tested = sum(1 for m in mut_analyses if m["pdl1_tested"])

    tumor_rows.append(("Molecular Testing, n (%)", "", None))
    tumor_rows.append(("  Tested", f"{tested_count} ({100*tested_count/n:.1f}%)", None))
    tumor_rows.append(("  Any Actionable Mutation", f"{positive_count} ({100*positive_count/n:.1f}%)", None))
    for gene in ["EGFR", "ALK", "KRAS", "ROS1", "BRAF", "MET", "RET", "NTRK", "ERBB2 (HER2)"]:
        c = gene_counter.get(gene, 0)
        if c > 0:
            tumor_rows.append((f"    {gene}", f"{c} ({100*c/n:.1f}%)", None))
    tumor_rows.append((f"  PD-L1 Tested", f"{pdl1_tested} ({100*pdl1_tested/n:.1f}%)", None))

    stats["sections"].append(("Tumor Characteristics", tumor_rows))

    # --- TREATMENT ---
    has_current_tx = sum(1 for p in patient_data
                        if p["current_therapy"] and len(p["current_therapy"]) > 0
                        and not all("none" in str(t).lower() for t in p["current_therapy"]))
    rad_yes = sum(1 for p in patient_data if str(p["radiation"]).upper().startswith("Y"))
    surg_eligible = sum(1 for p in patient_data
                       if isinstance(p["surgical_candidate"], dict)
                       and p["surgical_candidate"].get("eligible", False))

    tx_rows = []
    tx_rows.append(("Current Medical Therapy, n (%)", f"{has_current_tx} ({100*has_current_tx/n:.1f}%)", None))
    tx_rows.append(("Radiation Therapy, n (%)", f"{rad_yes} ({100*rad_yes/n:.1f}%)", None))
    tx_rows.append(("Surgical Candidate, n (%)", f"{surg_eligible} ({100*surg_eligible/n:.1f}%)", None))

    stats["sections"].append(("Treatment", tx_rows))

    # --- SAFETY ---
    dnr_yes = sum(1 for p in patient_data if str(p["dnr"]).upper().startswith("Y"))
    has_allergies = sum(1 for p in patient_data
                       if p["allergies"] and len(p["allergies"]) > 0
                       and not any(kw in str(p["allergies"]).upper()
                                  for kw in ["NKDA", "NO KNOWN", "NONE", "NKA"]))

    safety_rows = []
    safety_rows.append(("DNR Status, n (%)", f"{dnr_yes} ({100*dnr_yes/n:.1f}%)", None))
    safety_rows.append(("Known Drug Allergies, n (%)", f"{has_allergies} ({100*has_allergies/n:.1f}%)", None))

    stats["sections"].append(("Safety", safety_rows))

    # --- RECORD CHARACTERISTICS ---
    record_spans = []
    event_counts = []
    node_counts = []
    for p in patient_data:
        dr = p.get("date_range", {})
        if dr.get("start") and dr.get("end"):
            try:
                start = datetime.strptime(dr["start"], "%Y-%m-%d")
                end = datetime.strptime(dr["end"], "%Y-%m-%d")
                span_years = (end - start).days / 365.25
                if span_years >= 0:
                    record_spans.append(round(span_years, 1))
            except (ValueError, TypeError):
                pass
        if p.get("event_count", 0) > 0:
            event_counts.append(p["event_count"])
        if p.get("node_count", 0) > 0:
            node_counts.append(p["node_count"])

    rec_rows = []
    if record_spans:
        rs = sorted(record_spans)
        q25, q50, q75 = rs[int(len(rs)*0.25)], statistics.median(rs), rs[int(len(rs)*0.75)]
        rec_rows.append(("Record Time Span (years), median (IQR)", f"{q50:.1f} ({q25:.1f}–{q75:.1f})", None))
    if event_counts:
        ec = sorted(event_counts)
        q25, q50, q75 = ec[int(len(ec)*0.25)], statistics.median(ec), ec[int(len(ec)*0.75)]
        rec_rows.append(("Clinical Events, median (IQR)", f"{q50:,.0f} ({q25:,.0f}–{q75:,.0f})", None))
    if node_counts:
        nc = sorted(node_counts)
        q25, q50, q75 = nc[int(len(nc)*0.25)], statistics.median(nc), nc[int(len(nc)*0.75)]
        rec_rows.append(("Graph Nodes, median (IQR)", f"{q50:,.0f} ({q25:,.0f}–{q75:,.0f})", None))

    stats["sections"].append(("Record Characteristics", rec_rows))

    return stats


def format_markdown(stats: dict) -> str:
    """Format statistics as a publication-quality markdown table."""
    n = stats["n"]
    lines = []
    lines.append(f"## Table 1. Cohort Characteristics (N={n})")
    lines.append("")
    lines.append(f"| Characteristic | N={n} |")
    lines.append("|:---|---:|")

    for section_name, rows in stats["sections"]:
        lines.append(f"| **{section_name}** | |")
        for label, value, _ in rows:
            if value:
                lines.append(f"| {label} | {value} |")
            else:
                lines.append(f"| {label} | |")

    return "\n".join(lines)


def format_latex(stats: dict) -> str:
    """Format statistics as a LaTeX table."""
    n = stats["n"]
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Cohort Characteristics (N={n})}}")
    lines.append(r"\label{tab:cohort}")
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\toprule")
    lines.append(f"\\textbf{{Characteristic}} & \\textbf{{N={n}}} \\\\")
    lines.append(r"\midrule")

    for i, (section_name, rows) in enumerate(stats["sections"]):
        if i > 0:
            lines.append(r"\midrule")
        lines.append(f"\\textbf{{{section_name}}} & \\\\")
        for label, value, _ in rows:
            # Escape special LaTeX chars
            label_tex = label.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")
            value_tex = value.replace("%", r"\%").replace("–", "--")
            if value_tex:
                lines.append(f"{label_tex} & {value_tex} \\\\")
            else:
                lines.append(f"{label_tex} & \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate cohort description table from VISTA patient data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="Cohort manifest JSON file")
    parser.add_argument("--store-path", default="graph_store", help="Graph store path")
    parser.add_argument("--format", choices=["markdown", "latex", "both"], default="both",
                       help="Output format (default: both)")
    parser.add_argument("--output", default=None,
                       help="Output base filename (produces .md and/or .tex)")

    args = parser.parse_args()

    print(f"Loading patient data from: {args.manifest}")
    patient_data = load_patient_data(args.manifest, args.store_path)

    if not patient_data:
        print("No patient data found!")
        sys.exit(1)

    print(f"Generating cohort table for {len(patient_data)} patients...")
    stats = generate_table(patient_data)

    # Determine output paths
    base = args.output or "cohort_table"

    if args.format in ("markdown", "both"):
        md = format_markdown(stats)
        md_path = f"{base}.md"
        with open(md_path, "w") as f:
            f.write(md)
        print(f"\nMarkdown table saved to: {md_path}")
        print(md)

    if args.format in ("latex", "both"):
        tex = format_latex(stats)
        tex_path = f"{base}.tex"
        with open(tex_path, "w") as f:
            f.write(tex)
        print(f"\nLaTeX table saved to: {tex_path}")

    # Also dump raw stats as JSON for further analysis
    json_path = f"{base}_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Raw stats saved to: {json_path}")


if __name__ == "__main__":
    main()
