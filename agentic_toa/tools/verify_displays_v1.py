#!/usr/bin/env python3
"""Verifier-corrector for agentic-pipeline display items.

For each of the 16 MTB variables in a patient's patient_info.json, an Opus 4.6
verifier reads:
  - the current extracted value
  - the same EHR evidence the judge would see (deterministic context + XML chunks)

The verifier returns:
  {is_correct, confidence: high|med|low, correction, rationale}

Only confidence>=med corrections are applied. Originals preserved.

Usage:
    python agentic_toa/tools/verify_displays_v1.py --pid 136093825
        # writes patient_info_verified.json + verifier_audit.json

    python agentic_toa/tools/verify_displays_v1.py --pid 136093825 --variable Histology
        # verify a single variable
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import gsgpt
import quick_eval as qe
from eval_variables_v2 import EVAL_VARIABLES_V2

RUNS = ROOT / "agentic_toa" / "runs"


VERIFIER_PROMPT = """You are a strict clinical verifier checking a single tumor-board variable extraction against the EHR before it is shown to a clinician.

VARIABLE: {variable_name}
EXPECTED FORMAT: {variable_format}

CURRENT EXTRACTED VALUE (in patient_info.json):
{current_value}

EHR EVIDENCE (deterministic graph-search context + the truncated XML the system had access to):
{evidence}

Your job: determine whether the current extracted value is factually CORRECT against the EHR. Be strict about hallucinations, direction flips, and missing decision-relevant findings — but do NOT deduct for stylistic preferences (verbose vs terse, list order, format).

Specifically watch for:
  - Hallucinated entities (genes/drugs/dates/findings not in the EHR)
  - Direction flips (Yes when EHR shows No; No when EHR shows Yes / Suspected)
  - Wrong dates (most recent CT, most recent therapy stop date, etc.)
  - Conservative "No" / "Unknown" when EHR clearly documents Yes / a specific value
  - Including findings from a DIFFERENT cancer than the current diagnosis when the variable scope is "current"

Return JSON only:
{{
  "is_correct": true|false,
  "confidence": "high" | "med" | "low",
  "correction": "<new value to use, or null if is_correct=true>",
  "rationale": "<one or two sentences>"
}}

If is_correct=true, set correction=null.
Use "high" confidence only when the EHR evidence is unambiguous.
Use "low" when the chart itself is ambiguous (don't apply low-confidence corrections).
"""


def build_evidence_for_variable(pid: str, variable: str) -> str:
    """Use quick_eval's existing evidence assembly for a single variable."""
    xml_path = qe._resolve_truncated_xml(pid, cohort_name=None)
    if xml_path is None or not Path(xml_path).exists():
        return "(no XML evidence available)"

    # Get deterministic context (same as judge sees)
    try:
        from toa.graph_store import CohortGraphStore
        from toa.deterministic_retrieval import DeterministicRetriever
        from toa.backend import _format_deterministic_guidance
        g = CohortGraphStore("graph_store").load_patient_graph(pid)
        det = DeterministicRetriever(g).get_comprehensive_context()
        det_str = _format_deterministic_guidance(det)
    except Exception as e:
        det_str = f"(deterministic context unavailable: {e})"

    # Get the XML evidence — last 80k chars (where late-record content lives)
    full_xml = Path(xml_path).read_text()
    xml_chunk = full_xml[-80_000:]

    # Get demographics from <person>
    import xml.etree.ElementTree as ET
    demo = ""
    try:
        person = ET.parse(str(xml_path)).getroot().find(".//person")
        if person is not None:
            demo = ET.tostring(person, encoding="unicode")
    except Exception:
        pass

    parts = []
    if demo:
        parts.append("=== DEMOGRAPHICS ===\n" + demo)
    parts.append("=== DETERMINISTIC GRAPH-SEARCH CONTEXT ===\n" + det_str)
    parts.append("=== XML (final 80K chars, where late-record content lives) ===\n" + xml_chunk)
    return "\n\n".join(parts)


def verify_one(pid: str, var_def: dict, current_value: str, model: str) -> dict:
    evidence = build_evidence_for_variable(pid, var_def["name"])
    prompt = VERIFIER_PROMPT.format(
        variable_name=var_def["name"],
        variable_format=var_def.get("format", "(no format spec)"),
        current_value=current_value or "(empty)",
        evidence=evidence,
    )
    try:
        resp = gsgpt.chat(prompt, model=model, max_tokens=2048)
        text = resp.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif text.startswith("```"):
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        # tolerant JSON parse — find first complete object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            j = json.loads(m.group(0))
            return {
                "variable": var_def["name"],
                "current_value": current_value,
                "is_correct": bool(j.get("is_correct", False)),
                "confidence": str(j.get("confidence", "low")).lower(),
                "correction": j.get("correction"),
                "rationale": j.get("rationale", ""),
                "error": None,
            }
        return {"variable": var_def["name"], "current_value": current_value,
                "is_correct": None, "confidence": None, "correction": None,
                "rationale": "", "error": "no JSON in response"}
    except Exception as e:
        return {"variable": var_def["name"], "current_value": current_value,
                "is_correct": None, "confidence": None, "correction": None,
                "rationale": "", "error": str(e)}


def apply_corrections(patient_info: dict, audit_entries: list) -> dict:
    """Apply confidence>=med corrections to a deep copy of patient_info."""
    import copy
    pi = copy.deepcopy(patient_info)

    # Map variable_name → (section, key) using EVAL_VARIABLES_V2 if it has paths,
    # else try simple heuristics.
    SECTION_KEY = {
        "Date of Birth":      ("PATIENT DEMOGRAPHICS", "date_of_birth"),
        "Sex":                ("PATIENT DEMOGRAPHICS", "sex"),
        "Smoking Status":     ("PATIENT DEMOGRAPHICS", "smoking_history"),
        "ECOG Performance Status": ("PATIENT DEMOGRAPHICS", "ecog_performance_status"),
        "Allergies":          ("PATIENT DEMOGRAPHICS", "allergies"),
        "DNR":                ("PATIENT DEMOGRAPHICS", "dnr"),
        "Therapy Toxicity / Comorbidities": ("PATIENT DEMOGRAPHICS", "therapy_toxicities"),
        "Diagnosis":          ("TUMOR INFORMATION", "diagnosis"),
        "Histology":          ("TUMOR INFORMATION", "histology"),
        "Metastasis":         ("TUMOR INFORMATION", "metastasis_status"),
        "Lymph Node Involvement": ("TUMOR INFORMATION", "lymph_node_involvement"),
        "Genetic Testing Panel": ("TUMOR INFORMATION", "driver_mutations"),
        "Current Medical Therapy": ("TREATMENTS", "current"),
        "Previous Surgery":   ("TREATMENTS", "previous"),
        "Radiation Therapy":  ("TREATMENTS", "radiation_therapy"),
        "Date of Last CT":    ("TREATMENTS", "date_of_last_ct"),
    }

    applied = []
    for entry in audit_entries:
        if entry.get("is_correct") is not False:
            continue
        conf = entry.get("confidence", "low")
        if conf not in ("high", "med"):
            continue
        path = SECTION_KEY.get(entry["variable"])
        if not path:
            continue
        section, key = path
        if section not in pi:
            pi[section] = {}
        old = pi[section].get(key)
        new = entry["correction"]
        if new is None:
            continue
        pi[section][key] = new
        applied.append({"variable": entry["variable"], "old": old, "new": new,
                        "confidence": conf, "rationale": entry["rationale"]})
    return pi, applied


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--variable", default=None, help="If set, verify only this variable")
    ap.add_argument("--model", default="claude-opus-4-6")
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    pi_path = RUNS / args.pid / "patient_info.json"
    if not pi_path.exists():
        print(f"ERROR: no patient_info.json at {pi_path}", file=sys.stderr)
        return 2
    patient_info = json.load(open(pi_path))

    # Pull current extracted values via quick_eval
    extracted = qe.extract_variables_from_snapshot(args.pid, pi_path.parent, no_graph_fallbacks=True)

    vars_to_check = EVAL_VARIABLES_V2
    if args.variable:
        vars_to_check = [v for v in EVAL_VARIABLES_V2 if v["name"] == args.variable]
        if not vars_to_check:
            print(f"ERROR: unknown variable {args.variable!r}", file=sys.stderr)
            return 2

    audit_entries = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(verify_one, args.pid, v, str(extracted.get(v["name"], "")), args.model): v["name"]
                   for v in vars_to_check}
        for fut in as_completed(futures):
            res = fut.result()
            audit_entries.append(res)

    # Order audit entries by variable position
    order = {v["name"]: i for i, v in enumerate(EVAL_VARIABLES_V2)}
    audit_entries.sort(key=lambda e: order.get(e["variable"], 999))

    # Apply corrections
    pi_corrected, applied = apply_corrections(patient_info, audit_entries)
    out_pi = RUNS / args.pid / "patient_info_verified.json"
    out_audit = RUNS / args.pid / "verifier_audit.json"
    out_pi.write_text(json.dumps(pi_corrected, indent=2))
    out_audit.write_text(json.dumps({"audit": audit_entries, "applied": applied}, indent=2))

    n_inc = sum(1 for e in audit_entries if e.get("is_correct") is False)
    n_app = len(applied)
    print(f"PID {args.pid}: verified {len(audit_entries)} variables, flagged {n_inc} as incorrect, applied {n_app} corrections (confidence>=med)")
    for a in applied:
        print(f"  ✎ {a['variable']:<32}  ({a['confidence']})  {a['rationale'][:120]}")
    print(f"  → {out_pi.name}, {out_audit.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
