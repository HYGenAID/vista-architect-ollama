#!/usr/bin/env python3
"""Judge-driven corrector v2 — grouped by clinical domain.

Improvements over v1:
  - 4 verifier groups run in parallel per patient (Identity/Safety, Tumor Biology,
    Disease Spread, Treatment/Timeline). Each verifier sees its full group together,
    enabling cross-variable reasoning (e.g., histology must be consistent with
    diagnosis; LN status must be consistent with metastasis status).
  - Operates on raw patient_info.json STRUCTURE (not the eval-extractor's
    flattened strings) — preserves field types (str, list, dict).
  - Conservative: defaults to keep_original; corrections require explicit
    EHR-cited rationale.
  - Audit-only mode: --dry-run produces patient_info_judgefix_v2.json without
    overwriting patient_info.json.

Usage:
    python agentic_toa/tools/judge_driven_corrector_v2.py --pid 136093825 \
        --judge-eval-file eval_strict_fresh.json --threshold 8
"""
from __future__ import annotations
import argparse, copy, json, re, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import gsgpt
import quick_eval as qe

RUNS = ROOT / "agentic_toa" / "runs"

# Field paths in patient_info.json + expected type
FIELD_PATHS = {
    "Date of Birth":             ("PATIENT DEMOGRAPHICS", "date_of_birth",         "str"),
    "Sex":                       ("PATIENT DEMOGRAPHICS", "sex",                   "str"),
    "Smoking Status":            ("PATIENT DEMOGRAPHICS", "smoking_history",       "str"),
    "ECOG Performance Status":   ("PATIENT DEMOGRAPHICS", "ecog_performance_status","str"),
    "Allergies":                 ("PATIENT DEMOGRAPHICS", "allergies",             "list_of_str"),
    "DNR":                       ("PATIENT DEMOGRAPHICS", "dnr",                   "str"),
    "Therapy Toxicity / Comorbidities": ("PATIENT DEMOGRAPHICS", "therapy_toxicities", "str"),
    "Diagnosis":                 ("TUMOR INFORMATION",    "diagnosis",             "str"),
    "Histology":                 ("TUMOR INFORMATION",    "histology",             "str"),
    "Metastasis":                ("TUMOR INFORMATION",    "metastasis_status",     "str"),
    "Lymph Node Involvement":    ("TUMOR INFORMATION",    "lymph_node_involvement","str"),
    "Genetic Testing Panel":     ("TUMOR INFORMATION",    "driver_mutations",      "dict"),
    "Current Medical Therapy":   ("TREATMENTS",           "current",               "list_of_str"),
    "Previous Surgery":          ("TREATMENTS",           "previous",              "list_of_str"),
    "Radiation Therapy":         ("TREATMENTS",           "radiation_therapy",     "str"),
    "Date of Last CT":           ("TREATMENTS",           "date_of_last_ct",       "str"),
}

# Clinical-domain groupings — each group is verified together
GROUPS = {
    "Identity & Safety": [
        "Date of Birth", "Sex", "Smoking Status", "ECOG Performance Status",
        "Allergies", "DNR",
    ],
    "Tumor Biology": [
        "Diagnosis", "Histology", "Genetic Testing Panel",
    ],
    "Disease Spread": [
        "Metastasis", "Lymph Node Involvement",
    ],
    "Treatment & Timeline": [
        "Current Medical Therapy", "Previous Surgery", "Radiation Therapy",
        "Therapy Toxicity / Comorbidities", "Date of Last CT",
    ],
}


GROUP_PROMPT = """You are a clinical verifier checking a related group of tumor-board variable extractions against the EHR before they are shown to a clinician. Your group: **{group_name}**.

Variables in this group are related and should be internally consistent (e.g., histology must be compatible with diagnosis; lymph-node involvement should not contradict metastasis assessment; smoking status should not contradict ECOG/comorbidity context).

For each variable below, decide: keep_original (extraction is correct) OR correct (extraction is factually wrong, with an evidence-cited replacement). Be CONSERVATIVE — default to keep_original unless the EHR evidence unambiguously contradicts the current value.

Strict rules:
  - **Preserve the original structure/type.** If current_value is a list, return a list. If it's a dict, return a dict. If it's a string, return a string. Do not "summarize" a list into a string or vice versa.
  - **Address ONLY factual errors.** Do not reformat correct content. Stylistic preferences are not corrections.
  - **Honest 'Unknown' / 'No' / 'None' is correct** when the EHR truly does not document the field.
  - **Verbose accurate ≠ wrong.** If the current value contains extra correct context, keep it.
  - **Cite the chart.** Every correction must quote or paraphrase a specific note/finding.

VARIABLES IN THIS GROUP:
{variables_block}

EHR EVIDENCE (deterministic graph-search hits + final XML chunk):
{evidence}

Return JSON only, one entry per variable in the SAME order:
```json
{{
  "decisions": [
    {{
      "variable": "<variable name>",
      "keep_original": true | false,
      "corrected_value": <new value in the SAME type as current_value, or null if keep_original=true>,
      "rationale": "<one sentence with chart evidence>"
    }},
    ...
  ]
}}
```
"""


def build_evidence(pid: str) -> str:
    xml_path = qe._resolve_truncated_xml(pid, cohort_name=None)
    if xml_path is None or not Path(xml_path).exists():
        return "(no XML evidence available)"
    try:
        from toa.graph_store import CohortGraphStore
        from toa.deterministic_retrieval import DeterministicRetriever
        from toa.backend import _format_deterministic_guidance
        g = CohortGraphStore("graph_store").load_patient_graph(pid)
        det = DeterministicRetriever(g).get_comprehensive_context()
        det_str = _format_deterministic_guidance(det)
    except Exception as e:
        det_str = f"(deterministic context unavailable: {e})"
    full_xml = Path(xml_path).read_text()
    xml_chunk = full_xml[-100_000:]
    import xml.etree.ElementTree as ET
    demo = ""
    try:
        person = ET.parse(str(xml_path)).getroot().find(".//person")
        if person is not None:
            demo = ET.tostring(person, encoding="unicode")
    except Exception:
        pass
    parts = []
    if demo: parts.append("=== DEMOGRAPHICS ===\n" + demo)
    parts.append("=== DETERMINISTIC GRAPH-SEARCH CONTEXT ===\n" + det_str)
    parts.append("=== XML (final 100K chars) ===\n" + xml_chunk)
    return "\n\n".join(parts)


def verify_group(pid: str, group_name: str, variables: list[dict], evidence: str,
                 model: str) -> dict:
    vars_block = json.dumps(variables, indent=2, default=str)
    prompt = GROUP_PROMPT.format(
        group_name=group_name, variables_block=vars_block, evidence=evidence,
    )
    try:
        resp = gsgpt.chat(prompt, model=model, max_tokens=4096)
        text = resp.strip()
        if "```json" in text: text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif text.startswith("```"): text = text.split("```", 1)[1].split("```", 1)[0].strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {"group": group_name, "error": "no JSON", "raw": text[:300]}
        j = json.loads(m.group(0))
        return {"group": group_name, "decisions": j.get("decisions", []), "error": None}
    except Exception as e:
        return {"group": group_name, "error": str(e), "decisions": []}


def coerce_type(value, expected_type):
    if value is None: return None
    if expected_type == "str":
        return value if isinstance(value, str) else str(value)
    if expected_type == "list_of_str":
        if isinstance(value, list): return [str(x) for x in value]
        if isinstance(value, str): return [value]
        return None
    if expected_type == "dict":
        if isinstance(value, dict): return value
        return None
    return value


def apply_decisions(patient_info: dict, group_results: list[dict]) -> tuple[dict, list]:
    pi = copy.deepcopy(patient_info)
    applied = []
    for gr in group_results:
        for d in gr.get("decisions", []):
            if d.get("keep_original", True): continue
            var = d.get("variable")
            path = FIELD_PATHS.get(var)
            if not path: continue
            section, key, vtype = path
            new = coerce_type(d.get("corrected_value"), vtype)
            if new is None: continue
            if section not in pi: pi[section] = {}
            old = pi[section].get(key)
            # Defensive: don't apply if the correction is a degenerate empty value
            # for a non-empty original (avoid wiping out content)
            if (old and not new) or (isinstance(old, list) and isinstance(new, list) and len(new) == 0 and len(old) > 0):
                continue
            pi[section][key] = new
            applied.append({"group": gr["group"], "variable": var,
                            "old": old, "new": new, "rationale": d.get("rationale", "")})
    return pi, applied


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--judge-eval-file", default="eval_strict_fresh.json")
    ap.add_argument("--threshold", type=int, default=8)
    ap.add_argument("--model", default="claude-opus-4-6")
    ap.add_argument("--parallel-groups", type=int, default=4)
    args = ap.parse_args()

    pi_path = RUNS / args.pid / "patient_info.json"
    if not pi_path.exists():
        print(f"ERROR: no patient_info at {pi_path}", file=sys.stderr); return 2
    patient_info = json.load(open(pi_path))

    je_path = RUNS / args.pid / args.judge_eval_file
    if not je_path.exists():
        print(f"ERROR: no judge eval at {je_path}", file=sys.stderr); return 2
    judge_items = {r["variable"]: r for r in json.load(open(je_path))["results"]}

    # Filter groups to only those with at least one flagged variable
    flagged_vars = {v for v, item in judge_items.items() if item.get("score", 10) < args.threshold}
    if not flagged_vars:
        print(f"PID {args.pid}: no variables below threshold {args.threshold}; nothing to do.")
        out = RUNS / args.pid / "patient_info_judgefix_v2.json"
        out.write_text(json.dumps(patient_info, indent=2))
        return 0

    print(f"PID {args.pid}: {len(flagged_vars)} flagged: {sorted(flagged_vars)}", file=sys.stderr)

    # For each group that contains ≥1 flagged var, build the group payload with
    # ALL variables in that group (so the verifier sees the context, even for
    # ones the judge accepted — needed for cross-variable consistency reasoning).
    groups_to_run = []
    for group_name, group_vars in GROUPS.items():
        if not any(v in flagged_vars for v in group_vars): continue
        variables_payload = []
        for var in group_vars:
            path = FIELD_PATHS.get(var)
            if not path: continue
            section, key, vtype = path
            current = patient_info.get(section, {}).get(key)
            j = judge_items.get(var, {})
            variables_payload.append({
                "variable": var,
                "expected_type": vtype,
                "current_value": current,
                "judge_score": j.get("score"),
                "judge_explanation": j.get("explanation", "")[:400],
                "flagged_for_review": var in flagged_vars,
            })
        groups_to_run.append((group_name, variables_payload))

    print(f"  → {len(groups_to_run)} groups need verification: {[g[0] for g in groups_to_run]}", file=sys.stderr)

    evidence = build_evidence(args.pid)

    results = []
    with ThreadPoolExecutor(max_workers=args.parallel_groups) as pool:
        futures = [pool.submit(verify_group, args.pid, gn, gv, evidence, args.model)
                   for gn, gv in groups_to_run]
        for fut in as_completed(futures):
            results.append(fut.result())

    pi_fixed, applied = apply_decisions(patient_info, results)
    out_pi = RUNS / args.pid / "patient_info_judgefix_v2.json"
    out_audit = RUNS / args.pid / "corrector_v2_audit.json"
    out_pi.write_text(json.dumps(pi_fixed, indent=2))
    out_audit.write_text(json.dumps({
        "threshold": args.threshold, "flagged": sorted(flagged_vars),
        "group_results": results, "applied": applied,
    }, indent=2))

    print(f"PID {args.pid}: groups_run={len(groups_to_run)}, decisions_total={sum(len(g.get('decisions', [])) for g in results)}, applied={len(applied)}")
    for a in applied:
        rat = a.get("rationale", "")[:140]
        print(f"  ✎ [{a['group']:<22}] {a['variable']:<32}  {rat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
