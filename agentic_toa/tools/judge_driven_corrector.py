#!/usr/bin/env python3
"""Judge-driven corrector for agentic-pipeline display items.

Idea: the judge already flags wrong items AND explains why. The corrector just
acts on that specific feedback in the structurally-correct shape — no
second-guessing, no reformatting things the judge accepted.

Flow per patient:
  1. Read judge results (eval_strict_gemini.json) and patient_info.json
  2. For each variable with score < threshold:
        prompt = {
            variable, current_value, judge_rationale, judge_score,
            field_format_spec, evidence
        }
        → corrector LLM returns a structurally-correct replacement (or "keep")
  3. Apply replacements to patient_info.json with the right shape per field
  4. Write patient_info_judgefix.json + corrector_audit.json

Usage:
    python agentic_toa/tools/judge_driven_corrector.py --pid 135925902 \
        --judge-eval-file eval_strict_gemini.json --threshold 8

    python agentic_toa/tools/judge_driven_corrector.py --pid 135925902 \
        --threshold 8 --model claude-opus-4-6
"""
from __future__ import annotations
import argparse, copy, json, re, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import gsgpt
import quick_eval as qe
from eval_variables_v2 import EVAL_VARIABLES_V2

RUNS = ROOT / "agentic_toa" / "runs"

# Where each variable lives in patient_info.json and what shape it should be
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


CORRECTOR_PROMPT = """A judge has flagged a tumor-board variable extraction as incorrect. Your job is to produce a corrected value addressing the judge's specific feedback. Be conservative — only change what the judge identifies as wrong.

VARIABLE: {variable_name}
EXPECTED FORMAT (per the system schema): {field_format_spec}
TYPE: {value_type}

CURRENT EXTRACTED VALUE (from patient_info.json — note its STRUCTURE):
```json
{current_value_json}
```

JUDGE SCORE: {judge_score}/10
JUDGE FEEDBACK (this is what to address):
"{judge_rationale}"

RELEVANT EHR EVIDENCE (deterministic graph-search hits + final XML chunk):
{evidence}

Produce a corrected value. Rules:
  - Output MUST match the same type and structure as CURRENT EXTRACTED VALUE (string stays string; dict stays dict; list of strings stays list of strings).
  - Address ONLY the specific issue the judge raised. Do not reformat or restructure correct content.
  - Cite EHR evidence in the rationale (a phrase or date from the chart).
  - If the judge's complaint is purely about format/style rather than fact, set keep_original=true.
  - If the EHR genuinely doesn't support a corrected value (e.g., judge thinks something is missing but it really isn't documented), set keep_original=true.

Return JSON only:
{{
  "keep_original": true | false,
  "corrected_value": <new value in correct type/structure, or null if keep_original=true>,
  "rationale": "<one sentence citing chart evidence>"
}}
"""


def build_evidence(pid: str) -> str:
    """Reuse quick_eval's existing evidence assembly (same as judge sees)."""
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


def correct_one(pid: str, variable: str, judge_item: dict, current_value, evidence: str,
                model: str) -> dict:
    path = FIELD_PATHS.get(variable)
    if not path:
        return {"variable": variable, "skipped": True, "reason": "no field-path mapping"}
    section, key, value_type = path
    var_def = next((v for v in EVAL_VARIABLES_V2 if v["name"] == variable), {})
    fmt_spec = var_def.get("format", "(no spec)")

    current_json = json.dumps(current_value, indent=2) if current_value is not None else "null"

    prompt = CORRECTOR_PROMPT.format(
        variable_name=variable,
        field_format_spec=fmt_spec,
        value_type=value_type,
        current_value_json=current_json,
        judge_score=judge_item.get("score", "?"),
        judge_rationale=judge_item.get("explanation", "(none)"),
        evidence=evidence,
    )

    try:
        resp = gsgpt.chat(prompt, model=model, max_tokens=2048)
        text = resp.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif text.startswith("```"):
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {"variable": variable, "error": "no JSON in response", "raw": text[:300]}
        j = json.loads(m.group(0))
        keep = bool(j.get("keep_original", False))
        new_val = j.get("corrected_value") if not keep else None
        # Type-coerce the corrected value to match expected shape
        if not keep and new_val is not None:
            if value_type == "str" and not isinstance(new_val, str):
                new_val = str(new_val)
            elif value_type == "list_of_str":
                if isinstance(new_val, str):
                    # Single string — wrap in list ONLY if it makes sense; otherwise reject
                    new_val = [new_val]
                elif not isinstance(new_val, list):
                    return {"variable": variable, "error": f"expected list_of_str, got {type(new_val).__name__}", "raw_value": new_val}
                else:
                    new_val = [str(x) for x in new_val]
            elif value_type == "dict" and not isinstance(new_val, dict):
                return {"variable": variable, "error": f"expected dict, got {type(new_val).__name__}", "raw_value": new_val}
        return {
            "variable": variable,
            "judge_score": judge_item.get("score"),
            "judge_rationale": judge_item.get("explanation", ""),
            "current_value": current_value,
            "keep_original": keep,
            "corrected_value": new_val,
            "rationale": j.get("rationale", ""),
            "error": None,
        }
    except Exception as e:
        return {"variable": variable, "error": str(e), "current_value": current_value}


def apply_corrections(patient_info: dict, corrections: list) -> tuple[dict, list]:
    pi = copy.deepcopy(patient_info)
    applied = []
    for c in corrections:
        if c.get("error") or c.get("skipped") or c.get("keep_original"):
            continue
        if c.get("corrected_value") is None:
            continue
        path = FIELD_PATHS.get(c["variable"])
        if not path: continue
        section, key, _vt = path
        if section not in pi:
            pi[section] = {}
        old = pi[section].get(key)
        pi[section][key] = c["corrected_value"]
        applied.append({
            "variable": c["variable"],
            "old": old,
            "new": c["corrected_value"],
            "judge_score": c["judge_score"],
            "rationale": c["rationale"],
        })
    return pi, applied


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--judge-eval-file", default="eval_strict_gemini_150k.json",
                    help="Filename inside runs/<pid>/ with the judge scores")
    ap.add_argument("--threshold", type=int, default=8,
                    help="Score below this triggers correction")
    ap.add_argument("--model", default="claude-opus-4-6")
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()

    pi_path = RUNS / args.pid / "patient_info.json"
    if not pi_path.exists():
        print(f"ERROR: no patient_info at {pi_path}", file=sys.stderr); return 2
    patient_info = json.load(open(pi_path))

    je_path = RUNS / args.pid / args.judge_eval_file
    if not je_path.exists():
        print(f"ERROR: no judge eval at {je_path}", file=sys.stderr); return 2
    judge_eval = json.load(open(je_path))
    judge_items = {r["variable"]: r for r in judge_eval["results"]}

    flagged = [(var, item) for var, item in judge_items.items() if item.get("score", 10) < args.threshold]
    if not flagged:
        print(f"PID {args.pid}: nothing below threshold {args.threshold}; no corrections needed.")
        out = RUNS / args.pid / "patient_info_judgefix.json"
        out.write_text(json.dumps(patient_info, indent=2))
        return 0

    print(f"PID {args.pid}: {len(flagged)} flagged items below threshold {args.threshold}", file=sys.stderr)

    evidence = build_evidence(args.pid)

    # Pull current values directly from patient_info.json (raw structure, not eval-extractor strings)
    def get_current(var):
        path = FIELD_PATHS.get(var)
        if not path: return None
        section, key, _ = path
        return patient_info.get(section, {}).get(key)

    corrections = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(correct_one, args.pid, var, item, get_current(var), evidence, args.model): var
                   for var, item in flagged}
        for fut in as_completed(futures):
            res = fut.result()
            corrections.append(res)

    # Order by variable position
    order = {v["name"]: i for i, v in enumerate(EVAL_VARIABLES_V2)}
    corrections.sort(key=lambda c: order.get(c.get("variable", ""), 999))

    pi_fixed, applied = apply_corrections(patient_info, corrections)
    out_pi = RUNS / args.pid / "patient_info_judgefix.json"
    out_audit = RUNS / args.pid / "corrector_audit.json"
    out_pi.write_text(json.dumps(pi_fixed, indent=2))
    out_audit.write_text(json.dumps({
        "threshold": args.threshold,
        "flagged_count": len(flagged),
        "corrections": corrections,
        "applied": applied,
    }, indent=2))

    print(f"PID {args.pid}: flagged={len(flagged)}, attempted={len(corrections)}, applied={len(applied)}, kept_original={sum(1 for c in corrections if c.get('keep_original'))}, errored={sum(1 for c in corrections if c.get('error'))}")
    for a in applied:
        print(f"  ✎ {a['variable']:<32}  judge_score={a['judge_score']}  {a['rationale'][:140]}")
    print(f"  → {out_pi.name}, {out_audit.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
