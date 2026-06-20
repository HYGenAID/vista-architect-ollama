#!/usr/bin/env python3
"""
Quick automated evaluation of JSON outputs against XML evidence.

V2: Tests 17 tumor board variables (15 primary + 2 secondary) for accuracy.
Supports two modes:
  1. Standard mode (2 LLM calls): Extract variables from dashboard + judge vs XML
  2. Snapshot mode (1 LLM call): Read MTB_VARIABLES from patient_info.json, only judge

Uses an LLM judge (default: GPT-5) routed through the pluggable vista_llm
backend (see SETUP.md for backend configuration).
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
import time

sys.path.insert(0, str(Path(__file__).parent))
import gsgpt
from eval_variables_v2 import EVAL_VARIABLES_V2, PRIMARY_VARIABLES, VARIABLE_NAMES, CATEGORIES

# Default eval model
EVAL_MODEL = "gpt-5"

def call_llm(system_prompt: str, user_prompt: str, model: str = None, max_retries: int = 5) -> str:
    """Call LLM via gsgpt with automatic retry on rate limits."""
    model = model or EVAL_MODEL

    # Pass thinking_budget for Gemini Flash variants (default gsgpt auto-zeros them)
    extra = {}
    if 'gemini' in model.lower() and 'flash' in model.lower():
        extra['thinking_budget'] = globals().get('EVAL_THINKING', 1024)

    for attempt in range(max_retries):
        try:
            return gsgpt.chat(
                user_prompt,
                system=system_prompt if system_prompt else None,
                model=model,
                max_tokens=8192,
                **extra,
            )
        except Exception as e:
            error_str = str(e)
            if '429' in error_str and attempt < max_retries - 1:
                wait_time = 120
                print(f"  Rate limit hit (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries - 1}...")
                time.sleep(wait_time)
                continue
            elif ('timeout' in error_str.lower() or 'timed out' in error_str.lower()) and attempt < max_retries - 1:
                wait_time = 30
                print(f"  Timeout. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries - 1})")
                time.sleep(wait_time)
                continue
            else:
                raise

    raise Exception(f"Failed after {max_retries} retries")


# Use V2 variable definitions
EVAL_VARIABLES = EVAL_VARIABLES_V2


def extract_variable_from_json(variable_name: str, patient_jsons: dict) -> str:
    """Extract a specific variable from patient JSONs."""

    # This is a simplified extraction - GPT-5 will do the actual extraction
    # Just return relevant JSON sections

    if variable_name in ["Date of Birth", "Sex"]:
        return json.dumps(patient_jsons.get("patient_info", {}).get("demographics", {}))

    elif variable_name == "Diagnosis":
        return json.dumps({
            "patient_info": patient_jsons.get("patient_info", {}),
            "tumor_info": patient_jsons.get("tumor_info", {})
        })

    elif variable_name == "Histology":
        return json.dumps({
            "patient_info": patient_jsons.get("patient_info", {}),
            "tumor_info": patient_jsons.get("tumor_info", {})
        })

    elif variable_name in ["Metastasis", "Lymph Node Involvement"]:
        return json.dumps({
            "tumor_info": patient_jsons.get("tumor_info", {}),
            "summary": patient_jsons.get("summary", {})
        })

    elif variable_name == "Current Treatment":
        return json.dumps(patient_jsons.get("treatment_plans", {}))

    elif variable_name == "Previous Surgery":
        timeline = patient_jsons.get("timeline", {}).get("events", [])
        surgeries = [e for e in timeline if e.get("type") == "surgery"]
        return json.dumps({"surgeries": surgeries})

    elif variable_name == "Date of Last CT":
        timeline = patient_jsons.get("timeline", {}).get("events", [])
        cts = [e for e in timeline if e.get("type") == "imaging" and "ct" in e.get("description", "").lower()]
        return json.dumps({"ct_scans": cts})

    return "{}"


def extract_demographics_from_xml(xml_path: str) -> dict:
    """Extract demographics (DOB, Sex) from XML as ground truth."""
    root = ET.parse(xml_path).getroot()

    demographics = {}

    # Find person element
    person = root.find(".//person")
    if person is not None:
        # DOB
        birthdate = person.find("birthdate")
        if birthdate is not None and birthdate.text:
            demographics["dob"] = birthdate.text

        # Sex
        gender_elem = person.find(".//gender")
        if gender_elem is not None and gender_elem.text:
            demographics["sex"] = gender_elem.text

    return demographics


def extract_surgery_dates_from_timeline(timeline_jsonl_path: str) -> list:
    """Extract surgery dates from timeline_objects.jsonl."""
    if not Path(timeline_jsonl_path).exists():
        return []

    events = []
    with open(timeline_jsonl_path) as f:
        for line in f:
            events.append(json.loads(line))

    surgeries = [e for e in events if e.get("type") == "surgery"]
    return [s.get("date") for s in surgeries if s.get("date")]


def extract_xml_evidence_for_date(xml_path: str, date: str) -> str:
    """Extract XML entries from a specific date (filtered to relevant types)."""
    root = ET.parse(xml_path).getroot()
    evidence = []

    RELEVANT_TYPES = {'procedure', 'note', 'condition', 'observation'}

    for entry in root.findall(".//entry"):
        timestamp = entry.get("timestamp", "")
        if timestamp.startswith(date):
            events = entry.findall("event")
            has_relevant = any(e.get("type") in RELEVANT_TYPES for e in events)
            if has_relevant:
                entry_str = ET.tostring(entry, encoding="unicode")
                evidence.append(entry_str)

    return "\n\n".join(evidence)


def extract_final_chunk(xml_path: str, max_chars: int = 50000) -> str:
    """Extract final chunk of XML (often contains summary/recent info)."""
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_text = f.read()

    # Get last N chars
    return xml_text[-max_chars:] if len(xml_text) > max_chars else xml_text


def extract_xml_evidence_for_variable(xml_path: str, variable_name: str, surgery_dates: list = None) -> str:
    """Extract relevant XML evidence for a specific variable.

    Includes:
    1. Variable-specific evidence
    2. Evidence from surgery dates (if any)
    3. Final chunk of XML
    """

    root = ET.parse(xml_path).getroot()
    evidence_parts = []

    # Relevant event types (like streamlined TOA)
    RELEVANT_TYPES = {'procedure', 'note', 'condition', 'drug_exposure', 'observation', 'diagnostic'}

    # 1. Variable-specific evidence
    variable_evidence = []

    if variable_name in ["Date of Birth", "Sex"]:
        # Return demographics section
        person = root.find(".//person")
        if person is not None:
            variable_evidence.append(ET.tostring(person, encoding="unicode"))

    elif variable_name == "DNR":
        # Search for DNR mentions in notes/observations
        for entry in root.findall(".//entry"):
            events = entry.findall("event")
            for event in events:
                if event.get("type") in {"note", "observation"}:
                    content = event.text or ""
                    name = event.get("name", "")
                    if "dnr" in content.lower() or "dnr" in name.lower() or "do not resuscitate" in content.lower():
                        entry_str = ET.tostring(entry, encoding="unicode")
                        variable_evidence.append(entry_str)
                        break

    elif variable_name == "Previous Surgery":
        # Get procedure entries from surgery dates
        if surgery_dates:
            for date in surgery_dates:
                date_evidence = extract_xml_evidence_for_date(xml_path, date)
                if date_evidence:
                    variable_evidence.append(f"<!-- Surgery date: {date} -->\n{date_evidence}")

    elif variable_name in ["Metastasis", "Lymph Node Involvement", "Diagnosis", "Histology"]:
        # Get diagnostic and pathology notes (limited sample)
        count = 0
        for entry in root.findall(".//entry"):
            if count >= 20:  # Limit to first 20 relevant entries
                break
            events = entry.findall("event")
            has_relevant = any(e.get("type") in {"note", "condition", "observation"} for e in events)
            if has_relevant:
                entry_str = ET.tostring(entry, encoding="unicode")
                variable_evidence.append(entry_str)
                count += 1

    elif variable_name == "Date of Last CT":
        # Get imaging procedures
        for entry in root.findall(".//entry"):
            events = entry.findall("event")
            for event in events:
                if event.get("type") == "procedure" and "ct" in event.get("name", "").lower():
                    entry_str = ET.tostring(entry, encoding="unicode")
                    variable_evidence.append(entry_str)

    if variable_evidence:
        evidence_parts.append("=== VARIABLE-SPECIFIC EVIDENCE ===\n" + "\n\n".join(variable_evidence[:30]))

    # 2. Surgery date evidence (if applicable and not already included)
    if surgery_dates and variable_name != "Previous Surgery":
        surgery_evidence = []
        for date in surgery_dates[:2]:  # Max 2 surgery dates
            date_evidence = extract_xml_evidence_for_date(xml_path, date)
            if date_evidence:
                surgery_evidence.append(f"<!-- Surgery date: {date} -->\n{date_evidence[:10000]}")  # Limit per date

        if surgery_evidence:
            evidence_parts.append("\n\n=== SURGERY DATE EVIDENCE ===\n" + "\n\n".join(surgery_evidence))

    # 3. Final chunk
    final_chunk = extract_final_chunk(xml_path, max_chars=30000)
    evidence_parts.append(f"\n\n=== FINAL CHUNK (recent/summary info) ===\n{final_chunk}")

    return "\n\n".join(evidence_parts)


def extract_variables_from_snapshot(patient_id: str, json_dir: Path, no_graph_fallbacks: bool = False) -> dict:
    """
    Extract v2 MTB variables directly from pipeline outputs.

    Most variables come from patient_info.json (the core display item).
    Some (like Date of Last CT, Previous Surgery) come from the TOA timeline,
    since the timeline is the natural source for event-based data.

    Args:
        no_graph_fallbacks: If True, do not backfill missing values from
            the graph store. Use this when evaluating external snapshots
            (e.g., a RAG baseline) to prevent VISTA-only data from leaking
            into the evaluated output.

    Returns:
        Dict mapping variable name -> extracted value
    """
    patient_info_path = json_dir / "patient_info.json"
    if not patient_info_path.exists():
        print(f"  patient_info.json not found")
        return {v["name"]: "ERROR" for v in EVAL_VARIABLES}

    with open(patient_info_path) as f:
        patient_info = json.load(f)

    demographics = patient_info.get("PATIENT DEMOGRAPHICS", {})
    tumor_info_raw = patient_info.get("TUMOR INFORMATION", {})
    # Fix LLM key inconsistencies (leading spaces, misspellings of lymph_node_involvement)
    if isinstance(tumor_info_raw, dict):
        tumor_info = {}
        for k, v in tumor_info_raw.items():
            ck = k.strip()
            if ck != 'lymph_node_involvement' and 'node_involvement' in ck and ck.startswith(('l', 'y')):
                ck = 'lymph_node_involvement'
            tumor_info[ck] = v
    else:
        tumor_info = tumor_info_raw
    treatments = patient_info.get("TREATMENTS", {})

    result = {}

    # Demographics (from patient_info.json)
    result["Date of Birth"] = demographics.get("date_of_birth", "Unknown")
    result["Sex"] = demographics.get("sex", "Unknown")
    result["Smoking Status"] = demographics.get("smoking_history", "Unknown")

    # Tumor (from patient_info.json)
    result["Diagnosis"] = tumor_info.get("diagnosis", "Unknown")
    result["Histology"] = tumor_info.get("histology", "Unknown")
    result["Metastasis"] = tumor_info.get("metastasis_status", "Unknown")
    result["Lymph Node Involvement"] = tumor_info.get("lymph_node_involvement", "Unknown")

    # Genetic Testing Panel — only include genes with actual results (positive, negative, or percentage)
    # Filter out "Not tested" entries — they add no clinical info and confuse the judge
    driver_muts = tumor_info.get("driver_mutations", {})
    if isinstance(driver_muts, dict) and driver_muts:
        panel = []
        for gene, value in driver_muts.items():
            if isinstance(value, dict):
                # Nested dict (e.g. "Other": {"GNAS": "R021H", ...})
                for sub_gene, sub_val in value.items():
                    if sub_val and isinstance(sub_val, str):
                        panel.append(f"{sub_gene}: {sub_val}")
            elif value and isinstance(value, str) and value.lower() not in ("not tested", "not available", "unknown"):
                panel.append(f"{gene}: {value}")
        result["Genetic Testing Panel"] = "; ".join(panel) if panel else "Not tested"
    else:
        result["Genetic Testing Panel"] = "Not tested"

    # Clinical (from patient_info.json)
    result["ECOG Performance Status"] = demographics.get("ecog_performance_status", "Unknown")
    toxicities = demographics.get("therapy_toxicities", "None")
    conditions = demographics.get("previous_conditions", "")
    if toxicities in ("None", "none", "", None) and conditions:
        result["Therapy Toxicity / Comorbidities"] = f"Comorbidities: {conditions}"
    elif conditions and toxicities not in ("None", "none", "", None):
        result["Therapy Toxicity / Comorbidities"] = f"{toxicities}; Comorbidities: {conditions}"
    else:
        result["Therapy Toxicity / Comorbidities"] = toxicities if toxicities else "None"

    # Treatment: Previous Surgery — pass the full treatments.previous[] list to the judge.
    # The judge decides Yes/No (oncologic resection for current diagnosis) by inspecting
    # the list against the EHR. No keyword filtering in the eval extractor — that approach
    # gives both false negatives (missing rare surgical terms) and false positives
    # (matching "mediastinal" in a chemo description).
    prev_treatments = treatments.get("previous", [])
    if prev_treatments:
        joined = "; ".join(str(p)[:160] for p in prev_treatments[:15])
        result["Previous Surgery"] = f"previous_treatments_list: {joined}"
    else:
        result["Previous Surgery"] = "previous_treatments_list: (empty)"

    # Current Medical Therapy (from patient_info.json)
    current_tx = treatments.get("current", [])
    if current_tx:
        if isinstance(current_tx, list):
            result["Current Medical Therapy"] = "; ".join(str(t) for t in current_tx) if current_tx else "None"
        else:
            result["Current Medical Therapy"] = str(current_tx)
    else:
        result["Current Medical Therapy"] = "None"

    result["Radiation Therapy"] = treatments.get("radiation_therapy", "No")

    # Date of Last CT - primary source: patient_info.json (from deterministic ct_date_vector)
    # Timeline regex is unreliable (matches planned CTs, "CT recommended" in x-ray descriptions, etc.)
    ct_date = "Unknown"
    # Primary: patient_info.json date_of_last_ct (gpt-5 with CT date vector context)
    pi_ct = treatments.get("date_of_last_ct", "")
    if pi_ct and pi_ct.lower() not in ("unknown", "none", ""):
        ct_date = pi_ct  # Keep the date as the display LLM produced it (YYYY-MM-DD when available)
    # Fallback: graph store deterministic retrieval (structured imaging data)
    # Skipped for external snapshots (e.g. RAG baselines) to avoid leaking
    # VISTA-only graph data into the evaluated output.
    if ct_date == "Unknown" and not no_graph_fallbacks:
        try:
            from toa.graph_store import CohortGraphStore
            from toa.deterministic_retrieval import DeterministicRetriever
            store = CohortGraphStore("graph_store")
            graph = store.load_patient_graph(patient_id)
            retriever = DeterministicRetriever(graph)
            ct_dates = retriever.get_ct_date_vector()
            if ct_dates:
                ct_date = ct_dates[0][:7]  # Most recent, YYYY-MM
        except Exception:
            pass
    result["Date of Last CT"] = ct_date

    # Safety (from patient_info.json)
    allergies = demographics.get("allergies", [])
    if isinstance(allergies, list):
        result["Allergies"] = ", ".join(allergies) if allergies else "NKDA"
    else:
        result["Allergies"] = str(allergies) if allergies else "NKDA"

    result["DNR"] = demographics.get("dnr", "No")

    return result


def extract_variables_from_dashboard(patient_jsons: dict, timeline_jsonl_path: str, episodes_path: str, xml_path: str) -> dict:
    """Extract all 17 v2 variables from dashboard data (like a chat query).

    Mimics the app.py chat interface by providing:
    - Patient JSONs (patient_info, tumor_info, timeline, treatment_plans, summary)
    - Timeline objects
    - Episodes
    - Recent XML chunk

    Returns extracted values for all 17 variables.
    """

    # Read timeline objects
    timeline_events = []
    if Path(timeline_jsonl_path).exists():
        with open(timeline_jsonl_path) as f:
            for line in f:
                timeline_events.append(json.loads(line))

    # Read episodes
    episodes = []
    if Path(episodes_path).exists():
        with open(episodes_path) as f:
            episodes = json.load(f)

    # Get XML chunks: demographics + recent chunk
    root = ET.parse(xml_path).getroot()

    demographics_xml = ""
    person_elem = root.find(".//person")
    if person_elem is not None:
        demographics_xml = "=== DEMOGRAPHICS ===\n" + ET.tostring(person_elem, encoding="unicode") + "\n\n"

    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_text = f.read()
    recent_xml_chunk = xml_text[-120000:] if len(xml_text) > 120000 else xml_text
    recent_xml = demographics_xml + recent_xml_chunk

    # Build variable list for prompt
    var_descriptions = []
    for i, v in enumerate(EVAL_VARIABLES, 1):
        primary_tag = "" if v["primary"] else " [secondary]"
        var_descriptions.append(f"{i}. {v['name']} (format: {v['format']}){primary_tag}")

    system_prompt = f"""You are Vista, a medical AI assistant analyzing tumor board data.

Extract the following {len(EVAL_VARIABLES)} tumor board variables from the provided dashboard data:

{chr(10).join(var_descriptions)}

IMPORTANT:
- For DOB/Sex, always check XML demographics first
- For histology, provide only the basic type (adenocarcinoma, squamous, etc.)
- For CT date, ensure it's an actual CT scan, not other procedures
- For Metastasis/Lymph Nodes: Use Yes/Suspected/No
- For Genetic Testing Panel: Compare tested genes and results against XML. Accept equivalent naming (EML4-ALK = ALK fusion). VUS should be labeled as VUS. Negative results as Negative/Not detected. PD-L1 as percentage.
- For ECOG: Use 0-4 scale or "Unknown" if not documented
- For Smoking Status: Include pack-years if available, and current/former/never status
- For Radiation Therapy: Yes with details or No
- For Toxicity/Comorbidities: List major toxicities (grade >= 3) and serious comorbidities
- For DNR: If not documented, assume "No"

Be accurate and concise."""

    # Build JSON template
    json_template = {v["name"]: f"<{v['format']}>" for v in EVAL_VARIABLES}

    user_prompt = f"""
PATIENT JSONs (all):
{json.dumps(patient_jsons, indent=2)}

TIMELINE OBJECTS (all {len(timeline_events)} events):
{json.dumps(timeline_events, indent=2)}

EPISODES (all):
{json.dumps(episodes, indent=2)}

XML DATA (demographics + recent 120k chars):
{recent_xml[:120000]}

Extract all {len(EVAL_VARIABLES)} tumor board variables. Format as JSON:
{json.dumps(json_template, indent=2)}
"""

    response = call_llm(system_prompt, user_prompt)

    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        return json.loads(response)
    except Exception as e:
        print(f"  Failed to parse extraction response: {e}")
        return {var['name']: "ERROR" for var in EVAL_VARIABLES}


def judge_all_variables(extracted_values: dict, xml_evidence: str) -> dict:
    """Use GPT-5 to judge ALL v2 extracted variables against XML ground truth."""

    # Build variable list for prompt
    var_list = "\n".join([f"{i+1}. {v['name']} (expected format: {v['format']})" for i, v in enumerate(EVAL_VARIABLES)])

    # === GENERAL TRUTH-VALUE PRINCIPLE (model-conditional emphasis) ===
    # The judge should score whether the answer is TRUE given the EHR, not whether it
    # matches a preferred form, length, or ordering. Anthropic Opus models tend to
    # nitpick on form (single-word vs sentence, list ordering, terse vs verbose),
    # so we add an emphatic preamble when the judge is Opus.
    _eval_model = EVAL_MODEL.lower()
    _is_opus = 'opus' in _eval_model
    opus_emphasis = (
        "\n\nEXTRA EMPHASIS (because you are an Opus-family judge that has a known tendency "
        "to penalize form over substance): Score ONLY the truth value. Do NOT deduct for "
        "verbosity, terseness, list ordering, presence/absence of qualifiers, sentence vs "
        "single-word format, or any other purely stylistic difference. If the extracted "
        "answer is factually consistent with the EHR, score it 10 regardless of how it is "
        "phrased. Save your deductions for genuine factual errors, hallucinations, missed "
        "decision-relevant findings, and misleading content.\n"
    ) if _is_opus else ""

    system_prompt = f"""You are evaluating the quality of {len(EVAL_VARIABLES)} tumor board variables extracted from a clinical pipeline.

Compare the extracted values against the EHR ground truth provided.

For EACH variable, evaluate:
- Correctness: Correct/Incorrect/Partial/N/A
- Score: 1-10 (10=correct, 1=wrong, 5=partially correct)
- Brief explanation (1-2 sentences)
- XML value (ground truth value from XML)

=== CORE PRINCIPLE: JUDGE TRUTH, NOT FORM ===

You are scoring whether the answer is **factually correct given the EHR**, NOT whether it matches a preferred style, length, ordering, or completeness threshold. Concretely:

1. **Honest uncertainty is correct.** When the EHR genuinely does not document a field, "Unknown" or "Not documented" or "No" (where appropriate per the rules below) is a SCORE 10 answer — not a 7. The pipeline is being honest.
2. **Default-No is correct when not documented.** For binary/safety fields where the rubric specifies a default of "No" when undocumented (DNR, Metastasis with no findings, Radiation when none given), "No" without further qualification scores 10.
3. **Equivalent phrasings score the same.** "NKDA" = "No Known Allergies" = "None". "Former smoker, 30py, quit 2010" = "30 pack-year former smoker, quit 2010-04". Range vs single value for ECOG ("1-2" vs "1") both score 10 when both are documented.
4. **Verbose ≠ wrong.** Rich context alongside the core answer (e.g., "No — PET ruled out initial concern") scores the same as the bare answer ("No"). Do not deduct for *extra* clinically relevant detail.
5. **Order in lists does not matter** as long as the content is right and the most relevant items are recognizable.
6. **Form/style preferences are not deductions.** Do not deduct for: bullets vs prose, dates as YYYY-MM-DD vs YYYY-MM, "Yes (sites)" vs "Yes - sites", single-word vs sentence.

DEDUCT for: (a) factually wrong values, (b) hallucinations (entities not in the EHR), (c) missed decision-relevant findings actually documented in the EHR, (d) inverted clinical meaning (Full Code reported as DNR, "No" when EHR confirms "Yes", etc.), (e) made-up dates/genes/drugs.

=== VARIABLE-SPECIFIC SCOPE CLARIFICATIONS (only where the principle alone is ambiguous) ===

- **Metastasis / Lymph Node Involvement**: "Suspected" is correct when EHR evidence is suggestive but not definitive; "No" is correct if initial concern was resolved on follow-up imaging.
- **Previous Surgery — MANUSCRIPT SCOPE (Yes/No, oncologic resection for CURRENT diagnosis only)**:
  The extracted value is prefixed `previous_treatments_list:` and contains the full `treatments.previous[]` list from the patient_info JSON — a MIX of drug regimens, surgeries (current cancer + prior cancers + non-oncologic), and radiation entries. The list is provided as-is, intentionally without filtering, so you can evaluate against the EHR.
  Your task:
    1. Determine the ground truth from the EHR: did the patient undergo an **oncologic resection for the CURRENT cancer** (e.g., lobectomy/pneumonectomy for current NSCLC, thymectomy for current thymoma, craniotomy for current brain mets, colectomy/sigmoidectomy for current colon cancer, mediastinal mass resection, etc.)?
    2. Check whether such a current-cancer oncologic resection appears in the `previous_treatments_list`.
    3. Score:
       * 10 — list correctly contains the current-cancer resection AND the EHR documents one; OR list correctly contains no current-cancer resection AND the EHR documents none.
       * 5–6 — wrong direction (list omits a current-cancer resection that's in the EHR, or includes one that isn't).
       * 1–3 — list contains a fabricated current-cancer surgery (entity not in EHR).
  Do NOT deduct for: drug regimens being in the list (they're scored under Current Medical Therapy), prior-cancer surgeries being in the list (informational, not in scope), non-oncologic surgeries being in the list (informational, not in scope), surgeries NOT in the list if they're prior-cancer or non-oncologic.
- **Date of Last CT — SCOPE**: PET-CT counts as CT. Most recent in-house (from `ct_date_vector`) OR outside imaging in chart, whichever is more recent, is acceptable. Month-level (YYYY-MM) acceptable when day uncertain.
- **Genetic Testing Panel — SCOPE**: do not penalize molecular results that are clinically correct but not visible in the provided XML chunk (the chunk may be truncated). Only penalize factually wrong or hallucinated results.

Be concise but accurate.{opus_emphasis}"""

    user_prompt = f"""
EVALUATE THESE {len(EVAL_VARIABLES)} VARIABLES:
{var_list}

EXTRACTED VALUES:
{json.dumps(extracted_values, indent=2)}

XML GROUND TRUTH (demographics + surgery dates + final 120k chunk):
{xml_evidence}

For each variable, evaluate the extracted value against XML ground truth.

Format as JSON with ALL {len(EVAL_VARIABLES)} variables:
{{
  "Date of Birth": {{
    "correctness": "<Correct|Incorrect|Partial|N/A>",
    "score": <1-10>,
    "explanation": "<brief explanation>",
    "xml_value": "<ground truth from XML>"
  }},
  ... (all {len(EVAL_VARIABLES)} variables)
}}
"""

    response = call_llm(system_prompt, user_prompt)

    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        return json.loads(response)
    except Exception as e:
        print(f"  Failed to parse judge response: {e}")
        return {var['name']: {
            "correctness": "Error",
            "score": 0,
            "explanation": f"Parse error: {e}",
            "xml_value": ""
        } for var in EVAL_VARIABLES}


GRAPH_STORE_INDEX = None  # lazy-loaded

def _load_graph_store_index() -> dict:
    """Lazy-load the graph store index (TB dates for all patients)."""
    global GRAPH_STORE_INDEX
    if GRAPH_STORE_INDEX is None:
        index_path = Path("graph_store/index.json")
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
            GRAPH_STORE_INDEX = data.get("patients", {})
        else:
            GRAPH_STORE_INDEX = {}
    return GRAPH_STORE_INDEX


def _resolve_truncated_xml(patient_id: str, cohort_name: str = None) -> Path:
    """Find or generate a TB-date-truncated XML for evaluation ground truth.

    Priority:
    1. Cohort-specific truncated XML (if cohort_name provided)
    2. Any existing truncated XML for this patient (glob match)
    3. Auto-generate from graph store TB date
    4. None (error)
    """
    cohorts_dir = Path("patient_records/cohorts")
    full_xml = Path(f"patient_records/thoracic_xmls/{patient_id}.xml")

    # 1. Cohort-specific truncated XML
    if cohort_name:
        cohort_xml = cohorts_dir / f"{cohort_name}_{patient_id}.xml"
        if cohort_xml.exists():
            print(f"  Using truncated cohort XML: {cohort_xml.name}")
            return cohort_xml

    # 2. Any existing truncated XML for this patient
    existing = list(cohorts_dir.glob(f"*_{patient_id}.xml"))
    if existing:
        chosen = existing[0]
        print(f"  Using existing truncated XML: {chosen.name}")
        return chosen

    # 3. Auto-generate from graph store TB date
    gs_index = _load_graph_store_index()
    patient_meta = gs_index.get(patient_id)
    if patient_meta and full_xml.exists():
        tb_date_str = patient_meta.get("tb_date")
        if tb_date_str:
            from prepare_cohort import truncate_xml
            from datetime import datetime as dt
            import contextlib, io

            tb_date = dt.strptime(tb_date_str, "%Y-%m-%d")
            output_path = cohorts_dir / f"eval_{patient_id}.xml"
            cohorts_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Auto-generating truncated XML (TB date: {tb_date_str})...")
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                truncated_xml = truncate_xml(full_xml, tb_date)
            # Print truncation stats
            stats = buf.getvalue().strip()
            if stats:
                print(f"  {stats}")

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(truncated_xml)
            print(f"  Saved: {output_path.name} ({len(truncated_xml):,} chars)")
            return output_path

    # 4. No truncation possible
    if full_xml.exists():
        print(f"  WARNING: Using FULL XML (no TB date found) — eval may be inaccurate!")
        return full_xml

    return None


def evaluate_patient(patient_id: str, eval_from_snapshot: bool = False, cohort_name: str = None,
                     snapshot_dir: str = "temp_jsons", no_graph_fallbacks: bool = False):
    """Run evaluation for a single patient.

    Args:
        patient_id: Patient ID
        eval_from_snapshot: If True, read MTB_VARIABLES from patient_info.json (1 LLM call).
                           If False, extract via LLM first (2 LLM calls).
        cohort_name: If provided, use truncated cohort XML as ground truth.
        snapshot_dir: Root directory containing {patient_id}/patient_info.json snapshots.
                     Defaults to "temp_jsons". Use a different dir (e.g. "temp_jsons_rag_gpt5")
                     to evaluate external snapshots without touching VISTA's outputs.
        no_graph_fallbacks: If True, do not backfill missing snapshot values from the graph
                           store (prevents VISTA-derived data from inflating an external
                           baseline's score).
    """
    print(f"\n{'='*60}")
    print(f"Evaluating patient: {patient_id}")
    print(f"  Mode: {'snapshot' if eval_from_snapshot else 'standard'}")
    print(f"  Snapshot dir: {snapshot_dir}")
    if no_graph_fallbacks:
        print(f"  Graph fallbacks: DISABLED (external snapshot mode)")
    print(f"  Variables: {len(EVAL_VARIABLES)} ({len(PRIMARY_VARIABLES)} primary)")
    print(f"{'='*60}\n")

    # Paths
    json_dir = Path(f"{snapshot_dir}/{patient_id}")
    timeline_jsonl = json_dir / "timeline_objects.jsonl"
    episodes_path = json_dir / "episodes.json"

    # XML ground truth: MUST use TB-date-truncated XML (matches graph store input)
    xml_path = _resolve_truncated_xml(patient_id, cohort_name)
    if xml_path is None:
        print(f"  ERROR: No XML found for {patient_id}")
        return None

    # Extract variables (snapshot or standard mode)
    if eval_from_snapshot:
        print("  Reading MTB_VARIABLES from patient_info.json...")
        extracted_values = extract_variables_from_snapshot(patient_id, json_dir,
                                                           no_graph_fallbacks=no_graph_fallbacks)
    else:
        # Load patient JSONs
        patient_jsons = {}
        for json_file in ["patient_info.json", "tumor_info.json", "timeline.json", "treatment_plans.json", "summary.json"]:
            json_path = json_dir / json_file
            if json_path.exists():
                with open(json_path) as f:
                    patient_jsons[json_file.replace('.json', '')] = json.load(f)

        # Extract demographics from XML (for display)
        xml_demographics = extract_demographics_from_xml(str(xml_path))
        print(f"  XML DOB: {xml_demographics.get('dob', 'N/A')}, Sex: {xml_demographics.get('sex', 'N/A')}")

        print(f"\n  [Call 1/2] Extracting {len(EVAL_VARIABLES)} variables from dashboard...")
        extracted_values = extract_variables_from_dashboard(patient_jsons, str(timeline_jsonl), str(episodes_path), str(xml_path))

    print(f"  Extracted values:")
    for var_name, value in extracted_values.items():
        val_str = str(value)[:60]
        print(f"     {var_name}: {val_str}")

    # Build comprehensive XML ground truth evidence
    print("\n  Building XML ground truth evidence...")
    evidence_parts = []

    # 1. Demographics
    person_elem = ET.parse(str(xml_path)).getroot().find(".//person")
    if person_elem is not None:
        evidence_parts.append("=== DEMOGRAPHICS ===\n" + ET.tostring(person_elem, encoding="unicode"))

    # 2. Surgery evidence
    timeline_events = []
    if Path(timeline_jsonl).exists():
        with open(timeline_jsonl) as f:
            for line in f:
                timeline_events.append(json.loads(line))
    surgeries = [e for e in timeline_events if e.get("type") == "surgery"]
    if surgeries:
        surgery_date = max(s.get("date") for s in surgeries if s.get("date"))
        date_evidence = extract_xml_evidence_for_date(str(xml_path), surgery_date)
        if date_evidence:
            evidence_parts.append(f"\n=== SURGERY DATE: {surgery_date} ===\n{date_evidence[:20000]}")

    # 3. Deterministic retrieval (molecular/genetics + CT dates from graph)
    try:
        from toa.graph_store import CohortGraphStore
        from toa.deterministic_retrieval import DeterministicRetriever
        store = CohortGraphStore("graph_store")
        graph = store.load_patient_graph(patient_id)
        if graph is not None:
            retriever = DeterministicRetriever(graph)
            det_context = retriever.get_comprehensive_context()
            if det_context:
                evidence_parts.append(f"\n=== DETERMINISTIC RETRIEVAL (graph search) ===\n{det_context}")
                print(f"  Added deterministic retrieval context ({len(det_context)} chars)")
            # Add explicit CT date vector as ground truth for imaging dates
            ct_dates = retriever.get_ct_date_vector()
            if ct_dates:
                ct_vector_str = f"\n=== CT DATE VECTOR (ground truth from structured institutional records) ===\n"
                ct_vector_str += f"Most recent: {ct_dates[0]}\n"
                ct_vector_str += f"All dates ({len(ct_dates)}): {', '.join(ct_dates[:10])}"
                if len(ct_dates) > 10:
                    ct_vector_str += f" ... (+{len(ct_dates)-10} more)"
                evidence_parts.append(ct_vector_str)
                print(f"  Added CT date vector ({len(ct_dates)} dates, most recent: {ct_dates[0]})")
    except Exception as e:
        print(f"  Deterministic retrieval unavailable: {e}")

    # 4. Final chunk (120k of recent XML)
    final_chunk = extract_final_chunk(str(xml_path), max_chars=120000)
    evidence_parts.append(f"\n=== FINAL CHUNK (recent 120k chars) ===\n{final_chunk}")

    xml_evidence = "\n\n".join(evidence_parts)
    print(f"  Assembled {len(xml_evidence)} chars of XML ground truth")

    # Judge extracted values against XML ground truth
    call_num = "1/1" if eval_from_snapshot else "2/2"
    print(f"\n  [Call {call_num}] Judging values against XML ground truth...")
    all_judgments = judge_all_variables(extracted_values, xml_evidence)

    # Parse results
    print(f"\n  RESULTS:\n")
    results = []
    for variable in EVAL_VARIABLES:
        var_name = variable['name']
        judgment = all_judgments.get(var_name, {})

        correctness = judgment.get('correctness', 'N/A')
        score = judgment.get('score', 0)
        explanation = judgment.get('explanation', 'N/A')
        xml_value = judgment.get('xml_value', '')
        primary = variable.get('primary', True)

        tag = "" if primary else " [sec]"
        print(f"  {var_name:35s} {correctness:10s} {score:2d}/10{tag}  {explanation[:60]}")

        extracted_value = extracted_values.get(var_name, 'N/A')

        results.append({
            'patient_id': patient_id,
            'variable': var_name,
            'category': variable.get('category', ''),
            'primary': primary,
            'correctness': correctness,
            'score': score,
            'explanation': explanation,
            'xml_value': xml_value,
            'extracted_value': str(extracted_value)[:200] + ('...' if len(str(extracted_value)) > 200 else ''),
            'eval_mode': 'snapshot' if eval_from_snapshot else 'standard'
        })

    return results


def print_cohort_analysis(all_results: list):
    """Print per-variable and per-category analysis of evaluation results."""
    if not all_results:
        return

    print(f"\n{'='*70}")
    print("COHORT ANALYSIS")
    print(f"{'='*70}\n")

    num_patients = len(set(r['patient_id'] for r in all_results))
    print(f"Patients: {num_patients}")
    print(f"Total evaluations: {len(all_results)}")

    # Per-variable analysis
    print(f"\n--- Per-Variable Accuracy ---\n")
    print(f"{'Variable':<40s} {'Correct':>8s} {'Partial':>8s} {'Incorrect':>8s} {'Avg Score':>10s}")
    print("-" * 70)

    var_stats = {}
    for var in EVAL_VARIABLES:
        var_name = var['name']
        var_results = [r for r in all_results if r['variable'] == var_name]
        if not var_results:
            continue

        correct = sum(1 for r in var_results if r['correctness'] == 'Correct')
        partial = sum(1 for r in var_results if r['correctness'] == 'Partial')
        incorrect = sum(1 for r in var_results if r['correctness'] == 'Incorrect')
        avg_score = sum(r['score'] for r in var_results) / len(var_results)

        tag = "" if var.get('primary', True) else " [sec]"
        print(f"{var_name + tag:<40s} {correct:>8d} {partial:>8d} {incorrect:>8d} {avg_score:>10.1f}")

        var_stats[var_name] = {
            'correct': correct, 'partial': partial, 'incorrect': incorrect,
            'avg_score': avg_score, 'total': len(var_results)
        }

    # Per-category analysis
    print(f"\n--- Per-Category Accuracy ---\n")
    print(f"{'Category':<20s} {'Avg Score':>10s} {'Correct %':>10s}")
    print("-" * 45)

    for category, var_names in CATEGORIES.items():
        cat_results = [r for r in all_results if r['variable'] in var_names]
        if not cat_results:
            continue

        avg_score = sum(r['score'] for r in cat_results) / len(cat_results)
        correct_pct = sum(1 for r in cat_results if r['correctness'] == 'Correct') / len(cat_results) * 100

        print(f"{category:<20s} {avg_score:>10.1f} {correct_pct:>9.1f}%")

    # Primary vs secondary
    print(f"\n--- Primary vs Secondary ---\n")
    primary_results = [r for r in all_results if r.get('primary', True)]
    secondary_results = [r for r in all_results if not r.get('primary', True)]

    if primary_results:
        avg = sum(r['score'] for r in primary_results) / len(primary_results)
        correct_pct = sum(1 for r in primary_results if r['correctness'] == 'Correct') / len(primary_results) * 100
        print(f"Primary ({len(primary_results)} evals): avg {avg:.1f}/10, {correct_pct:.1f}% correct")

    if secondary_results:
        avg = sum(r['score'] for r in secondary_results) / len(secondary_results)
        correct_pct = sum(1 for r in secondary_results if r['correctness'] == 'Correct') / len(secondary_results) * 100
        print(f"Secondary ({len(secondary_results)} evals): avg {avg:.1f}/10, {correct_pct:.1f}% correct")


def main():
    parser = argparse.ArgumentParser(
        description="V2 automated evaluation of tumor board variables against XML ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate single patient (standard mode - 2 LLM calls)
    python quick_eval.py 136040534

    # Evaluate using pre-extracted MTB_VARIABLES (snapshot mode - 1 LLM call)
    python quick_eval.py --eval-from-snapshot 136040534

    # Evaluate cohort with output file
    python quick_eval.py --cohort patient_records/cohorts/eval30_manifest.json --output eval30_v2.json

    # Evaluate multiple patients
    python quick_eval.py --output results.json patient1 patient2 patient3
"""
    )

    parser.add_argument(
        "patient_ids", nargs="*",
        help="Patient IDs to evaluate"
    )
    parser.add_argument(
        "--cohort",
        help="Cohort manifest JSON file (evaluates all patients in cohort)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--eval-from-snapshot", action="store_true",
        help="Read MTB_VARIABLES from patient_info.json (1 LLM call per patient)"
    )
    parser.add_argument(
        "--snapshot-dir", default="temp_jsons",
        help="Root directory of {pid}/patient_info.json snapshots (default: temp_jsons). "
             "Use e.g. temp_jsons_rag_gpt5 to evaluate an external baseline."
    )
    parser.add_argument(
        "--no-graph-fallbacks", action="store_true",
        help="Do not backfill missing snapshot values from the graph store. "
             "Use when evaluating external baselines to prevent data leakage."
    )
    parser.add_argument(
        "--eval-model", default=EVAL_MODEL,
        help=f"Model for evaluation (default: {EVAL_MODEL})"
    )

    args = parser.parse_args()

    # Set eval model (module-level for call_llm default).
    # NOTE: when quick_eval.py runs as a script, __name__ == '__main__', so
    # `import quick_eval` brings in a SECOND copy of the module and the
    # original (running) EVAL_MODEL never gets touched. Patch the
    # running module's global directly.
    globals()['EVAL_MODEL'] = args.eval_model
    import quick_eval as _qe_alias
    _qe_alias.EVAL_MODEL = args.eval_model

    # Gather patient IDs
    patient_ids = list(args.patient_ids or [])

    cohort_name = None
    if args.cohort:
        with open(args.cohort, 'r') as f:
            manifest = json.load(f)
        cohort_patients = [p['patient_id'] for p in manifest['patients']]
        cohort_name = manifest.get('cohort_name')
        # Use raw PIDs — graph-store pipeline stores outputs under raw PIDs
        patient_ids.extend(cohort_patients)
        print(f"Loaded {len(cohort_patients)} patients from cohort: {args.cohort}")
        if cohort_name:
            print(f"  Using truncated XMLs: patient_records/cohorts/{cohort_name}_*.xml")

    if not patient_ids:
        parser.print_help()
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"eval_v2_results_{timestamp}.json")

    # Load existing results if file exists (resume capability)
    all_results = []
    completed_patients = set()
    if output_path.exists():
        print(f"Found existing results file: {output_path}")
        with open(output_path, 'r') as f:
            existing = json.load(f)
            if isinstance(existing, dict):
                all_results = existing.get('results', [])
            else:
                all_results = existing
            completed_patients = set(r['patient_id'] for r in all_results)
            print(f"  Already completed: {len(completed_patients)} patients")

    # Process each patient
    for i, pid in enumerate(patient_ids, 1):
        if pid in completed_patients:
            print(f"\n[{i}/{len(patient_ids)}] Skipping {pid} (already completed)")
            continue

        print(f"\n[{i}/{len(patient_ids)}] Processing {pid}...")
        results = evaluate_patient(pid, eval_from_snapshot=args.eval_from_snapshot, cohort_name=cohort_name,
                                   snapshot_dir=args.snapshot_dir, no_graph_fallbacks=args.no_graph_fallbacks)

        if results:
            all_results.extend(results)

            # Save after each patient (incremental)
            output_data = {
                "eval_version": "v2",
                "eval_model": args.eval_model,
                "eval_mode": "snapshot" if args.eval_from_snapshot else "standard",
                "num_variables": len(EVAL_VARIABLES),
                "timestamp": datetime.now().isoformat(),
                "results": all_results
            }
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"  Saved results for {pid} to {output_path}")

    # Final summary
    if all_results:
        print(f"\n{'='*60}")
        print(f"Evaluation complete!")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}\n")

        # Overall summary
        avg_score = sum(r['score'] for r in all_results) / len(all_results)
        correct_count = sum(1 for r in all_results if r['correctness'] == 'Correct')
        partial_count = sum(1 for r in all_results if r['correctness'] == 'Partial')
        incorrect_count = sum(1 for r in all_results if r['correctness'] == 'Incorrect')

        print(f"Overall Summary:")
        print(f"  Patients evaluated: {len(set(r['patient_id'] for r in all_results))}")
        print(f"  Variables evaluated: {len(all_results)}")
        print(f"  Average quality score: {avg_score:.1f}/10")
        print(f"  Correct: {correct_count}  Partial: {partial_count}  Incorrect: {incorrect_count}")

        # Per-variable analysis
        print_cohort_analysis(all_results)


if __name__ == "__main__":
    main()
