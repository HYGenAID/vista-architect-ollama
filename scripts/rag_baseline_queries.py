"""
RAG baseline: one natural-language query per MTB variable.

Each entry maps a variable (from eval_variables_v2.VARIABLE_NAMES) to:
  - `query`: the natural-language question posed to the RAG system
  - `format_hint`: short instruction appended to the user prompt so the
    answer lands in a shape the patient_info.json schema expects. Kept
    minimal on purpose: a fair RAG baseline is allowed to be told the
    output format, just not given retrieval hints or domain cheats.
  - `json_location`: (section, key) tuple. Sections are those consumed by
    quick_eval.extract_variables_from_snapshot: "PATIENT DEMOGRAPHICS",
    "TUMOR INFORMATION", "TREATMENTS".

Entries are listed in the same order as EVAL_VARIABLES_V2.
"""

QUERIES = {
    "Date of Birth": {
        "query": "What is the patient's date of birth?",
        "format_hint": "Answer with only the date in YYYY-MM-DD format, or \"Unknown\".",
        "json_location": ("PATIENT DEMOGRAPHICS", "date_of_birth"),
    },
    "Sex": {
        "query": "What is the patient's sex?",
        "format_hint": "Answer with a single word: Male, Female, or Unknown.",
        "json_location": ("PATIENT DEMOGRAPHICS", "sex"),
    },
    "Smoking Status": {
        "query": "What is the patient's smoking history? Include pack-years and whether they are a current, former, or never smoker.",
        "format_hint": "Answer in one short sentence, or \"Unknown\".",
        "json_location": ("PATIENT DEMOGRAPHICS", "smoking_history"),
    },

    "Diagnosis": {
        "query": "What is the patient's cancer diagnosis? Include cancer type, primary site, and stage if documented.",
        "format_hint": "Answer in one short phrase, or \"Unknown\".",
        "json_location": ("TUMOR INFORMATION", "diagnosis"),
    },
    "Histology": {
        "query": "What is the histologic subtype of the patient's cancer (e.g., adenocarcinoma, squamous cell carcinoma, small cell, carcinoid)?",
        "format_hint": "Answer in one short phrase, or \"Unknown\".",
        "json_location": ("TUMOR INFORMATION", "histology"),
    },
    "Metastasis": {
        "query": "Does the patient have metastatic disease? If yes, where are the metastases?",
        "format_hint": "Begin your answer with \"Yes\", \"No\", or \"Suspected\", optionally followed by sites.",
        "json_location": ("TUMOR INFORMATION", "metastasis_status"),
    },
    "Lymph Node Involvement": {
        "query": "Does the patient have lymph node involvement from the cancer?",
        "format_hint": "Answer with one word: Yes, No, or Suspected.",
        "json_location": ("TUMOR INFORMATION", "lymph_node_involvement"),
    },
    "Genetic Testing Panel": {
        "query": "What molecular and genetic testing has been done on the patient's tumor? List each gene tested (EGFR, KRAS, ALK, ROS1, BRAF, MET, RET, NTRK, HER2) and its result, as well as PD-L1 if reported.",
        "format_hint": "Answer as a semicolon-separated list like \"EGFR: L858R positive; KRAS: negative; PD-L1: 60%\", or \"Not tested\" if no molecular testing is found.",
        "json_location": ("TUMOR INFORMATION", "driver_mutations"),
    },

    "ECOG Performance Status": {
        "query": "What is the patient's most recent ECOG performance status?",
        "format_hint": "Answer with a single digit (0, 1, 2, 3, or 4), a range (e.g., 0-1), or \"Unknown\".",
        "json_location": ("PATIENT DEMOGRAPHICS", "ecog_performance_status"),
    },
    "Therapy Toxicity / Comorbidities": {
        "query": "What significant cancer treatment toxicities or major comorbidities does the patient have?",
        "format_hint": "Answer in one short phrase, or \"None documented\".",
        "json_location": ("PATIENT DEMOGRAPHICS", "therapy_toxicities"),
    },

    "Previous Surgery": {
        "query": "Has the patient had previous oncological surgery for their current cancer (such as lobectomy, wedge resection, pneumonectomy, or VATS resection)? If yes, what procedure and when?",
        "format_hint": "Answer in one short sentence that names the procedure if yes, or \"No\".",
        "json_location": ("TREATMENTS", "previous"),
    },
    "Current Medical Therapy": {
        "query": "What systemic cancer therapy (chemotherapy, immunotherapy, or targeted therapy) is the patient currently receiving?",
        "format_hint": "Answer in one short phrase naming the drug(s) and start date if known, or \"None\".",
        "json_location": ("TREATMENTS", "current"),
    },
    "Radiation Therapy": {
        "query": "Has the patient received radiation therapy for their cancer? If yes, what site, dose, and dates?",
        "format_hint": "Begin your answer with \"Yes\" or \"No\"; if yes, briefly include site, dose, and dates.",
        "json_location": ("TREATMENTS", "radiation_therapy"),
    },

    "Date of Last CT": {
        "query": "What is the date of the patient's most recent CT scan (chest, abdomen/pelvis, or PET-CT)?",
        "format_hint": "Answer with only the date in YYYY-MM-DD format, or \"Unknown\".",
        "json_location": ("TREATMENTS", "date_of_last_ct"),
    },

    "Allergies": {
        "query": "What drug allergies does the patient have?",
        "format_hint": "Answer with a comma-separated list of drug allergies, or \"NKDA\" if none are documented.",
        "json_location": ("PATIENT DEMOGRAPHICS", "allergies"),
    },
    "DNR": {
        "query": "Does the patient have a DNR (Do Not Resuscitate) order documented?",
        "format_hint": "Answer with one word: Yes or No.",
        "json_location": ("PATIENT DEMOGRAPHICS", "dnr"),
    },
}


SYSTEM_PROMPT = (
    "You are a clinical assistant. Answer the user's question using ONLY the "
    "provided excerpts from the patient's medical record. If the information "
    "is not present in the excerpts, answer \"Unknown\". Be concise."
)


def build_user_prompt(context: str, query: str, format_hint: str) -> str:
    return (
        f"Excerpts from patient's medical record:\n---\n{context}\n---\n\n"
        f"Question: {query}\n"
        f"{format_hint}\n"
        f"Answer:"
    )


TIMELINE_QUERY = {
    "query": (
        "List the major clinical events in this patient's history in chronological "
        "order. Include: cancer diagnoses and biopsies, molecular or pathology results, "
        "imaging studies (CT, MRI, PET-CT) with dates, systemic therapy starts and stops, "
        "surgeries, radiation therapy, hospitalizations, and disease progression events."
    ),
    "format_hint": (
        "Format each event on its own line as \"YYYY-MM-DD | TYPE | DESCRIPTION\", "
        "where TYPE is one of imaging, diagnostic, treatment, surgery, lab, adverse_effect. "
        "List events in chronological order (oldest first). If no events are found in the "
        "excerpts, answer \"No events found\"."
    ),
}
