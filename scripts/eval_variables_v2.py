"""
V2 Evaluation Variables for VISTA Architect MTB Assessment.

16 variables total: 14 primary (clinician-aligned) + 2 secondary (safety context).
Based on clinician feedback (Tim Ellis-Caleo, David Wu) from manuscript review.

Changes from v1 (12 variables):
  NEW:  Smoking Status, ECOG, Radiation Therapy, Toxicity/Comorbidities
  REMOVED: TNM Stage (not in v1 eval, not in clinical reviewer feedback, high ambiguity)
  KEPT: All 12 original variables (Allergies + DNR demoted to secondary)
"""

EVAL_VARIABLES_V2 = [
    # --- Demographics ---
    {
        "name": "Date of Birth",
        "category": "demographics",
        "extraction_method": "xml_demographics",
        "format": "YYYY-MM-DD",
        "deterministic_retrieval_method": None,
        "eval_comparison": "exact_match",
        "primary": True,
    },
    {
        "name": "Sex",
        "category": "demographics",
        "extraction_method": "xml_demographics",
        "format": "Male/Female",
        "deterministic_retrieval_method": None,
        "eval_comparison": "exact_match",
        "primary": True,
    },
    {
        "name": "Smoking Status",
        "category": "demographics",
        "extraction_method": "graph_search",
        "format": "pack-years and status (current/former/never) or Unknown",
        "deterministic_retrieval_method": "get_smoking_status",
        "eval_comparison": "semantic_match",
        "primary": True,
    },

    # --- Tumor ---
    {
        "name": "Diagnosis",
        "category": "tumor",
        "extraction_method": "json",
        "format": "Cancer type and site (e.g., NSCLC RUL, SCLC, mesothelioma)",
        "deterministic_retrieval_method": None,
        "eval_comparison": "semantic_match",
        "primary": True,
    },
    {
        "name": "Histology",
        "category": "tumor",
        "extraction_method": "json",
        "format": "Histologic subtype (adenocarcinoma, squamous, large cell, etc.)",
        "deterministic_retrieval_method": None,
        "eval_comparison": "semantic_match",
        "primary": True,
    },
    {
        "name": "Metastasis",
        "category": "tumor",
        "extraction_method": "graph_search",
        "format": "Yes/Suspected/No (+ sites if Yes)",
        "deterministic_retrieval_method": "get_latest_metastasis_mentions",
        "eval_comparison": "categorical_match",
        "primary": True,
    },
    {
        "name": "Lymph Node Involvement",
        "category": "tumor",
        "extraction_method": "graph_search",
        "format": "Yes/Suspected/No",
        "deterministic_retrieval_method": "get_latest_lymph_node_mentions",
        "eval_comparison": "categorical_match",
        "primary": True,
    },
    {
        "name": "Genetic Testing Panel",
        "category": "tumor",
        "extraction_method": "graph_search",
        "format": "Dict of tested genes with results, or Not tested",
        "deterministic_retrieval_method": "get_driver_mutation_mentions",
        "eval_comparison": "semantic_match",
        "primary": True,
    },
    # --- Clinical ---
    {
        "name": "ECOG Performance Status",
        "category": "clinical",
        "extraction_method": "graph_search",
        "format": "0-4, range (e.g. 0-1), or Unknown",
        "deterministic_retrieval_method": "get_ecog_status",
        "eval_comparison": "exact_match",
        "primary": True,
    },
    {
        "name": "Therapy Toxicity / Comorbidities",
        "category": "clinical",
        "extraction_method": "graph_search",
        "format": "Major toxicities (grade >= 3) and serious comorbidities, or None",
        "deterministic_retrieval_method": "get_toxicity_comorbidity_mentions",
        "eval_comparison": "semantic_match",
        "primary": True,
    },

    # --- Treatment ---
    {
        "name": "Previous Surgery",
        "category": "treatment",
        "extraction_method": "timeline",
        "format": "Yes/No (full surgical history relevant to TB planning: current-cancer resections + prior-cancer resections + major non-oncologic surgeries [joint replacement, hernia repair, cardiac/vascular, craniotomy, transplants, parotidectomy, etc.] + diagnostic surgical procedures [VATS biopsy, mediastinoscopy, pleurodesis]. EXCLUDE: radiation procedures [CyberKnife/SRS/Gamma Knife], port/mediport placement, bronchoscopy alone, FNA, cataract/dental/cosmetic procedures.)",
        "deterministic_retrieval_method": "get_surgical_history_mentions",
        "eval_comparison": "exact_match",
        "primary": True,
    },
    {
        "name": "Current Medical Therapy",
        "category": "treatment",
        "extraction_method": "json",
        "format": "Current oncological regimen or None",
        "deterministic_retrieval_method": "get_distinct_drug_exposures",
        "eval_comparison": "semantic_match",
        "primary": True,
    },
    {
        "name": "Radiation Therapy",
        "category": "treatment",
        "extraction_method": "graph_search",
        "format": "Yes (site, modality, dates) or No",
        "deterministic_retrieval_method": "get_radiation_therapy_mentions",
        "eval_comparison": "semantic_match",
        "primary": True,
    },

    # --- Imaging ---
    {
        "name": "Date of Last CT",
        "category": "imaging",
        "extraction_method": "timeline",
        "format": "YYYY-MM (month level)",
        "deterministic_retrieval_method": None,
        "eval_comparison": "date_match_month",
        "primary": True,
    },

    # --- Secondary / Safety Context ---
    {
        "name": "Allergies",
        "category": "safety",
        "extraction_method": "json",
        "format": "List of allergies, or None/NKDA",
        "deterministic_retrieval_method": "get_all_allergy_mentions",
        "eval_comparison": "semantic_match",
        "primary": False,
    },
    {
        "name": "DNR",
        "category": "safety",
        "extraction_method": "xml",
        "format": "Yes/No (if not documented, assume No)",
        "deterministic_retrieval_method": None,
        "eval_comparison": "exact_match",
        "primary": False,
    },
]

# Convenience accessors
PRIMARY_VARIABLES = [v for v in EVAL_VARIABLES_V2 if v["primary"]]
SECONDARY_VARIABLES = [v for v in EVAL_VARIABLES_V2 if not v["primary"]]
VARIABLE_NAMES = [v["name"] for v in EVAL_VARIABLES_V2]
PRIMARY_VARIABLE_NAMES = [v["name"] for v in PRIMARY_VARIABLES]

# Variables that use graph search (need DeterministicRetriever)
GRAPH_SEARCH_VARIABLES = [v for v in EVAL_VARIABLES_V2 if v["deterministic_retrieval_method"]]

# Category groupings for reporting
CATEGORIES = {
    "demographics": ["Date of Birth", "Sex", "Smoking Status"],
    "tumor": ["Diagnosis", "Histology", "Metastasis", "Lymph Node Involvement",
              "Genetic Testing Panel"],
    "clinical": ["ECOG Performance Status", "Therapy Toxicity / Comorbidities"],
    "treatment": ["Previous Surgery", "Current Medical Therapy", "Radiation Therapy"],
    "imaging": ["Date of Last CT"],
    "safety": ["Allergies", "DNR"],
}
