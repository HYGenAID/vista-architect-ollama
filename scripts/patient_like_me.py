#!/usr/bin/env python3
"""
Patient Like Me — Multi-Strategy Retrieval + LLM Clinical Judge.

Finds clinically similar patients from the VISTA cohort using a 4-stage
pipeline: Hard Gate → Multi-Strategy Retrieval → Context Assembly → LLM Judge.

Uses pre-computed PLM index (from build_plm_index.py) for fast retrieval,
then packs candidates into an LLM prompt for clinical similarity judgment.

Usage:
    # Find twins for a single patient
    python patient_like_me.py --pid 136020661

    # Retrieval only (no LLM call)
    python patient_like_me.py --pid 136020661 --retrieval-only

    # Custom index and output
    python patient_like_me.py --pid 136020661 --index outputs/plm/plm_index_600.json --output plm_result.json

    # Batch mode
    python patient_like_me.py --batch patient_records/cohorts/eval_v2_100_manifest.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

import gsgpt


# ---------------------------------------------------------------------------
# Bayesian-calibrated weights (Phase I, plm_bayesian_calibration.py)
# ---------------------------------------------------------------------------
#
# 11 structural-similarity features identical to those used by the Phase I
# Beta-regression model. Two scoring modes:
#   * "prior"     — original expert-encoded weights (manuscript Table 1).
#                   Used by retrieval before calibration was wired in.
#   * "posterior" — MAP estimates from outputs/plm/plm_calibration_results.json
#                   (30 index patients, 443 LLM-judged pairs, gemini-2.5-flash).
#
# Switch via CLI:  --calibrated [PATH]   (defaults to outputs/plm/plm_calibration_results.json)
#                  --prior                 (force original prior weights)

PRIOR_WEIGHTS = {
    "mutation_concordance": 3.0,
    "histology_match":      1.5,
    "pdl1_concordance":     0.5,
    "ln_concordance":       0.3,
    "line_similarity":      1.5,
    "intent_match":         1.0,
    "regimen_jaccard":      1.0,
    "surgery_match":        0.5,
    "radiation_match":      0.5,
    "age_similarity":       0.0,
    "smoking_match":        0.3,
}

# Module-level active weight set; mutated once at startup by configure_weights()
_ACTIVE_WEIGHTS: dict = dict(PRIOR_WEIGHTS)
_WEIGHT_MODE: str = "prior"  # "prior" or "posterior"
_WEIGHT_SOURCE: str = "manuscript-table-1"


def configure_weights(mode: str = "prior", calibration_path: str | None = None) -> tuple[dict, str, str]:
    """Set the active weight vector for the retrieval scorers.

    Args:
        mode: "prior" → original heuristic weights;
              "posterior" → load from a Phase I calibration JSON.
        calibration_path: required when mode="posterior".

    Returns the (weights, mode, source) tuple.
    """
    global _ACTIVE_WEIGHTS, _WEIGHT_MODE, _WEIGHT_SOURCE
    if mode == "prior":
        _ACTIVE_WEIGHTS = dict(PRIOR_WEIGHTS)
        _WEIGHT_MODE = "prior"
        _WEIGHT_SOURCE = "manuscript-table-1"
    elif mode == "posterior":
        if not calibration_path:
            raise ValueError("posterior mode requires calibration_path")
        with open(calibration_path) as f:
            cal = json.load(f)
        post = cal["posterior_weights"]
        # Sanity: must contain all 11 expected feature keys
        missing = set(PRIOR_WEIGHTS) - set(post)
        if missing:
            raise ValueError(f"calibration JSON missing weights: {sorted(missing)}")
        _ACTIVE_WEIGHTS = {k: float(post[k]) for k in PRIOR_WEIGHTS}
        _WEIGHT_MODE = "posterior"
        meta = cal.get("meta", {})
        _WEIGHT_SOURCE = (
            f"{Path(calibration_path).name} "
            f"(N={meta.get('n_index_patients','?')} idx, "
            f"{meta.get('n_pairs','?')} pairs, judge={meta.get('model','?')})"
        )
    else:
        raise ValueError(f"unknown weight mode: {mode}")
    return _ACTIVE_WEIGHTS, _WEIGHT_MODE, _WEIGHT_SOURCE


def get_active_weights() -> tuple[dict, str, str]:
    return dict(_ACTIVE_WEIGHTS), _WEIGHT_MODE, _WEIGHT_SOURCE


# ---------------------------------------------------------------------------
# Stage 1: Hard Gate
# ---------------------------------------------------------------------------

def hard_gate(index_patient: dict, plm_index: dict) -> list[dict]:
    """Filter to same diagnosis category and compatible metastatic status.

    Returns candidates that pass through both gates.
    Unknown metastatic status passes through (permissive).
    """
    candidates = []
    idx_diag = index_patient["diagnosis_category"]
    idx_met = index_patient["metastatic"]

    for pid, record in plm_index["patients"].items():
        if pid == index_patient["patient_id"]:
            continue
        # Gate 1: Same diagnosis category
        if record["diagnosis_category"] != idx_diag:
            continue
        # Gate 2: Compatible metastatic status
        if (idx_met != "Unknown" and
                record["metastatic"] != "Unknown" and
                idx_met != record["metastatic"]):
            continue
        candidates.append(record)
    return candidates


# ---------------------------------------------------------------------------
# Stage 2: Multi-Strategy Retrieval
# ---------------------------------------------------------------------------

def _mutation_concordance_feat(index_patient: dict, candidate: dict) -> float:
    """[0,1] concordance over jointly-tested genes; positive matches dominate.

    Same-positive contribution = 3, same-negative = 1, mismatch = -1.
    Normalized against max-positive-on-all-tested-genes = 4 * compared.
    Returns 0.3 when no joint testing is available (neutral).
    """
    idx_muts = index_patient["driver_mutations"]
    cand_muts = candidate["driver_mutations"]
    raw, compared = 0.0, 0
    for gene, idx_status in idx_muts.items():
        cand_status = cand_muts.get(gene, "not_tested")
        if idx_status == "not_tested" or cand_status == "not_tested":
            continue
        compared += 1
        if idx_status == cand_status:
            raw += 3.0 if idx_status == "positive" else 1.0
        else:
            raw -= 1.0
    if compared == 0:
        return 0.3
    return max(0.0, (raw + compared) / (4 * compared))


def _histology_match_feat(idx, cand) -> float:
    if idx["histology_category"] == cand["histology_category"] and idx["histology_category"] != "Other/NOS":
        return 1.0
    return 0.0


def _pdl1_concordance_feat(idx, cand) -> float:
    if idx["pdl1_status"] == "unknown" or cand["pdl1_status"] == "unknown":
        return 0.0
    if idx["pdl1_status"] == cand["pdl1_status"]:
        return 1.0
    if _pdl1_close(idx["pdl1_status"], cand["pdl1_status"]):
        return 0.5
    return 0.0


def _ln_concordance_feat(idx, cand) -> float:
    if idx["lymph_node"] == "Unknown" or cand["lymph_node"] == "Unknown":
        return 0.0
    return 1.0 if idx["lymph_node"] == cand["lymph_node"] else 0.0


def _weighted_combine(features: dict, weight_keys: list[str]) -> float:
    """Combine selected features with current active weights, normalized to [0,1].

    score = max(0, Σ w_i * x_i) / Σ max(0, w_i)
    Negative weights (e.g. PD-L1 posterior at -0.11) reduce the score honestly,
    matching the Bayesian model's interpretation.
    """
    weights = _ACTIVE_WEIGHTS
    num = sum(weights[k] * features[k] for k in weight_keys)
    denom = sum(max(0.0, weights[k]) for k in weight_keys)
    if denom <= 0:
        return 0.0
    return min(1.0, max(0.0, num / denom))


def biology_score(index_patient: dict, candidate: dict) -> float:
    """[0,1] biology similarity using calibrated weights for 4 features:
    mutation_concordance, histology_match, pdl1_concordance, ln_concordance.

    Weight values come from the active weight set (prior or Phase I posterior).
    """
    feats = {
        "mutation_concordance": _mutation_concordance_feat(index_patient, candidate),
        "histology_match":      _histology_match_feat(index_patient, candidate),
        "pdl1_concordance":     _pdl1_concordance_feat(index_patient, candidate),
        "ln_concordance":       _ln_concordance_feat(index_patient, candidate),
    }
    return _weighted_combine(feats, list(feats.keys()))


def _pdl1_close(a: str, b: str) -> bool:
    """Check if two PD-L1 statuses are close (adjacent categories)."""
    order = {"negative": 0, "low": 1, "high": 2}
    if a in order and b in order:
        return abs(order[a] - order[b]) <= 1
    return False


def _line_similarity_feat(idx, cand) -> float:
    li, lc = idx["current_line_number"], cand["current_line_number"]
    if li == lc:
        return 1.0
    if abs(li - lc) == 1:
        return 0.5
    return 0.0


def _intent_match_feat(idx, cand) -> float:
    if idx["current_intent"] == cand["current_intent"] and idx["current_intent"] != "unknown":
        return 1.0
    return 0.0


def _regimen_jaccard_feat(idx, cand) -> float:
    a, b = set(idx["regimen_classes"]), set(cand["regimen_classes"])
    if not (a or b):
        return 0.0
    return len(a & b) / len(a | b)


def _surgery_match_feat(idx, cand) -> float:
    return 1.0 if idx["had_surgery"] == cand["had_surgery"] else 0.0


def _radiation_match_feat(idx, cand) -> float:
    return 1.0 if idx["had_radiation"] == cand["had_radiation"] else 0.0


def _age_similarity_feat(idx, cand) -> float:
    """1.0 when ages within 5y, linearly decaying to 0 at 30y apart."""
    ai, ac = idx.get("age_at_tb"), cand.get("age_at_tb")
    if ai is None or ac is None:
        return 0.0
    diff = abs(float(ai) - float(ac))
    if diff <= 5: return 1.0
    if diff >= 30: return 0.0
    return max(0.0, 1.0 - (diff - 5) / 25.0)


def _smoking_match_feat(idx, cand) -> float:
    si, sc = (idx.get("smoking_category") or "Unknown"), (cand.get("smoking_category") or "Unknown")
    if si in ("Unknown", "?") or sc in ("Unknown", "?"):
        return 0.0
    return 1.0 if si == sc else 0.0


def trajectory_score(index_patient: dict, candidate: dict) -> float:
    """[0,1] trajectory similarity using calibrated weights for 5 features:
    line_similarity, intent_match, regimen_jaccard, surgery_match, radiation_match.
    """
    feats = {
        "line_similarity":  _line_similarity_feat(index_patient, candidate),
        "intent_match":     _intent_match_feat(index_patient, candidate),
        "regimen_jaccard":  _regimen_jaccard_feat(index_patient, candidate),
        "surgery_match":    _surgery_match_feat(index_patient, candidate),
        "radiation_match":  _radiation_match_feat(index_patient, candidate),
    }
    return _weighted_combine(feats, list(feats.keys()))


def composite_similarity(index_patient: dict, candidate: dict) -> tuple[float, dict]:
    """Full 11-feature similarity using all calibrated weights.

    Returns (score, feature_dict). Feature dict matches the Bayesian model
    so a future enhancement can also produce μ = sigmoid(β_0 + Σ β_i x_i).
    """
    feats = {
        "mutation_concordance": _mutation_concordance_feat(index_patient, candidate),
        "histology_match":      _histology_match_feat(index_patient, candidate),
        "pdl1_concordance":     _pdl1_concordance_feat(index_patient, candidate),
        "ln_concordance":       _ln_concordance_feat(index_patient, candidate),
        "line_similarity":      _line_similarity_feat(index_patient, candidate),
        "intent_match":         _intent_match_feat(index_patient, candidate),
        "regimen_jaccard":      _regimen_jaccard_feat(index_patient, candidate),
        "surgery_match":        _surgery_match_feat(index_patient, candidate),
        "radiation_match":      _radiation_match_feat(index_patient, candidate),
        "age_similarity":       _age_similarity_feat(index_patient, candidate),
        "smoking_match":        _smoking_match_feat(index_patient, candidate),
    }
    return _weighted_combine(feats, list(feats.keys())), feats


def narrative_scores(
    index_patient: dict,
    candidates: list[dict],
) -> list[tuple[str, float]]:
    """BM25-rank candidates by tb_note similarity to index patient.

    Returns list of (patient_id, bm25_score) sorted by score descending.
    """
    # Filter candidates that have tb_note tokens
    valid = [(c, c["tb_note_tokens"]) for c in candidates if c.get("tb_note_tokens")]
    if not valid or not index_patient.get("tb_note_tokens"):
        return []

    corpus = [tokens for _, tokens in valid]
    bm25 = BM25Okapi(corpus)
    query = index_patient["tb_note_tokens"]
    scores = bm25.get_scores(query)

    results = []
    for i, (cand, _) in enumerate(valid):
        results.append((cand["patient_id"], float(scores[i])))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def retrieve_candidates(
    index_patient: dict,
    gated_candidates: list[dict],
    top_k_per_net: int = 5,
) -> list[dict]:
    """Run all 3 retrieval nets, deduplicate, tag sources.

    Returns 10-15 candidates with retrieval metadata.
    """
    if not gated_candidates:
        return []

    # --- Net A: Biology Twins ---
    bio_scores = []
    for cand in gated_candidates:
        s = biology_score(index_patient, cand)
        bio_scores.append((cand["patient_id"], s, cand))
    bio_scores.sort(key=lambda x: x[1], reverse=True)
    bio_top = bio_scores[:top_k_per_net]

    # --- Net B: Trajectory Twins ---
    traj_scores = []
    for cand in gated_candidates:
        s = trajectory_score(index_patient, cand)
        traj_scores.append((cand["patient_id"], s, cand))
    traj_scores.sort(key=lambda x: x[1], reverse=True)
    traj_top = traj_scores[:top_k_per_net]

    # --- Net C: Narrative Twins ---
    narr_results = narrative_scores(index_patient, gated_candidates)
    narr_top_pids = {pid for pid, _ in narr_results[:top_k_per_net]}

    # --- Deduplicate and merge ---
    seen = {}  # pid -> {record, sources, scores}

    for pid, score, cand in bio_top:
        if pid not in seen:
            seen[pid] = {
                "record": cand,
                "sources": [],
                "scores": {},
            }
        seen[pid]["sources"].append("biology")
        seen[pid]["scores"]["biology"] = round(score, 3)

    for pid, score, cand in traj_top:
        if pid not in seen:
            seen[pid] = {
                "record": cand,
                "sources": [],
                "scores": {},
            }
        if "trajectory" not in seen[pid]["sources"]:
            seen[pid]["sources"].append("trajectory")
        seen[pid]["scores"]["trajectory"] = round(score, 3)

    # Build pid→cand lookup for narrative
    cand_by_pid = {c["patient_id"]: c for c in gated_candidates}
    for pid, score in narr_results[:top_k_per_net]:
        if pid not in seen:
            seen[pid] = {
                "record": cand_by_pid[pid],
                "sources": [],
                "scores": {},
            }
        if "narrative" not in seen[pid]["sources"]:
            seen[pid]["sources"].append("narrative")
        seen[pid]["scores"]["narrative"] = round(score, 3)

    # Build result list
    results = []
    for pid, data in seen.items():
        results.append({
            "patient_id": pid,
            "retrieval_sources": data["sources"],
            "retrieval_scores": data["scores"],
            **{k: v for k, v in data["record"].items()},
        })

    # Sort by number of nets that found them (multi-net hits first), then avg score
    results.sort(
        key=lambda r: (
            len(r["retrieval_sources"]),
            sum(r["retrieval_scores"].values()) / len(r["retrieval_scores"]),
        ),
        reverse=True,
    )

    return results


# ---------------------------------------------------------------------------
# Stage 3: Context Assembly
# ---------------------------------------------------------------------------

def _format_patient_compact(patient_info: dict, summary: dict, episodes: list) -> str:
    """Format a patient's data compactly for LLM context."""
    demo = patient_info.get("PATIENT DEMOGRAPHICS", {})
    tumor = patient_info.get("TUMOR INFORMATION", {})
    treat = patient_info.get("TREATMENTS", {})

    lines = []

    # Demographics
    age_sex = []
    if demo.get("sex"):
        age_sex.append(demo["sex"])
    if demo.get("ecog_performance_status"):
        age_sex.append(f"ECOG {demo['ecog_performance_status']}")
    if demo.get("smoking_history"):
        lines.append(f"Smoking: {demo['smoking_history']}")

    # Tumor
    if tumor.get("diagnosis"):
        lines.insert(0, f"Dx: {tumor['diagnosis']}")
    if tumor.get("histology"):
        lines.append(f"Histology: {tumor['histology']}")
    if tumor.get("tnm_staging"):
        lines.append(f"Stage: {tumor['tnm_staging']}")
    if tumor.get("metastasis_status"):
        lines.append(f"Metastatic: {tumor['metastasis_status']}")
    if tumor.get("lymph_node_involvement"):
        lines.append(f"LN: {tumor['lymph_node_involvement']}")

    # Mutations (compact)
    muts = tumor.get("driver_mutations", {})
    mut_parts = []
    for gene, val in muts.items():
        if val and isinstance(val, str) and "NOT TESTED" not in val.upper():
            mut_parts.append(f"{gene}: {val}")
    if mut_parts:
        lines.append(f"Mutations: {'; '.join(mut_parts)}")

    # Treatments
    current = treat.get("current", [])
    if current:
        lines.append(f"Current Tx: {'; '.join(str(t) for t in current)}")
    prev = treat.get("previous", [])
    if prev:
        lines.append(f"Prior Tx: {'; '.join(str(t) for t in prev)}")
    if treat.get("radiation_therapy"):
        lines.append(f"Radiation: {treat['radiation_therapy']}")

    # Demographics line at top
    if age_sex:
        lines.insert(1, f"Demographics: {', '.join(age_sex)}")

    # Treatment trajectory from episodes
    tx_episodes = [ep for ep in episodes if ep.get("kind") == "treatment_line"]
    if tx_episodes:
        trajectory_parts = []
        for ep in sorted(tx_episodes, key=lambda e: e.get("line_number", 0)):
            part = f"L{ep.get('line_number', '?')}: {ep.get('treatment', 'unknown')}"
            if ep.get("intent"):
                part += f" ({ep['intent']})"
            if ep.get("termination_reason"):
                part += f" → {ep['termination_reason']}"
            trajectory_parts.append(part)
        lines.append(f"Trajectory: {' | '.join(trajectory_parts)}")

    # Summary
    if isinstance(summary, dict) and summary.get("summary"):
        lines.append(f"TB Note: {summary['summary']}")

    return "\n".join(lines)


def assemble_context(
    index_patient_id: str,
    candidates: list[dict],
    plm_index: dict,
) -> str:
    """Build the LLM prompt with index patient + all candidates."""
    # Load prompt template
    prompt_path = Path("prompts/patient_like_me.txt")
    if prompt_path.exists():
        template = prompt_path.read_text()
    else:
        template = _default_prompt_template()

    # Load index patient data
    idx_info, idx_summary, idx_episodes = _load_patient_artifacts(index_patient_id)
    index_text = _format_patient_compact(idx_info, idx_summary, idx_episodes)

    # Format candidates
    candidate_blocks = []
    for i, cand in enumerate(candidates, 1):
        cand_pid = cand["patient_id"]
        cand_info, cand_summary, cand_episodes = _load_patient_artifacts(cand_pid)
        cand_text = _format_patient_compact(cand_info, cand_summary, cand_episodes)

        sources_str = " + ".join(cand["retrieval_sources"])
        scores_str = ", ".join(f"{k}={v}" for k, v in cand["retrieval_scores"].items())
        header = f"### Candidate {i} (ID: {cand_pid}) — Retrieved by: {sources_str} (scores: {scores_str})"

        candidate_blocks.append(f"{header}\n{cand_text}")

    candidates_text = "\n\n".join(candidate_blocks)

    # Fill template
    prompt = template.replace("{index_patient_text}", index_text)
    prompt = prompt.replace("{candidates_text}", candidates_text)
    prompt = prompt.replace("{n_candidates}", str(len(candidates)))
    prompt = prompt.replace("{index_patient_id}", index_patient_id)

    # Add index patient PLM record context for the judge
    idx_record = plm_index["patients"].get(index_patient_id, {})
    context_parts = [
        f"Diagnosis category: {idx_record.get('diagnosis_category', 'unknown')}",
        f"Metastatic: {idx_record.get('metastatic', 'unknown')}",
        f"Positive mutations: {', '.join(idx_record.get('positive_mutations', [])) or 'None'}",
        f"Current line: {idx_record.get('current_line_number', 0)}",
        f"Intent: {idx_record.get('current_intent', 'unknown')}",
    ]
    prompt = prompt.replace("{index_patient_context}", "; ".join(context_parts))

    return prompt


def _load_patient_artifacts(pid: str) -> tuple[dict, dict, list]:
    """Load patient_info.json, summary.json, episodes.json for a patient."""
    base = Path(f"temp_jsons/{pid}")

    info = {}
    info_path = base / "patient_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)

    summary = {}
    summary_path = base / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    episodes = []
    episodes_path = base / "episodes.json"
    if episodes_path.exists():
        with open(episodes_path) as f:
            episodes = json.load(f)

    return info, summary, episodes


def _default_prompt_template() -> str:
    """Fallback prompt template if file doesn't exist."""
    return """You are a thoracic oncology clinical similarity expert.

## INDEX PATIENT (ID: {index_patient_id})
Key features: {index_patient_context}

{index_patient_text}

## CANDIDATE PATIENTS ({n_candidates} candidates)

{candidates_text}

## YOUR TASK

Evaluate each candidate for TRUE clinical similarity to the index patient.
Clinical similarity means: same disease biology AND comparable treatment
decision point — patients a tumor board would consider "similar cases."

For each candidate, assess:
1. Disease biology alignment (diagnosis, histology, mutations, staging)
2. Treatment trajectory alignment (same line, same decision point)
3. Key differences that matter clinically

Select the 1-5 BEST clinical twins. For each selected twin:
- Explain WHY they are a good match (specific shared features)
- Note KEY DIFFERENCES (what's not the same)
- Describe their CLINICAL TRAJECTORY (what happened in their treatment journey)
- Rate similarity: "Strong Twin" or "Partial Twin"

Output valid JSON only:
{
  "twins": [
    {
      "patient_id": "...",
      "similarity_rating": "Strong Twin or Partial Twin",
      "rationale": "Both EGFR+ adenocarcinoma, both on 2nd line after...",
      "key_differences": "Candidate is 15 years younger, had prior surgery",
      "trajectory_summary": "Started osimertinib, progressed after 18mo...",
      "relevance_to_decision": "Shows outcome of switching to chemo-IO after EGFR TKI"
    }
  ],
  "cohort_context": "Of N candidates with same diagnosis, X had similar mutations...",
  "no_match_note": null
}"""


# ---------------------------------------------------------------------------
# Stage 4: LLM Clinical Judge
# ---------------------------------------------------------------------------

def find_clinical_twins(
    index_patient_id: str,
    plm_index: dict,
    model: str = "gpt-5",
    top_k_per_net: int = 5,
) -> dict:
    """Full pipeline: gate → retrieve → assemble → judge.

    Returns structured result with twins, metadata, and timing.
    """
    t0 = time.perf_counter()

    if index_patient_id not in plm_index["patients"]:
        return {"error": f"Patient {index_patient_id} not in PLM index"}

    index_patient = plm_index["patients"][index_patient_id]

    # Stage 1: Hard gate
    t1 = time.perf_counter()
    gated = hard_gate(index_patient, plm_index)
    gate_time = time.perf_counter() - t1

    # Stage 2: Multi-strategy retrieval
    t2 = time.perf_counter()
    candidates = retrieve_candidates(index_patient, gated, top_k_per_net)
    retrieval_time = time.perf_counter() - t2

    if not candidates:
        return {
            "index_patient_id": index_patient_id,
            "twins": [],
            "no_match_note": "No candidates passed hard gate",
            "candidates_after_gate": len(gated),
            "candidates_retrieved": 0,
        }

    # Stage 3: Context assembly
    t3 = time.perf_counter()
    prompt = assemble_context(index_patient_id, candidates, plm_index)
    assembly_time = time.perf_counter() - t3

    # Stage 4: LLM judge
    t4 = time.perf_counter()
    try:
        response = gsgpt.chat(
            prompt,
            system="You are a thoracic oncology clinical similarity expert. "
                   "Output valid JSON only, no markdown fences.",
            model=model,
        )
        llm_time = time.perf_counter() - t4

        # Parse JSON from response
        result = _parse_json_response(response)
    except Exception as e:
        llm_time = time.perf_counter() - t4
        result = {
            "twins": [],
            "error": f"LLM call failed: {e}",
        }

    total_time = time.perf_counter() - t0

    # Add metadata
    result["index_patient_id"] = index_patient_id
    result["candidates_after_gate"] = len(gated)
    result["candidates_retrieved"] = len(candidates)
    result["retrieval_details"] = [
        {
            "patient_id": c["patient_id"],
            "sources": c["retrieval_sources"],
            "scores": c["retrieval_scores"],
        }
        for c in candidates
    ]
    result["timing"] = {
        "gate_s": round(gate_time, 3),
        "retrieval_s": round(retrieval_time, 3),
        "assembly_s": round(assembly_time, 3),
        "llm_s": round(llm_time, 3),
        "total_s": round(total_time, 3),
    }
    result["model"] = model
    result["weight_mode"] = _WEIGHT_MODE
    result["weight_source"] = _WEIGHT_SOURCE
    result["active_weights"] = dict(_ACTIVE_WEIGHTS)

    return result


def _parse_json_response(response: str) -> dict:
    """Parse JSON from LLM response, handling markdown fences."""
    text = response.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # Remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {
            "twins": [],
            "parse_error": "Could not parse LLM response as JSON",
            "raw_response": text[:2000],
        }


# ---------------------------------------------------------------------------
# Retrieval-only mode
# ---------------------------------------------------------------------------

def retrieval_only(
    index_patient_id: str,
    plm_index: dict,
    top_k_per_net: int = 5,
) -> dict:
    """Run retrieval pipeline without LLM judge."""
    if index_patient_id not in plm_index["patients"]:
        return {"error": f"Patient {index_patient_id} not in PLM index"}

    index_patient = plm_index["patients"][index_patient_id]

    # Stage 1: Hard gate
    gated = hard_gate(index_patient, plm_index)

    # Stage 2: Retrieval
    candidates = retrieve_candidates(index_patient, gated, top_k_per_net)

    return {
        "index_patient_id": index_patient_id,
        "index_patient_summary": {
            "diagnosis_category": index_patient["diagnosis_category"],
            "metastatic": index_patient["metastatic"],
            "histology_category": index_patient["histology_category"],
            "positive_mutations": index_patient["positive_mutations"],
            "current_line_number": index_patient["current_line_number"],
            "current_intent": index_patient["current_intent"],
        },
        "candidates_after_gate": len(gated),
        "candidates_retrieved": len(candidates),
        "candidates": [
            {
                "patient_id": c["patient_id"],
                "retrieval_sources": c["retrieval_sources"],
                "retrieval_scores": c["retrieval_scores"],
                "diagnosis_category": c["diagnosis_category"],
                "histology_category": c["histology_category"],
                "metastatic": c["metastatic"],
                "positive_mutations": c["positive_mutations"],
                "current_line_number": c["current_line_number"],
                "current_intent": c["current_intent"],
                "regimen_classes": c["regimen_classes"],
                "age_at_tb": c["age_at_tb"],
                "smoking_category": c["smoking_category"],
                "ecog": c["ecog"],
                "tb_note": c.get("tb_note", "")[:200] + "..." if len(c.get("tb_note", "")) > 200 else c.get("tb_note", ""),
            }
            for c in candidates
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Patient Like Me — find clinically similar patients",
    )
    parser.add_argument("--pid", help="Index patient ID")
    parser.add_argument("--batch", help="Manifest JSON for batch mode")
    parser.add_argument("--index", default="plm_index.json", help="PLM index file")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--top-k", type=int, default=5, help="Top K per retrieval net")
    parser.add_argument("--model", default="gpt-5", help="LLM model for judge")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip LLM judge, show retrieval results only")
    parser.add_argument("--calibrated", nargs="?", const="outputs/plm/plm_calibration_results.json",
                        default=None,
                        help="Use Phase I Bayesian-calibrated posterior weights. "
                             "Optional value is the path to outputs/plm/plm_calibration_results.json "
                             "(default: ./plm_calibration_results.json).")
    parser.add_argument("--prior", action="store_true",
                        help="Force original prior weights (default if --calibrated not given).")
    args = parser.parse_args()

    # Configure weight mode before any retrieval call
    if args.calibrated and not args.prior:
        weights, mode, source = configure_weights("posterior", args.calibrated)
    else:
        weights, mode, source = configure_weights("prior")
    print(f"Retrieval weights: mode={mode}  source={source}")
    print("  " + "  ".join(f"{k}={v:+.2f}" for k, v in weights.items()))

    if not args.pid and not args.batch:
        parser.error("Provide --pid for single patient or --batch for batch mode")

    # Load PLM index
    print(f"Loading PLM index from: {args.index}")
    with open(args.index) as f:
        plm_index = json.load(f)
    print(f"  {plm_index['n_patients']} patients in index")

    if args.pid:
        # Single patient mode
        print(f"\nFinding clinical twins for patient {args.pid}...")

        if args.retrieval_only:
            result = retrieval_only(args.pid, plm_index, args.top_k)
            _print_retrieval_result(result)
        else:
            result = find_clinical_twins(
                args.pid, plm_index, model=args.model, top_k_per_net=args.top_k,
            )
            _print_twins_result(result)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {args.output}")

    elif args.batch:
        # Batch mode
        with open(args.batch) as f:
            manifest = json.load(f)
        pids = [p["patient_id"] for p in manifest["patients"]]
        print(f"\nBatch mode: {len(pids)} patients")

        results = []
        for i, pid in enumerate(pids, 1):
            print(f"\n[{i}/{len(pids)}] Processing {pid}...")
            if pid not in plm_index["patients"]:
                print(f"  Skipped (not in PLM index)")
                continue

            if args.retrieval_only:
                result = retrieval_only(pid, plm_index, args.top_k)
            else:
                result = find_clinical_twins(
                    pid, plm_index, model=args.model, top_k_per_net=args.top_k,
                )
            results.append(result)

            n_twins = len(result.get("twins", []))
            n_cands = result.get("candidates_retrieved", 0)
            print(f"  Gate: {result.get('candidates_after_gate', 0)} → "
                  f"Retrieved: {n_cands} → Twins: {n_twins}")

        output = args.output or "plm_batch_results.json"
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBatch results saved to: {output}")


def _print_retrieval_result(result: dict):
    """Pretty-print retrieval results."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    idx = result["index_patient_summary"]
    print(f"\n--- Index Patient ---")
    print(f"  Diagnosis: {idx['diagnosis_category']} | Histology: {idx['histology_category']}")
    print(f"  Metastatic: {idx['metastatic']} | Mutations: {', '.join(idx['positive_mutations']) or 'None'}")
    print(f"  Treatment line: {idx['current_line_number']} | Intent: {idx['current_intent']}")

    print(f"\n--- Gate: {result['candidates_after_gate']} candidates passed ---")
    print(f"--- Retrieved: {result['candidates_retrieved']} candidates ---\n")

    for i, c in enumerate(result["candidates"], 1):
        sources = " + ".join(c["retrieval_sources"])
        scores = ", ".join(f"{k}={v}" for k, v in c["retrieval_scores"].items())
        print(f"  {i}. [{c['patient_id']}] via {sources} ({scores})")
        print(f"     {c['diagnosis_category']} {c['histology_category']} | "
              f"Met: {c['metastatic']} | Muts: {', '.join(c['positive_mutations']) or 'None'}")
        print(f"     Line {c['current_line_number']} ({c['current_intent']}) | "
              f"ECOG {c['ecog']} | Age {c['age_at_tb']}")
        if c.get("tb_note"):
            print(f"     Note: {c['tb_note'][:120]}...")
        print()


def _print_twins_result(result: dict):
    """Pretty-print LLM judge results."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    timing = result.get("timing", {})
    print(f"\n--- Results (model: {result.get('model', '?')}) ---")
    print(f"  Gate: {result['candidates_after_gate']} | "
          f"Retrieved: {result['candidates_retrieved']} | "
          f"Twins found: {len(result.get('twins', []))}")
    print(f"  Timing: gate={timing.get('gate_s', 0)}s, "
          f"retrieval={timing.get('retrieval_s', 0)}s, "
          f"assembly={timing.get('assembly_s', 0)}s, "
          f"llm={timing.get('llm_s', 0)}s, "
          f"total={timing.get('total_s', 0)}s")

    for i, twin in enumerate(result.get("twins", []), 1):
        print(f"\n  Twin {i}: [{twin['patient_id']}] — {twin.get('similarity_rating', '?')}")
        print(f"    Rationale: {twin.get('rationale', 'N/A')}")
        print(f"    Differences: {twin.get('key_differences', 'N/A')}")
        print(f"    Trajectory: {twin.get('trajectory_summary', 'N/A')}")
        print(f"    Relevance: {twin.get('relevance_to_decision', 'N/A')}")

    if result.get("cohort_context"):
        print(f"\n  Cohort context: {result['cohort_context']}")
    if result.get("no_match_note"):
        print(f"\n  No match: {result['no_match_note']}")


if __name__ == "__main__":
    main()
