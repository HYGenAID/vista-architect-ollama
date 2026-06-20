#!/usr/bin/env python3
"""
Bayesian RLAIF Calibration for Patient Like Me weights.

Uses LLM-as-judge (gemini-2.5-flash via Vertex AI) to score patient pairs,
then fits a Bayesian Beta Regression to learn optimal feature weights
from informative clinical priors.

Phase I of the PLM ML roadmap (docs/plm/plm.tex).

Usage:
    python plm_bayesian_calibration.py --index outputs/plm/plm_index_1180.json
    python plm_bayesian_calibration.py --index outputs/plm/plm_index_1180.json --n-index 30 --seed 42
    python plm_bayesian_calibration.py --skip-llm --pairs outputs/plm/plm_calibration_results.json
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Local imports
import gsgpt
from patient_like_me import hard_gate

# ---------------------------------------------------------------------------
# Feature names and Bayesian priors
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "mutation_concordance",
    "histology_match",
    "pdl1_concordance",
    "ln_concordance",
    "line_similarity",
    "intent_match",
    "regimen_jaccard",
    "surgery_match",
    "radiation_match",
    "age_similarity",
    "smoking_match",
]

# Informative priors: N(mean, sd) for each feature weight
PRIOR_MEANS = np.array([3.0, 1.5, 0.5, 0.3, 1.5, 1.0, 1.0, 0.5, 0.5, 0.0, 0.3])
PRIOR_SDS = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5])

# Intercept prior: N(-2.0, 1.0) -> baseline similarity ~12%
INTERCEPT_PRIOR_MEAN = -2.0
INTERCEPT_PRIOR_SD = 1.0

# Precision prior: log(nu) ~ N(log(20), 1.0)
NU_PRIOR_LOG_MEAN = np.log(20)
NU_PRIOR_LOG_SD = 1.0


# ---------------------------------------------------------------------------
# Feature vector computation
# ---------------------------------------------------------------------------

def compute_feature_vector(index_rec: dict, cand_rec: dict) -> np.ndarray:
    """Compute 11-dim feature vector for an (index, candidate) pair.

    All features are encoded as similarity: higher = more similar.
    """
    features = np.zeros(11)

    # 1. Mutation concordance (weighted)
    idx_muts = index_rec.get("driver_mutations", {})
    cand_muts = cand_rec.get("driver_mutations", {})
    score, compared = 0.0, 0
    for gene in idx_muts:
        idx_s = idx_muts[gene]
        cand_s = cand_muts.get(gene, "not_tested")
        if idx_s == "not_tested" or cand_s == "not_tested":
            continue
        compared += 1
        if idx_s == cand_s:
            score += 3.0 if idx_s == "positive" else 1.0
        else:
            score -= 1.0
    if compared > 0:
        features[0] = max(0.0, (score + compared) / (4 * compared))
    else:
        features[0] = 0.5  # neutral

    # 2. Histology match
    features[1] = 1.0 if (
        index_rec["histology_category"] == cand_rec["histology_category"]
        and index_rec["histology_category"] != "Other/NOS"
    ) else 0.0

    # 3. PD-L1 concordance
    idx_pdl1 = index_rec.get("pdl1_status", "unknown")
    cand_pdl1 = cand_rec.get("pdl1_status", "unknown")
    if idx_pdl1 == "unknown" or cand_pdl1 == "unknown":
        features[2] = 0.5  # neutral
    elif idx_pdl1 == cand_pdl1:
        features[2] = 1.0
    else:
        order = {"negative": 0, "low": 1, "high": 2}
        if idx_pdl1 in order and cand_pdl1 in order:
            features[2] = 0.5 if abs(order[idx_pdl1] - order[cand_pdl1]) <= 1 else 0.0
        else:
            features[2] = 0.0

    # 4. Lymph node concordance
    idx_ln = index_rec.get("lymph_node", "Unknown")
    cand_ln = cand_rec.get("lymph_node", "Unknown")
    if idx_ln == "Unknown" or cand_ln == "Unknown":
        features[3] = 0.5
    else:
        features[3] = 1.0 if idx_ln == cand_ln else 0.0

    # 5. Line similarity: 1 - min(|delta|, 5)/5
    idx_line = index_rec.get("current_line_number", 0)
    cand_line = cand_rec.get("current_line_number", 0)
    features[4] = 1.0 - min(abs(idx_line - cand_line), 5) / 5.0

    # 6. Intent match
    idx_intent = index_rec.get("current_intent", "unknown")
    cand_intent = cand_rec.get("current_intent", "unknown")
    if idx_intent == "unknown" or cand_intent == "unknown":
        features[5] = 0.5
    else:
        features[5] = 1.0 if idx_intent == cand_intent else 0.0

    # 7. Regimen class Jaccard
    idx_classes = set(index_rec.get("regimen_classes", []))
    cand_classes = set(cand_rec.get("regimen_classes", []))
    if idx_classes or cand_classes:
        union = idx_classes | cand_classes
        intersection = idx_classes & cand_classes
        features[6] = len(intersection) / len(union) if union else 0.0
    else:
        features[6] = 0.5  # both empty = neutral

    # 8. Surgery match
    features[7] = 1.0 if index_rec.get("had_surgery") == cand_rec.get("had_surgery") else 0.0

    # 9. Radiation match
    features[8] = 1.0 if index_rec.get("had_radiation") == cand_rec.get("had_radiation") else 0.0

    # 10. Age similarity: 1 - min(|delta|, 30)/30
    idx_age = index_rec.get("age_at_tb")
    cand_age = cand_rec.get("age_at_tb")
    if idx_age and cand_age:
        features[9] = 1.0 - min(abs(idx_age - cand_age), 30) / 30.0
    else:
        features[9] = 0.5

    # 11. Smoking match
    idx_smoke = index_rec.get("smoking_category", "Unknown")
    cand_smoke = cand_rec.get("smoking_category", "Unknown")
    if idx_smoke == "Unknown" or cand_smoke == "Unknown":
        features[10] = 0.5
    else:
        features[10] = 1.0 if idx_smoke == cand_smoke else 0.0

    return features


# ---------------------------------------------------------------------------
# Stratified index patient sampling
# ---------------------------------------------------------------------------

def sample_index_patients(plm_index: dict, n: int = 30, seed: int = 42) -> list[str]:
    """Select n stratified index patients covering clinical diversity."""
    rng = np.random.default_rng(seed)
    patients = list(plm_index["patients"].values())

    # Define strata
    strata = {}
    for p in patients:
        diag = p["diagnosis_category"]
        met = p["metastatic"]
        line = p["current_line_number"]

        if diag == "NSCLC":
            if met == "Yes":
                key = "NSCLC_met_lo" if line <= 1 else "NSCLC_met_hi"
            elif met == "No":
                key = "NSCLC_nomet_lo" if line <= 1 else "NSCLC_nomet_hi"
            else:
                key = "NSCLC_unk"
        elif diag in ("Thymoma/Thymic",):
            key = "Thymoma"
        elif diag == "Mesothelioma":
            key = "Mesothelioma"
        elif diag == "SCLC":
            key = "SCLC"
        else:
            key = "Other"

        strata.setdefault(key, []).append(p["patient_id"])

    # Allocate samples proportionally with minimum 1
    total = sum(len(v) for v in strata.values())
    allocation = {}
    remaining = n
    for key, pids in sorted(strata.items(), key=lambda x: len(x[1])):
        alloc = max(1, round(n * len(pids) / total))
        alloc = min(alloc, len(pids), remaining)
        allocation[key] = alloc
        remaining -= alloc

    # If we have remaining slots, add to largest strata
    if remaining > 0:
        for key in sorted(strata.keys(), key=lambda k: len(strata[k]), reverse=True):
            add = min(remaining, len(strata[key]) - allocation[key])
            allocation[key] += add
            remaining -= add
            if remaining == 0:
                break

    # Sample from each stratum
    selected = []
    for key, count in allocation.items():
        pool = strata[key]
        chosen = rng.choice(pool, size=min(count, len(pool)), replace=False)
        selected.extend(chosen.tolist())

    print(f"\nStratified sampling: {len(selected)} index patients")
    for key in sorted(allocation.keys()):
        print(f"  {key}: {allocation[key]}/{len(strata[key])}")

    return selected[:n]


# ---------------------------------------------------------------------------
# Candidate retrieval (reuse existing hard gate + simplified scoring)
# ---------------------------------------------------------------------------

def get_candidates_for_index(index_pid: str, plm_index: dict, top_k: int = 15) -> list[dict]:
    """Get top-k candidate patients for an index patient."""
    index_rec = plm_index["patients"][index_pid]

    # Hard gate
    gated = hard_gate(index_rec, plm_index)
    if not gated:
        return []

    # Score all gated candidates using current biology + trajectory
    scored = []
    for cand in gated:
        fv = compute_feature_vector(index_rec, cand)
        # Simple composite score for ranking (just to select diverse candidates)
        composite = float(np.mean(fv[:7]))  # biology + trajectory features
        scored.append((cand, composite, fv))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Take top-k but also include some lower-ranked for diversity
    top = scored[:top_k - 3] if len(scored) > top_k else scored
    # Add 3 mid-range candidates for score diversity
    if len(scored) > top_k:
        mid = len(scored) // 2
        mid_candidates = scored[mid:mid+3]
        top.extend(mid_candidates)

    return [
        {
            "patient_id": cand["patient_id"],
            "record": cand,
            "feature_vector": fv.tolist(),
        }
        for cand, _, fv in top
    ]


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

def format_patient_summary(rec: dict) -> str:
    """Format a PLM index record as compact text for the LLM judge."""
    muts = rec.get("driver_mutations", {})
    mut_parts = []
    for gene, status in muts.items():
        if status == "positive":
            mut_parts.append(f"{gene}+")
        elif status == "negative":
            mut_parts.append(f"{gene}-")
    mut_str = ", ".join(mut_parts) if mut_parts else "Not tested"

    regimens = ", ".join(rec.get("regimen_classes", [])) or "None"
    tb_note = rec.get("tb_note", "")
    if len(tb_note) > 400:
        tb_note = tb_note[:400] + "..."

    lines = [
        f"{rec['diagnosis_category']} {rec['histology_category']} | Metastatic: {rec['metastatic']} | LN: {rec.get('lymph_node', 'Unknown')}",
        f"Mutations: {mut_str} | PD-L1: {rec.get('pdl1_status', 'unknown')}",
        f"Line {rec.get('current_line_number', 0)} ({rec.get('current_intent', 'unknown')}) | Regimens: {regimens}",
        f"Surgery: {'Yes' if rec.get('had_surgery') else 'No'} | Radiation: {'Yes' if rec.get('had_radiation') else 'No'} | Age: {int(rec['age_at_tb']) if rec.get('age_at_tb') else '?'} | ECOG: {rec.get('ecog', '?')} | Smoking: {rec.get('smoking_category', 'Unknown')}",
    ]
    if tb_note:
        lines.append(f"TB Note: {tb_note}")
    return "\n".join(lines)


def _extract_first_json_array(text: str) -> str:
    """Extract the first complete JSON array from text using bracket counting."""
    start = text.find('[')
    if start == -1:
        raise ValueError("No JSON array found in text")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\':
            if in_string:
                escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                return text[start:i+1]

    raise ValueError("Unclosed JSON array")


def judge_candidates(
    index_pid: str,
    candidates: list[dict],
    plm_index: dict,
    model: str = "gemini-2.5-flash",
) -> list[dict]:
    """Call LLM judge to score candidates for an index patient.

    Returns list of {candidate_id, score, rationale}.
    """
    index_rec = plm_index["patients"][index_pid]

    # Load prompt template
    prompt_path = Path("prompts/plm_judge_numeric.txt")
    template = prompt_path.read_text()

    # Format index patient
    index_summary = format_patient_summary(index_rec)

    # Format candidates
    cand_blocks = []
    for i, cand in enumerate(candidates, 1):
        cand_summary = format_patient_summary(cand["record"])
        cand_blocks.append(f"### Candidate {i} (ID: {cand['patient_id']})\n{cand_summary}")
    candidates_block = "\n\n".join(cand_blocks)

    # Fill template
    prompt = template.replace("{index_patient_summary}", index_summary)
    prompt = prompt.replace("{candidates_block}", candidates_block)

    # Call LLM
    response = gsgpt.chat(
        prompt,
        system="You are a thoracic oncology clinical similarity expert. Output ONLY a JSON array.",
        model=model,
    )

    # Parse response
    try:
        json_str = _extract_first_json_array(response)
        scores = json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  WARNING: Failed to parse judge response for {index_pid}: {e}")
        print(f"  Response (first 500 chars): {response[:500]}")
        return []

    return scores


def run_judge_parallel(
    index_pids: list[str],
    all_candidates: dict,
    plm_index: dict,
    model: str = "gemini-2.5-flash",
    max_workers: int = 15,
) -> list[dict]:
    """Run LLM judge for all index patients in parallel.

    Returns list of {index_pid, candidate_id, score, rationale, feature_vector}.
    """
    all_pairs = []

    def _judge_one(pid):
        cands = all_candidates.get(pid, [])
        if not cands:
            return pid, []
        scores = judge_candidates(pid, cands, plm_index, model=model)
        return pid, scores

    print(f"\nRunning LLM judge on {len(index_pids)} index patients ({max_workers} workers)...")
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_judge_one, pid): pid for pid in index_pids}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                pid_result, scores = future.result()
                # Match scores to candidates
                cands = all_candidates.get(pid_result, [])
                cand_lookup = {c["patient_id"]: c for c in cands}

                for s in scores:
                    cand_id = s.get("candidate_id", "")
                    score_val = s.get("score", 0.0)
                    rationale = s.get("rationale", "")

                    if cand_id in cand_lookup:
                        all_pairs.append({
                            "index_pid": pid_result,
                            "candidate_pid": cand_id,
                            "score": float(np.clip(score_val, 0.01, 0.99)),
                            "rationale": rationale,
                            "feature_vector": cand_lookup[cand_id]["feature_vector"],
                        })

                n_matched = sum(1 for s in scores if s.get("candidate_id") in cand_lookup)
                print(f"  {pid_result}: {n_matched}/{len(cands)} candidates scored")
            except Exception as e:
                print(f"  {pid}: FAILED - {e}")

    elapsed = time.perf_counter() - t0
    print(f"  Judge complete: {len(all_pairs)} pairs in {elapsed:.1f}s")
    return all_pairs


# ---------------------------------------------------------------------------
# Bayesian Beta Regression (MAP + Laplace)
# ---------------------------------------------------------------------------

def neg_log_posterior(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Negative log-posterior for Beta regression with Gaussian priors."""
    from scipy.special import gammaln

    n_features = len(FEATURE_NAMES)
    beta = params[:n_features]
    beta_0 = params[n_features]
    log_nu = params[n_features + 1]
    nu = np.exp(log_nu)

    # Mean via logistic link
    eta = X @ beta + beta_0
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -20, 20)))
    mu = np.clip(mu, 1e-6, 1 - 1e-6)

    # Beta distribution parameters
    a = mu * nu
    b = (1 - mu) * nu
    a = np.clip(a, 1e-6, 1e6)
    b = np.clip(b, 1e-6, 1e6)

    # Beta log-likelihood
    ll = np.sum(
        gammaln(nu) - gammaln(a) - gammaln(b)
        + (a - 1) * np.log(np.clip(y, 1e-10, 1.0))
        + (b - 1) * np.log(np.clip(1 - y, 1e-10, 1.0))
    )

    # Gaussian priors on feature weights
    log_prior_beta = -0.5 * np.sum(((beta - PRIOR_MEANS) / PRIOR_SDS) ** 2)

    # Prior on intercept
    log_prior_b0 = -0.5 * ((beta_0 - INTERCEPT_PRIOR_MEAN) / INTERCEPT_PRIOR_SD) ** 2

    # Prior on log(nu)
    log_prior_nu = -0.5 * ((log_nu - NU_PRIOR_LOG_MEAN) / NU_PRIOR_LOG_SD) ** 2

    return -(ll + log_prior_beta + log_prior_b0 + log_prior_nu)


def neg_log_posterior_grad(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of negative log-posterior (numerical)."""
    from scipy.optimize import approx_fprime
    return approx_fprime(params, neg_log_posterior, 1e-5, X, y)


def fit_beta_regression(X: np.ndarray, y: np.ndarray) -> dict:
    """Fit Beta regression via MAP + Laplace approximation.

    Returns posterior means, SDs, and diagnostics.
    """
    from scipy.optimize import minimize

    n_features = len(FEATURE_NAMES)
    n_params = n_features + 2  # features + intercept + log(nu)

    # Initialize at prior means
    x0 = np.concatenate([PRIOR_MEANS, [INTERCEPT_PRIOR_MEAN], [NU_PRIOR_LOG_MEAN]])

    print(f"\nFitting Beta regression: {X.shape[0]} pairs, {n_features} features...")
    t0 = time.perf_counter()

    result = minimize(
        neg_log_posterior,
        x0,
        args=(X, y),
        method='L-BFGS-B',
        jac=neg_log_posterior_grad,
        options={'maxiter': 1000, 'ftol': 1e-8},
    )

    elapsed = time.perf_counter() - t0
    print(f"  Optimization: {'converged' if result.success else 'FAILED'} in {elapsed:.2f}s")
    print(f"  Iterations: {result.nit}, Final loss: {result.fun:.2f}")

    # Extract MAP estimates
    map_beta = result.x[:n_features]
    map_intercept = result.x[n_features]
    map_log_nu = result.x[n_features + 1]

    # Laplace approximation: posterior covariance = inverse Hessian
    print("  Computing Hessian for Laplace approximation...")
    hessian = _compute_hessian(neg_log_posterior, result.x, X, y)
    try:
        posterior_cov = np.linalg.inv(hessian)
        posterior_sds = np.sqrt(np.abs(np.diag(posterior_cov)))
    except np.linalg.LinAlgError:
        print("  WARNING: Hessian singular, using approximate SDs")
        # Use L-BFGS-B inverse Hessian approximation
        if hasattr(result, 'hess_inv'):
            h_inv = result.hess_inv
            if hasattr(h_inv, 'todense'):
                posterior_sds = np.sqrt(np.abs(np.diag(h_inv.todense())))
            else:
                posterior_sds = np.sqrt(np.abs(np.diag(h_inv)))
        else:
            posterior_sds = PRIOR_SDS * 0.8  # fallback

    return {
        "posterior_means": map_beta,
        "posterior_intercept": float(map_intercept),
        "posterior_log_nu": float(map_log_nu),
        "posterior_sds": posterior_sds[:n_features],
        "intercept_sd": float(posterior_sds[n_features]) if len(posterior_sds) > n_features else 0.5,
        "converged": result.success,
        "n_iterations": result.nit,
        "final_loss": float(result.fun),
    }


def _compute_hessian(
    func, x0: np.ndarray, *args, eps: float = 1e-4
) -> np.ndarray:
    """Compute Hessian via central differences."""
    n = len(x0)
    hess = np.zeros((n, n))
    f0 = func(x0, *args)

    for i in range(n):
        for j in range(i, n):
            x_pp = x0.copy(); x_pp[i] += eps; x_pp[j] += eps
            x_pm = x0.copy(); x_pm[i] += eps; x_pm[j] -= eps
            x_mp = x0.copy(); x_mp[i] -= eps; x_mp[j] += eps
            x_mm = x0.copy(); x_mm[i] -= eps; x_mm[j] -= eps

            hess[i, j] = (func(x_pp, *args) - func(x_pm, *args)
                          - func(x_mp, *args) + func(x_mm, *args)) / (4 * eps * eps)
            hess[j, i] = hess[i, j]

    return hess


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_prior_vs_posterior(
    posterior_means: np.ndarray,
    posterior_sds: np.ndarray,
    output_path: str = "docs/plm/prior_vs_posterior.png",
):
    """Forest plot comparing prior and posterior weight distributions."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    n = len(FEATURE_NAMES)
    y_pos = np.arange(n)

    # Prior (blue)
    ax.errorbar(PRIOR_MEANS, y_pos + 0.15, xerr=2*PRIOR_SDS,
                fmt='o', color='#2196F3', markersize=8, capsize=4,
                label='Prior (clinical domain knowledge)', linewidth=1.5)

    # Posterior (orange)
    ax.errorbar(posterior_means, y_pos - 0.15, xerr=2*posterior_sds,
                fmt='s', color='#FF9800', markersize=8, capsize=4,
                label='Posterior (RLAIF-calibrated)', linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.replace('_', ' ').title() for name in FEATURE_NAMES],
                       fontsize=11)
    ax.set_xlabel('Weight (95% Credible Interval)', fontsize=12)
    ax.set_title('Bayesian Prior vs Posterior Feature Weights\n'
                 '(Phase I: RLAIF Calibration via Gemini-2.5-Flash Judge)',
                 fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_feature_importance(
    posterior_means: np.ndarray,
    output_path: str = "docs/plm/feature_importance.png",
):
    """Barplot of absolute posterior weight values, ranked."""
    import matplotlib.pyplot as plt

    abs_weights = np.abs(posterior_means)
    order = np.argsort(abs_weights)[::-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(order)))

    bars = ax.bar(
        range(len(order)),
        abs_weights[order],
        color=colors,
        edgecolor='white',
        linewidth=0.5,
    )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [FEATURE_NAMES[i].replace('_', ' ').title() for i in order],
        rotation=45, ha='right', fontsize=10,
    )
    ax.set_ylabel('|Posterior Weight|', fontsize=12)
    ax.set_title('Feature Importance Ranking\n'
                 '(Bayesian Beta Regression Posterior Weights)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, abs_weights[order]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_calibration_scatter(
    X: np.ndarray,
    y: np.ndarray,
    posterior_means: np.ndarray,
    posterior_intercept: float,
    output_path: str = "docs/plm/calibration_scatter.png",
):
    """Scatter plot of predicted vs actual judge scores."""
    import matplotlib.pyplot as plt

    eta = X @ posterior_means + posterior_intercept
    y_pred = 1.0 / (1.0 + np.exp(-np.clip(eta, -20, 20)))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y, y_pred, alpha=0.5, s=30, color='#2196F3', edgecolors='white', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

    # Correlation
    corr = float(np.corrcoef(y, y_pred)[0, 1])
    rmse = float(np.sqrt(np.mean((y - y_pred)**2)))
    ax.text(0.05, 0.92, f'r = {corr:.3f}\nRMSE = {rmse:.3f}',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('LLM Judge Score (Actual)', fontsize=12)
    ax.set_ylabel('Beta Regression (Predicted)', fontsize=12)
    ax.set_title('Calibration: Predicted vs Actual Similarity Scores',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    return corr, rmse


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def save_calibration_table(
    posterior_means: np.ndarray,
    posterior_sds: np.ndarray,
    output_path: str = "docs/plm/calibration_table.md",
):
    """Save markdown table of prior vs posterior weights."""
    lines = [
        "# Bayesian RLAIF Calibration Results",
        "",
        "| Feature | Prior Mean | Prior SD | Posterior Mean | Posterior SD | Change |",
        "|---------|-----------|---------|---------------|-------------|--------|",
    ]
    for i, name in enumerate(FEATURE_NAMES):
        change = posterior_means[i] - PRIOR_MEANS[i]
        direction = "+" if change > 0 else ""
        lines.append(
            f"| {name.replace('_', ' ').title()} | "
            f"{PRIOR_MEANS[i]:.2f} | {PRIOR_SDS[i]:.2f} | "
            f"{posterior_means[i]:.2f} | {posterior_sds[i]:.2f} | "
            f"{direction}{change:.2f} |"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n")
    print(f"  Saved: {output_path}")

    # Also print to console
    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian RLAIF calibration for PLM weights",
    )
    parser.add_argument("--index", default="outputs/plm/plm_index_1180.json",
                        help="PLM index file")
    parser.add_argument("--n-index", type=int, default=30,
                        help="Number of index patients to sample")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Candidates per index patient")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="LLM model for judge")
    parser.add_argument("--max-workers", type=int, default=15,
                        help="Parallel LLM workers")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM calls, load pairs from file")
    parser.add_argument("--pairs", default="outputs/plm/plm_calibration_results.json",
                        help="Pairs file (for --skip-llm mode)")
    parser.add_argument("--output", default="outputs/plm/plm_calibration_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("VISTA PLM: Bayesian RLAIF Weight Calibration (Phase I)")
    print("=" * 70)

    # Load PLM index
    print(f"\nLoading PLM index: {args.index}")
    with open(args.index) as f:
        plm_index = json.load(f)
    print(f"  {plm_index['n_patients']} patients in index")

    if args.skip_llm:
        # Load pre-computed pairs
        print(f"\nLoading pre-computed pairs: {args.pairs}")
        with open(args.pairs) as f:
            saved = json.load(f)
        pairs = saved["pairs"]
        print(f"  {len(pairs)} pairs loaded")
    else:
        # Step 1: Sample index patients
        index_pids = sample_index_patients(plm_index, n=args.n_index, seed=args.seed)

        # Step 2: Get candidates for each index patient
        print(f"\nRetrieving candidates (top-{args.top_k} per index patient)...")
        all_candidates = {}
        total_cands = 0
        for pid in index_pids:
            cands = get_candidates_for_index(pid, plm_index, top_k=args.top_k)
            all_candidates[pid] = cands
            total_cands += len(cands)
        print(f"  Total: {total_cands} (index, candidate) pairs across {len(index_pids)} index patients")

        # Step 3: Run LLM judge
        pairs = run_judge_parallel(
            index_pids, all_candidates, plm_index,
            model=args.model, max_workers=args.max_workers,
        )

        if not pairs:
            print("\nERROR: No pairs scored. Check LLM connectivity.")
            sys.exit(1)

    # Step 4: Prepare data for regression
    X = np.array([p["feature_vector"] for p in pairs])
    y = np.array([p["score"] for p in pairs])
    y = np.clip(y, 0.01, 0.99)  # Beta boundary safety

    print(f"\nData summary:")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Score range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Score mean: {y.mean():.3f}, median: {np.median(y):.3f}")
    print(f"  Score SD: {y.std():.3f}")

    # Step 5: Fit Beta regression
    results = fit_beta_regression(X, y)

    posterior_means = results["posterior_means"]
    posterior_sds = results["posterior_sds"]

    # Step 6: Generate outputs
    print("\n--- Generating outputs ---")

    # Plots
    plot_prior_vs_posterior(posterior_means, posterior_sds)
    plot_feature_importance(posterior_means)
    corr, rmse = plot_calibration_scatter(
        X, y, posterior_means, results["posterior_intercept"]
    )

    # Table
    save_calibration_table(posterior_means, posterior_sds)

    # Save full results JSON
    output = {
        "meta": {
            "n_index_patients": args.n_index,
            "n_pairs": len(pairs),
            "seed": args.seed,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "prior_weights": {name: float(PRIOR_MEANS[i]) for i, name in enumerate(FEATURE_NAMES)},
        "prior_sds": {name: float(PRIOR_SDS[i]) for i, name in enumerate(FEATURE_NAMES)},
        "posterior_weights": {name: float(posterior_means[i]) for i, name in enumerate(FEATURE_NAMES)},
        "posterior_sds": {name: float(posterior_sds[i]) for i, name in enumerate(FEATURE_NAMES)},
        "intercept": {
            "prior_mean": INTERCEPT_PRIOR_MEAN,
            "posterior_mean": results["posterior_intercept"],
        },
        "precision_nu": {
            "posterior_log_nu": results["posterior_log_nu"],
            "posterior_nu": float(np.exp(results["posterior_log_nu"])),
        },
        "diagnostics": {
            "converged": results["converged"],
            "n_iterations": results["n_iterations"],
            "final_loss": results["final_loss"],
            "correlation": corr,
            "rmse": rmse,
        },
        "feature_importance_ranking": [
            FEATURE_NAMES[i]
            for i in np.argsort(np.abs(posterior_means))[::-1]
        ],
        "score_distribution": {
            "mean": float(y.mean()),
            "median": float(np.median(y)),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        },
        "pairs": pairs,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Full results saved: {args.output}")

    # Summary
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"\n  Pairs scored: {len(pairs)}")
    print(f"  Convergence: {'Yes' if results['converged'] else 'NO'}")
    print(f"  Correlation (pred vs actual): {corr:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"\n  Top 3 most important features:")
    importance_order = np.argsort(np.abs(posterior_means))[::-1]
    for rank, idx in enumerate(importance_order[:3], 1):
        print(f"    {rank}. {FEATURE_NAMES[idx]}: {posterior_means[idx]:.2f} "
              f"(prior: {PRIOR_MEANS[idx]:.2f}, change: {posterior_means[idx]-PRIOR_MEANS[idx]:+.2f})")

    print(f"\n  Outputs:")
    print(f"    docs/plm/prior_vs_posterior.png")
    print(f"    docs/plm/feature_importance.png")
    print(f"    docs/plm/calibration_scatter.png")
    print(f"    docs/plm/calibration_table.md")
    print(f"    {args.output}")


if __name__ == "__main__":
    main()
