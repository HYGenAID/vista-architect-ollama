#!/usr/bin/env bash
# run_cohort_parallel.sh — Parallel patient processing for VISTA Architect
#
# Usage:
#   ./run_cohort_parallel.sh [OPTIONS]
#
# Options:
#   -m, --manifest FILE    Cohort manifest JSON (required)
#   -j, --jobs N           Max parallel jobs (default: 3)
#   -n, --first N          Process only first N patients from manifest
#   -f, --force            Force reprocessing even if outputs exist
#   --skip-eval            Skip evaluation after processing
#   --eval-only            Run only evaluation (skip processing)
#   -h, --help             Show this help
#
# Examples:
#   ./run_cohort_parallel.sh -m patient_records/cohorts/eval_v2_100_manifest.json -n 20 -j 3
#   ./run_cohort_parallel.sh -m patient_records/cohorts/eval_v2_100_manifest.json --eval-only

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MANIFEST=""
JOBS=3
FIRST_N=0       # 0 = all
FORCE=""
SKIP_EVAL=false
EVAL_ONLY=false
CHUNK_MODEL=""
EPISODE_MODEL=""
INFO_MODEL=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/outputs/logs/cohort_run"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--manifest) MANIFEST="$2"; shift 2 ;;
        -j|--jobs)     JOBS="$2"; shift 2 ;;
        -n|--first)    FIRST_N="$2"; shift 2 ;;
        -f|--force)    FORCE="--force"; shift ;;
        --force-clean) FORCE="--force-clean"; shift ;;
        --skip-eval)   SKIP_EVAL=true; shift ;;
        --eval-only)   EVAL_ONLY=true; shift ;;
        --chunk-model)   CHUNK_MODEL="$2"; shift 2 ;;
        --episode-model) EPISODE_MODEL="$2"; shift 2 ;;
        --info-model)    INFO_MODEL="$2"; shift 2 ;;
        -h|--help)
            head -20 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$MANIFEST" ]]; then
    echo "Error: --manifest is required"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "Error: Manifest not found: $MANIFEST"
    exit 1
fi

# ── Extract patient IDs from manifest ─────────────────────────────────────────
COHORT_NAME=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['cohort_name'])")
ALL_PIDS=($(python3 -c "
import json
m = json.load(open('$MANIFEST'))
for p in m['patients']:
    print(p['patient_id'])
"))

TOTAL=${#ALL_PIDS[@]}

if [[ $FIRST_N -gt 0 && $FIRST_N -lt $TOTAL ]]; then
    PIDS=("${ALL_PIDS[@]:0:$FIRST_N}")
else
    PIDS=("${ALL_PIDS[@]}")
    FIRST_N=$TOTAL
fi

# ── Setup logging ─────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
RUN_LOG="${LOG_DIR}/${COHORT_NAME}_run_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUN_LOG"
}

log "═══════════════════════════════════════════════════════════════"
log "VISTA Cohort Parallel Runner"
log "═══════════════════════════════════════════════════════════════"
log "Cohort:     $COHORT_NAME"
log "Manifest:   $MANIFEST"
log "Patients:   ${#PIDS[@]} / $TOTAL"
log "Parallel:   $JOBS"
log "Force:      ${FORCE:-none}"
[[ -n "$CHUNK_MODEL" ]]   && log "Chunk:      $CHUNK_MODEL"
[[ -n "$EPISODE_MODEL" ]] && log "Episode:    $EPISODE_MODEL"
[[ -n "$INFO_MODEL" ]]    && log "Info:       $INFO_MODEL"
log "Eval:       $(if $SKIP_EVAL; then echo 'skip'; elif $EVAL_ONLY; then echo 'only'; else echo 'after processing'; fi)"
log "Log:        $RUN_LOG"
log "═══════════════════════════════════════════════════════════════"

# ── Process a single patient ──────────────────────────────────────────────────
process_patient() {
    local pid=$1
    local idx=$2
    local total=$3
    local patient_log="${LOG_DIR}/${pid}_${TIMESTAMP}.log"
    local output_dir="${SCRIPT_DIR}/temp_jsons/${pid}"
    local start_time=$(date +%s)

    # Check if already processed (patient_info.json exists)
    if [[ -z "$FORCE" && -f "${output_dir}/patient_info.json" ]]; then
        echo "[$(date '+%H:%M:%S')] [${idx}/${total}] SKIP ${pid} — already processed" | tee -a "$RUN_LOG"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] [${idx}/${total}] START ${pid}" | tee -a "$RUN_LOG"

    # Build model flags
    local model_flags=""
    [[ -n "$CHUNK_MODEL" ]]   && model_flags="$model_flags --chunk-model $CHUNK_MODEL"
    [[ -n "$EPISODE_MODEL" ]] && model_flags="$model_flags --episode-model $EPISODE_MODEL"
    [[ -n "$INFO_MODEL" ]]    && model_flags="$model_flags --info-model $INFO_MODEL"

    if python3 "${SCRIPT_DIR}/prepare_patient.py" \
        --pid "$pid" \
        --from-graph-store \
        $FORCE \
        $model_flags \
        > "$patient_log" 2>&1; then
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo "[$(date '+%H:%M:%S')] [${idx}/${total}] DONE  ${pid} (${elapsed}s)" | tee -a "$RUN_LOG"
        return 0
    else
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo "[$(date '+%H:%M:%S')] [${idx}/${total}] FAIL  ${pid} (${elapsed}s) — see ${patient_log}" | tee -a "$RUN_LOG"
        return 1
    fi
}

export -f process_patient
export SCRIPT_DIR LOG_DIR TIMESTAMP RUN_LOG FORCE CHUNK_MODEL EPISODE_MODEL INFO_MODEL

# ── Run processing ────────────────────────────────────────────────────────────
if [[ "$EVAL_ONLY" != "true" ]]; then
    log ""
    log "Starting parallel processing (${#PIDS[@]} patients, $JOBS parallel)..."
    log ""

    STARTED=$(date +%s)

    # Track success/fail/skip counts
    SUCCESS=0
    FAILED=0
    SKIPPED=0

    # Use a FIFO-based parallel job scheduler
    FIFO=$(mktemp -u)
    mkfifo "$FIFO"
    exec 3<>"$FIFO"
    rm "$FIFO"

    # Pre-fill the job slot tokens
    for ((i=0; i<JOBS; i++)); do
        echo >&3
    done

    # Track PIDs of background processes
    declare -A BG_PIDS
    IDX=0

    for pid in "${PIDS[@]}"; do
        IDX=$((IDX + 1))

        # Wait for a job slot
        read -u 3

        (
            process_patient "$pid" "$IDX" "${#PIDS[@]}" || true
            echo >&3  # Release job slot (must always run, even on failure)
        ) &
        BG_PIDS[$!]=$pid
    done

    # Wait for all background jobs
    FAIL_COUNT=0
    for bg_pid in "${!BG_PIDS[@]}"; do
        if ! wait "$bg_pid" 2>/dev/null; then
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done

    exec 3>&-

    ENDED=$(date +%s)
    WALL_TIME=$((ENDED - STARTED))
    WALL_MIN=$((WALL_TIME / 60))
    WALL_SEC=$((WALL_TIME % 60))

    log ""
    log "═══════════════════════════════════════════════════════════════"
    log "Processing complete"
    log "Wall time:  ${WALL_MIN}m ${WALL_SEC}s"
    log "Failures:   ${FAIL_COUNT}"
    log "═══════════════════════════════════════════════════════════════"

    # Count actual outcomes by checking outputs
    DONE_COUNT=0
    MISS_COUNT=0
    for pid in "${PIDS[@]}"; do
        if [[ -f "${SCRIPT_DIR}/temp_jsons/${pid}/patient_info.json" ]]; then
            DONE_COUNT=$((DONE_COUNT + 1))
        else
            MISS_COUNT=$((MISS_COUNT + 1))
            log "  MISSING output: $pid"
        fi
    done
    log "Patients with output: ${DONE_COUNT}/${#PIDS[@]}"
    if [[ $MISS_COUNT -gt 0 ]]; then
        log "Patients missing output: ${MISS_COUNT}"
    fi
fi

# ── Run evaluation ────────────────────────────────────────────────────────────
if [[ "$SKIP_EVAL" != "true" ]]; then
    log ""
    log "═══════════════════════════════════════════════════════════════"
    log "Running evaluation..."
    log "═══════════════════════════════════════════════════════════════"

    # Build list of PIDs that have outputs
    EVAL_PIDS=()
    for pid in "${PIDS[@]}"; do
        if [[ -f "${SCRIPT_DIR}/temp_jsons/${pid}/patient_info.json" ]]; then
            EVAL_PIDS+=("$pid")
        fi
    done

    if [[ ${#EVAL_PIDS[@]} -eq 0 ]]; then
        log "No patients with output to evaluate!"
    else
        EVAL_OUTPUT="${SCRIPT_DIR}/eval_v2_results_${TIMESTAMP}.json"
        EVAL_LOG="${LOG_DIR}/${COHORT_NAME}_eval_${TIMESTAMP}.log"

        log "Evaluating ${#EVAL_PIDS[@]} patients..."
        log "Output: $EVAL_OUTPUT"

        if python3 "${SCRIPT_DIR}/quick_eval.py" \
            "${EVAL_PIDS[@]}" \
            --eval-from-snapshot \
            -o "$EVAL_OUTPUT" \
            > "$EVAL_LOG" 2>&1; then
            log "Evaluation complete! Results: $EVAL_OUTPUT"

            # Print summary from eval results
            python3 -c "
import json, sys
try:
    with open('$EVAL_OUTPUT') as f:
        data = json.load(f)
    results = data if isinstance(data, list) else data.get('results', data.get('evaluations', []))
    if not results:
        print('  No results found in output')
        sys.exit(0)

    # Per-patient scores
    patient_scores = {}
    for r in results:
        pid = str(r.get('patient_id', 'unknown'))
        score = r.get('score', r.get('correctness_score', 0))
        if pid not in patient_scores:
            patient_scores[pid] = []
        patient_scores[pid].append(score)

    total_correct = sum(1 for r in results if r.get('score', r.get('correctness_score', 0)) >= 9)
    total = len(results)
    avg_scores = {pid: sum(s)/len(s) for pid, s in patient_scores.items()}
    overall_avg = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0

    print(f'  Patients evaluated: {len(avg_scores)}')
    print(f'  Total variables:    {total}')
    print(f'  Correct (>=9):      {total_correct}/{total} ({100*total_correct/total:.1f}%)')
    print(f'  Average score:      {overall_avg:.1f}/10')

    # Per-variable breakdown
    var_scores = {}
    for r in results:
        var = r.get('variable', 'unknown')
        score = r.get('score', r.get('correctness_score', 0))
        if var not in var_scores:
            var_scores[var] = []
        var_scores[var].append(score)

    print()
    print('  Per-variable averages:')
    for var in sorted(var_scores.keys()):
        scores = var_scores[var]
        avg = sum(scores) / len(scores)
        wrong = sum(1 for s in scores if s < 9)
        status = f' ({wrong} wrong)' if wrong else ''
        print(f'    {var:.<40s} {avg:.1f}{status}')
except Exception as e:
    print(f'  Could not parse results: {e}')
" 2>&1 | tee -a "$RUN_LOG"
        else
            log "Evaluation failed — see $EVAL_LOG"
        fi
    fi
fi

log ""
log "All done. Full log: $RUN_LOG"
