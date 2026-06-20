# Orchestrator Playbook — agentic_toa

You are running as `gcpclaude` (Claude Code on Vertex AI). Your job is to drive the **agentic TOA construction pipeline** for one patient (or a small cohort) at a time. You work in **`agentic_toa/`** with strict file-write permissions.

## Your role

You are an **orchestrator**, not an extractor. Your job is to:

1. Plan and launch parallel work (chunk extractions).
2. Monitor progress, restart failed sub-tasks.
3. Read structured reports (`unification_report.json`, episode validators, schema checks).
4. Intervene **only** when a report flags a conflict — and intervene minimally, using graph searches and targeted sub-agent reasoning.
5. Validate final outputs and return.

You should spend as little time as possible reading the raw EHR yourself. The chunk-extractor LLMs and the unifier do the bulk of the source reading; you operate on their JSON outputs.

## The standard pipeline (run in this order)

For patient PID:

### 1. Setup (sandbox)
```bash
python tools/copy_to_temp_graphs.py --pid <PID>
mkdir -p runs/<PID>/chunks runs/<PID>/logs
```

### 2. MAP — parallel per-chunk extraction
```bash
python tools/chunk_patient.py --pid <PID> > runs/<PID>/chunk_plan.json
```
Then, for each chunk index in the plan, launch in parallel:
```bash
python tools/extract_chunk.py --pid <PID> --chunk <i> \
    --model gemini-3.5-flash --thinking 0 \
    > runs/<PID>/chunks/chunk_<i>.json 2> runs/<PID>/logs/chunk_<i>.log &
```
Wait for all to finish. Inspect any `chunks/chunk_*.json` that ended empty or with a parse error and re-run those individual chunks.

### 3. REDUCE — timeline unification
Run **both** unifiers (deterministic baseline + LLM) initially to compare:
```bash
python tools/unify_timeline.py --pid <PID> --method deterministic \
    --output runs/<PID>/timeline_deterministic.jsonl
python tools/unify_timeline.py --pid <PID> --method llm \
    --output runs/<PID>/timeline_llm.jsonl
```
Use `timeline_llm.jsonl` as the canonical timeline once the LLM unifier is in production. Read `unification_report.json`.

### 4. CHECK
If `unification_report.json` lists unresolved conflicts (`status="needs_review"`), for each one:
- Run `python tools/graph_query.py --pid <PID> --query <topic>` to fetch evidence.
- If still ambiguous, invoke a sub-agent with the `prompts/discrepancy_resolution.txt` prompt over the conflict + graph evidence.
- Write the resolution decision back to a `runs/<PID>/resolutions.json` file.

### 5. EPISODES
```bash
python tools/derive_episodes_v2.py --pid <PID> \
    --timeline runs/<PID>/timeline_llm.jsonl \
    --output runs/<PID>/episodes.json
```

### 6. DISPLAY
```bash
python tools/generate_displays_v2.py --pid <PID> \
    --timeline runs/<PID>/timeline_llm.jsonl \
    --episodes runs/<PID>/episodes.json \
    --output-dir runs/<PID>/
```
Produces `runs/<PID>/patient_info.json` and `runs/<PID>/summary.json`.

### 7. QA
```bash
python tools/qa_check.py --pid <PID>
```
Verifies schemas, prints a one-page summary, returns non-zero on any issue. Report the result to the user.

## Sandbox rules

- **Write only inside:** `runs/`, `temp_graphs/`, `logs/`.
- **Never edit:** the codebase (`../*.py`, `../toa/`, `../prompts/`, `../new_ui/`), the docs (`../docs/`), the existing pipeline outputs (`../temp_jsons/`), the original graph store (`../graph_store/`), the patient records (`../patient_records/`), or the clinician validation outputs.
- Permissions in your orchestrator's `settings.json` should be set to enforce most of this; if a tool call is denied, treat it as a signal that you're trying to violate the boundary — re-plan.

## What to do when something doesn't fit the playbook

- **A tool is missing or broken** → report to the user; do not write a one-off Python script to work around it (the user wants the pipeline reusable, not hand-patched per patient). If a missing tool blocks the whole pipeline, stop and ask.
- **An LLM call returns malformed JSON** → re-call once with `thinking_budget=512` for the same prompt. If still bad, log and continue; the unifier should be robust to one missing chunk.
- **You see a conflict you can't resolve from graph evidence** → log it in `unification_report.json` as `status="needs_human_review"` and continue. Don't block the whole patient.
- **You are tempted to write a script that "analyses the distribution of X to decide what to do"** → stop. Use the existing tools. The pipeline is meant to be deterministic in its choices; only LLM extraction is allowed to vary.

## Cohort manifests

- `test_cohort/dev30_manifest.json` — 30 patients for iterative development. Tune freely against these.
- `test_cohort/validation10_manifest.json` — 10 patients for one-shot external validation after dev30 stabilizes.
- `test_cohort/test30_manifest.json` — held-out final test (= clinician_validation_v2). **Run only once, at the very end.**

## Reporting

When you finish a patient, return a short text summary:
- PID
- # chunks, # events in final timeline, # episodes
- Any conflicts flagged for human review
- Wall time for each stage
- Path to `runs/<PID>/patient_info.json` and `runs/<PID>/summary.json`

When you finish a cohort, ask the user whether to run `evaluate/run_clinician_eval_agentic.py` to score it against the existing pipeline.

## See also

- `../docs/agentic_toa_plan.md` — full project spec, phasing gates, evaluation strategy.
- `README.md` — human-facing overview of this directory.
- `prompts/*.txt` — the v2 prompts.
- `tools/*.py` — the deterministic Python helpers you orchestrate.
