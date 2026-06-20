# agentic_toa — alternative VISTA Architect TOA pipeline driven by gcpclaude

**Status:** under active construction (May 2026). The existing `prepare_patient.py` / `toa/backend.py` pipeline is **untouched** and remains the production path. This directory is a parallel implementation that uses `gcpclaude` (Claude Code on Vertex AI) as the orchestrator and Gemini-3.5-Flash on Vertex for the heavy LLM lifts.

## Design at a glance

```
[user] → gcpclaude (orchestrator: claude-sonnet-4-6 on Vertex; cwd = agentic_toa/)
   ▼
   1. SETUP        → tools/copy_to_temp_graphs.py --pid X    # sandbox snapshot
   2. MAP          → for chunk i: tools/extract_chunk.py --pid X --chunk i
                       (gemini-3.5-flash, thinking_budget=0; parallel via bash &)
                       Emits chunks/chunk_i.json: {events, before_chunk_mentions, background}
   3. REDUCE       → tools/unify_timeline.py --pid X
                       v1: deterministic merge.
                       v2: LLM unifier (gemini-3.5-flash) over deterministic merge +
                           raw chunks + graph-search context.
                       Emits timeline_objects.jsonl + unification_report.json.
   4. CHECK        → agent reads unification_report.json; runs tools/graph_query.py
                       or spawns a Claude sub-agent for any unresolved conflicts.
   5. EPISODES     → tools/derive_episodes_v2.py --pid X    (single LLM call)
   6. DISPLAY      → tools/generate_displays_v2.py --pid X  (single LLM call)
   7. QA           → agent verifies schemas + returns
```

The agent reads source EHR text as little as possible — it works on the structured intermediates and only consults the graph or raw chunks when the reports flag a conflict.

Full spec: [`../docs/agentic_toa_plan.md`](../docs/agentic_toa_plan.md).

## Usage

**CLI (one-shot):**
```bash
python agentic_toa/cli/agentic_prepare_patient.py --pid 135982524 --max-chunks 3
```

**Interactive:**
```bash
cd agentic_toa
gcpclaude
> Prepare patient 135982524 using the dev profile.
```

The `CLAUDE.md` in this directory is auto-loaded by gcpclaude and tells it how to drive the pipeline.

## Directory layout

| | |
|---|---|
| `prompts/` | New v2 prompts (per-chunk, unifier, episodes, displays, discrepancy resolution). Existing prompts under `../toa/prompts/` and `../prompts/` are not touched. |
| `tools/` | Deterministic Python CLIs the agent invokes. |
| `cli/` | User-facing entry points (CLI mode). |
| `runs/` | Per-patient working directory; one subdir per PID. Gitignored. |
| `temp_graphs/` | Read-only Lumia-graph snapshots copied from `../graph_store/graphs/` at cohort setup. Gitignored. |
| `logs/` | Orchestrator + tool logs. Gitignored. |
| `test_cohort/` | Manifests for the dev/validation/test cohorts. |
| `evaluate/` | Per-variable A/B vs existing pipeline + `quick_eval --eval-from-snapshot` wrappers. |

## Cohort discipline

| Cohort | Size | Purpose |
|---|---:|---|
| `dev30` | 30 | Iterate prompts, thresholds, unifier rules. Free to tune. |
| `validation10` | 10 | External validation **after** dev30 stable. Run once. |
| `test30` | 30 | Held-out final test (= `clinician_validation_v2` cohort). Run **only** at the very end; no tuning after the result. |

All three are disjoint. Manifests live in `test_cohort/`.

## Safety

- The agent works only on **copies** in `temp_graphs/`. The original `../graph_store/`, `../patient_records/`, and `../temp_jsons/` are never modified.
- The agent **cannot** edit the existing codebase, prompts, or docs — the orchestrator's `settings.json` denies Edit/Write outside `agentic_toa/runs/`, `agentic_toa/temp_graphs/`, `agentic_toa/logs/`.
- LLM traffic is routed through whichever backend is configured in the environment (see `../SETUP.md`). For PHI-safe deployments, set `VISTA_LLM_BACKEND=vertex` and point `GOOGLE_CLOUD_PROJECT` at a private project you control.

## Models

| Role | Model | Thinking |
|---|---|---|
| Orchestrator | `claude-sonnet-4-6` (Vertex, via gcpclaude) | n/a |
| Per-chunk extraction | `gemini-3.5-flash` (Vertex) | 0 in dev |
| Timeline unification (v2 LLM mode) | `gemini-3.5-flash` | 0 in dev |
| Episode synthesis | `gemini-3.5-flash` | 0 in dev |
| Display items | `gemini-3.5-flash` | 0 in dev |
| Discrepancy resolution | `claude-sonnet-4-6` (in-session sub-agent) | n/a |

Per-model location overrides live in `gsgpt.VERTEX_MODEL_LOCATION` (gemini-3.5-flash is served only from `global`; gemini-2.5-flash from `us-central1`).
