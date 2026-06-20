# Agentic TOA Construction Pipeline — Project Spec

**Status:** Phase 0 scaffolding starting May 2026.
**Scope:** alternative pipeline that runs alongside the existing `prepare_patient.py` / `toa/backend.py` flow. The existing pipeline is **untouched**. Once the agentic version is validated on the held-out clinician_validation_v2 cohort, we may unify them as a CLI option.

## Constraints baked in from user direction

1. **Cohort discipline (no data leakage):** dev iteration uses a *separate* 30-patient cohort. A 10-patient external validation set comes next. Only after that runs cleanly do we run on the 30-patient `clinician_validation_v2` cohort. This protects the headline number from "we tuned to the test set" criticism.
2. **Gemini-only via Vertex.** No GPT in the agentic pipeline — large rate-limited proxies bottleneck real parallelism. Everything routes through Vertex AI in a user-controlled GCP project (PHI-safe when the project is locked down appropriately; see `../SETUP.md`).
3. **`gemini-3.5-flash` as the default model for every LLM step** (chunk extraction, unification, episodes, display items). GA since 2026-05-19; near-Pro intelligence with thinking; 1M token context. Claude Sonnet/Opus on Vertex can be swapped in later if Gemini falls short on specific steps.
4. **Thinking budget = 0 for dev** (Flash-speed). After Phase 4 stabilizes, sweep `[0, 1024, 4096]` to test if reasoning helps unification or chunk grounding.
5. **Unification: LLM-led, deterministic fallback.** User hypothesis: deterministic merge will fail because chunks won't agree on dates/details — too much room for hallucination/duplication. Build deterministic FIRST as a baseline diagnostic, then layer an LLM unifier on top that takes the deterministic merge + raw chunks + graph-search context and produces the final timeline.
6. **Graph safety:** `graph_store/` is **never touched**. The agent operates exclusively on copies in `agentic_toa/temp_graphs/{pid}.graphml`. Same applies to original XMLs and existing pipeline outputs in `temp_jsons/`.
7. **Existing pipeline outputs untouched.** Agentic outputs live in `agentic_toa/runs/{pid}/`, not `temp_jsons/{pid}/`.
8. **Case 09 (pid 135933241) excluded from dev cohort** — already in `clinician_validation_v2` so it's held out for the final test anyway.

## Pipeline architecture

```
[user] → gcpclaude (orchestrator: claude-sonnet-4-6 via Vertex; working dir agentic_toa/)
   ▼
   1. SETUP        → tools/copy_to_temp_graphs.py --pid X
   ▼
   2. MAP          → tools/chunk_patient.py --pid X            (plans chunks)
                     for each chunk i:
                       tools/extract_chunk.py --pid X --chunk i --model gemini-3.5-flash
                       (gemini-3.5-flash with thinking_budget=0;
                        runs in parallel via background bash; agent monitors,
                        restarts failed chunks)
                     each chunk emits: events, before_chunk_mentions, background
                     each chunk gets graph-search context from
                       DeterministicRetriever (smoking, mets, mutations, drugs,
                       ECOG, allergies, imaging dates) — same as existing pipeline
   ▼
   3. REDUCE       → tools/unify_timeline.py --pid X --method {deterministic|llm}
                     v1: deterministic merge (dedupe, prefer in-chunk over before-chunk)
                     v2 (default after Phase 2): LLM unifier (gemini-3.5-flash) that
                       consumes deterministic-merge + raw chunks + graph context and
                       emits final timeline_objects.jsonl + unification_report.json
   ▼
   4. CHECK        → agent reads unification_report.json
                     if discrepancies flagged → invokes tools/graph_query.py
                       or spawns Claude sub-agent (Task tool) with the targeted
                       prompts/discrepancy_resolution.txt prompt
   ▼
   5. EPISODES     → tools/derive_episodes_v2.py --pid X
                     single LLM call (gemini-3.5-flash) on the unified timeline
                     using prompts/episode_synthesis_v2.txt
   ▼
   6. DISPLAY      → tools/generate_displays_v2.py --pid X
                     single LLM call (gemini-3.5-flash) producing
                     patient_info.json + summary.json
                     using prompts/display_items_v2.txt
   ▼
   7. QA           → agent verifies schemas, summarizes for user, returns
```

The agent reads source EHR text as little as possible — only when a downstream report surfaces a conflict that can't be resolved from structured intermediates.

## New prompts (live in `agentic_toa/prompts/`, never overwrite the old ones)

| File | Purpose |
|---|---|
| `chunk_events_v2.txt` | Per-chunk extraction without cross-chunk timeline carry-over. Emits 3 arrays: `events` (in-window, high-trust facts), `before_chunk_mentions` (events mentioned in this chunk's text but dated before the chunk window — flagged for the unifier with lower trust on dates), `background` (pre-cancer baseline). |
| `timeline_unification_v2.txt` | LLM unifier prompt. Inputs: deterministic merge, raw chunk JSONs, graph-search context. Output: final timeline_objects.jsonl + reconciliation notes. |
| `episode_synthesis_v2.txt` | Like the existing episodes prompt but without the `patient_state` block (display step handles that), and with a stricter "no carry-over events between lines" rule since the unified timeline is already clean. |
| `display_items_v2.txt` | Combined patient_info + summary, same shape as the existing combined prompt, fed unified timeline + episodes. |
| `discrepancy_resolution.txt` | Targeted prompt the agent invokes only when the unification_report flags an unresolved conflict (e.g., two chunks claim different dates for the same event). Fed conflicting evidence + relevant graph_query output. |

## Models matrix

| Role | Model | Thinking |
|---|---|---|
| Orchestrator (gcpclaude session) | `claude-sonnet-4-6` (Vertex) | n/a |
| Per-chunk extraction | `gemini-3.5-flash` (Vertex) | 0 in dev → sweep later |
| Timeline unification (LLM) | `gemini-3.5-flash` | 0 → sweep later |
| Discrepancy resolution sub-agent | `claude-sonnet-4-6` (in-session Task tool) | n/a |
| Episode synthesis | `gemini-3.5-flash` | 0 → sweep later |
| Display items | `gemini-3.5-flash` | 0 → sweep later |
| **Eval judge (canonical from Phase 2 on)** | **`claude-opus-4-6` (Vertex)** | n/a |

**Locked-in pipeline decisions from Phase 1 single-patient smoke test:**

- **`source_text` STAYS ON in the final pipeline.** The structured-only variant scored 8.81/10 vs the with-source-text variant's 9.31/10 (Opus 4.6 judge). The richness in prose-heavy fields (Therapy Toxicity / Comorbidities, Diagnosis context, TB note quality) matters for the actual tumor-board display purpose — even when binary MTB-variable scores agree. `--no-source-text` remains as an A/B switch for future ablation studies.
- **Eval judge swap: Gemini 3.5 Flash → Claude Opus 4.6.** Phase 1 showed Gemini Flash marked all three variants (existing pipeline, agentic+src, agentic-no-src) at 10.0/10 — too lenient to detect meaningful differences. Opus 4.6 scored the same outputs 9.00 / 9.31 / 8.81 and correctly caught the source_text-driven differences on Metastasis, LN Involvement, and Therapy Toxicity. Use Opus as the canonical judge for all agentic evaluations from Phase 2 onward.
- **Agentic pipeline beats existing pipeline on the single Phase 1 patient: 9.31 vs 9.00 (Opus judge).** Small lift on n=1 but the lift is in the right places (Metastasis and LN — the variables that benefit most from the cleaner unified-timeline grounding).

**Wiring changes made:** `gsgpt.VERTEX_MODELS` now includes `gemini-3.5-flash`, `gemini-3.1-flash-lite`. `gsgpt.VERTEX_ANTHROPIC_MODELS` (new set) includes `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-5`, routed through `AnthropicVertex` SDK with the same Stanford GCP project. `gsgpt.VERTEX_MODEL_LOCATION` maps Anthropic + Gemini-3.x models to the `global` endpoint. `gsgpt.chat()` now defaults `thinking_budget=0` for Gemini Flash variants so callers passing small `max_tokens` don't get empty responses. A pre-existing bug in `quick_eval.py`'s `--eval-model` plumbing (module-aliasing under `__name__=='__main__'`) was patched in passing.

## Sandboxing

`~/.claude.gcp/settings.json` gets a `permissions` block:
- **Allow:** Read everywhere; Write/Edit limited to `agentic_toa/runs/**`, `agentic_toa/temp_graphs/**`, `agentic_toa/logs/**`; Bash for specific allowed commands (`python agentic_toa/tools/*.py …`, `ls`, `cat`, `tail`, `mkdir`, `cp`, `head`).
- **Deny:** Write/Edit on `graph_store/**`, `toa/**`, `temp_jsons/**`, `patient_records/**`, top-level `*.py`, `*.tex`, `docs/**`; any `rm -rf`; any `git` write.

Result: agent **cannot** modify the codebase, the original graphs, the existing pipeline outputs, the manuscript, or other docs. It can read everything and write only inside its own working area.

## Cohorts and graph copies

Three disjoint patient sets, all drawn from the 1,180-patient cohort:

| Set | Size | Purpose | Source |
|---|---:|---|---|
| **dev30** | 30 | Iterate the pipeline; tune prompts, thresholds, unifier rules | Stratified sample (seed=2026, by score-bucket × diagnosis-category), **excluding** all 30 `clinician_validation_v2` PIDs |
| **validation10** | 10 | External validation before touching the test set | Stratified sample (seed=2027), excluding dev30 AND clinician_validation_v2 |
| **test30** | 30 | Final headline number, run ONCE | `clinician_validation_v2/_pid_mapping.json` (the existing held-out set) |

**Graph copies:** for all 70 patients across dev30+validation10+test30, `cp graph_store/graphs/{pid}.graphml agentic_toa/temp_graphs/{pid}.graphml` at the start of each cohort run. Original `graph_store/` is never touched. (The current Lumia graphs already exclude TOA layer state — TOA outputs live in `temp_jsons/` — so the copies are "TOA-free" by virtue of being copies of the Lumia-only graphs.)

## File layout (all new, no overlap with existing)

```
vista_architect/agentic_toa/
├── README.md                              # human-facing overview
├── CLAUDE.md                              # orchestration playbook (auto-loaded by gcpclaude)
├── prompts/
│   ├── chunk_events_v2.txt
│   ├── timeline_unification_v2.txt
│   ├── episode_synthesis_v2.txt
│   ├── display_items_v2.txt
│   └── discrepancy_resolution.txt
├── tools/                                 # deterministic Python CLIs the agent invokes
│   ├── copy_to_temp_graphs.py
│   ├── chunk_patient.py
│   ├── extract_chunk.py
│   ├── unify_timeline.py
│   ├── derive_episodes_v2.py
│   ├── generate_displays_v2.py
│   └── graph_query.py
├── cli/
│   ├── agentic_prepare_patient.py         # CLI: builds gcpclaude prompt + runs it
│   └── agentic_prepare_cohort.py
├── runs/{pid}/                            # gitignored; per-patient working area
├── temp_graphs/{pid}.graphml              # gitignored
├── logs/                                  # gitignored
├── test_cohort/
│   ├── dev30_manifest.json
│   ├── validation10_manifest.json
│   └── test30_manifest.json               # symlink to clinician_validation_v2/_pid_mapping.json equivalent
└── evaluate/
    ├── compare_to_existing.py             # per-variable A/B vs existing pipeline outputs
    └── run_clinician_eval_agentic.py      # quick_eval --eval-from-snapshot wrapper
```

## Two invocation modes

**CLI / non-interactive (production):**
```bash
python agentic_toa/cli/agentic_prepare_patient.py --pid 135982524 --max-chunks 3
# under the hood: builds the orchestration prompt and runs
#   gcpclaude -p "<orchestration prompt>" --cwd agentic_toa/
```

**Interactive:**
```bash
cd vista_architect/agentic_toa && gcpclaude
> Prepare patient 135982524 using the dev profile.
```

The `CLAUDE.md` auto-loaded in the working dir tells the orchestrator its role, tools, and the standard pipeline order.

## Phasing (small → big, with explicit gates)

| Phase | Cohort | Goal | Gate to next phase |
|---|---|---|---|
| **0** | none | Scaffold: dirs, CLAUDE.md, permissions, gemini-3.5-flash wired into gsgpt, dev30+validation10 manifests, graph copies | `gcpclaude -p "hello"` from `agentic_toa/` sees the playbook; gemini-3.5-flash round-trip works via gsgpt.chat |
| **1** | **1 patient** (TBD — pick from dev30; small, pre-diagnostic-ish NSCLC) | Single-patient end-to-end: MAP → deterministic REDUCE → episodes → display. Confirms wiring. | Valid `patient_info.json` + `summary.json` produced; parse cleanly into `quick_eval.extract_variables_from_snapshot()` |
| **2** | **3–5 patients** from dev30 | Add the LLM unifier; compare deterministic vs LLM merges side by side. Lock the unifier choice. | LLM unifier produces ≥ deterministic-equivalent timeline quality on all 3–5 patients |
| **3** | **10 patients** from dev30 | Smoke: no crashes, no UX truncations, schema integrity holds at modest scale; parallel-chunk concurrency settings finalized | All 10 patients complete cleanly; per-patient wall time stable |
| **4** | **30 patients (full dev30)** | Tune until per-variable scores match or beat existing pipeline. **Allowed** to iterate prompts / thresholds based on what we learn from dev30. | Agentic overall mean ≥ existing-pipeline mean on dev30; no per-variable regression worse than −0.5 |
| **5** | **validation10** | Run once; no further tuning. Sanity check that improvements weren't dev30-specific. | Validation10 within ±0.3 of dev30 overall mean |
| **6** | **clinician_validation_v2 (30 pts)** | **Headline number, single run, no tuning.** | report whatever comes out — this is the manuscript figure |

After Phase 6 we know whether the agentic pipeline is publishable. If yes, expose it as `prepare_patient.py --agentic` in a separate unification task.

## Evaluation

For each phase that scores patients (Phase 1+):

1. Run agentic pipeline → `agentic_toa/runs/{pid}/patient_info.json` + `summary.json`
2. `python quick_eval.py --eval-from-snapshot --snapshot-dir agentic_toa/runs --no-graph-fallbacks <pids>` → judge scores per variable
3. `python agentic_toa/evaluate/compare_to_existing.py --pids <pids>` → per-variable side-by-side vs existing pipeline scores from `eval_v2_1180_merged_results.json`

Expected behaviour: the agentic pipeline should help most on variables where the existing pipeline currently loses points to extraction-time fragmentation — most plausibly **Date of Last CT, Genetic Testing Panel completeness, Current Medical Therapy on combination regimens, Previous Surgery (where the existing pipeline missed a major resection on Case 13)**.

## Open items I'll resolve during Phase 0

1. **Single dev patient for Phase 1.** Pick from dev30 — small record, pre-diagnostic or early-stage NSCLC. Will report the chosen PID at end of Phase 0.
2. **Thinking budget sweep.** Add a `--thinking N` flag to `extract_chunk.py` and to the orchestrator prompt; default 0 in dev. Sweep happens in Phase 4.
3. **CLI argument shape for `agentic_prepare_patient.py`.** Final design after Phase 1's first end-to-end run; will mirror the existing `prepare_patient.py --pid X --from-graph-store` ergonomics where possible.
