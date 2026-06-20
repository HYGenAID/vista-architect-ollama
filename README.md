# VISTA Architect

A graph-database-oriented health AI system for multidisciplinary tumor boards.

VISTA Architect converts longitudinal EHR records into a two-tier knowledge
graph — a source-faithful **MEDS Graph** of raw clinical events plus a
clinically organized **Timeline Object Architecture (TOA) Graph** of
deduplicated events and episodes — and serves any downstream application
(dashboards, agents, similarity retrieval, automated note writing) from the
pre-computed graph rather than re-retrieving raw documents per query. On a
held-out cohort of 1,180 thoracic-oncology patients at Stanford Medicine,
VISTA Architect extracted 15 tumor-board-salient variables at **96.4%
accuracy** (17,063 / 17,700 evaluations), substantially above a matched
BM25 RAG baseline (≈67%) and recent agentic-LLM benchmarks (84%).

This repository accompanies the preprint *"VISTA Architect: A graph
database-oriented health AI system demonstrated in multidisciplinary tumor
boards"*. It contains the backend (graph construction, TOA event extraction,
episode synthesis, deterministic graph-resident queries, the agentic build
pipeline, the RAG baseline, the patient-similarity module, and example
prompts). The production frontend and the institution-specific runtime
prompts are intentionally **not** part of this release.

## Quickstart (one patient, ~2 minutes)

1. Install:

   ```bash
   pip install -e .
   ```

2. Pick an LLM backend. Four are supported out of the box; pick whichever
   you have an API key for:

   ```bash
   # OpenAI
   export VISTA_LLM_BACKEND=openai
   export OPENAI_API_KEY=sk-...

   # Anthropic
   export VISTA_LLM_BACKEND=anthropic
   export ANTHROPIC_API_KEY=sk-ant-...

   # Google Gemini (public AI Studio)
   export VISTA_LLM_BACKEND=gemini
   export GOOGLE_API_KEY=AIza...

   # Google Vertex AI (what the paper used)
   export VISTA_LLM_BACKEND=vertex
   export GOOGLE_CLOUD_PROJECT=your-gcp-project
   export GOOGLE_CLOUD_LOCATION=us-central1
   gcloud auth application-default login
   ```

3. Run the one-patient demo:

   ```bash
   bash examples/run_demo.sh
   ```

   Expected output: a built MEDS Graph (deterministic, no LLM calls), a TOA
   event timeline, and a `patient_info.json` describing the synthetic demo
   patient. Total wall time ≈ 2 minutes against any of the four backends.

A 5-line round-trip against each backend lives in
`vista_llm/examples/quickstart_*.py`.

## Repository layout

| Directory | Contents |
|---|---|
| `toa/` | MEDS Graph builder, TOA event extraction, episode synthesis, deterministic graph-resident queries |
| `agentic_toa/` | Agentic build pipeline (per-stage runners, orchestrator playbook, tool surface) |
| `prompts/` | Example LLM prompts (patient-info synthesis, summary, timeline, PLM judge) — sufficient to apply the pipeline to MEDS-formatted data from any source |
| `vista_llm/` | Pluggable LLM client with four backends (OpenAI, Anthropic, Gemini, Vertex) |
| `scripts/` | End-to-end runners (`prepare_patient.py`, `build_graph_store.py`, `quick_eval.py`, `rag_baseline_eval.py`, `patient_like_me.py`, etc.) |
| `examples/` | Single-patient demo (`demo1.xml`, `run_demo.sh`) |
| `docs/` | Architecture, deterministic retrieval, agentic pipeline references |
| `gsgpt.py`, `llm_client.py` | Back-compat shims that route through `vista_llm` |

## What's NOT in this release

Per the manuscript's *Code and Data Availability* statement:

- **Production frontend.** The Dash/Plotly user interface and the React
  demo UI used in clinical-workflow walkthroughs are not part of this
  repository.
- **The exact production extraction prompts** used for the thoracic
  oncology configuration. The example prompts in `prompts/` and
  `toa/prompts/` are sufficient to apply the pipeline to MEDS-formatted
  data and reproduce the architectural results, but are not bit-identical
  to the live production iteration.
- **Patient data.** No EHR data, intermediate per-patient artifacts, or
  evaluation outputs are included. The demo runs against a synthetic
  example record at `examples/demo1.xml`.

## External dependencies

- **`meds2text`** — Stanford's MEDS XML library, used by the MEDS Graph
  builder. Public release at https://github.com/VISTA-Stanford/meds2text.
- **API keys** for whichever LLM backend you pick (see Quickstart).
- **Google Cloud SDK** if you use the Vertex AI backend
  (`gcloud auth application-default login`).

See `SETUP.md` for the full setup matrix.

## License

Apache License 2.0 — see `LICENSE`.

## Citation

If you use VISTA Architect in your work, please cite the preprint
(see `CITATION.cff`).
