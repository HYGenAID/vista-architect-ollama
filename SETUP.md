# SETUP

## 1. Install the package

```bash
git clone https://github.com/VISTA-Stanford/vista-architect.git
cd vista-architect
pip install -e .
```

Optional extras:

```bash
pip install -e ".[agentic]"   # Vertex AI Gemini + Anthropic
pip install -e ".[eval]"      # scipy + pandas for evaluation scoring
pip install -e ".[dev]"       # pytest + ruff
```

## 2. Pick an LLM backend

The pluggable `vista_llm/` package supports four backends, selected via the
`VISTA_LLM_BACKEND` env var. Pick whichever vendor you have an API key for.

### Backend: OpenAI (default)

```bash
export VISTA_LLM_BACKEND=openai
export OPENAI_API_KEY=sk-...
# Optional: OPENAI_BASE_URL for OpenAI-compatible gateways (vLLM, Together, etc.)
```

Supported model identifiers include `gpt-4.1`, `gpt-5`, `gpt-5-mini`,
`gpt-5-nano`, `gpt-4o`. Reasoning models (`gpt-5*`, `o1`, `o3`) use
`max_completion_tokens` and ignore `temperature`.

### Backend: Anthropic

```bash
export VISTA_LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

Supported model identifiers include `claude-sonnet-4-5`, `claude-opus-4-5`,
`claude-haiku-4-5`, and earlier Sonnet/Opus generations.

### Backend: Google Gemini (public AI Studio)

```bash
export VISTA_LLM_BACKEND=gemini
export GOOGLE_API_KEY=AIza...
```

Supported model identifiers include `gemini-2.5-flash`, `gemini-2.5-pro`,
`gemini-2.5-flash-lite`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`.

### Backend: Google Vertex AI (the configuration used in the paper)

Requires a Google Cloud project with Vertex AI enabled, plus Application
Default Credentials (run `gcloud auth application-default login` once).

```bash
pip install "anthropic[vertex]"
export VISTA_LLM_BACKEND=vertex
export GOOGLE_CLOUD_PROJECT=your-gcp-project
export GOOGLE_CLOUD_LOCATION=us-central1
gcloud auth application-default login
```

Supported models include all the Gemini identifiers above (served via
Vertex's `google.genai` client) plus Claude family identifiers
(`claude-sonnet-4-5`, `claude-opus-4-5`, etc.) served via `AnthropicVertex`.
For Anthropic-on-Vertex, the region defaults to `global`; override with
`VERTEX_ANTHROPIC_REGION=...`.

## 3. Smoke-test the backend

```bash
python vista_llm/examples/quickstart_${VISTA_LLM_BACKEND}.py
```

Expected output: a `vista_llm.info()` diagnostic line plus a one-sentence
answer to the question "what is the TNM staging system?".

## 4. External dependencies (not on PyPI)

- **`meds2text`** — Stanford's MEDS XML library, used by the MEDS Graph
  builder (`toa/xml_to_graph_hierarchical.py`). Install from source:

  ```bash
  pip install git+https://github.com/VISTA-Stanford/meds2text.git
  ```

- **`gcpclaude`** (optional — needed only for the agentic pipeline in
  `agentic_toa/`) — a thin shell launcher around the Anthropic Claude Code
  CLI that points it at a Vertex AI endpoint. Set the following before
  launching `claude`:

  ```bash
  export CLAUDE_CODE_USE_VERTEX=1
  export ANTHROPIC_VERTEX_PROJECT_ID=your-gcp-project
  export CLOUD_ML_REGION=global
  export ANTHROPIC_DEFAULT_OPUS_MODEL=claude-opus-4-5
  export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-sonnet-4-5
  ```

  Then run `claude` from `agentic_toa/`. It will read `agentic_toa/CLAUDE.md`
  as its orchestrator playbook.

## 5. Reproducibility matrix

| Mode | Backend | Models used in the paper |
|---|---|---|
| Sequential build | OpenAI or Vertex | `gpt-4.1` (chunk extraction), `gpt-5` (patient-info synthesis), `gpt-5` (LLM-as-judge) |
| Agentic build | Vertex | `claude-opus-4-5` (orchestrator + display synthesis), `gemini-3.5-flash` (per-chunk extraction + episode synthesis) |
| RAG baseline | OpenAI | `gpt-4.1` or `gpt-5` (answer generation) |
| Patient-Like-Me | Any | One LLM judge call per query; any backend works |

All decoding settings are vendor defaults (temperature 1.0, no explicit
seed). `thinking_budget=0` is explicitly set on Gemini Flash for
short-output calls to avoid extended-thinking token usage.

## 6. Running the demo

```bash
bash examples/run_demo.sh
```

The demo builds a MEDS Graph from `examples/demo1.xml` (deterministic, no
LLM call), then runs the sequential TOA pipeline against the configured
backend. Total wall time ≈ 2 minutes.
