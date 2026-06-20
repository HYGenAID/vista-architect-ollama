"""vista_llm — pluggable LLM client for VISTA Architect.

A single env var picks the backend:

    VISTA_LLM_BACKEND ∈ {openai, anthropic, gemini, vertex}

Each backend reads its own credentials from per-vendor env vars:
    OPENAI_API_KEY        — for `openai`
    ANTHROPIC_API_KEY     — for `anthropic`
    GOOGLE_API_KEY        — for `gemini` (public Google AI Studio API)
    GOOGLE_CLOUD_PROJECT  — for `vertex` (plus `GOOGLE_CLOUD_LOCATION`,
                            requires `gcloud auth application-default login`)

Usage:

    import vista_llm
    response = vista_llm.chat("Summarize NSCLC staging", model="auto")

The `model` argument can be:
    - "auto"        — pick a sensible default for the active backend
    - a full model identifier (e.g. "gpt-5", "claude-opus-4-5",
      "gemini-2.5-flash", "gemini-3.5-flash") — routed to the active backend
      iff the model is supported there.
"""
from .routing import chat, chat_with_history, get_active_backend, info

__all__ = ["chat", "chat_with_history", "get_active_backend", "info"]
