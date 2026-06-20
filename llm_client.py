"""Centralized LLM client for VISTA Architect.

This preprint version is a thin wrapper around the `vista_llm` package's
backend dispatch. Existing call sites importing `chat`, `chat_with_history`,
and `get_llm_mode_info` continue to work; the active backend is selected
via the `VISTA_LLM_BACKEND` env var (see SETUP.md).
"""
from __future__ import annotations
from typing import List, Dict, Optional

import vista_llm


def chat(prompt: str, *, model: Optional[str] = None, system: Optional[str] = None,
         max_tokens: int = 2048, temperature: float = 1.0, **kwargs) -> str:
    return vista_llm.chat(prompt, model=model, system=system,
                          max_tokens=max_tokens, temperature=temperature, **kwargs)


def chat_with_history(messages: List[Dict[str, str]], *,
                      model: Optional[str] = None, max_tokens: int = 2048,
                      temperature: float = 1.0, **kwargs) -> str:
    return vista_llm.chat_with_history(messages, model=model,
                                       max_tokens=max_tokens,
                                       temperature=temperature, **kwargs)


def get_llm_mode_info() -> Dict:
    """Diagnostic backend info (backend name, default model, no credentials).

    Returns the same shape as `vista_llm.info()`. Suitable for surfacing in
    a UI or status endpoint without leaking secrets.
    """
    return vista_llm.info()


