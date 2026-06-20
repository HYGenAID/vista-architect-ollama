"""Backend dispatch — picks one of the four backends based on env config."""
from __future__ import annotations
import os
from functools import lru_cache
from typing import List, Dict, Optional

from .base import LLMBackend


_BACKENDS = {
    "openai":    ("openai_public",    "OpenAIPublicBackend"),
    "anthropic": ("anthropic_public", "AnthropicPublicBackend"),
    "gemini":    ("gemini_public",    "GeminiPublicBackend"),
    "vertex":    ("vertex",           "VertexBackend"),
}


@lru_cache(maxsize=1)
def get_active_backend() -> LLMBackend:
    """Resolve the active backend from `VISTA_LLM_BACKEND` (default: openai)."""
    name = os.environ.get("VISTA_LLM_BACKEND", "openai").strip().lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown VISTA_LLM_BACKEND={name!r}; pick one of {sorted(_BACKENDS)}"
        )
    module_name, class_name = _BACKENDS[name]
    module = __import__(f"vista_llm.backends.{module_name}", fromlist=[class_name])
    return getattr(module, class_name)()


def chat(prompt: str, **kwargs) -> str:
    """Single-turn completion against the active backend."""
    return get_active_backend().chat(prompt, **kwargs)


def chat_with_history(messages: List[Dict[str, str]], **kwargs) -> str:
    """Multi-turn completion against the active backend."""
    return get_active_backend().chat_with_history(messages, **kwargs)


def info() -> Dict:
    """Diagnostic info about the active backend (no credentials)."""
    return get_active_backend().info()
