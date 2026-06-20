"""gsgpt — thin back-compat shim that re-routes through the pluggable
`vista_llm` package.

Existing code (`toa/backend.py`, `quick_eval.py`, `prepare_patient.py`,
`rag_baseline_eval.py`, `patient_like_me.py`, etc.) calls into `gsgpt`
directly. This shim preserves those call sites while sending all traffic
through `vista_llm`, so the active backend is controlled by the
`VISTA_LLM_BACKEND` env var.

The original gsgpt.py in the dev environment wired institution-specific
backends; this preprint version intentionally omits that wiring. See
SETUP.md for backend configuration.
"""
from __future__ import annotations
from typing import List, Dict, Optional

import vista_llm


def chat(prompt: str, *, model: Optional[str] = None, system: Optional[str] = None,
         max_tokens: int = 2048, temperature: float = 1.0, **kwargs) -> str:
    """Single-turn completion. See vista_llm.chat for full signature."""
    return vista_llm.chat(prompt, model=model, system=system,
                          max_tokens=max_tokens, temperature=temperature, **kwargs)


def chat_with_history(messages: List[Dict[str, str]], *,
                      model: Optional[str] = None, max_tokens: int = 2048,
                      temperature: float = 1.0, **kwargs) -> str:
    """Multi-turn completion. See vista_llm.chat_with_history."""
    return vista_llm.chat_with_history(messages, model=model,
                                       max_tokens=max_tokens,
                                       temperature=temperature, **kwargs)


def info() -> Dict:
    """Return diagnostic info about the active vista_llm backend."""
    return vista_llm.info()


class Conversation:
    """Lightweight multi-turn helper, matching the original API surface."""

    def __init__(self, model: Optional[str] = None,
                 system: Optional[str] = None):
        self.model = model
        self._messages: List[Dict[str, str]] = []
        if system:
            self._messages.append({"role": "system", "content": system})

    def chat(self, prompt: str, **kwargs) -> str:
        self._messages.append({"role": "user", "content": prompt})
        reply = vista_llm.chat_with_history(self._messages, model=self.model, **kwargs)
        self._messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        """Reset history but keep the system prompt if there was one."""
        if self._messages and self._messages[0]["role"] == "system":
            self._messages = [self._messages[0]]
        else:
            self._messages = []
