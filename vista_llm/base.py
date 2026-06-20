"""Backend interface — every concrete backend implements this."""
from __future__ import annotations
from typing import List, Dict, Optional


class LLMBackend:
    """Minimal LLM backend contract.

    Subclasses must implement `chat()` and `chat_with_history()` and set
    `default_model` to a sensible per-backend default.
    """

    name: str = "abstract"
    default_model: str = ""

    def chat(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        **kwargs,
    ) -> str:
        """Single-turn completion. Returns the assistant's text response."""
        raise NotImplementedError

    def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        **kwargs,
    ) -> str:
        """Multi-turn completion. `messages` is a list of {role, content} dicts
        where role ∈ {system, user, assistant}.
        """
        raise NotImplementedError

    def info(self) -> Dict:
        """Return diagnostic metadata about the active backend (no credentials)."""
        return {"backend": self.name, "default_model": self.default_model}
