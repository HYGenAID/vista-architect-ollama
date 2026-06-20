"""Public OpenAI backend (api.openai.com).

Reads OPENAI_API_KEY. Optionally OPENAI_BASE_URL for OpenAI-compatible
gateways (e.g. local vLLM, Together, OpenRouter — same wire protocol).
"""
from __future__ import annotations
import os
from typing import List, Dict, Optional

from ..base import LLMBackend


class OpenAIPublicBackend(LLMBackend):
    name = "openai"
    default_model = "gpt-4.1"

    # Reasoning-family models do not accept `temperature` and use
    # `max_completion_tokens` instead of `max_tokens`.
    _REASONING = {"gpt-5", "gpt-5-mini", "gpt-5-nano", "o1", "o1-mini", "o3", "o3-mini"}

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI backend requires `openai>=1.0`. Install with: pip install openai"
            ) from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (required by VISTA_LLM_BACKEND=openai).")
        kw = {"api_key": api_key}
        base = os.environ.get("OPENAI_BASE_URL")
        if base:
            kw["base_url"] = base
        self._client = OpenAI(**kw)

    def _build_kwargs(self, model: str, max_tokens: int, temperature: float):
        if model in self._REASONING:
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens, "temperature": temperature}

    def chat(self, prompt, *, model=None, system=None, max_tokens=2048, temperature=1.0, **_):
        model = model or self.default_model
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=model, messages=messages,
            **self._build_kwargs(model, max_tokens, temperature),
        )
        return resp.choices[0].message.content or ""

    def chat_with_history(self, messages, *, model=None, max_tokens=2048, temperature=1.0, **_):
        model = model or self.default_model
        resp = self._client.chat.completions.create(
            model=model, messages=messages,
            **self._build_kwargs(model, max_tokens, temperature),
        )
        return resp.choices[0].message.content or ""
