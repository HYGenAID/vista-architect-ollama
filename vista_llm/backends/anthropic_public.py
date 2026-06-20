"""Public Anthropic backend (api.anthropic.com). Reads ANTHROPIC_API_KEY."""
from __future__ import annotations
import os
from typing import List, Dict, Optional

from ..base import LLMBackend


class AnthropicPublicBackend(LLMBackend):
    name = "anthropic"
    default_model = "claude-sonnet-4-5"

    def __init__(self):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic backend requires `anthropic>=0.40`. Install with: pip install anthropic"
            ) from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (required by VISTA_LLM_BACKEND=anthropic).")
        self._client = Anthropic(api_key=api_key)

    @staticmethod
    def _split_system(messages):
        """Anthropic's API takes `system` as a top-level arg, not a message role."""
        system_parts, conv = [], []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        return ("\n\n".join(system_parts) or None), conv

    def chat(self, prompt, *, model=None, system=None, max_tokens=2048, temperature=1.0, **_):
        model = model or self.default_model
        kw = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                  messages=[{"role": "user", "content": prompt}])
        if system:
            kw["system"] = system
        resp = self._client.messages.create(**kw)
        return resp.content[0].text if resp.content else ""

    def chat_with_history(self, messages, *, model=None, max_tokens=2048, temperature=1.0, **_):
        model = model or self.default_model
        system, conv = self._split_system(messages)
        kw = dict(model=model, max_tokens=max_tokens, temperature=temperature, messages=conv)
        if system:
            kw["system"] = system
        resp = self._client.messages.create(**kw)
        return resp.content[0].text if resp.content else ""
