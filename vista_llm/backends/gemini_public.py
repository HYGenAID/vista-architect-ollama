"""Public Google Gemini (AI Studio) backend. Reads GOOGLE_API_KEY."""
from __future__ import annotations
import os
from typing import List, Dict, Optional

from ..base import LLMBackend


class GeminiPublicBackend(LLMBackend):
    name = "gemini"
    default_model = "gemini-2.5-flash"

    def __init__(self):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "Gemini backend requires `google-genai>=1.0`. Install with: pip install google-genai"
            ) from e
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set (required by VISTA_LLM_BACKEND=gemini).")
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _to_gemini_contents(messages):
        """Convert OpenAI-style messages to Gemini's contents/system_instruction split."""
        sys_parts = []
        contents = []
        for m in messages:
            if m["role"] == "system":
                sys_parts.append(m["content"])
            else:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return ("\n\n".join(sys_parts) or None), contents

    def _gen_config(self, max_tokens, temperature, system):
        from google.genai import types
        kw = dict(max_output_tokens=max_tokens, temperature=temperature)
        if system:
            kw["system_instruction"] = system
        return types.GenerateContentConfig(**kw)

    def chat(self, prompt, *, model=None, system=None, max_tokens=2048, temperature=1.0, **_):
        model = model or self.default_model
        resp = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=self._gen_config(max_tokens, temperature, system),
        )
        return (resp.text or "")

    def chat_with_history(self, messages, *, model=None, max_tokens=2048, temperature=1.0, **_):
        model = model or self.default_model
        system, contents = self._to_gemini_contents(messages)
        resp = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=self._gen_config(max_tokens, temperature, system),
        )
        return (resp.text or "")
