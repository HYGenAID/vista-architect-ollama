"""Google Vertex AI backend — Gemini + Anthropic models served from a GCP project.

This is the configuration the VISTA Architect paper used. Reads
GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION; requires ADC credentials
(`gcloud auth application-default login`).

Routes Gemini-family models to Vertex's `google.genai` client and
Claude-family models to Vertex's `AnthropicVertex` client.
"""
from __future__ import annotations
import os
from typing import List, Dict, Optional

from ..base import LLMBackend


def _is_anthropic_model(name: str) -> bool:
    return name.startswith(("claude-",))


def _is_gemini_model(name: str) -> bool:
    return name.startswith(("gemini-",))


class VertexBackend(LLMBackend):
    name = "vertex"
    default_model = "gemini-2.5-flash"

    def __init__(self):
        self._project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not self._project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is not set (required by VISTA_LLM_BACKEND=vertex). "
                "Run `gcloud auth application-default login` and export the project."
            )
        self._gemini = None
        self._anthropic = None

    # ---- lazy clients ----
    def _get_gemini(self):
        if self._gemini is not None:
            return self._gemini
        try:
            from google import genai
        except ImportError as e:
            raise ImportError("Vertex Gemini requires `google-genai`. Install with: pip install google-genai") from e
        self._gemini = genai.Client(vertexai=True, project=self._project, location=self._location)
        return self._gemini

    def _get_anthropic(self):
        if self._anthropic is not None:
            return self._anthropic
        try:
            from anthropic import AnthropicVertex
        except ImportError as e:
            raise ImportError("Vertex Anthropic requires `anthropic[vertex]`. Install with: pip install 'anthropic[vertex]'") from e
        # Anthropic models on Vertex are typically served from a global region
        region = os.environ.get("VERTEX_ANTHROPIC_REGION", "global")
        self._anthropic = AnthropicVertex(project_id=self._project, region=region)
        return self._anthropic

    # ---- Gemini path ----
    @staticmethod
    def _to_gemini_contents(messages):
        sys_parts, contents = [], []
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

    def _chat_gemini(self, model, prompt=None, messages=None, system=None,
                     max_tokens=2048, temperature=1.0):
        client = self._get_gemini()
        if messages is None:
            contents = prompt
        else:
            system, contents = self._to_gemini_contents(messages)
        resp = client.models.generate_content(
            model=model, contents=contents,
            config=self._gen_config(max_tokens, temperature, system),
        )
        return (resp.text or "")

    # ---- Anthropic path ----
    @staticmethod
    def _split_system(messages):
        sys_parts, conv = [], []
        for m in messages:
            if m["role"] == "system":
                sys_parts.append(m["content"])
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        return ("\n\n".join(sys_parts) or None), conv

    def _chat_anthropic(self, model, prompt=None, messages=None, system=None,
                        max_tokens=2048, temperature=1.0):
        client = self._get_anthropic()
        if messages is None:
            msgs = [{"role": "user", "content": prompt}]
        else:
            system, msgs = self._split_system(messages)
        kw = dict(model=model, max_tokens=max_tokens, temperature=temperature, messages=msgs)
        if system:
            kw["system"] = system
        resp = client.messages.create(**kw)
        return resp.content[0].text if resp.content else ""

    # ---- public ----
    def chat(self, prompt, *, model=None, system=None, max_tokens=2048, temperature=1.0, **_):
        model = model or self.default_model
        if _is_anthropic_model(model):
            return self._chat_anthropic(model, prompt=prompt, system=system,
                                        max_tokens=max_tokens, temperature=temperature)
        if _is_gemini_model(model):
            return self._chat_gemini(model, prompt=prompt, system=system,
                                     max_tokens=max_tokens, temperature=temperature)
        raise ValueError(f"Vertex backend does not recognize model {model!r}; "
                         "use a gemini-* or claude-* identifier.")

    def chat_with_history(self, messages, *, model=None, max_tokens=2048, temperature=1.0, **_):
        model = model or self.default_model
        if _is_anthropic_model(model):
            return self._chat_anthropic(model, messages=messages,
                                        max_tokens=max_tokens, temperature=temperature)
        if _is_gemini_model(model):
            return self._chat_gemini(model, messages=messages,
                                     max_tokens=max_tokens, temperature=temperature)
        raise ValueError(f"Vertex backend does not recognize model {model!r}; "
                         "use a gemini-* or claude-* identifier.")

    def info(self):
        return {"backend": self.name, "default_model": self.default_model,
                "project": self._project, "location": self._location}
