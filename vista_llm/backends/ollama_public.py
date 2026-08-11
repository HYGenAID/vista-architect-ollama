from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import LLMBackend

# ---------------------------------------------------------------------------
# Ollama native API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "gemma4:31b-it-q8_0"
    api_key: Optional[str] = None
    timeout: float = 300.0

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        return cls(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.environ.get("OLLAMA_MODEL", "gemma4:31b-it-q8_0"),
            api_key=os.environ.get("OLLAMA_API_KEY"),
            timeout=float(os.environ.get("OLLAMA_TIMEOUT", "300")),
        )


class OllamaBackend(LLMBackend):
    name = "ollama"
    default_model = "gemma4:31b-it-q8_0"

    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        try:
            import requests
        except ImportError as e:
            raise ImportError(
                "Ollama backend requires `requests`. Install with: pip install requests"
            ) from e

        if config is not None:
            if any(value is not None for value in (base_url, model, api_key, timeout)):
                raise ValueError(
                    "Pass either `config=` or individual Ollama parameters, not both."
                )
            self.config = config
        else:
            env_config = OllamaConfig.from_env()
            self.config = OllamaConfig(
                base_url=base_url or env_config.base_url,
                model=model or env_config.model,
                api_key=api_key if api_key is not None else env_config.api_key,
                timeout=timeout if timeout is not None else env_config.timeout,
            )

        self._requests = requests
        self.default_model = self.config.model
        self._base_url = self.config.base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "OllamaBackend":
        return cls(config=OllamaConfig.from_env())

    def _url(self, path: str) -> str:
        """
        Supports either:
          base_url="http://localhost:11434"
        or:
          base_url="http://localhost:11434/api"
        """
        path = path.lstrip("/")

        if self._base_url.endswith("/api"):
            return f"{self._base_url}/{path}"

        return f"{self._base_url}/api/{path}"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}

        # Local Ollama does not require authentication. API keys are useful
        # for ollama.com or authenticated proxies.
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        return headers

    def _build_options(
        self,
        *,
        max_tokens: int,
        temperature: float,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        merged = dict(options or {})

        # Ollama equivalent of OpenAI-style max_tokens.
        merged.setdefault("num_predict", max_tokens)
        merged.setdefault("temperature", temperature)

        return merged

    def _chat_request(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str],
        max_tokens: int,
        temperature: float,
        options: Optional[Dict[str, Any]] = None,
        format: Optional[Any] = None,
        keep_alive: Optional[str] = None,
        think: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> str:
        model = model or self.default_model

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": self._build_options(
                max_tokens=max_tokens,
                temperature=temperature,
                options=options,
            ),
        }

        if format is not None:
            payload["format"] = format

        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        if think is not None:
            payload["think"] = think

        resp = self._requests.post(
            self._url("/chat"),
            headers=self._headers(),
            json=payload,
            timeout=timeout or self.config.timeout,
        )

        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Ollama API request failed: HTTP {resp.status_code}: {resp.text}"
            ) from e

        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"Ollama API error: {data['error']}")

        message = data.get("message") or {}
        return message.get("content") or ""

    def chat(
        self,
        prompt,
        *,
        model=None,
        system=None,
        max_tokens=2048,
        temperature=0.8,
        options=None,
        format=None,
        keep_alive=None,
        think=None,
        timeout=None,
        **_,
    ):
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        return self._chat_request(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            options=options,
            format=format,
            keep_alive=keep_alive,
            think=think,
            timeout=timeout,
        )

    def chat_with_history(
        self,
        messages,
        *,
        model=None,
        max_tokens=2048,
        temperature=0.8,
        options=None,
        format=None,
        keep_alive=None,
        think=None,
        timeout=None,
        **_,
    ):
        return self._chat_request(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            options=options,
            format=format,
            keep_alive=keep_alive,
            think=think,
            timeout=timeout,
        )