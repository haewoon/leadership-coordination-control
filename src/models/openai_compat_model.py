"""OpenAI-compatible Chat Completions client for a configurable endpoint.

Speaks the standard OpenAI Chat Completions API (``/chat/completions``) to any
compatible gateway whose base URL is supplied via the ``LLM_BASE_URL`` environment
variable or the ``base_url`` config field. Kept separate from ``OpenAIModel``
(which uses the newer Responses API) so self-hosted or third-party
OpenAI-compatible servers can be used without affecting that provider.

The API key is read from the environment variable named by ``api_key_env``
(default ``LLM_API_KEY``), falling back to a key file if one is configured.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.models.base_model import BaseModel, ModelOutput


class OpenAICompatModel(BaseModel):
    DEFAULT_HOST = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")

    def __init__(
        self,
        name: str,
        base_url: str | None = None,
        generation: dict[str, object] | None = None,
        api_key_env: str = "LLM_API_KEY",
        api_key_path: str | None = None,
        max_retries: int = 3,
        system_prompt: str | None = None,
    ) -> None:
        self.name = name
        self.base_url = (base_url or self.DEFAULT_HOST).rstrip("/")
        self.generation = generation or {}
        self.max_retries = max_retries
        self.system_prompt = system_prompt
        self.ssl_context = self._build_ssl_context()
        self.api_key = self._resolve_api_key(api_key_env, api_key_path)

    def _resolve_api_key(self, api_key_env: str, api_key_path: str | None) -> str:
        # Prefer the environment variable.
        env_value = os.getenv(api_key_env)
        if env_value:
            return env_value.strip()
        # Fall back to a key file if one is configured or present at the repo root.
        candidate_paths: list[Path] = []
        if api_key_path:
            candidate_paths.append(Path(api_key_path))
        repo_root = Path(__file__).resolve().parents[2]  # repository root
        candidate_paths.append(repo_root / ".llm_key")
        for path in candidate_paths:
            if path.exists():
                return path.read_text().strip()
        raise RuntimeError(
            f"Missing API key. Set {api_key_env} or place the token in .llm_key "
            f"(checked: {[str(p) for p in candidate_paths]})."
        )

    def _build_ssl_context(self) -> ssl.SSLContext:
        ca_bundle = os.getenv("SSL_CERT_FILE")
        if ca_bundle:
            return ssl.create_default_context(cafile=ca_bundle)
        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return ssl.create_default_context()

    def _build_payload(self, prompt: str) -> dict[str, object]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, object] = {
            "model": self.name,
            "messages": messages,
            "max_tokens": self.generation.get("max_tokens", 256),
        }
        temperature = self.generation.get("temperature")
        if temperature is not None:
            payload["temperature"] = temperature
        top_p = self.generation.get("top_p")
        if top_p is not None:
            payload["top_p"] = top_p
        # Provider-specific passthrough fields (e.g. OpenRouter provider
        # pinning). Merged verbatim into the request body when present.
        extra_body = self.generation.get("extra_body")
        if isinstance(extra_body, dict):
            payload.update(extra_body)
        return payload

    def generate(self, prompt: str, metadata: dict[str, object] | None = None) -> ModelOutput:
        payload = self._build_payload(prompt)
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=300, context=self.ssl_context) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                try:
                    error_body = exc.read().decode("utf-8")
                except Exception:
                    error_body = ""
                # Retry on 5xx and on transient 401/408/429 (some gateways briefly
                # flap auth/rate-limit under load).
                if exc.code in {401, 408, 429} or exc.code >= 500:
                    if attempt < self.max_retries:
                        time.sleep(2**attempt)
                        last_error = exc
                        continue
                raise RuntimeError(
                    f"API request failed with HTTP {exc.code} at {self.base_url}. "
                    f"Response body: {error_body}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    last_error = exc
                    continue
                raise RuntimeError(
                    f"Failed to reach {self.base_url}. Check network access."
                ) from exc
        else:
            raise RuntimeError(
                f"API request failed after {self.max_retries + 1} attempts."
            ) from last_error

        choices = body.get("choices") or []
        text = ""
        finish_reason = "stop"
        if choices:
            msg = choices[0].get("message", {})
            text = msg.get("content", "") or ""
            finish_reason = choices[0].get("finish_reason", "stop")
        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", max(1, len(prompt) // 4))
        completion_tokens = usage.get("completion_tokens", max(1, len(text) // 4))
        return ModelOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )
