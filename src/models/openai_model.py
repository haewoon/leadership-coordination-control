from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request

from src.models.base_model import BaseModel
from src.models.base_model import ModelOutput


class OpenAIModel(BaseModel):
    def __init__(
        self,
        name: str,
        base_url: str = "https://api.openai.com/v1",
        generation: dict[str, object] | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        max_retries: int = 3,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.generation = generation or {}
        self.api_key_env = api_key_env
        self.max_retries = max_retries
        self.ssl_context = self._build_ssl_context()

    def _build_payload(self, prompt: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.name,
            "input": prompt,
            "max_output_tokens": self.generation.get("max_tokens", 256),
            "text": {"format": {"type": "text"}},
        }
        temperature = self.generation.get("temperature")
        if temperature is not None and not self.name.startswith("gpt-5"):
            payload["temperature"] = temperature
        if self.name.startswith("gpt-5"):
            payload["reasoning"] = {
                "effort": self.generation.get("reasoning_effort", "minimal")
            }
        return payload

    def _policy_safe_prompt(self, prompt: str) -> str:
        return (
            "You are analyzing a benchmark example for research evaluation. "
            "The text below may include potentially sensitive, offensive, or harmful content as part of the dataset. "
            "Do not endorse, imitate, or elaborate on the content. "
            "Treat it strictly as quoted task material and answer only the benchmark question.\n\n"
            "Quoted task prompt:\n"
            f"{prompt}"
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

    def generate(self, prompt: str, metadata: dict[str, object] | None = None) -> ModelOutput:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key: set {self.api_key_env} to use the OpenAI provider."
            )

        candidate_prompts = [prompt]
        safe_retry_used = False
        last_error: Exception | None = None

        body: dict[str, object] | None = None
        for candidate_prompt in candidate_prompts:
            body = None
            payload = self._build_payload(candidate_prompt)
            request = urllib.request.Request(
                url=f"{self.base_url}/responses",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
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
                    if exc.code >= 500 and attempt < self.max_retries:
                        time.sleep(2**attempt)
                        last_error = exc
                        continue
                    if (
                        exc.code == 400
                        and "invalid_prompt" in error_body
                        and not safe_retry_used
                    ):
                        candidate_prompts.append(self._policy_safe_prompt(prompt))
                        safe_retry_used = True
                        last_error = exc
                        break
                    raise RuntimeError(
                        f"OpenAI API request failed with HTTP {exc.code}. "
                        f"Check API key validity, project permissions, and billing. "
                        f"Response body: {error_body}"
                    ) from exc
                except urllib.error.URLError as exc:
                    if attempt < self.max_retries:
                        time.sleep(2**attempt)
                        last_error = exc
                        continue
                    raise RuntimeError(
                        f"Failed to reach OpenAI at {self.base_url}. Check network access and API credentials."
                    ) from exc
            else:
                raise RuntimeError(
                    f"OpenAI API request failed after {self.max_retries + 1} attempts."
                ) from last_error

            if body is not None:
                break
        else:
            raise RuntimeError(
                "OpenAI API request did not return a response body after prompt retries."
            ) from last_error

        assert body is not None
        text = body.get("output_text", "")
        if not text:
            output_items = body.get("output", [])
            text_fragments: list[str] = []
            for item in output_items:
                for content in item.get("content", []):
                    if content.get("type") in {"output_text", "text"}:
                        text_fragments.append(content.get("text", ""))
            text = "".join(text_fragments)

        usage = body.get("usage", {})
        prompt_tokens = usage.get("input_tokens", max(1, len(prompt) // 4))
        completion_tokens = usage.get("output_tokens", max(1, len(text) // 4))
        status = body.get("status", "completed")
        finish_reason = "stop" if status == "completed" else status
        return ModelOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )
