from __future__ import annotations

import json
import urllib.error
import urllib.request

from src.models.base_model import BaseModel, ModelOutput


class OllamaModel(BaseModel):
    def __init__(
        self,
        name: str,
        host: str = "http://localhost:11434",
        generation: dict[str, object] | None = None,
    ) -> None:
        self.name = name
        self.host = host.rstrip("/")
        self.generation = generation or {}

    def generate(self, prompt: str, metadata: dict[str, object] | None = None) -> ModelOutput:
        payload = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": self.generation.get("think", False),
            "options": {
                "temperature": self.generation.get("temperature", 0.2),
                "num_predict": self.generation.get("max_tokens", 256),
            },
        }
        request = urllib.request.Request(
            url=f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to reach Ollama at {self.host}. Is `ollama serve` running and is model `{self.name}` pulled?"
            ) from exc

        message = body.get("message", {})
        text = message.get("content", "")
        prompt_tokens = body.get("prompt_eval_count", max(1, len(prompt) // 4))
        completion_tokens = body.get("eval_count", max(1, len(text) // 4))
        finish_reason = "stop" if body.get("done", True) else "incomplete"
        return ModelOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )
