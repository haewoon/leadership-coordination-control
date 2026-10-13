from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


class BaseModel:
    def generate(self, prompt: str, metadata: dict[str, object] | None = None) -> ModelOutput:
        raise NotImplementedError
