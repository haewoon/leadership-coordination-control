from __future__ import annotations

from src.models.base_model import BaseModel


class LocalModel(BaseModel):
    def generate(self, prompt: str, metadata: dict[str, object] | None = None):
        raise NotImplementedError("LocalModel is a placeholder in this skeleton.")
