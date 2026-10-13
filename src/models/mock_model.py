from __future__ import annotations

from src.models.base_model import BaseModel, ModelOutput


class MockModel(BaseModel):
    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, prompt: str, metadata: dict[str, object] | None = None) -> ModelOutput:
        metadata = metadata or {}
        round_idx = metadata.get("round_idx", 0)
        agent_id = metadata.get("agent_id", "agent")
        if "Transactional" in prompt:
            text = f"ACCEPT: mock answer from {agent_id} at round {round_idx}"
        elif "Transformational" in prompt:
            text = f"Exploratory answer from {agent_id} at round {round_idx}"
        elif "Leader" in prompt:
            text = f"Leader instruction from {agent_id} at round {round_idx}"
        else:
            text = f"Mock answer from {agent_id} at round {round_idx}"
        return ModelOutput(
            text=text,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(text) // 4),
        )
