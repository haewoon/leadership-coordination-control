from __future__ import annotations

from dataclasses import dataclass

from src.agents.base_agent import BaseAgent


@dataclass
class LLMAgent(BaseAgent):
    profile_name: str | None = None
    persona: str | None = None
