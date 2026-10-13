from __future__ import annotations

from dataclasses import dataclass

from src.agents.llm_agent import LLMAgent


@dataclass
class LeaderAgent(LLMAgent):
    authority_level: str = "leader"
