from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BaseAgent:
    agent_id: str
    role: str
    history: list[str] = field(default_factory=list)
