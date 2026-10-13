from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskInstance:
    task_id: str
    task_type: str
    question: str
    gold_answer: str | None = None
    choices: list[dict[str, str]] = field(default_factory=list)
    shared_context: str | None = None
    agent_contexts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
