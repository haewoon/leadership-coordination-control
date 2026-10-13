from __future__ import annotations

from pathlib import Path

from src.tasks.base_task import TaskInstance
from src.utils.io import load_jsonl


def load_integration_tasks(path: Path, limit: int | None = None) -> list[TaskInstance]:
    rows = load_jsonl(path, limit)
    return [
        TaskInstance(
            task_id=row["task_id"],
            task_type="integration",
            question=row["question"],
            gold_answer=row.get("gold_answer"),
            agent_contexts=row.get("agent_contexts", {}),
            metadata={"dataset": "integration_task"},
        )
        for row in rows
    ]
