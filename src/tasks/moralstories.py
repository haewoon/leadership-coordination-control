from __future__ import annotations

from pathlib import Path

from src.tasks.base_task import TaskInstance
from src.utils.io import load_jsonl


def load_moralstories(path: Path, limit: int | None = None) -> list[TaskInstance]:
    rows = load_jsonl(path, limit)
    return [
        TaskInstance(
            task_id=row["task_id"],
            task_type="closed_ended",
            question=row["question"],
            gold_answer=row.get("gold_answer"),
            choices=row.get("choices", []),
            metadata=row.get("metadata", {"source": "moralstories", "dataset": "moralstories"}),
        )
        for row in rows
    ]
