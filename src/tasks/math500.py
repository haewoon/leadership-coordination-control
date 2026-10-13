from __future__ import annotations

from pathlib import Path

from src.tasks.base_task import TaskInstance
from src.utils.io import load_jsonl


def load_math500(path: Path, limit: int | None = None) -> list[TaskInstance]:
    """Load MATH-500 hard-math word problems (Level 5, simple-numeric subset).

    Expected JSONL row format:
        {
            "task_id": str,
            "question": str,
            "gold_answer": str (numeric string),
            "metadata": {"source": "math500", "level": int, "subject": str, ...}
        }
    """
    rows = load_jsonl(path, limit)
    return [
        TaskInstance(
            task_id=row["task_id"],
            task_type="open_ended_numeric",
            question=row["question"],
            gold_answer=row.get("gold_answer"),
            metadata=row.get("metadata", {"source": "math500"}),
        )
        for row in rows
    ]
