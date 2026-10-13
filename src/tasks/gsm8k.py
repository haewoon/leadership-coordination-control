from __future__ import annotations

from pathlib import Path

from src.tasks.base_task import TaskInstance
from src.utils.io import load_jsonl


def load_gsm8k(path: Path, limit: int | None = None) -> list[TaskInstance]:
    """Load GSM8K math word problems.

    Expected JSONL row format:
        {
            "task_id": str,
            "question": str,
            "gold_answer": str (numeric string, e.g., "42"),
            "metadata": {"source": "gsm8k", ...}
        }
    """
    rows = load_jsonl(path, limit)
    return [
        TaskInstance(
            task_id=row["task_id"],
            task_type="open_ended_numeric",
            question=row["question"],
            gold_answer=row.get("gold_answer"),
            metadata=row.get("metadata", {"source": "gsm8k"}),
        )
        for row in rows
    ]
