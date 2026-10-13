from __future__ import annotations

import json
from pathlib import Path

from src.logging.trace_logger import TraceLogger


class RunLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_logger = TraceLogger(self.output_dir / "trace.jsonl")
        self.summary_path = self.output_dir / "run_summaries.jsonl"

    def log_event(self, record: object) -> None:
        self.trace_logger.log(record)

    def log_summary(self, summary: dict[str, object]) -> None:
        with self.summary_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary) + "\n")
