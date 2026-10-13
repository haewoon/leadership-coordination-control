from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize AlphaNLI into jsonl.")
    parser.add_argument("--input-jsonl", required=True, help="Path to raw AlphaNLI jsonl file.")
    parser.add_argument("--input-labels", required=True, help="Path to raw AlphaNLI label file.")
    parser.add_argument("--output", required=True, help="Path to normalized jsonl output.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_rows(jsonl_path: Path, labels_path: Path) -> list[dict]:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(labels):
        raise ValueError(f"Row/label mismatch: {len(rows)} rows vs {len(labels)} labels")

    normalized = []
    for idx, (row, label) in enumerate(zip(rows, labels, strict=True)):
        gold_answer = {"1": "A", "2": "B"}.get(label)
        if gold_answer is None:
            continue
        normalized.append(
            {
                "task_id": row.get("story_id", f"alphanli_{idx}"),
                "question": (
                    "Given the two observations, choose the more plausible hypothesis that explains the situation.\n"
                    f"Observation 1: {row['obs1']}\n"
                    f"Observation 2: {row['obs2']}"
                ),
                "gold_answer": gold_answer,
                "choices": [
                    {"label": "A", "text": row["hyp1"]},
                    {"label": "B", "text": row["hyp2"]},
                ],
                "metadata": {
                    "source": "alphanli",
                    "obs1": row["obs1"],
                    "obs2": row["obs2"],
                },
            }
        )
    return normalized


def write_jsonl(path: Path, rows: list[dict], limit: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = rows[:limit] if limit is not None else rows
    with path.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input_jsonl), Path(args.input_labels))
    write_jsonl(Path(args.output), rows, limit=args.limit)


if __name__ == "__main__":
    main()
