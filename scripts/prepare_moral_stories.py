from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Moral Stories into closed-ended jsonl splits.")
    parser.add_argument("--input", required=True, help="Path to moral_stories_full.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for normalized train/validation/test jsonl files")
    return parser.parse_args()


def stable_bucket(task_id: str) -> str:
    digest = hashlib.md5(task_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    if bucket == 0:
        return "validation"
    if bucket == 1:
        return "test"
    return "train"


def stable_choice_order(task_id: str, moral_action: str, immoral_action: str) -> tuple[list[dict[str, str]], str]:
    digest = hashlib.md5((task_id + ":choice").encode("utf-8")).hexdigest()
    moral_first = int(digest[:8], 16) % 2 == 0
    if moral_first:
        return (
            [
                {"label": "A", "text": moral_action},
                {"label": "B", "text": immoral_action},
            ],
            "A",
        )
    return (
        [
            {"label": "A", "text": immoral_action},
            {"label": "B", "text": moral_action},
        ],
        "B",
    )


def normalize_row(row: dict) -> tuple[str, dict]:
    task_id = row["ID"]
    choices, gold_answer = stable_choice_order(
        task_id=task_id,
        moral_action=row["moral_action"],
        immoral_action=row["immoral_action"],
    )
    question = (
        f"Norm: {row['norm']}\n"
        f"Situation: {row['situation']}\n"
        f"Intention: {row['intention']}\n"
        "Question: Which action is more appropriate given the norm, situation, and intention?"
    )
    normalized = {
        "task_id": task_id,
        "question": question,
        "gold_answer": gold_answer,
        "choices": choices,
        "metadata": {
            "source": "moralstories",
            "norm": row["norm"],
            "situation": row["situation"],
            "intention": row["intention"],
            "moral_action": row["moral_action"],
            "immoral_action": row["immoral_action"],
            "moral_consequence": row["moral_consequence"],
            "immoral_consequence": row["immoral_consequence"],
        },
    }
    return stable_bucket(task_id), normalized


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buckets = {"train": [], "validation": [], "test": []}
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            split, normalized = normalize_row(row)
            buckets[split].append(normalized)

    for split, rows in buckets.items():
        out_path = output_dir / f"moralstories_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(split, len(rows), out_path)


if __name__ == "__main__":
    main()
