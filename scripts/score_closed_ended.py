from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.accuracy import exact_match, extract_final_answer, normalize_prediction
from src.utils.io import load_config, load_jsonl


def load_gold_by_task_id(config: dict, root: Path) -> dict[str, dict[str, str]]:
    gold = {}
    for task_cfg in config["tasks"]:
        for row in load_jsonl(root / task_cfg["path"]):
            gold[row["task_id"]] = {
                "gold_answer": row.get("gold_answer"),
                "question": row["question"],
                "source": row.get("metadata", {}).get("source", task_cfg["name"]),
            }
    return gold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    output_dir = ROOT / config["output_dir"]
    summaries = load_jsonl(output_dir / "run_summaries.jsonl")
    gold_by_task_id = load_gold_by_task_id(config, ROOT)

    aggregates: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for row in summaries:
        gold = gold_by_task_id[row["task_id"]]["gold_answer"]
        extracted = extract_final_answer(row["final_answer"])
        normalized = normalize_prediction(row["final_answer"], gold)
        score = exact_match(row["final_answer"], gold)
        condition = row["condition"]
        aggregates[condition]["count"] += 1
        aggregates[condition]["correct"] += score
        aggregates[condition]["rounds"] += row["rounds_used"]
        print(
            json.dumps(
                {
                    "task_id": row["task_id"],
                    "condition": condition,
                    "gold": gold,
                    "extracted": extracted,
                    "normalized": normalized,
                    "correct": score,
                },
                ensure_ascii=False,
            )
        )

    print("\n=== aggregate ===")
    for condition, stats in sorted(aggregates.items()):
        count = int(stats["count"])
        accuracy = stats["correct"] / count if count else 0.0
        avg_rounds = stats["rounds"] / count if count else 0.0
        print(
            json.dumps(
                {
                    "condition": condition,
                    "n": count,
                    "accuracy": round(accuracy, 4),
                    "avg_rounds": round(avg_rounds, 2),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
