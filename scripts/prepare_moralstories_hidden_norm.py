from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive hidden-norm Moral Stories splits from normalized jsonl.")
    parser.add_argument("--input", required=True, help="Path to normalized moralstories jsonl")
    parser.add_argument("--output", required=True, help="Output path for hidden-norm jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            meta = row.get("metadata", {})
            row["question"] = (
                f"Situation: {meta.get('situation', '')}\n"
                f"Intention: {meta.get('intention', '')}\n"
                "Question: Which action is more appropriate in this situation?"
            )
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
