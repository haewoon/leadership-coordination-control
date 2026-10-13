from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an ambiguity-heavy Social Chemistry 101 subset.")
    parser.add_argument("--input", required=True, help="Path to social_chemistry_101 parquet")
    parser.add_argument("--output", required=True, help="Output jsonl path")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-agree", type=float, default=3.0)
    parser.add_argument("--min-agree", type=float, default=2.0)
    parser.add_argument("--min-polarity", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    df = df[df["split"] == "dev"].copy()
    df = df[df["action"].notna() & df["situation"].notna()]
    df = df[df["action-agree"].notna() & df["action-moral-judgment"].notna()]
    df = df[
        (df["action-agree"] <= args.max_agree)
        & (df["action-agree"] >= args.min_agree)
        & (df["action-moral-judgment"].abs() >= args.min_polarity)
    ].copy()
    df["ambiguity_score"] = (args.max_agree - df["action-agree"]).abs()
    df = df.sort_values(["action-agree", "ambiguity_score"], ascending=[True, True]).head(args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for idx, row in df.iterrows():
            acceptable = int(row["action-moral-judgment"]) > 0
            gold_answer = "A" if acceptable else "B"
            question = (
                f"Situation: {row['situation']}\n"
                f"Action under consideration: {row['action']}\n"
                "Question: In this situation, is the action socially and morally acceptable?"
            )
            normalized = {
                "task_id": f"socialchem_{idx}",
                "question": question,
                "gold_answer": gold_answer,
                "choices": [
                    {"label": "A", "text": "acceptable"},
                    {"label": "B", "text": "unacceptable"},
                ],
                "metadata": {
                    "source": "social_chemistry_101",
                    "rot": row["rot"],
                    "action_agree": float(row["action-agree"]),
                    "action_moral_judgment": float(row["action-moral-judgment"]),
                    "rot_agree": float(row["rot-agree"]) if pd.notna(row["rot-agree"]) else None,
                    "area": row["area"],
                },
            }
            fh.write(json.dumps(normalized, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
