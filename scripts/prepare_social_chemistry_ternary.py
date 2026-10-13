from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a ternary Social Chemistry 101 subset.")
    parser.add_argument("--input", required=True, help="Path to social_chemistry_101 parquet")
    parser.add_argument("--output", required=True, help="Output jsonl path")
    parser.add_argument("--per-label", type=int, default=20)
    parser.add_argument("--min-agree", type=float, default=2.0)
    parser.add_argument("--max-agree", type=float, default=3.0)
    parser.add_argument(
        "--areas",
        nargs="*",
        default=None,
        help="Optional list of areas to keep, e.g. amitheasshole rocstories",
    )
    return parser.parse_args()


def label_from_judgment(value: float) -> str:
    if value > 0:
        return "A"
    if value < 0:
        return "C"
    return "B"


def choice_text(label: str) -> str:
    return {
        "A": "acceptable",
        "B": "context-dependent / unclear",
        "C": "unacceptable",
    }[label]


def question_for(row: pd.Series) -> str:
    return (
        f"Situation: {row['situation']}\n"
        f"Action under consideration: {row['action']}\n"
        "Question: How should this action be judged in this situation?\n"
        "Choose the best option based on ordinary social and moral reasoning."
    )


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    df = df[df["split"] == "dev"].copy()
    df = df[df["action"].notna() & df["situation"].notna()]
    df = df[df["action-agree"].notna() & df["action-moral-judgment"].notna()]
    if args.areas:
        df = df[df["area"].isin(args.areas)].copy()
    df = df[(df["action-agree"] >= args.min_agree) & (df["action-agree"] <= args.max_agree)].copy()

    df["gold_answer"] = df["action-moral-judgment"].apply(label_from_judgment)

    # Prefer low-agreement neutral items and moderate-agreement polar items.
    df["selection_score"] = df.apply(
        lambda row: (
            abs(2.5 - float(row["action-agree"]))
            + (0.0 if row["gold_answer"] == "B" else 0.25)
            + (0.0 if pd.notna(row["rot-agree"]) and float(row["rot-agree"]) <= 3.0 else 0.25)
        ),
        axis=1,
    )

    groups = {}
    for label in ["A", "B", "C"]:
        groups[label] = (
            df[df["gold_answer"] == label]
            .sort_values("selection_score", ascending=True)
            .head(args.per_label)
            .to_dict(orient="records")
        )

    interleaved: list[dict] = []
    for idx in range(args.per_label):
        for label in ["A", "B", "C"]:
            if idx < len(groups[label]):
                interleaved.append(groups[label][idx])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for idx, row in enumerate(interleaved):
            normalized = {
                "task_id": f"socialchem3_{idx}",
                "question": question_for(row),
                "gold_answer": row["gold_answer"],
                "choices": [
                    {"label": "A", "text": choice_text("A")},
                    {"label": "B", "text": choice_text("B")},
                    {"label": "C", "text": choice_text("C")},
                ],
                "metadata": {
                    "source": "social_chemistry_101",
                    "rot": row["rot"],
                    "area": row["area"],
                    "rot_categorization": row["rot-categorization"],
                    "rot_moral_foundations": row["rot-moral-foundations"],
                    "action_agree": float(row["action-agree"]),
                    "action_moral_judgment": float(row["action-moral-judgment"]),
                    "rot_agree": float(row["rot-agree"]) if pd.notna(row["rot-agree"]) else None,
                },
            }
            fh.write(json.dumps(normalized, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
