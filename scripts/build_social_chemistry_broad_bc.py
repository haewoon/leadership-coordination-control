from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_AREAS = ["confessions", "amitheasshole", "dearabby"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a scalable broad B/C Social Chemistry set.")
    parser.add_argument("--input", required=True, help="Path to social_chemistry_101 parquet")
    parser.add_argument("--output", required=True, help="Output jsonl path")
    parser.add_argument("--notes", required=True, help="Output notes path")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--min-agree", type=float, default=2.0)
    parser.add_argument("--max-agree", type=float, default=3.0)
    parser.add_argument("--areas", nargs="*", default=DEFAULT_AREAS)
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


def stable_task_id(row: pd.Series) -> str:
    short_id = str(row["situation-short-id"]).replace("/", "_")
    return f"socialchem_bc_{row['area']}_{short_id}"


def priority_score(row: pd.Series, area_priority: dict[str, int]) -> tuple[float, int, str]:
    agree = float(row["action-agree"])
    closeness = abs(2.5 - agree)
    area_rank = area_priority.get(str(row["area"]), 99)
    short_id = str(row["situation-short-id"])
    return (closeness, area_rank, short_id)


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    df = df[df["split"] == args.split].copy()
    df = df[df["area"].isin(args.areas)].copy()
    df = df[df["action"].notna() & df["situation"].notna()]
    df = df[df["action-agree"].notna() & df["action-moral-judgment"].notna()]
    df = df[(df["action-agree"] >= args.min_agree) & (df["action-agree"] <= args.max_agree)].copy()
    df["gold_answer"] = df["action-moral-judgment"].apply(label_from_judgment)
    df = df[df["gold_answer"].isin(["B", "C"])].copy()

    area_priority = {name: idx for idx, name in enumerate(args.areas)}
    df["task_id"] = df.apply(stable_task_id, axis=1)
    df["priority_tuple"] = df.apply(lambda row: priority_score(row, area_priority), axis=1)
    df = df.sort_values("priority_tuple", ascending=True)

    target_b = args.limit // 2
    target_c = args.limit - target_b
    pools: dict[str, dict[str, list[dict]]] = {label: {} for label in ["B", "C"]}
    for label in ["B", "C"]:
        for area in args.areas:
            area_rows = df[(df["gold_answer"] == label) & (df["area"] == area)].to_dict(orient="records")
            pools[label][area] = area_rows

    selected_rows: list[dict] = []
    seen_ids: set[str] = set()
    b_count = 0
    c_count = 0
    target_by_label = {"B": target_b, "C": target_c}
    counts_by_label = {"B": 0, "C": 0}
    counts_by_area: dict[str, int] = {area: 0 for area in args.areas}

    while len(selected_rows) < args.limit:
        progressed = False
        for label in ["B", "C"]:
            if counts_by_label[label] >= target_by_label[label]:
                continue
            for area in args.areas:
                while pools[label][area]:
                    row = pools[label][area].pop(0)
                    if row["task_id"] in seen_ids:
                        continue
                    normalized = {
                        "task_id": row["task_id"],
                        "question": question_for(pd.Series(row)),
                        "gold_answer": label,
                        "choices": [
                            {"label": "A", "text": choice_text("A")},
                            {"label": "B", "text": choice_text("B")},
                            {"label": "C", "text": choice_text("C")},
                        ],
                        "metadata": {
                            "source": "social_chemistry_101",
                            "area": row["area"],
                            "rot": row["rot"],
                            "rot_categorization": row["rot-categorization"],
                            "rot_moral_foundations": row["rot-moral-foundations"],
                            "action_agree": float(row["action-agree"]),
                            "action_moral_judgment": float(row["action-moral-judgment"]),
                            "rot_agree": float(row["rot-agree"]) if pd.notna(row["rot-agree"]) else None,
                            "selection_split": args.split,
                        },
                    }
                    selected_rows.append(normalized)
                    seen_ids.add(row["task_id"])
                    counts_by_label[label] += 1
                    counts_by_area[area] += 1
                    if label == "B":
                        b_count += 1
                    else:
                        c_count += 1
                    progressed = True
                    break
                if len(selected_rows) >= args.limit or counts_by_label[label] >= target_by_label[label]:
                    break
            if len(selected_rows) >= args.limit:
                break
        if not progressed:
            break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in selected_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    notes_lines = [
        f"# Social Chemistry Broad BC {len(selected_rows)}",
        "",
        "This file is the scalable broad social-norm ambiguity evaluation set.",
        "Construction rule:",
        "- keep only B/C ternary labels;",
        f"- keep areas: {', '.join(args.areas)};",
        f"- keep split: {args.split};",
        f"- keep action-agree in [{args.min_agree}, {args.max_agree}];",
        "- prioritize lower-agreement items and preserve area diversity by fixed area ordering;",
        "- balance B and C labels as closely as possible at the requested target size;",
        "- use this set as the broad anti-cherrypicking social evaluation layer for 100+ scale.",
        "",
        f"Final size: {len(selected_rows)}",
        f"B items: {b_count}",
        f"C items: {c_count}",
        "Area counts:",
    ]
    for area in args.areas:
        notes_lines.append(f"- {area}: {counts_by_area.get(area, 0)}")
    Path(args.notes).write_text("\n".join(notes_lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(selected_rows), "b_items": b_count, "c_items": c_count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
