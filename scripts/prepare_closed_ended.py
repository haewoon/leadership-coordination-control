from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize CommonsenseQA or StrategyQA into jsonl.")
    parser.add_argument("--dataset", choices=["commonsenseqa", "strategyqa", "socialiqa"], required=True)
    parser.add_argument("--input", required=True, help="Path to the raw dataset file.")
    parser.add_argument("--labels", default=None, help="Optional labels file for datasets with separate gold labels.")
    parser.add_argument("--output", required=True, help="Path to the normalized jsonl output file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging.")
    return parser.parse_args()


def normalize_commonsenseqa(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        q = row["question"]
        if isinstance(q, dict):
            question_text = q["stem"]
            raw_choices = q.get("choices", [])
            choices = [
                {"label": choice["label"], "text": choice["text"]}
                for choice in raw_choices
            ]
        else:
            question_text = q
            raw_choices = row.get("choices", {})
            labels = raw_choices.get("label", [])
            texts = raw_choices.get("text", [])
            choices = [
                {"label": label, "text": text}
                for label, text in zip(labels, texts, strict=False)
            ]
        normalized.append(
            {
                "task_id": row["id"],
                "question": question_text,
                "gold_answer": row.get("answerKey"),
                "choices": choices,
                "metadata": {"source": "commonsenseqa"}
            }
        )
    return normalized


def normalize_strategyqa(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for idx, row in enumerate(rows):
        task_id = row.get("qid") or row.get("id") or f"strategyqa_{idx}"
        answer = row.get("answer")
        if isinstance(answer, bool):
            gold_answer = "yes" if answer else "no"
        else:
            gold_answer = str(answer).lower() if answer is not None else None
        normalized.append(
            {
                "task_id": str(task_id),
                "question": row["question"],
                "gold_answer": gold_answer,
                "choices": [{"label": "A", "text": "yes"}, {"label": "B", "text": "no"}],
                "metadata": {"source": "strategyqa"},
            }
        )
    return normalized


def normalize_socialiqa(rows: list[dict[str, Any]], labels: list[str] | None = None) -> list[dict[str, Any]]:
    normalized = []
    for idx, row in enumerate(rows):
        task_id = row.get("qid") or row.get("id") or f"socialiqa_{idx}"
        label = row.get("label")
        if label is None and labels is not None and idx < len(labels):
            label = labels[idx]
        if label in {"1", "2", "3"}:
            gold_answer = chr(ord("A") + int(label) - 1)
        else:
            gold_answer = row.get("gold_answer")
        context = row.get("context", "").strip()
        question = row.get("question", "").strip()
        prompt = f"{context}\nQuestion: {question}" if context else question
        choices = [
            {"label": "A", "text": row.get("answerA", "")},
            {"label": "B", "text": row.get("answerB", "")},
            {"label": "C", "text": row.get("answerC", "")},
        ]
        normalized.append(
            {
                "task_id": str(task_id),
                "question": prompt,
                "gold_answer": gold_answer,
                "choices": choices,
                "metadata": {"source": "socialiqa"},
            }
        )
    return normalized


def load_raw_rows(dataset: str, path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if dataset == "commonsenseqa":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if dataset == "socialiqa":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected StrategyQA format in {path}")


def write_jsonl(path: Path, rows: list[dict[str, Any]], limit: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = rows[:limit] if limit is not None else rows
    with path.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    labels_path = Path(args.labels) if args.labels else None
    output_path = Path(args.output)
    raw_rows = load_raw_rows(args.dataset, input_path)
    if args.dataset == "commonsenseqa":
        normalized = normalize_commonsenseqa(raw_rows)
    elif args.dataset == "socialiqa":
        labels = None
        if labels_path is not None:
            labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        normalized = normalize_socialiqa(raw_rows, labels=labels)
    else:
        normalized = normalize_strategyqa(raw_rows)
    write_jsonl(output_path, normalized, limit=args.limit)


if __name__ == "__main__":
    main()
