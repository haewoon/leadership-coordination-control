from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download closed-ended datasets from Hugging Face.")
    parser.add_argument("--dataset", choices=["commonsenseqa", "strategyqa"], required=True)
    parser.add_argument("--split", default="train", help="Dataset split to download.")
    parser.add_argument("--output", required=True, help="Output path for raw dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of examples.")
    return parser.parse_args()


def download_rows(dataset: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    if dataset == "commonsenseqa":
        ds = load_dataset("tau/commonsense_qa", split=split)
    else:
        ds = load_dataset("voidful/StrategyQA", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(row) for row in ds]


def write_rows(dataset: str, rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if dataset == "commonsenseqa":
        with output.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    else:
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = download_rows(dataset=args.dataset, split=args.split, limit=args.limit)
    write_rows(args.dataset, rows, Path(args.output))


if __name__ == "__main__":
    main()
