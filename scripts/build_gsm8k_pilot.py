"""Download GSM8K test set and convert to our task format.

Source: OpenAI grade-school-math repo (public).
URL: https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl

Each row in source:
    {"question": "...", "answer": "...explanation...\\n#### N"}

Output row format (data/processed/gsm8k_test.jsonl):
    {
        "task_id": "gsm8k_<idx>",
        "question": "...",
        "gold_answer": "N",
        "metadata": {"source": "gsm8k", "solution": "..."}
    }

Also creates a pilot split (data/splits/gsm8k_pilot_50.jsonl) with the first 50 items.
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
ANSWER_RE = re.compile(r"####\s*([-+]?[\d,\.]+)")

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"


def download_gsm8k(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return
    print(f"Downloading GSM8K test set to {dest}")
    urllib.request.urlretrieve(GSM8K_URL, dest)
    print(f"  downloaded ({dest.stat().st_size} bytes)")


def parse_gsm8k_row(idx: int, row: dict) -> dict:
    question = row["question"].strip()
    answer_blob = row["answer"]
    m = ANSWER_RE.search(answer_blob)
    if not m:
        raise ValueError(f"row {idx}: no #### N pattern in answer: {answer_blob[-200:]!r}")
    gold_raw = m.group(1).replace(",", "").strip()
    # Normalize to integer string if it's a whole number; else keep float string.
    try:
        gold_float = float(gold_raw)
        if gold_float.is_integer():
            gold = str(int(gold_float))
        else:
            gold = str(gold_float)
    except ValueError:
        gold = gold_raw
    return {
        "task_id": f"gsm8k_{idx:04d}",
        "question": question,
        "gold_answer": gold,
        "metadata": {
            "source": "gsm8k",
            "split": "test",
            "solution": answer_blob.strip(),
        },
    }


def main() -> None:
    raw_path = RAW_DIR / "gsm8k_test.jsonl"
    download_gsm8k(raw_path)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DIR / "gsm8k_test.jsonl"
    written = 0
    with raw_path.open() as src, processed_path.open("w") as dst:
        for idx, line in enumerate(src):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out = parse_gsm8k_row(idx, row)
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} rows to {processed_path}")

    # Pilot split: first 50 items.
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    pilot_path = SPLITS_DIR / "gsm8k_pilot_50.jsonl"
    n_pilot = 50
    with processed_path.open() as src, pilot_path.open("w") as dst:
        for i, line in enumerate(src):
            if i >= n_pilot:
                break
            dst.write(line)
    print(f"Wrote pilot split (n={n_pilot}) to {pilot_path}")


if __name__ == "__main__":
    main()
