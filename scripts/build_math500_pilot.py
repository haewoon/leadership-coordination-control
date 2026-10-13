"""Build a hard-math pilot from MATH-500 (Hendrycks et al.) for the leadership rebuttal.

Source: https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl
(Originally derived from the MATH benchmark, OpenAI PRM800K curation.)

Source row format:
    {
        "problem": str,
        "solution": str,
        "answer": str,    # LaTeX-formatted; for our pilot we filter to simple-numeric.
        "subject": str,
        "level": int (1-5),
        "unique_id": str,
    }

We filter to Level 5 problems with simple-numeric (parseable as float) answers,
then take the first 50. This gives the hardest available problems where our
numeric exact_match evaluator is well-defined.

Output JSONL row format (data/processed/math500_test.jsonl):
    {
        "task_id": "math500_<unique_id_slug>",
        "question": problem,
        "gold_answer": normalized numeric string,
        "metadata": {"source": "math500", "level": int, "subject": str,
                     "solution": str, "answer_raw": str}
    }

Pilot split (data/splits/math500_hard_50.jsonl): first 50 Level-5 simple-numeric items.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

MATH500_URL = (
    "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
)

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"

PILOT_N = 50
TARGET_LEVEL = 5


def download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest} already exists ({dest.stat().st_size} bytes)")
        return
    print(f"Downloading MATH-500 test set to {dest}")
    urllib.request.urlretrieve(MATH500_URL, dest)
    print(f"  downloaded ({dest.stat().st_size} bytes)")


def is_simple_numeric(ans: str) -> tuple[bool, str | None]:
    """Return (is_simple, normalized_numeric_str). Strips $, \\, commas."""
    s = ans.replace(",", "").replace("$", "").replace("\\", "").strip()
    try:
        v = float(s)
        if v.is_integer():
            return True, str(int(v))
        return True, f"{v:g}"
    except ValueError:
        return False, None


def slug_from_uid(uid: str) -> str:
    return uid.replace("test/", "").replace("/", "_").replace(".json", "")


def main() -> None:
    raw_path = RAW_DIR / "math500_test.jsonl"
    download(raw_path)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DIR / "math500_test.jsonl"

    written = 0
    with raw_path.open() as src, processed_path.open("w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            is_simple, normalized = is_simple_numeric(row["answer"])
            if not is_simple:
                continue
            out = {
                "task_id": f"math500_{slug_from_uid(row['unique_id'])}",
                "question": row["problem"].strip(),
                "gold_answer": normalized,
                "metadata": {
                    "source": "math500",
                    "level": row["level"],
                    "subject": row["subject"],
                    "answer_raw": row["answer"],
                    "solution": row["solution"],
                },
            }
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} simple-numeric items to {processed_path}")

    # Pilot split: first PILOT_N Level-TARGET_LEVEL items.
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    pilot_path = SPLITS_DIR / f"math500_hard_{PILOT_N}.jsonl"
    selected = []
    with processed_path.open() as src:
        for line in src:
            row = json.loads(line)
            if row["metadata"]["level"] != TARGET_LEVEL:
                continue
            selected.append(row)
            if len(selected) >= PILOT_N:
                break

    with pilot_path.open("w") as dst:
        for row in selected:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote pilot split (n={len(selected)}, level={TARGET_LEVEL}) to {pilot_path}")


if __name__ == "__main__":
    main()
