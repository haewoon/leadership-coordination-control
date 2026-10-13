from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_POOL = ROOT / "data" / "splits" / "alphanli_reference_pool_100.jsonl"
OUTPUT = ROOT / "data" / "splits" / "alphanli_scale_step_100.jsonl"
NOTES = ROOT / "data" / "splits" / "alphanli_scale_step_100_notes.md"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    rows = load_jsonl(REF_POOL)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# AlphaNLI Scale-Step 100",
        "",
        "This file is the 100-item scale-step evaluation set for the abductive regime.",
        "Construction rule:",
        "- start from the 100-item stable AlphaNLI reference pool;",
        "- apply no additional item-level exclusions at this stage;",
        "- treat this set as the broad anti-cherrypicking evaluation layer above the defended curated subset;",
        "- use this result to decide whether 100-scale preserves the same task-policy pattern as 30 and 50.",
        "",
        f"Final scale-step size: {len(rows)}",
    ]
    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
