from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_POOL = ROOT / "data" / "splits" / "alphanli_reference_pool_50.jsonl"
OUTPUT = ROOT / "data" / "splits" / "alphanli_scale_step_50.jsonl"
NOTES = ROOT / "data" / "splits" / "alphanli_scale_step_50_notes.md"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    rows = load_jsonl(REF_POOL)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# AlphaNLI Scale-Step 50",
        "",
        "This file is the anti-cherrypicking scale-step evaluation set for the abductive regime.",
        "Construction rule:",
        "- start from the existing 50-item AlphaNLI reference pool;",
        "- apply no additional item-level exclusions at this scale step;",
        "- use this set as the broad reference evaluation while keeping the defended 30-item curated set for the main mechanistic comparison;",
        "- if the directional result survives here, then a larger 100-scale expansion is justified.",
        "",
        f"Final scale-step size: {len(rows)}",
        "",
        "This artifact is intentionally broader and noisier than the defended curated set.",
        "Its purpose is to test whether the AlphaNLI regime-level pattern survives beyond the stricter curated slice.",
    ]
    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
