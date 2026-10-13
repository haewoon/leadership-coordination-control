from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLOSED_ENDED = ROOT / "data" / "splits" / "closed_ended_candidate_curated_100.jsonl"
ALPHANLI = ROOT / "data" / "splits" / "alphanli_scale_step_100.jsonl"
SOCIAL = ROOT / "data" / "splits" / "social_chemistry_broad_bc_100_v2.jsonl"

OUTPUT = ROOT / "data" / "splits" / "mixed_benchmark_150.jsonl"
NOTES = ROOT / "data" / "splits" / "mixed_benchmark_150_notes.md"

SEED = 25


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def sample_rows(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    copied = list(rows)
    rng.shuffle(copied)
    return copied[:n]


def main() -> None:
    rng = random.Random(SEED)

    closed_rows = load_rows(CLOSED_ENDED)
    alpha_rows = load_rows(ALPHANLI)
    social_rows = load_rows(SOCIAL)

    csqa_rows = [row for row in closed_rows if row.get("metadata", {}).get("source") == "commonsenseqa"]
    strategy_rows = [row for row in closed_rows if row.get("metadata", {}).get("source") == "strategyqa"]

    mixed_rows = []
    mixed_rows.extend(sample_rows(csqa_rows, 25, rng))
    mixed_rows.extend(sample_rows(strategy_rows, 25, rng))
    mixed_rows.extend(sample_rows(alpha_rows, 50, rng))
    mixed_rows.extend(sample_rows(social_rows, 50, rng))

    rng.shuffle(mixed_rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as fh:
        for row in mixed_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    for row in mixed_rows:
        source = row.get("metadata", {}).get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1

    notes = [
        "# Mixed Benchmark 150",
        "",
        "Construction:",
        "- 25 CommonsenseQA items sampled from the 100-scale closed-ended candidate set",
        "- 25 StrategyQA items sampled from the same closed-ended candidate set",
        "- 50 AlphaNLI items sampled from the 100-scale AlphaNLI set",
        "- 50 Social Chemistry items sampled from the broad BC 100 v2 set",
        "",
        f"Sampling seed: `{SEED}`",
        "",
        "Source counts:",
    ]
    for source, count in sorted(counts.items()):
        notes.append(f"- `{source}`: {count}")

    NOTES.write_text("\n".join(notes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
