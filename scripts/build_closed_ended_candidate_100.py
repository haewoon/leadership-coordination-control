from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURATED6 = ROOT / "data" / "splits" / "final_challenge_curated_6.jsonl"
CSQA_REF = ROOT / "data" / "splits" / "commonsenseqa_reference_pool_100.jsonl"
SQA_REF = ROOT / "data" / "splits" / "strategyqa_reference_pool_100.jsonl"
OUTPUT = ROOT / "data" / "splits" / "closed_ended_candidate_curated_100.jsonl"
NOTES = ROOT / "data" / "splits" / "closed_ended_candidate_curated_100_notes.md"


EXCLUDED_IDS = {
    "4c1cb0e95b99f72d55c068ba0255c54d",
    "f375ae73d22eb099f403128708e21962",
    "e239a7c6e9a2efc98ccb",
    "f8cc260e3863b0be4b81",
    "661919f362d276f8196a590cbf40edf1",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    curated = load_jsonl(CURATED6)
    csqa_ref = load_jsonl(CSQA_REF)
    sqa_ref = load_jsonl(SQA_REF)

    rows = list(curated)
    kept_ids = {row["task_id"] for row in rows}
    current_csqa = sum(row.get("metadata", {}).get("source") == "commonsenseqa" for row in rows)
    current_sqa = sum(row.get("metadata", {}).get("source") == "strategyqa" for row in rows)

    target_csqa = 50
    target_sqa = 50
    added = []

    for row in csqa_ref:
        task_id = row["task_id"]
        if task_id in kept_ids or task_id in EXCLUDED_IDS:
            continue
        if current_csqa >= target_csqa:
            break
        rows.append(row)
        added.append(row)
        kept_ids.add(task_id)
        current_csqa += 1

    for row in sqa_ref:
        task_id = row["task_id"]
        if task_id in kept_ids or task_id in EXCLUDED_IDS:
            continue
        if current_sqa >= target_sqa:
            break
        rows.append(row)
        added.append(row)
        kept_ids.add(task_id)
        current_sqa += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Closed-Ended Candidate Curated 100",
        "",
        "This file is the 100-item scale-step candidate set for the shared-information closed-ended regime.",
        "Construction rule:",
        "- keep the defended curated 6 anchor items;",
        "- extend with stable reference-pool items from CommonsenseQA and StrategyQA;",
        "- maintain a balanced mix of 50 CommonsenseQA and 50 StrategyQA items;",
        "- preserve the same small exclusion list used in earlier scale steps for benchmark-fragile or low-diagnostic items;",
        "- use this set to test whether the closed-ended regime story stabilizes at conference-scale sample size.",
        "",
        f"Final candidate size: {len(rows)}",
        f"CommonsenseQA items: {current_csqa}",
        f"StrategyQA items: {current_sqa}",
    ]
    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_rows": len(rows),
                "commonsenseqa": current_csqa,
                "strategyqa": current_sqa,
                "output": str(OUTPUT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
