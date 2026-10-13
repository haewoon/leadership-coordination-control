from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURATED6 = ROOT / "data" / "splits" / "final_challenge_curated_6.jsonl"
CSQA_REF = ROOT / "data" / "splits" / "commonsenseqa_reference_pool_30.jsonl"
SQA_REF = ROOT / "data" / "splits" / "strategyqa_reference_pool_30.jsonl"
OUTPUT = ROOT / "data" / "splits" / "closed_ended_candidate_curated_30.jsonl"
NOTES = ROOT / "data" / "splits" / "closed_ended_candidate_curated_30_notes.md"


EXCLUDED_IDS = {
    "4c1cb0e95b99f72d55c068ba0255c54d",  # neck vs jewelry-store mismatch
    "f375ae73d22eb099f403128708e21962",  # lower disagreement value
    "e239a7c6e9a2efc98ccb",              # niche medical/complementary medicine artifact
    "f8cc260e3863b0be4b81",              # valid but lower diagnostic value
    "661919f362d276f8196a590cbf40edf1",  # spouse-first-action item is annotation-fragile
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def main() -> None:
    curated = load_jsonl(CURATED6)
    csqa_ref = load_jsonl(CSQA_REF)
    sqa_ref = load_jsonl(SQA_REF)

    rows = list(curated)
    kept_ids = {row["task_id"] for row in rows}
    current_csqa = sum(row.get("metadata", {}).get("source") == "commonsenseqa" for row in rows)
    current_sqa = sum(row.get("metadata", {}).get("source") == "strategyqa" for row in rows)

    target_csqa = 15
    target_sqa = 15
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
        "# Closed-Ended Candidate Curated 30",
        "",
        "This file is a conference-scale candidate set for the convergence-heavy closed-ended regime.",
        "Construction rule:",
        "- keep the existing defended curated 6;",
        "- extend with high-scoring stable reference-pool items from CommonsenseQA and StrategyQA;",
        "- maintain a balanced mix of approximately 15 CommonsenseQA and 15 StrategyQA items;",
        "- exclude previously flagged benchmark-fragile or lower-diagnostic items;",
        "- require final human review before the set becomes the paper's final curated closed-ended benchmark.",
        "",
        f"Final candidate size: {len(rows)}",
        f"CommonsenseQA items: {current_csqa}",
        f"StrategyQA items: {current_sqa}",
        "",
        "## Added beyond the original curated 6",
    ]
    for row in added:
        meta = row.get("metadata", {})
        lines.extend(
            [
                f"- `{row['task_id']}`",
                f"  - source: `{meta.get('source')}`; gold: `{row['gold_answer']}`; challenge score: `{meta.get('challenge_score')}`",
                f"  - details: `{meta.get('challenge_details')}`",
            ]
        )

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
