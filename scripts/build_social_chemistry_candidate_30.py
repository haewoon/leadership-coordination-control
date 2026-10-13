from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_POOL = ROOT / "data" / "processed" / "social_chemistry_ternary_social_conflict_dev.jsonl"
CURATED20 = ROOT / "data" / "splits" / "social_chemistry_candidate_bc_20.jsonl"
OUTPUT = ROOT / "data" / "splits" / "social_chemistry_candidate_bc_30.jsonl"
NOTES = ROOT / "data" / "splits" / "social_chemistry_candidate_bc_30_notes.md"


EXCLUDED_IDS = {
    "socialchem3_2",
    "socialchem3_7",
    "socialchem3_10",
    "socialchem3_11",
    "socialchem3_13",
    "socialchem3_16",
    "socialchem3_19",
    "socialchem3_22",
    "socialchem3_26",
    "socialchem3_28",
    "socialchem3_31",
    "socialchem3_34",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def candidate_priority(row: dict) -> tuple[int, float, str]:
    meta = row.get("metadata", {})
    agree = float(meta.get("action_agree", 3.0))
    area = str(meta.get("area", ""))
    cats = str(meta.get("rot_categorization", ""))
    category_bonus = 1 if "|" in cats else 0
    label = row["gold_answer"]
    label_rank = {"B": 0, "C": 1}.get(label, 2)
    area_rank = {"amitheasshole": 0, "dearabby": 1, "confessions": 2}.get(area, 3)
    return (label_rank, area_rank, agree, f"{-category_bonus}:{row['task_id']}")


def main() -> None:
    curated20 = load_jsonl(CURATED20)
    ref_pool = load_jsonl(REF_POOL)

    rows = list(curated20)
    kept_ids = {row["task_id"] for row in rows}
    added = []

    current_b = sum(row["gold_answer"] == "B" for row in rows)
    current_c = sum(row["gold_answer"] == "C" for row in rows)

    target_size = 30
    candidates = [
        row
        for row in ref_pool
        if row["task_id"] not in kept_ids
        and row["task_id"] not in EXCLUDED_IDS
        and row["gold_answer"] in {"B", "C"}
    ]
    candidates.sort(key=candidate_priority)

    # Preserve B items whenever available, then fill remaining slots with C items.
    b_rows = [row for row in candidates if row["gold_answer"] == "B"]
    c_rows = [row for row in candidates if row["gold_answer"] == "C"]

    for row in b_rows:
        if len(rows) >= target_size:
            break
        rows.append(row)
        added.append(row)
        current_b += 1

    for row in c_rows:
        if len(rows) >= target_size:
            break
        rows.append(row)
        added.append(row)
        current_c += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Social Chemistry Candidate BC 30",
        "",
        "This file is the next scale-step candidate set for the social-norm ambiguity regime.",
        "Construction rule:",
        "- start from the defended BC 20 candidate set;",
        "- add the remaining defensible B/C items from the ternary social-conflict reference pool;",
        "- preserve the original exclusion list for annotation-sensitive, semantically detached, or overly broad items;",
        "- prefer B items first because they are the scarcest and most diagnostic for inclusive control;",
        "- use this set as the 30-item scale step before any move toward 40 or 50.",
        "",
        f"Final candidate size: {len(rows)}",
        f"B items: {current_b}",
        f"C items: {current_c}",
        "",
        "## Added beyond BC 20",
    ]
    for row in added:
        meta = row.get("metadata", {})
        lines.extend(
            [
                f"- `{row['task_id']}`",
                f"  - gold: `{row['gold_answer']}`; action agree: `{meta.get('action_agree')}`",
                f"  - category: `{meta.get('rot_categorization')}`",
                f"  - rot: {meta.get('rot')}",
            ]
        )

    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "b_items": current_b,
                "c_items": current_c,
                "output": str(OUTPUT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
