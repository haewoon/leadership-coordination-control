from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_POOL = ROOT / "data" / "processed" / "social_chemistry_ternary_social_conflict_dev.jsonl"
CURATED8 = ROOT / "data" / "splits" / "social_chemistry_curated_bc_8.jsonl"
OUTPUT = ROOT / "data" / "splits" / "social_chemistry_candidate_bc_20.jsonl"
NOTES = ROOT / "data" / "splits" / "social_chemistry_candidate_bc_20_notes.md"


EXCLUDED_IDS = {
    "socialchem3_2",   # needing space from family -> annotation-sensitive negative
    "socialchem3_11",  # confronting someone -> annotation-sensitive negative
    "socialchem3_22",  # neutral label conflicts with ordinary reading
    "socialchem3_34",  # honesty-in-relationship neutral label is too broad/obvious
    "socialchem3_10",  # not working / spend your own time too broad
    "socialchem3_13",  # relaxing / taking it easy too broad
    "socialchem3_16",  # loving family too detached from concrete conflict
    "socialchem3_31",  # honoring contracts too semantically detached
    "socialchem3_7",   # abandoning family for own good too broad for stable B label
    "socialchem3_10",  # get a house to live in too broad / advice-like
    "socialchem3_13",  # being bothered by things in life too broad
    "socialchem3_19",  # feeling your emotions unapologetically too broad
    "socialchem3_28",  # always paying off student loans too detached from scenario
    "socialchem3_26",  # doing stupid things too generic
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def candidate_priority(row: dict) -> tuple[int, float, str]:
    # Prefer B/C items, lower agreement, richer categories, and AITA first.
    label = row["gold_answer"]
    meta = row.get("metadata", {})
    agree = float(meta.get("action_agree", 3.0))
    cats = str(meta.get("rot_categorization", ""))
    category_bonus = 1 if "|" in cats else 0
    area = str(meta.get("area", ""))
    label_rank = {"B": 0, "C": 1, "A": 2}.get(label, 3)
    area_rank = {"amitheasshole": 0, "dearabby": 1, "confessions": 2}.get(area, 3)
    return (label_rank, area_rank, agree, f"{-category_bonus}:{row['task_id']}")


def main() -> None:
    curated8 = load_jsonl(CURATED8)
    ref_pool = load_jsonl(REF_POOL)

    kept_ids = {row["task_id"] for row in curated8}
    rows = list(curated8)
    added = []

    current_b = sum(row["gold_answer"] == "B" for row in rows)
    current_c = sum(row["gold_answer"] == "C" for row in rows)
    target_b = 10
    target_c = 10

    candidates_by_label = {"B": [], "C": []}
    for row in ref_pool:
        if row["task_id"] in kept_ids or row["task_id"] in EXCLUDED_IDS:
            continue
        label = row["gold_answer"]
        if label in candidates_by_label:
            candidates_by_label[label].append(row)

    for label in ["B", "C"]:
        candidates_by_label[label].sort(key=candidate_priority)

    need_b = max(0, target_b - current_b)
    need_c = max(0, target_c - current_c)

    for row in candidates_by_label["B"][:need_b]:
        rows.append(row)
        added.append(row)
    for row in candidates_by_label["C"][:need_c]:
        rows.append(row)
        added.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Social Chemistry Candidate BC 20",
        "",
        "This file is a conference-scale candidate set for the social-norm ambiguity regime.",
        "Construction rule:",
        "- keep the current defended curated BC 8;",
        "- extend with additional B/C items from the ternary AITA reference pool;",
        "- exclude items already flagged as annotation-sensitive, semantically detached, or too broad;",
        "- require final human review before the set becomes the paper's final curated social-norm benchmark.",
        "",
        f"Final candidate size: {len(rows)}",
        "",
        "## Added beyond the original curated BC 8",
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
    print(json.dumps({"candidate_rows": len(rows), "added_rows": len(added), "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
