from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_POOL = ROOT / "data" / "splits" / "alphanli_reference_pool_50.jsonl"
CURATED10 = ROOT / "data" / "splits" / "alphanli_curated_challenge_10.jsonl"
OUTPUT = ROOT / "data" / "splits" / "alphanli_candidate_curated_30.jsonl"
NOTES = ROOT / "data" / "splits" / "alphanli_candidate_curated_30_notes.md"


EXCLUDED_IDS = {
    "3717fabf-07c4-44f4-8a6c-d7361b1b3bef-1",
    "3eeb7000-7f6a-43b5-83ce-e994ad8d982a-1",
    "57d1f6e1-cdbc-4df6-9841-5dba89d3e63e2",
    "81d092b2-13f3-4a93-ae36-94d529676611-1",
    "f32987e4-b8b0-49a2-b060-e6589bf1a1dd-1",
    "9f413689-2fef-4547-a3b2-4f6be1faf9f3-1",
    "b0ff7467-b109-456c-8dc9-e9e957e56098-1",
    "904a5fb1-14da-489e-92c0-ab9895910595-1",
    "dec3ca83-5ba2-402a-9374-e2290dc6e183-1",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def main() -> None:
    curated10 = load_jsonl(CURATED10)
    ref_pool = load_jsonl(REF_POOL)

    kept_ids = {row["task_id"] for row in curated10}
    rows = list(curated10)
    added = []
    for row in ref_pool:
        task_id = row["task_id"]
        if task_id in kept_ids or task_id in EXCLUDED_IDS:
            continue
        rows.append(row)
        added.append(row)
        if len(rows) == 30:
            break

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# AlphaNLI Candidate Curated 30",
        "",
        "This file is a conference-scale candidate set, not yet the final curated evaluation set.",
        "Construction rule:",
        "- keep the existing defended curated 10;",
        "- extend with high-scoring reference-pool items from the stable challenge pool;",
        "- exclude items already flagged in the curated-10 notes as obvious, artifact-heavy, or low-value;",
        "- require final human review before the set becomes the paper's final curated AlphaNLI benchmark.",
        "",
        f"Final candidate size: {len(rows)}",
        "",
        "## Added beyond the original curated 10",
    ]
    for row in added:
        meta = row.get("metadata", {})
        details = meta.get("challenge_details", {})
        lines.extend(
            [
                f"- `{row['task_id']}`",
                f"  - gold: `{row['gold_answer']}`; challenge score: `{meta.get('challenge_score')}`",
                f"  - support balance: `{details.get('support_balance')}`; hypothesis similarity: `{details.get('hypothesis_similarity')}`",
            ]
        )

    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_rows": len(rows), "added_rows": len(added), "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
