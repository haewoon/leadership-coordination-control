"""Build two adversarial-ambiguity regimes for the B (necessity-not-sufficiency) probe.

These regimes are chosen to test whether a leadership controller beats *flat*
interaction where round-0 is unreliable AND the failure mode is ambiguity-driven
premature consensus (a tempting-wrong answer), as opposed to hard reasoning
(MATH, where flat already recovers).

  anli_r3_100      : Adversarial NLI round 3 (3-way entail/neutral/contradict),
                     constructed adversarially so a tempting-wrong label fools models.
  winogrande_xl_100: Twin-sentence pronoun/blank disambiguation (binary), where a
                     surface word-association heuristic gives a tempting-wrong option.

Both are written in the schema the schema-driven `commonsenseqa` loader reads
(task_id, question, gold_answer, choices). Deterministic (first-N), no shuffle.

Run: ../.venv/bin/python scripts/build_ambiguity_regimes.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "splits"
N = 100


def build_anli() -> None:
    ds = load_dataset("facebook/anli", split="test_r3")
    label_to_letter = {0: "A", 1: "B", 2: "C"}  # 0 entail, 1 neutral, 2 contradiction
    choices = [
        {"label": "A", "text": "entailment (the premise entails the hypothesis)"},
        {"label": "B", "text": "neutral (the premise neither entails nor contradicts the hypothesis)"},
        {"label": "C", "text": "contradiction (the premise contradicts the hypothesis)"},
    ]
    # Balance labels: first-N of r3 is skewed (few contradictions); take an even
    # quota per label so gold is not trivially gameable by a constant guess.
    per_label = {0: N // 3 + (N % 3), 1: N // 3, 2: N // 3}  # ~34/33/33
    taken = {0: 0, 1: 0, 2: 0}
    rows = []
    for ex in ds:
        lab = ex["label"]
        if lab not in taken or taken[lab] >= per_label[lab]:
            continue
        taken[lab] += 1
        q = (
            f"Premise: {ex['premise']}\n"
            f"Hypothesis: {ex['hypothesis']}\n\n"
            "What is the logical relationship from the premise to the hypothesis?"
        )
        rows.append({
            "task_id": f"anli_r3_{ex['uid']}",
            "question": q,
            "gold_answer": label_to_letter[lab],
            "choices": choices,
            "metadata": {"source": "anli", "round": "r3", "split": "test"},
        })
        if sum(taken.values()) >= N:
            break
    _write("anli_r3_100.jsonl", rows)


def build_winogrande() -> None:
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    rows = []
    n = 0
    for ex in ds:
        if not ex.get("answer") or ex["answer"] not in ("1", "2"):
            continue
        q = (
            "Fill in the blank ( _ ) with the option that makes the sentence most sensible.\n"
            f"Sentence: {ex['sentence']}\n\n"
            "Which option best fills the blank?"
        )
        choices = [
            {"label": "A", "text": ex["option1"]},
            {"label": "B", "text": ex["option2"]},
        ]
        rows.append({
            "task_id": f"winogrande_{n:04d}",
            "question": q,
            "gold_answer": "A" if ex["answer"] == "1" else "B",
            "choices": choices,
            "metadata": {"source": "winogrande", "config": "xl", "split": "validation"},
        })
        n += 1
        if n >= N:
            break
    _write("winogrande_xl_100.jsonl", rows)


def _write(name: str, rows: list[dict]) -> None:
    path = OUT / name
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    dist = Counter(r["gold_answer"] for r in rows)
    print(f"wrote {path}  ({len(rows)} items; gold dist {dict(sorted(dist.items()))})")


if __name__ == "__main__":
    build_anli()
    build_winogrande()
