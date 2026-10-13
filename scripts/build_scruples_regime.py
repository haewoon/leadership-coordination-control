"""Build a contested moral-judgment regime from Scruples (r/AmITheAsshole anecdotes)
for the B salvage swing. Unlike ANLI/Winogrande (objective gold, majority-of-3 robust),
Scruples verdicts are genuinely contested community judgments, so the 3-agent round-0
majority should be UNRELIABLE -- the necessary condition for a leadership-vs-flat gain.

Task (3-way, mirroring the social-norm structure with an explicit "contested" bucket):
  A = the author was in the wrong            (AITA verdict AUTHOR / "YTA")
  B = the author was NOT in the wrong        (verdict OTHER / "NTA")
  C = mixed or unclear                       (EVERYBODY / NOBODY / INFO)

Selection: contested items only (plurality fraction < 0.7 over >= 5 votes), moderate
length, balanced ~34/33/33 across A/B/C. Deterministic scan order (no shuffle).

Run: ../.venv/bin/python scripts/build_scruples_regime.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "splits"
N = 99
PER = {"A": 33, "B": 33, "C": 33}
BUCKET = {"AUTHOR": "A", "OTHER": "B", "EVERYBODY": "C", "NOBODY": "C", "INFO": "C"}
CHOICES = [
    {"label": "A", "text": "the author was in the wrong"},
    {"label": "B", "text": "the author was not in the wrong (the other party was, or no one)"},
    {"label": "C", "text": "mixed or genuinely unclear (everyone at fault, no one at fault, or not enough information)"},
]


def clean(s: str, cap: int = 1000) -> str:
    s = re.sub(r"&amp;#x200B;|&amp;|&#x200B;|&gt;|&lt;", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:cap] + ("..." if len(s) > cap else "")


def main() -> None:
    ds = load_dataset("metaeval/scruples", split="test")
    taken = {"A": 0, "B": 0, "C": 0}
    rows = []
    for ex in ds:
        scores = ex.get("label_scores") or {}
        total = sum(scores.values())
        if total < 5:
            continue
        plurality = max(scores.values()) / total
        if plurality >= 0.7:  # keep contested items only
            continue
        b = BUCKET.get(ex["label"])
        if b is None or taken[b] >= PER[b]:
            continue
        text = clean(ex["text"])
        if len(text) < 80:
            continue
        taken[b] += 1
        title = clean(ex.get("title") or "", 160)
        q = (
            "The following is a real post from Reddit's r/AmITheAsshole, where the community "
            "votes on who was in the wrong. Read the situation and judge what the community's "
            "majority verdict was.\n\n"
            f"Title: {title}\n"
            f"Situation: {text}\n\n"
            "What was the community's majority verdict?"
        )
        rows.append({
            "task_id": f"scruples_{ex['id']}",
            "question": q,
            "gold_answer": b,
            "choices": CHOICES,
            "metadata": {"source": "scruples", "orig_label": ex["label"],
                         "plurality": round(plurality, 3), "votes": total},
        })
        if sum(taken.values()) >= N:
            break
    path = OUT / "scruples_contested_99.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    dist = Counter(r["gold_answer"] for r in rows)
    med_plur = sorted(r["metadata"]["plurality"] for r in rows)[len(rows) // 2]
    print(f"wrote {path}  ({len(rows)} items; gold dist {dict(sorted(dist.items()))}; "
          f"median plurality {med_plur})")


if __name__ == "__main__":
    main()
