from __future__ import annotations


def score_integration_quality(answer: str) -> float:
    # Placeholder rubric score.
    return min(1.0, max(0.0, len(answer) / 100.0))
