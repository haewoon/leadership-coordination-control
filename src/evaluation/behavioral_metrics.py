from __future__ import annotations

from collections import Counter


def control_actions_per_round(events: list[dict]) -> float:
    round_counts = Counter(event["round_idx"] for event in events if event["event_type"] == "control_action")
    if not round_counts:
        return 0.0
    return sum(round_counts.values()) / len(round_counts)
