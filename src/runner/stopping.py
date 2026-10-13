from __future__ import annotations


def should_stop_for_budget(team_state: object, round_budget: int) -> bool:
    return getattr(team_state, "round_idx", 0) + 1 >= round_budget
