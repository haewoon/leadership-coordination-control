from __future__ import annotations


def total_tokens(events: list[dict]) -> int:
    return sum(event.get("prompt_tokens", 0) + event.get("completion_tokens", 0) for event in events)
