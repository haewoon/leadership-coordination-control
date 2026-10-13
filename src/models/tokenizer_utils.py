from __future__ import annotations


def rough_token_count(text: str) -> int:
    return max(1, len(text.split()))
