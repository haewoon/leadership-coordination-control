from __future__ import annotations

from uuid import uuid4


def short_id() -> str:
    return uuid4().hex[:8]
