from __future__ import annotations

import json


def to_pretty_json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)
