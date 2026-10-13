from __future__ import annotations

from typing import Sequence


def build_shards(question: str, evidence: Sequence[str], n_agents: int = 3) -> dict[str, str]:
    shards: dict[str, str] = {}
    for idx in range(n_agents):
        shard = evidence[idx::n_agents]
        shards[f"agent_{idx}"] = f"Question: {question}\nEvidence:\n- " + "\n- ".join(shard)
    return shards
