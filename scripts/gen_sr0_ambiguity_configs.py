"""Generate sr0 configs for the B (necessity-not-sufficiency) ambiguity probe:
3 models x {anli_r3, winogrande} x 5 conditions x 3 seeds, shared round-0.

Run: python3 scripts/gen_sr0_ambiguity_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "src" / "config"

MODELS = {
    "gpt_oss_120b": "gpt-oss-120b",
    "llama_4_scout": "llama-4-scout",
    "gemma_4_31b": "gemma-4-31B-it",
}
CONDITIONS = ["flat", "theory_free_control", "transactional", "transformational", "situational"]

PROFILES = {
    "anli": [
        {"name": "counterexample_first", "instruction": "Prefer the relationship that survives the strongest counterexample to the hypothesis."},
        {"name": "surface_match", "instruction": "Favor the relationship most directly supported by the literal premise wording, without adding outside assumptions."},
        {"name": "plausibility_first", "instruction": "Prefer the relationship that best fits ordinary real-world expectations about the premise and hypothesis."},
    ],
    "winogrande": [
        {"name": "counterexample_first", "instruction": "Prefer the option that still makes sense when you check it against the rest of the sentence."},
        {"name": "surface_match", "instruction": "Favor the option whose meaning most directly fits the sentence wording without adding extra assumptions."},
        {"name": "plausibility_first", "instruction": "Prefer the option that best fits ordinary real-world expectations."},
    ],
}

REGIMES = {
    "anli": ("commonsenseqa", "data/splits/anli_r3_100.jsonl", 100),
    "winogrande": ("commonsenseqa", "data/splits/winogrande_xl_100.jsonl", 100),
}


def main() -> None:
    written = []
    for mslug, mname in MODELS.items():
        for regime, (task_name, path, limit) in REGIMES.items():
            name = f"sr0_{mslug}_{regime}"
            body = {
                "experiment_name": name,
                "output_dir": f"outputs/runs/{name}",
                "model": {"provider": "openai_compat", "name": mname},
                "shared_round0": True,
                "warmup_prompt": "Reply with exactly: READY",
                "team_size": 3,
                "agent_profiles": PROFILES[regime],
                "seeds": [0, 1, 2],
                "conditions": CONDITIONS,
                "tasks": [{"name": task_name, "path": path, "limit": limit}],
                "round_budget": {"closed_ended": 3, "integration": 4},
                "generation": {"temperature": 0.35, "max_tokens": 512},
                "runner": {"agent_parallelism": 3},
            }
            (CFG / f"{name}.json").write_text(json.dumps(body, indent=2) + "\n")
            written.append(name)
    print(f"Wrote {len(written)} configs:")
    for n in written:
        print(f"  src/config/{n}.json")


if __name__ == "__main__":
    main()
