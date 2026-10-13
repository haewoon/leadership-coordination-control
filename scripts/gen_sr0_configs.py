"""Generate shared-round-0 (sr0) self-hosted configs for the clean core re-run.

3 core open-weight models x {4 main regimes (7 conditions, 3 seeds) +
MATH-500 (6 conditions, 5 seeds)}, all with shared_round0=true. Profiles and
task defs mirror the existing selfhosted_*.json configs per regime; max_tokens bumped
(512 main / 1024 math) because gpt-oss emits hidden reasoning tokens.

Run: python3 scripts/gen_sr0_configs.py
Writes src/config/sr0_<model>_<regime>.json and prints the list.
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

MAIN_CONDITIONS = [
    "flat", "theory_free_control", "transactional", "transformational",
    "situational", "transactional_accept_only", "transformational_directives_only",
]
MATH_CONDITIONS = [
    "flat", "theory_free_control", "transactional", "transformational",
    "situational", "transactional_accept_only",
]

P = {
    "alphanli": [
        {"name": "counterexample_first", "instruction": "Prefer the answer that survives the strongest causal counterexample."},
        {"name": "surface_match", "instruction": "Favor the answer that most directly matches the observations without adding extra assumptions."},
        {"name": "plausibility_first", "instruction": "Prefer the answer that best fits ordinary causal expectations in real life."},
    ],
    "closed": [
        {"name": "counterexample_first", "instruction": "Prefer the answer that survives the strongest counterexample."},
        {"name": "surface_match", "instruction": "Favor the answer that most directly matches the question and choices without adding extra assumptions."},
        {"name": "plausibility_first", "instruction": "Prefer the answer that best matches ordinary expectations in real life."},
    ],
    "social": [
        {"name": "counterexample_first", "instruction": "Prefer the answer that survives the strongest social counterexample."},
        {"name": "surface_match", "instruction": "Favor the answer that most directly matches the situation and action wording without adding extra assumptions."},
        {"name": "plausibility_first", "instruction": "Prefer the answer that best matches ordinary social expectations in real life."},
    ],
    "mixed": [
        {"name": "counterexample_first", "instruction": "Prefer the answer that survives the strongest counterexample supported by the prompt."},
        {"name": "surface_match", "instruction": "Favor the answer that most directly matches the question and choices without adding extra assumptions."},
        {"name": "plausibility_first", "instruction": "Prefer the answer that best matches ordinary expectations from the given information."},
    ],
    "math500": [
        {"name": "step_by_step", "instruction": "Work the problem step by step, write the arithmetic and algebra explicitly, and double-check each step before committing to the final number."},
        {"name": "estimate_first", "instruction": "Form a rough estimate or sanity-check bound first, then compute the exact value and verify it lies in the expected range."},
        {"name": "verify_units", "instruction": "Track the meaning of each quantity throughout (counts, magnitudes, units), and confirm the final number matches what the question asks for."},
    ],
}

# regime -> (task_name, split_path, limit)
MAIN_REGIMES = {
    "alphanli": ("alphanli", "data/splits/alphanli_scale_step_100.jsonl", 100),
    "closed": ("commonsenseqa", "data/splits/closed_ended_candidate_curated_100.jsonl", 100),
    "social": ("social_chemistry", "data/splits/social_chemistry_broad_bc_100.jsonl", 100),
    "mixed": ("mixed_benchmark", "data/splits/mixed_benchmark_150.jsonl", 150),
}


def write_cfg(name: str, body: dict) -> None:
    (CFG / f"{name}.json").write_text(json.dumps(body, indent=2) + "\n")


def main() -> None:
    written = []
    for mslug, mname in MODELS.items():
        # main matrix (4 regimes)
        for regime, (task_name, path, limit) in MAIN_REGIMES.items():
            name = f"sr0_{mslug}_{regime}"
            body = {
                "experiment_name": name,
                "output_dir": f"outputs/runs/{name}",
                "model": {"provider": "openai_compat", "name": mname},
                "shared_round0": True,
                "warmup_prompt": "Reply with exactly: READY",
                "team_size": 3,
                "agent_profiles": P[regime],
                "seeds": [0, 1, 2],
                "conditions": MAIN_CONDITIONS,
                "tasks": [{"name": task_name, "path": path, "limit": limit}],
                "round_budget": {"closed_ended": 3, "integration": 4},
                "generation": {"temperature": 0.35, "max_tokens": 512},
                "runner": {"agent_parallelism": 3},
            }
            write_cfg(name, body)
            written.append(name)
        # MATH-500 (5 seeds, 6 conditions)
        name = f"sr0_{mslug}_math500"
        body = {
            "experiment_name": name,
            "output_dir": f"outputs/runs/{name}",
            "model": {"provider": "openai_compat", "name": mname},
            "shared_round0": True,
            "warmup_prompt": "Reply with exactly: READY",
            "team_size": 3,
            "agent_profiles": P["math500"],
            "seeds": [0, 1, 2, 3, 4],
            "conditions": MATH_CONDITIONS,
            "tasks": [{"name": "math500", "path": "data/splits/math500_hard_50.jsonl", "limit": 50}],
            "round_budget": {"closed_ended": 2, "integration": 4, "open_ended_numeric": 3},
            "generation": {"temperature": 0.35, "max_tokens": 1024},
            "runner": {"agent_parallelism": 3},
        }
        write_cfg(name, body)
        written.append(name)

    print(f"Wrote {len(written)} configs:")
    for n in written:
        print(f"  src/config/{n}.json")


if __name__ == "__main__":
    main()
