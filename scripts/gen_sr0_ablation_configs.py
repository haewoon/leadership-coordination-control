"""Generate shared-round-0 self-hosted configs for the appendix ablations (re-run of the
qwen-only ablations on the 3 self-hosted models): MAD baseline + situational-random-switch,
team-size 5, round-budget 6, inclusive (focused-social).

Run: python3 scripts/gen_sr0_ablation_configs.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "src" / "config"
MODELS = {"gpt_oss_120b": "gpt-oss-120b", "llama_4_scout": "llama-4-scout", "gemma_4_31b": "gemma-4-31B-it"}

P = {
    "alphanli": [
        {"name": "counterexample_first", "instruction": "Prefer the answer that survives the strongest causal counterexample."},
        {"name": "surface_match", "instruction": "Favor the answer that most directly matches the observations without adding extra assumptions."},
        {"name": "plausibility_first", "instruction": "Prefer the answer that best fits ordinary causal expectations in real life."},
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
}
EXTRA5 = [
    {"name": "evidence_aggregator", "instruction": "Weigh the cumulative evidence across the inputs before deciding."},
    {"name": "minority_defender", "instruction": "Argue for the less popular interpretation when it is still defensible."},
]
SPLIT = {
    "alphanli": ("alphanli", "data/splits/alphanli_scale_step_100.jsonl", 100),
    "social": ("social_chemistry", "data/splits/social_chemistry_broad_bc_100.jsonl", 100),
    "mixed": ("mixed_benchmark", "data/splits/mixed_benchmark_150.jsonl", 150),
}
MAIN5 = ["flat", "theory_free_control", "transactional", "transformational", "situational"]


def base(name, mname, profiles, conditions, regime, seeds, round_budget=None, team_size=3):
    tn, path, lim = SPLIT[regime]
    return {
        "experiment_name": name, "output_dir": f"outputs/runs/{name}",
        "model": {"provider": "openai_compat", "name": mname}, "shared_round0": True,
        "warmup_prompt": "Reply with exactly: READY", "team_size": team_size,
        "agent_profiles": profiles, "seeds": seeds, "conditions": conditions,
        "tasks": [{"name": tn, "path": path, "limit": lim}],
        "round_budget": round_budget or {"closed_ended": 3, "integration": 4},
        "generation": {"temperature": 0.35, "max_tokens": 512},
        "runner": {"agent_parallelism": 3},
    }


def main():
    written = []
    for slug, mname in MODELS.items():
        # MAD + situational_random_switch (alphanli, social, mixed)
        for r in ["alphanli", "social", "mixed"]:
            n = f"sr0_{slug}_madsr_{r}"
            written.append(n)
            (CFG / f"{n}.json").write_text(json.dumps(
                base(n, mname, P[r], MAIN5 + ["mad", "situational_random_switch"], r, [0, 1, 2]), indent=2) + "\n")
        # team-size 5 (alphanli, social)
        for r in ["alphanli", "social"]:
            n = f"sr0_{slug}_team5_{r}"
            written.append(n)
            (CFG / f"{n}.json").write_text(json.dumps(
                base(n, mname, P[r] + EXTRA5, MAIN5, r, [0, 1, 2], team_size=5), indent=2) + "\n")
        # round-budget 6 (alphanli, social)
        for r in ["alphanli", "social"]:
            n = f"sr0_{slug}_rb6_{r}"
            written.append(n)
            (CFG / f"{n}.json").write_text(json.dumps(
                base(n, mname, P[r], MAIN5, r, [0, 1, 2], round_budget={"closed_ended": 6, "integration": 6}), indent=2) + "\n")
        # inclusive focused-social
        n = f"sr0_{slug}_inclusive_social"
        written.append(n)
        (CFG / f"{n}.json").write_text(json.dumps(
            base(n, mname, P["social"], ["flat", "situational", "inclusive"], "social", [0, 1, 2]), indent=2) + "\n")
    print(f"Wrote {len(written)} ablation configs:")
    for n in written:
        print("  ", n)


if __name__ == "__main__":
    main()
