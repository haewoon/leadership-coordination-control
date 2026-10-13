"""Situational sensitivity-analysis configs: social regime x 3 models, with the main
situational controller and three threshold/lexicon variants, all sharing round-0."""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; CFG = ROOT / "src" / "config"
MODELS = {"gpt_oss_120b": "gpt-oss-120b", "llama_4_scout": "llama-4-scout", "gemma_4_31b": "gemma-4-31B-it"}
CONDS = ["flat", "situational", "situational_split3", "situational_lenient", "situational_altlex"]
PROFILES = [
    {"name": "counterexample_first", "instruction": "Prefer the answer that survives the strongest social counterexample."},
    {"name": "surface_match", "instruction": "Favor the answer that most directly matches the situation and action wording without adding extra assumptions."},
    {"name": "plausibility_first", "instruction": "Prefer the answer that best matches ordinary social expectations in real life."},
]
def main():
    written = []
    for mslug, mname in MODELS.items():
        name = f"sr0_{mslug}_social_sens"
        body = {"experiment_name": name, "output_dir": f"outputs/runs/{name}",
                "model": {"provider": "openai_compat", "name": mname}, "shared_round0": True,
                "warmup_prompt": "Reply with exactly: READY", "team_size": 3,
                "agent_profiles": PROFILES, "seeds": [0, 1, 2], "conditions": CONDS,
                "tasks": [{"name": "social_chemistry", "path": "data/splits/social_chemistry_broad_bc_100.jsonl", "limit": 100}],
                "round_budget": {"closed_ended": 3, "integration": 4},
                "generation": {"temperature": 0.35, "max_tokens": 512}, "runner": {"agent_parallelism": 3}}
        (CFG / f"{name}.json").write_text(json.dumps(body, indent=2) + "\n"); written.append(name)
    print("wrote:", *written, sep="\n  ")
if __name__ == "__main__": main()
