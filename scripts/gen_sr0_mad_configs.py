"""Full-matrix MAD baseline configs: MAD compared against flat/situational/transformational
on the same shared round-0, across all 12 (model, regime) main-matrix combinations.
Reuses the main-matrix profiles/regimes from gen_sr0_configs."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.gen_sr0_configs import MODELS, MAIN_REGIMES, P  # noqa: E402
CFG = ROOT / "src" / "config"
CONDS = ["flat", "mad", "situational", "transformational"]

def main():
    written = []
    for mslug, mname in MODELS.items():
        for regime, (task_name, path, limit) in MAIN_REGIMES.items():
            name = f"sr0_{mslug}_{regime}_mad"
            body = {"experiment_name": name, "output_dir": f"outputs/runs/{name}",
                    "model": {"provider": "openai_compat", "name": mname}, "shared_round0": True,
                    "warmup_prompt": "Reply with exactly: READY", "team_size": 3,
                    "agent_profiles": P[regime], "seeds": [0, 1, 2], "conditions": CONDS,
                    "tasks": [{"name": task_name, "path": path, "limit": limit}],
                    "round_budget": {"closed_ended": 3, "integration": 4},
                    "generation": {"temperature": 0.35, "max_tokens": 512}, "runner": {"agent_parallelism": 3}}
            (CFG / f"{name}.json").write_text(json.dumps(body, indent=2) + "\n"); written.append(name)
    print(f"wrote {len(written)} configs"); [print("  "+n) for n in written]

if __name__ == "__main__": main()
