"""Generate sr0 unconstrained-coordinator baseline configs.

For each main-matrix combination (3 models x 4 regimes) this derives a
`sr0_<model>_<regime>_coord.json` from the existing canonical sr0 config:
same tasks/seeds/round budgets/generation settings, conditions replaced with
["llm_coordinator"], and `shared_round0_source` pointing at the original run
directory so the baseline starts from the *identical* logged round 0 (the
shared-round-0 attribution design carries over to this separately-executed
condition).

Usage: PYTHONPATH=. python3 scripts/gen_sr0_coord_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "src" / "config"

MODELS = ["gpt_oss_120b", "llama_4_scout", "gemma_4_31b"]
REGIMES = ["alphanli", "closed", "mixed", "social"]

# llama-4-scout was removed from the primary self-hosted endpoint on
# 2026-06-30, so its coordinator-baseline runs go through OpenRouter pinned to
# a single serving provider (DeepInfra) for stack consistency. Round 0 is
# still injected verbatim from the original self-hosted logs via
# `shared_round0_source`, so the shared-round-0 attribution design holds; the
# serving-stack change for rounds 1+ is disclosed in the paper/rebuttal.
OPENROUTER_OVERRIDES = {
    "llama_4_scout": {
        "name": "meta-llama/llama-4-scout",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPEN_ROUTER_API_KEY",
        "extra_body": {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}},
    }
}


def main() -> None:
    for model in MODELS:
        for regime in REGIMES:
            base_name = f"sr0_{model}_{regime}"
            base_path = CONFIG_DIR / f"{base_name}.json"
            cfg = json.loads(base_path.read_text())
            assert cfg.get("shared_round0"), f"{base_name} is not a shared-round0 config"
            cfg["experiment_name"] = f"{base_name}_coord"
            cfg["output_dir"] = f"outputs/runs/{base_name}_coord"
            cfg["conditions"] = ["llm_coordinator"]
            cfg["shared_round0_source"] = f"outputs/runs/{base_name}"
            override = OPENROUTER_OVERRIDES.get(model)
            if override:
                cfg["model"]["name"] = override["name"]
                cfg["model"]["base_url"] = override["base_url"]
                cfg["model"]["api_key_env"] = override["api_key_env"]
                cfg["generation"]["extra_body"] = override["extra_body"]
            out_path = CONFIG_DIR / f"{base_name}_coord.json"
            out_path.write_text(json.dumps(cfg, indent=2) + "\n")
            print(f"wrote {out_path.name}")


if __name__ == "__main__":
    main()
