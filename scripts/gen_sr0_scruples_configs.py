"""Generate sr0 configs for the Scruples contested moral-judgment regime (B salvage)."""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; CFG = ROOT / "src" / "config"
MODELS = {"gpt_oss_120b": "gpt-oss-120b", "llama_4_scout": "llama-4-scout", "gemma_4_31b": "gemma-4-31B-it"}
CONDS = ["flat", "theory_free_control", "transactional", "transformational", "situational"]
PROFILES = [
    {"name": "counterexample_first", "instruction": "Prefer the verdict that survives the strongest counterargument from the other side's perspective."},
    {"name": "surface_match", "instruction": "Favor the verdict most directly supported by what the author actually did, without adding outside assumptions."},
    {"name": "plausibility_first", "instruction": "Prefer the verdict that best matches how an ordinary, fair-minded community would judge the situation."},
]
def main():
    written=[]
    for mslug, mname in MODELS.items():
        name=f"sr0_{mslug}_scruples"
        body={"experiment_name":name,"output_dir":f"outputs/runs/{name}","model":{"provider": "openai_compat","name":mname},
              "shared_round0":True,"warmup_prompt":"Reply with exactly: READY","team_size":3,
              "agent_profiles":PROFILES,"seeds":[0,1,2],"conditions":CONDS,
              "tasks":[{"name":"social_chemistry","path":"data/splits/scruples_contested_99.jsonl","limit":99}],
              "round_budget":{"closed_ended":3,"integration":4},
              "generation":{"temperature":0.35,"max_tokens":512},"runner":{"agent_parallelism":3}}
        (CFG/f"{name}.json").write_text(json.dumps(body,indent=2)+"\n"); written.append(name)
    print("wrote:", *written, sep="\n  ")
if __name__=="__main__": main()
