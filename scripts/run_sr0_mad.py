"""Run the 12 full-matrix MAD configs. Targeted; never touches other runs."""
from __future__ import annotations
import time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_sr0_core import run_one, CONFIG_DIR
MODELS = ["gpt_oss_120b", "llama_4_scout", "gemma_4_31b"]
REGIMES = ["alphanli", "closed", "social", "mixed"]
NAMES = [f"sr0_{m}_{r}_mad" for m in MODELS for r in REGIMES]
def main():
    configs=[CONFIG_DIR/f"{n}.json" for n in NAMES]
    print(f"[{time.strftime('%H:%M:%S')}] launching {len(configs)} MAD configs", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs={ex.submit(run_one,c):c for c in configs}
        for fut in as_completed(futs):
            name,status,dt=fut.result(); print(f"[{time.strftime('%H:%M:%S')}] {name}: {status} ({dt:.0f}s)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)
if __name__=="__main__": main()
