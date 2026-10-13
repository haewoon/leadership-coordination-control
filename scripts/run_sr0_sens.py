"""Run the 3 situational sensitivity configs (social regime). Targeted; never touches other runs."""
from __future__ import annotations
import time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_sr0_core import run_one, CONFIG_DIR
NAMES = ["sr0_gpt_oss_120b_social_sens", "sr0_llama_4_scout_social_sens", "sr0_gemma_4_31b_social_sens"]
def main():
    configs=[CONFIG_DIR/f"{n}.json" for n in NAMES]
    print(f"[{time.strftime('%H:%M:%S')}] launching {len(configs)} sensitivity configs", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs={ex.submit(run_one,c):c for c in configs}
        for fut in as_completed(futs):
            name,status,dt=fut.result(); print(f"[{time.strftime('%H:%M:%S')}] {name}: {status} ({dt:.0f}s)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)
if __name__=="__main__": main()
