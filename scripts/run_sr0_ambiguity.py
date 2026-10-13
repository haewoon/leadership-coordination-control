"""Targeted launcher: run ONLY the 6 B-probe ambiguity configs (sr0_*_anli, sr0_*_winogrande).
Avoids the global sr0_*.json glob so existing complete runs are never touched.
Run in bg: PYTHONPATH=. nohup ../.venv/bin/python scripts/run_sr0_ambiguity.py > /tmp/sr0_ambiguity.log 2>&1 &
"""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_sr0_core import run_one, CONFIG_DIR

NAMES = [
    "sr0_gpt_oss_120b_anli", "sr0_gpt_oss_120b_winogrande",
    "sr0_llama_4_scout_anli", "sr0_llama_4_scout_winogrande",
    "sr0_gemma_4_31b_anli", "sr0_gemma_4_31b_winogrande",
]


def main() -> None:
    configs = [CONFIG_DIR / f"{n}.json" for n in NAMES]
    print(f"[{time.strftime('%H:%M:%S')}] launching {len(configs)} ambiguity configs", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run_one, c): c for c in configs}
        for fut in as_completed(futs):
            name, status, dt = fut.result()
            print(f"[{time.strftime('%H:%M:%S')}] {name}: {status} ({dt:.0f}s)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
