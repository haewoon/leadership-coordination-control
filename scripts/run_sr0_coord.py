"""Orchestrate the unconstrained-coordinator baseline runs on self-hosted.

Runs all src/config/sr0_*_coord.json with bounded concurrency. Idempotent and
resumable with the same rules as run_sr0_core.py: a config whose
run_summaries.jsonl already has the expected number of rows is skipped;
otherwise its (possibly partial) output dir is removed and rerun fresh.

Run in background:
  PYTHONPATH=. nohup python3 scripts/run_sr0_coord.py > /tmp/sr0_coord.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "src" / "config"
MAX_WORKERS = int(os.environ.get("SR0_WORKERS", "6"))


def expected_rows(cfg: dict) -> int:
    n_tasks = sum(t.get("limit") or 0 for t in cfg["tasks"])
    return n_tasks * len(cfg["seeds"]) * len(cfg["conditions"])


def is_complete(cfg: dict) -> bool:
    p = ROOT / cfg["output_dir"] / "run_summaries.jsonl"
    if not p.exists():
        return False
    n = sum(1 for _ in p.open())
    return n >= expected_rows(cfg)


def run_one(cfg_path: Path) -> tuple[str, str, float]:
    cfg = json.loads(cfg_path.read_text())
    name = cfg["experiment_name"]
    t0 = time.time()
    if is_complete(cfg):
        return (name, "SKIP (complete)", 0.0)
    out = ROOT / cfg["output_dir"]
    if out.exists():
        shutil.rmtree(out)
    log = Path("/tmp") / f"sr0_{name}.log"
    with log.open("w") as fh:
        proc = subprocess.run(
            [sys.executable, "-m", "src.main", "--config", str(cfg_path.relative_to(ROOT))],
            cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
    dt = time.time() - t0
    ok = proc.returncode == 0 and is_complete(cfg)
    return (name, "OK" if ok else f"FAIL(rc={proc.returncode}, see {log})", dt)


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "sr0_*_coord.json"
    configs = sorted(CONFIG_DIR.glob(pattern))
    print(f"[{time.strftime('%H:%M:%S')}] {len(configs)} coord configs; max_workers={MAX_WORKERS}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one, c): c for c in configs}
        for fut in as_completed(futs):
            name, status, dt = fut.result()
            done += 1
            print(f"[{time.strftime('%H:%M:%S')}] ({done}/{len(configs)}) {name}: {status} ({dt:.0f}s)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
