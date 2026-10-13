"""Score paper-12 accept_only runs for the 3 models using gold answers."""
import json, sys, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.accuracy import exact_match, normalize_prediction

# paper-12 task splits used by all 3 accept_only configs
SPLITS = {
    "alphanli": "data/splits/alphanli_scale_step_100.jsonl",
    "social_chemistry": "data/splits/social_chemistry_broad_bc_100.jsonl",
    "commonsenseqa": "data/splits/closed_ended_candidate_curated_100.jsonl",
    "mixed_benchmark": "data/splits/mixed_benchmark_150.jsonl",
}

def load_gold():
    g = {}
    for task, path in SPLITS.items():
        with open(ROOT / path) as f:
            for line in f:
                row = json.loads(line)
                g[row["task_id"]] = (task, row.get("gold_answer"))
    return g

GOLD = load_gold()

def score_run(path, model):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    # group by (task, seed)
    grp = defaultdict(list)
    skipped = 0
    for r in rows:
        tid = r["task_id"]
        if tid not in GOLD:
            skipped += 1
            continue
        task, gold = GOLD[tid]
        seed = r["seed"]
        pred = r.get("final_answer", "") or ""
        correct = exact_match(pred, gold)
        grp[(task, seed)].append(1 if correct else 0)
    if skipped:
        print(f"  WARNING: {skipped} task_ids missing from gold")
    # per task: mean across seeds
    per_task = defaultdict(dict)
    for (t, s), vals in grp.items():
        per_task[t][s] = sum(vals)/len(vals)
    print(f"\n=== {model} (transactional_accept_only) ===")
    print(f"  {'task':22s}  seeds={'[0,1,2]':25s} mean")
    for t in ["alphanli","social_chemistry","commonsenseqa","mixed_benchmark"]:
        if t not in per_task: continue
        seeds = sorted(per_task[t].keys())
        accs = [per_task[t][s] for s in seeds]
        mean = sum(accs)/len(accs)
        print(f"  {t:22s}  {str(['%.3f'%a for a in accs]):25s} {mean:.3f}")
    return per_task

for model, path in [
    ('qwen3.5',   ROOT / 'outputs/runs/qwen35_paper12_accept_only/run_summaries.jsonl'),
    ('gpt-5-mini',ROOT / 'outputs/runs/gpt5mini_paper12_accept_only/run_summaries.jsonl'),
    ('llama3.1',  ROOT / 'outputs/runs/llama31_paper12_accept_only/run_summaries.jsonl'),
]:
    score_run(path, model)
