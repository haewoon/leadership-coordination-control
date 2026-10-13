"""Check rounds_used for transactional across all 12 paper-12 cells."""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Map (model, task) -> list of run dirs holding the original transactional data
# Tasks: alphanli, social, closed (commonsenseqa), mixed
RUN_DIRS = {
    ('qwen3.5', 'alphanli'):     ['alphanli_scale_step_100_core_qwen35'],
    ('qwen3.5', 'social'):       ['social_chemistry_broad_bc_100_core_qwen35','social_chemistry_broad_bc_100_v2_core_qwen35'],
    ('qwen3.5', 'closed'):       ['closed_ended_candidate_100_core_qwen35'],
    ('qwen3.5', 'mixed'):        ['mixed_benchmark_150_core_qwen35'],
    ('llama3.1','alphanli'):     ['alphanli_scale_step_100_llama31_full_seeds012','alphanli_scale_step_100_llama31_seed0'],
    ('llama3.1','social'):       ['social_chemistry_broad_bc_100_llama31_full_seeds012'],
    ('llama3.1','closed'):       ['closed_ended_candidate_100_llama31_full_seeds012','closed_ended_candidate_100_llama31_seed0'],
    ('llama3.1','mixed'):        ['mixed_benchmark_150_llama31_seed0','mixed_benchmark_150_llama31_seeds12'],
    ('gpt-5-mini','alphanli'):   ['alphanli_scale_step_100_openai_seed0','alphanli_scale_step_100_openai_seeds12'],
    ('gpt-5-mini','social'):     ['social_chemistry_broad_bc_100_openai_full_seeds012'],
    ('gpt-5-mini','closed'):     ['closed_ended_candidate_100_openai_seed0','closed_ended_candidate_100_openai_seeds12'],
    ('gpt-5-mini','mixed'):      ['mixed_benchmark_150_openai_seeds12_v3','mixed_benchmark_150_openai_spotcheck_seed0_v2','mixed_benchmark_150_openai_spotcheck_seed0'],
}

print(f"{'model':10s} {'task':10s} {'n':>5s}  {'r=1':>5s} {'r=2':>5s} {'r=3+':>5s}  {'%r>=2':>6s}")
print("-"*60)
for (model, task), dirs in RUN_DIRS.items():
    counter = Counter()
    seeds_seen = set()
    for d in dirs:
        p = ROOT / 'outputs/runs' / d / 'run_summaries.jsonl'
        if not p.exists(): continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get('condition') != 'transactional': continue
                counter[r.get('rounds_used',0)] += 1
                seeds_seen.add(r.get('seed'))
    n = sum(counter.values())
    if n == 0:
        print(f"{model:10s} {task:10s} (no data)")
        continue
    r1 = counter.get(1, 0)
    r2 = counter.get(2, 0)
    r3p = sum(v for k,v in counter.items() if k>=3)
    pct_2plus = (r2+r3p)/n*100
    print(f"{model:10s} {task:10s} {n:>5d}  {r1:>5d} {r2:>5d} {r3p:>5d}  {pct_2plus:>5.1f}%  seeds={sorted(seeds_seen)}")
