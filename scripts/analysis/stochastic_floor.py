"""Per-item 3-seed agreement rates across (model, regime, condition).

Quantifies each model's stochastic floor: of items run with seeds [0,1,2]
under identical prompt+policy, what fraction produce 3-way identical
normalized final answers?

Output: per-cell agreement rates and model-level averages. Used by
notes/bass_two_pillar_paper12.md §4 to ground the "run-to-run variance"
claim quantitatively.
"""
import json, sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.accuracy import normalize_prediction, exact_match

# Map (model, regime) -> (run_dirs, split_path)
CELLS = {
    ('qwen3.5','alphanli'):     (['alphanli_scale_step_100_core_qwen35'],
                                  'data/splits/alphanli_scale_step_100.jsonl'),
    ('qwen3.5','social'):       (['social_chemistry_broad_bc_100_core_qwen35',
                                   'social_chemistry_broad_bc_100_v2_core_qwen35'],
                                  'data/splits/social_chemistry_broad_bc_100.jsonl'),
    ('qwen3.5','closed'):       (['closed_ended_candidate_100_core_qwen35'],
                                  'data/splits/closed_ended_candidate_curated_100.jsonl'),
    ('qwen3.5','mixed'):        (['mixed_benchmark_150_core_qwen35'],
                                  'data/splits/mixed_benchmark_150.jsonl'),
    ('llama3.1','alphanli'):    (['alphanli_scale_step_100_llama31_full_seeds012',
                                   'alphanli_scale_step_100_llama31_seed0'],
                                  'data/splits/alphanli_scale_step_100.jsonl'),
    ('llama3.1','social'):      (['social_chemistry_broad_bc_100_llama31_full_seeds012'],
                                  'data/splits/social_chemistry_broad_bc_100.jsonl'),
    ('llama3.1','closed'):      (['closed_ended_candidate_100_llama31_full_seeds012',
                                   'closed_ended_candidate_100_llama31_seed0'],
                                  'data/splits/closed_ended_candidate_curated_100.jsonl'),
    ('llama3.1','mixed'):       (['mixed_benchmark_150_llama31_seed0',
                                   'mixed_benchmark_150_llama31_seeds12'],
                                  'data/splits/mixed_benchmark_150.jsonl'),
    ('gpt-5-mini','alphanli'):  (['alphanli_scale_step_100_openai_seed0',
                                   'alphanli_scale_step_100_openai_seeds12'],
                                  'data/splits/alphanli_scale_step_100.jsonl'),
    ('gpt-5-mini','social'):    (['social_chemistry_broad_bc_100_openai_full_seeds012'],
                                  'data/splits/social_chemistry_broad_bc_100.jsonl'),
    ('gpt-5-mini','closed'):    (['closed_ended_candidate_100_openai_seed0',
                                   'closed_ended_candidate_100_openai_seeds12'],
                                  'data/splits/closed_ended_candidate_curated_100.jsonl'),
    ('gpt-5-mini','mixed'):     (['mixed_benchmark_150_openai_seeds12_v3',
                                   'mixed_benchmark_150_openai_spotcheck_seed0_v2'],
                                  'data/splits/mixed_benchmark_150.jsonl'),
}

CONDITIONS = ['flat','theory_free_control','transactional','transformational','situational']

def load_gold(path):
    g = {}
    with open(ROOT / path) as f:
        for line in f:
            r = json.loads(line)
            g[r['task_id']] = r.get('gold_answer')
    return g

def collect_answers(run_dirs):
    """Returns dict[(condition, task_id)] -> dict[seed -> normalized_answer]"""
    out = defaultdict(lambda: defaultdict(str))
    for d in run_dirs:
        p = ROOT / 'outputs/runs' / d / 'run_summaries.jsonl'
        if not p.exists(): continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                cond = r.get('condition')
                if cond not in CONDITIONS: continue
                # store raw answer; normalize per task after we know gold
                out[(cond, r['task_id'])][r['seed']] = r.get('final_answer','')
    return out

def cell_agreement(answers, gold_map):
    """Per condition, return (n_items_with_3seeds, n_3way_agree, n_2way_agree_min).
    Agreement uses exact_match against gold for normalization (matches paper scoring)."""
    by_cond = defaultdict(lambda: {'n':0, 'agree3':0, 'agree2':0, 'no_agreement':0})
    for (cond, tid), seed_ans in answers.items():
        if tid not in gold_map: continue
        if len(seed_ans) < 3: continue  # need all 3 seeds
        gold = gold_map[tid]
        # Normalize each answer; we treat two answers as the same iff
        # exact_match(a,gold) == exact_match(b,gold) AND the normalized
        # extracted token is the same. Simpler: normalize_prediction then string compare.
        norm = [normalize_prediction(seed_ans[s], gold) for s in sorted(seed_ans.keys())]
        c = Counter(norm)
        top, n = c.most_common(1)[0]
        d = by_cond[cond]
        d['n'] += 1
        if n == 3:
            d['agree3'] += 1
        elif n == 2:
            d['agree2'] += 1
        else:
            d['no_agreement'] += 1
    return by_cond

# Run
print(f"{'model':10s} {'regime':10s} {'cond':22s} {'n':>4s} {'3-way':>7s} {'2-of-3':>7s} {'split':>7s}")
print("-"*75)
model_totals = defaultdict(lambda: defaultdict(int))  # model -> {agree3,agree2,split,n}
for (model, regime), (run_dirs, split_path) in CELLS.items():
    gold = load_gold(split_path)
    ans = collect_answers(run_dirs)
    cell = cell_agreement(ans, gold)
    for cond in CONDITIONS:
        d = cell.get(cond)
        if not d or d['n']==0: continue
        n = d['n']
        a3 = d['agree3']/n; a2 = d['agree2']/n; split = d['no_agreement']/n
        print(f"{model:10s} {regime:10s} {cond:22s} {n:>4d} {a3:>7.3f} {a2:>7.3f} {split:>7.3f}")
        model_totals[model]['agree3'] += d['agree3']
        model_totals[model]['agree2'] += d['agree2']
        model_totals[model]['split']  += d['no_agreement']
        model_totals[model]['n']      += n

print("\n=== Model-level totals (pooled across regimes × conditions) ===")
print(f"{'model':10s}  {'n':>5s}  {'3-way':>7s}  {'2-of-3':>7s}  {'split':>7s}  {'stoch.floor*':>13s}")
for m, t in model_totals.items():
    n = t['n']
    a3 = t['agree3']/n; a2 = t['agree2']/n; sp = t['split']/n
    # stochastic floor proxy = 1 - 3way agreement (fraction of items where at least one seed disagrees)
    floor = 1 - a3
    print(f"{m:10s}  {n:>5d}  {a3:>7.3f}  {a2:>7.3f}  {sp:>7.3f}  {floor:>12.3f}")
print("\n* stoch.floor proxy = 1 - 3-way agreement rate.")
print("  Interpreted as fraction of items where re-rolling a seed at temp 0.35")
print("  produces a different normalized answer at least once in 3 tries.")
