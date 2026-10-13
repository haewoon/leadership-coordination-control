"""Compare llama3.1 mixed accept_only vs original transactional, per-item."""
import json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.accuracy import exact_match

def load_gold():
    g = {}
    with open(ROOT / "data/splits/mixed_benchmark_150.jsonl") as f:
        for line in f:
            r = json.loads(line)
            g[r["task_id"]] = (r.get("gold_answer"), r.get("metadata",{}).get("source"))
    return g

GOLD = load_gold()

def score_path(path, want_cond, label):
    """Returns dict[(task_id, seed)] -> (pred, correct)"""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("condition") != want_cond: continue
            tid = r["task_id"]
            if tid not in GOLD: continue
            gold, source = GOLD[tid]
            pred = r.get("final_answer","") or ""
            correct = 1 if exact_match(pred, gold) else 0
            out[(tid, r["seed"])] = (pred, correct, source, r.get("rounds_used"), r.get("accepted_count"))
    return out

# Original transactional - aggregate across both runs
orig = {}
for p in [
    ROOT / "outputs/runs/mixed_benchmark_150_llama31_seed0/run_summaries.jsonl",
    ROOT / "outputs/runs/mixed_benchmark_150_llama31_seeds12/run_summaries.jsonl",
]:
    o = score_path(p, "transactional", "orig")
    orig.update(o)

accept = score_path(
    ROOT / "outputs/runs/llama31_paper12_accept_only/run_summaries.jsonl",
    "transactional_accept_only", "accept"
)

print(f"original transactional: {len(orig)} items")
print(f"accept_only: {len(accept)} items")

# Overall acc
def acc(d): 
    vals = [v[1] for v in d.values()]
    return sum(vals)/len(vals) if vals else 0
print(f"\nOverall: orig={acc(orig):.3f}  accept_only={acc(accept):.3f}  Δ={acc(accept)-acc(orig):+.3f}")

# Per source
by_src_o = defaultdict(list); by_src_a = defaultdict(list)
for k,v in orig.items():
    by_src_o[v[2]].append(v[1])
for k,v in accept.items():
    by_src_a[v[2]].append(v[1])
print(f"\n{'source':25s} {'orig':>6s} {'acc':>6s}    {'Δ':>6s}  {'nO':>4s} {'nA':>4s}")
for src in sorted(set(by_src_o.keys()) | set(by_src_a.keys())):
    o = sum(by_src_o[src])/len(by_src_o[src]) if by_src_o[src] else 0
    a = sum(by_src_a[src])/len(by_src_a[src]) if by_src_a[src] else 0
    print(f"{str(src)[:25]:25s} {o:.3f} {a:.3f}   {a-o:+.3f}  {len(by_src_o[src]):4d} {len(by_src_a[src]):4d}")

# Flip analysis: items present in both, where they disagree
common = set(orig.keys()) & set(accept.keys())
print(f"\nCommon (task,seed) pairs: {len(common)}")
flips = {'orig_correct_accept_wrong':0, 'orig_wrong_accept_correct':0, 'both_correct':0, 'both_wrong':0}
flip_examples = defaultdict(list)
for k in common:
    oc = orig[k][1]; ac = accept[k][1]
    src = orig[k][2]
    if oc and not ac:
        flips['orig_correct_accept_wrong'] += 1
        flip_examples['orig_correct_accept_wrong'].append((k, src))
    elif not oc and ac:
        flips['orig_wrong_accept_correct'] += 1
        flip_examples['orig_wrong_accept_correct'].append((k, src))
    elif oc and ac:
        flips['both_correct'] += 1
    else:
        flips['both_wrong'] += 1
print(f"  both correct:       {flips['both_correct']}")
print(f"  orig✓ accept✗ (lost): {flips['orig_correct_accept_wrong']}")
print(f"  orig✗ accept✓ (gained): {flips['orig_wrong_accept_correct']}")
print(f"  both wrong:         {flips['both_wrong']}")

# Source breakdown of losses
loss_by_src = defaultdict(int); gain_by_src = defaultdict(int)
for k,src in flip_examples['orig_correct_accept_wrong']:
    loss_by_src[src] += 1
for k,src in flip_examples['orig_wrong_accept_correct']:
    gain_by_src[src] += 1
print(f"\nLosses (orig✓ → accept✗) by source:")
for src in sorted(set(loss_by_src.keys()) | set(gain_by_src.keys())):
    print(f"  {str(src)[:25]:25s}  loss={loss_by_src[src]:3d}  gain={gain_by_src[src]:3d}  net={gain_by_src[src]-loss_by_src[src]:+d}")

# Rounds_used distribution
def rounds_dist(d):
    c = defaultdict(int)
    for v in d.values(): c[v[3]] += 1
    return dict(c)
print(f"\nRounds used distribution:")
print(f"  orig:   {sorted(rounds_dist(orig).items())}")
print(f"  accept: {sorted(rounds_dist(accept).items())}")
