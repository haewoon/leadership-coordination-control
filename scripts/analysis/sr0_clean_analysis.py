"""Clean-baseline analysis of the shared-round-0 (sr0) core re-run.

With shared round-0, the round-0-vote (R0-vote) is a TRUE shared baseline:
every condition starts from the identical round-0, so `final - R0vote` is a
clean, confound-free measure of what each controller's interaction (rounds 1+)
adds. This script recomputes, per (model, regime):

  R0-vote acc, per-condition final acc, gain = final - R0vote (matched paired
  permutation p), lock-in, recovery, mean rounds; F1 reducibility (within +-4pp
  count), F2 gain cells (>=5pp, BH-corrected across leadership cells).

Run after the core run completes:
  PYTHONPATH=. python3 scripts/analysis/sr0_clean_analysis.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.accuracy import exact_match, normalize_prediction  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402
from scripts.analysis.reviewer_followups import (  # noqa: E402
    round0_majority_answer,
    paired_perm_p,
)

MODELS = {
    "gpt_oss_120b": "gpt-oss-120b",
    "llama_4_scout": "llama-4-scout",
    "gemma_4_31b": "gemma-4-31B-it",
}
REGIME_SPLIT = {
    "alphanli": "alphanli_scale_step_100.jsonl",
    "closed": "closed_ended_candidate_curated_100.jsonl",
    "social": "social_chemistry_broad_bc_100.jsonl",
    "mixed": "mixed_benchmark_150.jsonl",
    "math500": "math500_hard_50.jsonl",
}
MAIN_CONDS = ["flat", "theory_free_control", "transactional", "transformational",
              "situational", "transactional_accept_only", "transformational_directives_only"]
LEADERSHIP = ["transactional", "transformational", "situational"]


def bh(pairs):
    """Benjamini-Hochberg. pairs: list of (key, p). Returns {key: q}."""
    m = len(pairs)
    order = sorted(range(m), key=lambda i: pairs[i][1])
    q = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        idx = m - rank + 1
        val = min(prev, pairs[i][1] * m / idx)
        q[i] = val
        prev = val
    return {pairs[i][0]: q[i] for i in range(m)}


def analyze(model_slug, regime):
    d = ROOT / "outputs" / "runs" / f"sr0_{model_slug}_{regime}"
    ps, pt = d / "run_summaries.jsonl", d / "trace.jsonl"
    if not ps.exists():
        return None
    summaries = load_jsonl(ps)
    events = load_jsonl(pt) if pt.exists() else []
    gold = {r["task_id"]: r["gold_answer"] for r in load_jsonl(ROOT / "data" / "splits" / REGIME_SPLIT[regime])}
    ev_by_run = defaultdict(list)
    for e in events:
        ev_by_run[e["run_id"]].append(e)

    # shared R0-vote per (task_id, seed) from flat round-0 (identical across conds)
    r0maj = {}
    for s in summaries:
        if s.get("condition") != "flat":
            continue
        key = (s["task_id"], s.get("seed", 0))
        if key in r0maj:
            continue
        g = gold.get(s["task_id"])
        if g is None:
            continue
        m = round0_majority_answer(ev_by_run.get(s["run_id"], []), g)
        if m is not None:
            r0maj[key] = m
    norm_gold = {tid: normalize_prediction(g, g) for tid, g in gold.items()}
    r0_scores = [int(r0maj[k] == norm_gold[k[0]]) for k in r0maj if k[0] in norm_gold]
    r0_acc = sum(r0_scores) / len(r0_scores) if r0_scores else float("nan")

    # explore-action episodes from trace
    explored = defaultdict(set)  # condition -> set(run_id with >=1 explore)
    for e in events:
        if e.get("event_type") == "control_action" and e.get("payload") == "explore":
            explored[e["condition"]].add(e["run_id"])

    rows = {}
    for cond in MAIN_CONDS:
        diffs, finals, locks, rec_num, rec_den, rounds, n_expl, n_ep = [], [], [], 0, 0, [], 0, 0
        for s in summaries:
            if s.get("condition") != cond:
                continue
            key = (s["task_id"], s.get("seed", 0))
            g = gold.get(s["task_id"])
            if g is None or key not in r0maj:
                continue
            fc = int(exact_match(s["final_answer"], g))
            r0c = int(r0maj[key] == norm_gold.get(s["task_id"]))
            fn = normalize_prediction(s["final_answer"], g)
            finals.append(fc)
            diffs.append(fc - r0c)
            locks.append(int(fn == r0maj[key]))
            if not r0c:
                rec_den += 1
                rec_num += fc
            rounds.append(s.get("rounds_used", 0))
            if s["run_id"] in explored.get(cond, set()):
                n_expl += 1
            n_ep += 1
        if not finals:
            continue
        _, p = paired_perm_p(diffs)
        rows[cond] = {
            "final": sum(finals) / len(finals),
            "gain": sum(finals) / len(finals) - r0_acc,
            "p": p,
            "lockin": sum(locks) / len(locks),
            "recovery": (rec_num / rec_den) if rec_den else float("nan"),
            "explore": n_expl / n_ep,
            "rounds": sum(rounds) / len(rounds),
            "n": n_ep,
        }
    return {"r0_acc": r0_acc, "n_r0": len(r0_scores), "conds": rows}


def main():
    main_regimes = ["alphanli", "closed", "social", "mixed"]
    print("=" * 120)
    print("SHARED-ROUND-0 CLEAN ANALYSIS  (gain = final - shared_R0vote; clean interaction effect)")
    print("=" * 120)
    gain_tests = []  # (model, regime, controller, gain, p)
    f1 = {"within4": 0, "tot": 0, "deltas": []}
    for mslug in MODELS:
        for regime in main_regimes:
            res = analyze(mslug, regime)
            if not res:
                print(f"  [pending] sr0_{mslug}_{regime}")
                continue
            print(f"\n## {MODELS[mslug]} / {regime}   R0-vote={res['r0_acc']:.3f} (n={res['n_r0']})")
            print(f"   {'cond':30s} {'final':>6s} {'gain':>7s} {'p':>7s} {'lockin':>7s} {'recov':>7s} {'expl':>6s} {'rnds':>5s}")
            for cond, r in res["conds"].items():
                star = "*" if (r["gain"] >= 0.05) else " "
                print(f"   {cond:30s} {r['final']:>6.3f} {r['gain']:>+7.3f}{star} {r['p']:>7.3f} "
                      f"{r['lockin']:>7.3f} {r['recovery']:>7.3f} {r['explore']:>6.3f} {r['rounds']:>5.2f}")
                if cond == "transactional":
                    f1["tot"] += 1
                    f1["deltas"].append(round(r["gain"], 3))
                    if abs(r["gain"]) <= 0.04:
                        f1["within4"] += 1
                if cond in LEADERSHIP:
                    gain_tests.append(((MODELS[mslug], regime, cond), r["gain"], r["p"]))
    # F1
    print("\n" + "=" * 120)
    print(f"F1 (transactional final vs shared R0-vote): within +-4pp on {f1['within4']}/{f1['tot']} cells")
    print(f"    deltas: {f1['deltas']}")
    # F2 with BH
    cells = [(k, g, p) for (k, g, p) in gain_tests if g >= 0.05]
    print(f"\nF2 gain cells (>=5pp over shared R0-vote): {len(cells)} of {len(gain_tests)} leadership cells")
    if gain_tests:
        qmap = bh([(k, p) for (k, g, p) in gain_tests])
        for k, g, p in sorted(gain_tests, key=lambda x: -x[1])[:12]:
            tag = "GAIN" if g >= 0.05 else "    "
            print(f"    {tag} {k[0]:14s} {k[1]:9s} {k[2]:16s} gain={g:+.3f} p={p:.3f} q_BH={qmap[k]:.3f}")


if __name__ == "__main__":
    main()
