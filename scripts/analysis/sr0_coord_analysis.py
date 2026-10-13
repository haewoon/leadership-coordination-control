"""Analysis of the unconstrained-coordinator (llm_coordinator) baseline runs.

For each (model, regime) combination this scores the coordinator condition with
the same conventions as sr0_clean_analysis.py and compares it, per matched
(task_id, seed) episode, against the shared round-0 vote and against the
original run's flat / situational conditions (paired permutation tests on the
per-episode correctness differences; the shared round 0 is injected verbatim
from the original logs, so all comparisons are confound-free at round 0).

Run after the coord runs complete:
  PYTHONPATH=. python3 scripts/analysis/sr0_coord_analysis.py
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
from scripts.analysis.sr0_clean_analysis import MODELS, REGIME_SPLIT, bh  # noqa: E402

MAIN_REGIMES = ["alphanli", "closed", "social", "mixed"]


def episode_scores(run_dir: Path, gold: dict, conditions: list[str]) -> dict:
    """condition -> {(task_id, seed): {'correct': int, 'final_norm': str, 'rounds': int, 'run_id': str}}"""
    summaries = load_jsonl(run_dir / "run_summaries.jsonl")
    out: dict[str, dict] = defaultdict(dict)
    for s in summaries:
        cond = s.get("condition")
        if cond not in conditions:
            continue
        g = gold.get(s["task_id"])
        if g is None:
            continue
        key = (s["task_id"], s.get("seed", 0))
        out[cond][key] = {
            "correct": int(exact_match(s["final_answer"], g)),
            "final_norm": normalize_prediction(s["final_answer"], g),
            "rounds": s.get("rounds_used", 0),
            "run_id": s["run_id"],
        }
    return out


def shared_r0_majority(run_dir: Path, gold: dict) -> dict:
    """(task_id, seed) -> normalized round-0 majority answer, from the flat rows."""
    summaries = load_jsonl(run_dir / "run_summaries.jsonl")
    events = load_jsonl(run_dir / "trace.jsonl")
    ev_by_run = defaultdict(list)
    for e in events:
        ev_by_run[e["run_id"]].append(e)
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
    return r0maj


def analyze(mslug: str, regime: str):
    orig_dir = ROOT / "outputs" / "runs" / f"sr0_{mslug}_{regime}"
    coord_dir = ROOT / "outputs" / "runs" / f"sr0_{mslug}_{regime}_coord"
    if not (coord_dir / "run_summaries.jsonl").exists():
        return None
    gold = {r["task_id"]: r["gold_answer"] for r in load_jsonl(ROOT / "data" / "splits" / REGIME_SPLIT[regime])}
    norm_gold = {tid: normalize_prediction(g, g) for tid, g in gold.items()}
    r0maj = shared_r0_majority(orig_dir, gold)
    coord = episode_scores(coord_dir, gold, ["llm_coordinator"])["llm_coordinator"]
    ref = episode_scores(orig_dir, gold, ["flat", "situational"])

    keys = sorted(k for k in coord if k in r0maj)
    r0_correct = {k: int(r0maj[k] == norm_gold[k[0]]) for k in keys}
    r0_acc = sum(r0_correct.values()) / len(keys)
    coord_acc = sum(coord[k]["correct"] for k in keys) / len(keys)
    _, p_vote = paired_perm_p([coord[k]["correct"] - r0_correct[k] for k in keys])

    res = {
        "n": len(keys),
        "r0_acc": r0_acc,
        "coord_acc": coord_acc,
        "gain_vote": coord_acc - r0_acc,
        "p_vote": p_vote,
        "lockin": sum(int(coord[k]["final_norm"] == r0maj[k]) for k in keys) / len(keys),
        "rounds": sum(coord[k]["rounds"] for k in keys) / len(keys),
    }
    den = sum(1 for k in keys if not r0_correct[k])
    res["recovery"] = (sum(coord[k]["correct"] for k in keys if not r0_correct[k]) / den) if den else float("nan")
    den_b = sum(1 for k in keys if r0_correct[k])
    res["breakage"] = (sum(1 - coord[k]["correct"] for k in keys if r0_correct[k]) / den_b) if den_b else float("nan")
    for cond in ("flat", "situational"):
        common = [k for k in keys if k in ref.get(cond, {})]
        if not common:
            continue
        diffs = [coord[k]["correct"] - ref[cond][k]["correct"] for k in common]
        _, p = paired_perm_p(diffs)
        res[f"delta_{cond}"] = sum(diffs) / len(common)
        res[f"p_{cond}"] = p
    return res


def main() -> None:
    print("=" * 118)
    print("UNCONSTRAINED-COORDINATOR BASELINE  (shared round 0 injected from original logs; paired comparisons)")
    print("=" * 118)
    header = (f"   {'combination':28s} {'n':>4s} {'R0vote':>7s} {'coord':>6s} {'gainV':>7s} {'pV':>6s} "
              f"{'dFlat':>7s} {'pF':>6s} {'dSitu':>7s} {'pS':>6s} {'lockin':>7s} {'recov':>6s} {'brk':>6s} {'rnds':>5s}")
    print(header)
    tests_vote = []
    for mslug in MODELS:
        for regime in MAIN_REGIMES:
            r = analyze(mslug, regime)
            name = f"{MODELS[mslug]}/{regime}"
            if r is None:
                print(f"   {name:28s} [pending]")
                continue
            print(f"   {name:28s} {r['n']:>4d} {r['r0_acc']:>7.3f} {r['coord_acc']:>6.3f} "
                  f"{r['gain_vote']:>+7.3f} {r['p_vote']:>6.3f} "
                  f"{r.get('delta_flat', float('nan')):>+7.3f} {r.get('p_flat', float('nan')):>6.3f} "
                  f"{r.get('delta_situational', float('nan')):>+7.3f} {r.get('p_situational', float('nan')):>6.3f} "
                  f"{r['lockin']:>7.3f} {r['recovery']:>6.3f} {r['breakage']:>6.3f} {r['rounds']:>5.2f}")
            tests_vote.append(((MODELS[mslug], regime), r["gain_vote"], r["p_vote"]))
    if tests_vote:
        qmap = bh([(k, p) for (k, _, p) in tests_vote])
        gains = [(k, g, p) for (k, g, p) in tests_vote if abs(g) >= 0.05]
        print(f"\n   |gain| >= 5pp vs shared R0-vote: {len(gains)} of {len(tests_vote)} combinations "
              f"(BH q-values across all {len(tests_vote)}):")
        for k, g, p in sorted(tests_vote, key=lambda x: -abs(x[1])):
            print(f"     {k[0]:14s} {k[1]:9s} gain={g:+.3f} p={p:.3f} q_BH={qmap[k]:.3f}")


if __name__ == "__main__":
    main()
