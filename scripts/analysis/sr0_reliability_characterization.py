"""Characterize the unreliable-round-0 regime DENSELY from existing sr0 data.

Decomposes each controller's gain over the shared round-0 vote into:
  gain ~= P(r0 wrong) * recovery  -  P(r0 right) * breakage
where
  recovery = P(final correct | round-0 majority wrong)
  breakage = P(final wrong   | round-0 majority correct)
Shows the gain is a systematic function of round-0 reliability, not 2 lucky cells.

Pools across the 12 main-matrix (model,regime) combinations.
Run: PYTHONPATH=. python3 scripts/analysis/sr0_reliability_characterization.py
"""
from __future__ import annotations
import sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.accuracy import exact_match, normalize_prediction  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402
from scripts.analysis.reviewer_followups import round0_majority_answer  # noqa: E402
from scripts.analysis.sr0_clean_analysis import MODELS  # noqa: E402

REG = ["alphanli", "closed", "social", "mixed"]
SPL = {"alphanli": "alphanli_scale_step_100.jsonl", "closed": "closed_ended_candidate_curated_100.jsonl",
       "social": "social_chemistry_broad_bc_100.jsonl", "mixed": "mixed_benchmark_150.jsonl"}
CTRL = ["flat", "theory_free_control", "transactional", "transformational", "situational"]


def cell(mslug, regime):
    d = ROOT / "outputs/runs" / f"sr0_{mslug}_{regime}"
    if not (d / "run_summaries.jsonl").exists():
        return None
    S = load_jsonl(d / "run_summaries.jsonl")
    gold = {r["task_id"]: r["gold_answer"] for r in load_jsonl(ROOT / "data/splits" / SPL[regime])}
    ng = {t: normalize_prediction(g, g) for t, g in gold.items()}
    E = collections.defaultdict(list)
    for e in load_jsonl(d / "trace.jsonl"):
        if e["event_type"] == "agent_output" and e["round_idx"] == 0:
            E[e["run_id"]].append(e)
    # shared r0 majority per (task,seed) from flat
    r0 = {}
    for s in S:
        if s["condition"] != "flat":
            continue
        m = round0_majority_answer(E.get(s["run_id"], []), gold[s["task_id"]])
        if m is not None:
            r0[(s["task_id"], s["seed"])] = m
    pw = sum(1 for k in r0 if r0[k] != ng[k[0]]) / len(r0)  # P(round-0 wrong)
    out = {"p_wrong": pw, "r0_acc": 1 - pw, "ctrl": {}}
    for c in CTRL:
        rec_n = rec_d = brk_n = brk_d = 0
        for s in S:
            if s["condition"] != c:
                continue
            k = (s["task_id"], s["seed"])
            if k not in r0:
                continue
            fc = int(exact_match(s["final_answer"], gold[s["task_id"]]))
            if r0[k] != ng[k[0]]:        # round-0 wrong
                rec_d += 1; rec_n += fc
            else:                         # round-0 right
                brk_d += 1; brk_n += (1 - fc)
        out["ctrl"][c] = {
            "recovery": rec_n / rec_d if rec_d else float("nan"),
            "breakage": brk_n / brk_d if brk_d else float("nan"),
            "rec_d": rec_d, "brk_d": brk_d,
        }
    return out


def main():
    print("=" * 100)
    print("RELIABILITY CHARACTERIZATION: gain ~= P(r0 wrong)*recovery - P(r0 right)*breakage")
    print("=" * 100)
    pooled = {c: {"rec_n": 0, "rec_d": 0, "brk_n": 0, "brk_d": 0} for c in CTRL}
    print(f"{'model/regime':26s} {'P(r0wrong)':>10s} | per controller: recovery / breakage")
    for mslug, mname in MODELS.items():
        for regime in REG:
            r = cell(mslug, regime)
            if not r:
                continue
            line = f"{mname[:14]+'/'+regime:26s} {r['p_wrong']:>10.3f} | "
            for c in CTRL:
                cc = r["ctrl"][c]
                if c in ("transactional", "transformational", "situational"):
                    line += f"{c[:5]}:{cc['recovery']:.2f}/{cc['breakage']:.2f} "
                # accumulate pooled (weight by counts) for ALL conditions
                pooled[c]["rec_n"] += cc["recovery"] * cc["rec_d"] if cc["rec_d"] else 0
                pooled[c]["rec_d"] += cc["rec_d"]
                pooled[c]["brk_n"] += cc["breakage"] * cc["brk_d"] if cc["brk_d"] else 0
                pooled[c]["brk_d"] += cc["brk_d"]
            print(line)
    print("\nPOOLED across all 12 main-matrix combinations (item-weighted):")
    print(f"  {'controller':18s} {'recovery':>9s} {'breakage':>9s}  (n_wrong / n_right items)")
    for c in CTRL:
        p = pooled[c]
        rec = p["rec_n"] / p["rec_d"] if p["rec_d"] else float("nan")
        brk = p["brk_n"] / p["brk_d"] if p["brk_d"] else float("nan")
        print(f"  {c:18s} {rec:>9.3f} {brk:>9.3f}  ({p['rec_d']} / {p['brk_d']})")
    print("\nReading: a controller's gain over the round-0 vote is positive only where P(r0 wrong)")
    print("is large enough that recovery outweighs breakage. situational has the highest recovery;")
    print("transactional ~never moves (recovery~breakage~0), i.e. it IS the vote.")


if __name__ == "__main__":
    main()
