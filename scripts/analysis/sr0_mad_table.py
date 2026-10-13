"""Full-matrix MAD baseline table: MAD vs flat / situational / transformational on the
same shared round-0, across all 12 (model, regime) combinations. Confirms MAD does not
beat the leadership controllers and that the contingency boundary holds against the
strong peer-debate baseline.

Run: PYTHONPATH=. python3 scripts/analysis/sr0_mad_table.py
"""
from __future__ import annotations
import collections
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.accuracy import exact_match, normalize_prediction  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402
from scripts.analysis.reviewer_followups import round0_majority_answer, paired_perm_p  # noqa: E402

SPLIT = {"alphanli": "data/splits/alphanli_scale_step_100.jsonl",
         "closed": "data/splits/closed_ended_candidate_curated_100.jsonl",
         "social": "data/splits/social_chemistry_broad_bc_100.jsonl",
         "mixed": "data/splits/mixed_benchmark_150.jsonl"}
RLAB = {"alphanli": "AlphaNLI", "closed": "Closed", "social": "Social", "mixed": "Mixed"}
CONDS = ["flat", "mad", "situational", "transformational"]
MODELS = [("gpt_oss_120b", "gpt-oss-120b"), ("llama_4_scout", "llama-4-scout"), ("gemma_4_31b", "gemma-4-31B")]
REGIMES = ["alphanli", "closed", "social", "mixed"]


def cell(mslug, regime):
    d = ROOT / "outputs/runs" / f"sr0_{mslug}_{regime}_mad"
    if not (d / "run_summaries.jsonl").exists():
        return None
    S = load_jsonl(d / "run_summaries.jsonl")
    gold = {r["task_id"]: r["gold_answer"] for r in load_jsonl(ROOT / SPLIT[regime])}
    ng = {t: normalize_prediction(g, g) for t, g in gold.items()}
    E = collections.defaultdict(list)
    for e in load_jsonl(d / "trace.jsonl"):
        if e["event_type"] == "agent_output" and e["round_idx"] == 0:
            E[e["run_id"]].append(e)
    r0 = {}
    for s in S:
        if s["condition"] != "flat":
            continue
        mm = round0_majority_answer(E[s["run_id"]], gold[s["task_id"]])
        if mm is not None:
            r0[(s["task_id"], s["seed"])] = mm
    vote = sum(int(r0[k] == ng[k[0]]) for k in r0) / len(r0) if r0 else float("nan")
    per = {}
    for c in CONDS:
        runs = {(s["task_id"], s["seed"]): int(exact_match(s["final_answer"], gold[s["task_id"]]))
                for s in S if s["condition"] == c}
        per[c] = runs
    acc = {c: (sum(per[c].values()) / len(per[c]) if per[c] else float("nan")) for c in CONDS}
    # MAD vs situational (paired)
    k = set(per["mad"]) & set(per["situational"])
    d_ms, p_ms = paired_perm_p([per["mad"][x] - per["situational"][x] for x in k]) if k else (float("nan"), float("nan"))
    return vote, acc, d_ms, p_ms


def main():
    print("% ===== full-matrix MAD baseline =====")
    print("\\begin{tabular}{llccccc}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{Regime} & R0-vote & Flat & MAD & Situ. & Transf. \\\\")
    print("\\midrule")
    mad_beats_best = 0
    rows_checked = 0
    for mslug, mlab in MODELS:
        for regime in REGIMES:
            r = cell(mslug, regime)
            if r is None:
                print(f"% [pending] {mslug} {regime}")
                continue
            vote, acc, d_ms, p_ms = r
            best_lead = max(acc["situational"], acc["transformational"])
            rows_checked += 1
            if acc["mad"] > best_lead + 1e-9:
                mad_beats_best += 1
            print(f"\\texttt{{{mlab}}} & {RLAB[regime]} & {vote:.3f} & {acc['flat']:.3f} & "
                  f"{acc['mad']:.3f} & {acc['situational']:.3f} & {acc['transformational']:.3f} \\\\")
        print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\n% MAD beats the better leadership controller in {mad_beats_best}/{rows_checked} combinations.")
    # plain summary
    print("\n% --- plain summary ---")
    for mslug, mlab in MODELS:
        for regime in REGIMES:
            r = cell(mslug, regime)
            if r is None:
                continue
            vote, acc, d_ms, p_ms = r
            print(f"{mlab:13s} {regime:9s} vote={vote:.3f} flat={acc['flat']:.3f} mad={acc['mad']:.3f} "
                  f"sit={acc['situational']:.3f} transf={acc['transformational']:.3f} | "
                  f"MAD-sit={acc['mad']-acc['situational']:+.3f}(p={p_ms:.2f})")


if __name__ == "__main__":
    main()
