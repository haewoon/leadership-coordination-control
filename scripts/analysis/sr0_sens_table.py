"""Situational sensitivity-analysis table: on the social regime, does the situational
gain survive perturbing the reopening thresholds and the conflict-marker lexicon?

Emits a LaTeX table: per model, round-0 vote + flat + the four situational arms
(main / split3 / lenient / altlex), each with accuracy, gain over flat, and recovery.

Run: PYTHONPATH=. python3 scripts/analysis/sr0_sens_table.py
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

SPLIT = "data/splits/social_chemistry_broad_bc_100.jsonl"
ARMS = ["situational", "situational_split3", "situational_lenient", "situational_altlex"]
ALAB = {"situational": "main", "situational_split3": "split-3-only",
        "situational_lenient": "lenient", "situational_altlex": "alt-lexicon"}
MODELS = [("gpt_oss_120b", "gpt-oss-120b"), ("llama_4_scout", "llama-4-scout"), ("gemma_4_31b", "gemma-4-31B")]


def cell(mslug):
    d = ROOT / "outputs/runs" / f"sr0_{mslug}_social_sens"
    S = load_jsonl(d / "run_summaries.jsonl")
    gold = {r["task_id"]: r["gold_answer"] for r in load_jsonl(ROOT / SPLIT)}
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
    vote = sum(int(r0[k] == ng[k[0]]) for k in r0) / len(r0)
    per, rec = {}, {}
    for c in ["flat"] + ARMS:
        runs, rn, rd = {}, 0, 0
        for s in S:
            if s["condition"] != c:
                continue
            key = (s["task_id"], s["seed"])
            runs[key] = int(exact_match(s["final_answer"], gold[s["task_id"]]))
            if key in r0 and r0[key] != ng[key[0]]:
                rd += 1; rn += runs[key]
        per[c] = runs
        rec[c] = rn / rd if rd else float("nan")
    acc = {c: sum(per[c].values()) / len(per[c]) for c in per}
    gains = {}
    for c in ARMS:
        k = set(per[c]) & set(per["flat"])
        d_, p = paired_perm_p([per[c][x] - per["flat"][x] for x in k])
        gains[c] = (acc[c] - acc["flat"], p)
    return vote, acc, rec, gains


def main():
    print("% ===== situational sensitivity (social regime) =====")
    print("\\begin{tabular}{llccc}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{Situational variant} & Acc.\\ & $\\Delta$ vs flat & Recov. \\\\")
    print("\\midrule")
    for mslug, mlab in MODELS:
        vote, acc, rec, gains = cell(mslug)
        print(f"\\texttt{{{mlab}}} & (flat / round-0 vote {vote:.3f}) & {acc['flat']:.3f} & --- & {rec['flat']:.3f} \\\\")
        for c in ARMS:
            g, p = gains[c]
            sig = "\\sigstar" if p < 0.05 and g >= 0.05 else ""
            print(f" & {ALAB[c]} & {acc[c]:.3f} & ${g:+.3f}${sig} & {rec[c]:.3f} \\\\")
        print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")
    # plain summary for the write-up
    print("\n% --- plain summary ---")
    for mslug, mlab in MODELS:
        vote, acc, rec, gains = cell(mslug)
        gvals = [gains[c][0] for c in ARMS]
        print(f"{mlab}: flat={acc['flat']:.3f} vote={vote:.3f} | "
              + " ".join(f"{ALAB[c]}={acc[c]:.3f}({gains[c][0]:+.3f})" for c in ARMS)
              + f" | gain range [{min(gvals):+.3f},{max(gvals):+.3f}]")


if __name__ == "__main__":
    main()
