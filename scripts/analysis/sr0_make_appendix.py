"""Emit LaTeX for the main-data appendix tables of the sr0 rewrite:
  A1 Mixed-benchmark source breakdown (3 models in one table).
  A2 Stopping rounds (3 models x 4 regimes).
  A3 Pairwise paired-permutation highlights (recomputed on sr0 data).

Run: PYTHONPATH=. python3 scripts/analysis/sr0_make_appendix.py
"""
from __future__ import annotations
import sys, json, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.accuracy import exact_match, normalize_prediction  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402
from scripts.analysis.reviewer_followups import paired_perm_p, round0_majority_answer  # noqa: E402
from scripts.analysis.sr0_clean_analysis import analyze, MODELS  # noqa: E402

MLAB = {"gpt-oss-120b": "\\texttt{gpt-oss-120b}", "llama-4-scout": "\\texttt{llama-4-scout}", "gemma-4-31B-it": "\\texttt{gemma-4-31B}"}
SLUG = {v: k for k, v in MODELS.items()}
COND = [("flat", "Flat"), ("theory_free_control", "Theory-free control"), ("transactional", "Transactional leadership"),
        ("transformational", "Transformational leadership"), ("situational", "Situational leadership")]
SRC = {"commonsenseqa": "CSQA", "strategyqa": "StratQA", "alphanli": "AlphaNLI", "social_chemistry_101": "SocialChem"}


def mixed_breakdown():
    src_of = {r["task_id"]: (r.get("metadata") or {}).get("source") for r in load_jsonl(ROOT / "data/splits/mixed_benchmark_150.jsonl")}
    gold = {r["task_id"]: r["gold_answer"] for r in load_jsonl(ROOT / "data/splits/mixed_benchmark_150.jsonl")}
    print("% ===== A1 mixed source breakdown (3 models) =====")
    print("\\begin{tabular}{ll" + "ccccc" + "}\\toprule")
    print("\\textbf{Model} & \\textbf{Condition} & \\textbf{AlphaNLI} & \\textbf{CSQA} & \\textbf{SocialChem} & \\textbf{StratQA} & \\textbf{Worst} \\\\ \\midrule")
    order = ["alphanli", "commonsenseqa", "social_chemistry_101", "strategyqa"]
    for mname in MODELS.values():
        S = load_jsonl(ROOT / "outputs/runs" / f"sr0_{SLUG[mname]}_mixed" / "run_summaries.jsonl")
        for ci, (c, lab) in enumerate(COND):
            per = {s: [] for s in order}
            for s in S:
                if s["condition"] != c:
                    continue
                so = src_of.get(s["task_id"])
                if so in per:
                    per[so].append(int(exact_match(s["final_answer"], gold[s["task_id"]])))
            accs = {s: (sum(v) / len(v) if v else float("nan")) for s, v in per.items()}
            worst = min(a for a in accs.values() if a == a)
            head = MLAB[mname] if ci == 0 else ""
            print(f"{head} & {lab} & {accs['alphanli']:.3f} & {accs['commonsenseqa']:.3f} & {accs['social_chemistry_101']:.3f} & {accs['strategyqa']:.3f} & {worst:.3f} \\\\")
        print("\\midrule")
    print("\\bottomrule\\end{tabular}\n")


def stopping_rounds():
    print("% ===== A2 stopping rounds (3 models x 4 regimes) =====")
    print("\\begin{tabular}{ll cccc}\\toprule")
    print("\\textbf{Model} & \\textbf{Condition} & Closed & AlphaNLI & Social & Mixed \\\\ \\midrule")
    REG = ["closed", "alphanli", "social", "mixed"]
    for mname in MODELS.values():
        res = {r: analyze(SLUG[mname], r) for r in REG}
        for ci, (c, lab) in enumerate(COND):
            cells = []
            for r in REG:
                v = res[r]["conds"].get(c) if res[r] else None
                cells.append(f"{v['rounds']:.2f}" if v else "---")
            head = MLAB[mname] if ci == 0 else ""
            print(f"{head} & {lab} & " + " & ".join(cells) + " \\\\")
        print("\\midrule")
    print("\\bottomrule\\end{tabular}\n")


def pairwise():
    """Recompute representative significant pairs on sr0 data (accuracy + lock-in)."""
    REGSPLIT = {"social": "social_chemistry_broad_bc_100.jsonl", "mixed": "mixed_benchmark_150.jsonl",
                "alphanli": "alphanli_scale_step_100.jsonl", "closed": "closed_ended_candidate_curated_100.jsonl"}
    print("% ===== A3 pairwise highlights (recomputed) =====")
    print("\\begin{tabular}{lllcc}\\toprule")
    print("\\textbf{Model / Regime} & \\textbf{Pair (A vs.\\ B)} & \\textbf{Metric} & \\textbf{Diff} & \\textbf{$p$} \\\\ \\midrule")
    jobs = [("llama-4-scout", "social", "situational", "transactional", "acc"),
            ("llama-4-scout", "social", "transformational", "transactional", "acc"),
            ("llama-4-scout", "social", "situational", "flat", "acc"),
            ("gpt-oss-120b", "social", "situational", "transactional", "acc"),
            ("llama-4-scout", "mixed", "transactional", "situational", "lockin"),
            ("gpt-oss-120b", "mixed", "transactional", "situational", "lockin"),
            ("gemma-4-31B-it", "mixed", "transactional", "situational", "lockin")]
    for mname, regime, A, B, metric in jobs:
        d = ROOT / "outputs/runs" / f"sr0_{SLUG[mname]}_{regime}"
        S = load_jsonl(d / "run_summaries.jsonl")
        gold = {r["task_id"]: r["gold_answer"] for r in load_jsonl(ROOT / "data/splits" / REGSPLIT[regime])}
        if metric == "lockin":
            E = collections.defaultdict(list)
            for e in load_jsonl(d / "trace.jsonl"):
                if e["event_type"] == "agent_output" and e["round_idx"] == 0:
                    E[e["run_id"]].append(e)
            r0 = {}
            for s in S:
                if s["condition"] != "flat":
                    continue
                m = round0_majority_answer(E[s["run_id"]], gold[s["task_id"]])
                if m is not None:
                    r0[(s["task_id"], s["seed"])] = m
        def vec(cond):
            out = {}
            for s in S:
                if s["condition"] != cond:
                    continue
                g = gold.get(s["task_id"])
                if g is None:
                    continue
                k = (s["task_id"], s["seed"])
                if metric == "acc":
                    out[k] = int(exact_match(s["final_answer"], g))
                else:
                    if k in r0:
                        out[k] = int(normalize_prediction(s["final_answer"], g) == r0[k])
            return out
        va, vb = vec(A), vec(B)
        keys = set(va) & set(vb)
        diffs = [va[k] - vb[k] for k in keys]
        obs, p = paired_perm_p(diffs)
        mlab = "lock-in" if metric == "lockin" else "acc"
        ps = "<0.0001" if p < 1e-4 else f"{p:.3f}"
        print(f"{MLAB[mname]} / {regime} & {A} vs.\\ {B} & {mlab} & ${obs:+.3f}$ & ${ps}$ \\\\")
    print("\\bottomrule\\end{tabular}\n")


if __name__ == "__main__":
    mixed_breakdown(); stopping_rounds(); pairwise()
