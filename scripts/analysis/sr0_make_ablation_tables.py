"""LaTeX for the appendix ablation tables from the self-hosted sr0 ablation re-runs:
  MAD baseline + situational-random-switch; team-size 5; round-budget 6; inclusive.
Run: PYTHONPATH=. python3 scripts/analysis/sr0_make_ablation_tables.py
Gracefully skips configs whose run is not yet complete.
"""
from __future__ import annotations
import sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.accuracy import exact_match, normalize_prediction  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402
from scripts.analysis.reviewer_followups import round0_majority_answer, paired_perm_p  # noqa: E402

MODELS = {"gpt_oss_120b": "gpt-oss-120b", "llama_4_scout": "llama-4-scout", "gemma_4_31b": "gemma-4-31B-it"}
MLAB = {k: f"\\texttt{{{v}}}" for k, v in [("gpt_oss_120b", "gpt-oss-120b"), ("llama_4_scout", "llama-4-scout"), ("gemma_4_31b", "gemma-4-31B")]}
SPL = {"alphanli": "alphanli_scale_step_100.jsonl", "social": "social_chemistry_broad_bc_100.jsonl", "mixed": "mixed_benchmark_150.jsonl"}


def load(dirname):
    d = ROOT / "outputs/runs" / dirname
    p = d / "run_summaries.jsonl"
    return (load_jsonl(p) if p.exists() else None), d


def gold_for(regime):
    return {r["task_id"]: r["gold_answer"] for r in load_jsonl(ROOT / "data/splits" / SPL[regime])}


def acc(S, gold, cond):
    v = [int(exact_match(s["final_answer"], gold[s["task_id"]])) for s in S if s["condition"] == cond and s["task_id"] in gold]
    return (sum(v) / len(v)) if v else float("nan")


def lockin_explore(S, d, gold, cond):
    E = collections.defaultdict(list); expl = set()
    for e in load_jsonl(d / "trace.jsonl"):
        if e["event_type"] == "agent_output" and e["round_idx"] == 0:
            E[e["run_id"]].append(e)
        if e["event_type"] == "control_action" and e.get("payload") == "explore":
            expl.add(e["run_id"])
    lk, ne, n = [], 0, 0
    for s in S:
        if s["condition"] != cond:
            continue
        m = round0_majority_answer(E.get(s["run_id"], []), gold[s["task_id"]])
        if m is not None:
            lk.append(int(normalize_prediction(s["final_answer"], gold[s["task_id"]]) == m))
        ne += s["run_id"] in expl; n += 1
    return (sum(lk) / len(lk) if lk else float("nan")), (ne / n if n else float("nan"))


def t_madsr():
    print("% ===== MAD baseline + situational-random-switch =====")
    print("\\begin{tabular}{ll ccc}\\toprule")
    print("\\textbf{Model} & \\textbf{Condition} & AlphaNLI & Social & Mixed \\\\ \\midrule")
    rows = ["flat", "transactional", "situational", "mad", "situational_random_switch"]
    rl = {"flat": "Flat", "transactional": "Transactional", "situational": "Situational", "mad": "MAD (peer-to-peer)", "situational_random_switch": "Situational random-switch"}
    ok = True
    for slug in MODELS:
        Ss = {}
        for r in ["alphanli", "social", "mixed"]:
            S, _ = load(f"sr0_{slug}_madsr_{r}")
            if S is None:
                ok = False
            Ss[r] = S
        for ci, c in enumerate(rows):
            cells = []
            for r in ["alphanli", "social", "mixed"]:
                cells.append(f"{acc(Ss[r], gold_for(r), c):.3f}" if Ss[r] else "---")
            head = MLAB[slug] if ci == 0 else ""
            print(f"{head} & {rl[c]} & " + " & ".join(cells) + " \\\\")
        print("\\midrule")
    print("\\bottomrule\\end{tabular}")
    print("% complete" if ok else "% PARTIAL")
    print()


def t_inclusive():
    print("% ===== Inclusive (focused social) =====")
    print("\\begin{tabular}{lccc}\\toprule")
    print("\\textbf{Model} & Flat & Situational & Inclusive \\\\ \\midrule")
    for slug in MODELS:
        S, _ = load(f"sr0_{slug}_inclusive_social")
        if S is None:
            print(f"{MLAB[slug]} & --- & --- & --- \\\\"); continue
        g = gold_for("social")
        print(f"{MLAB[slug]} & {acc(S,g,'flat'):.3f} & {acc(S,g,'situational'):.3f} & {acc(S,g,'inclusive'):.3f} \\\\")
    print("\\bottomrule\\end{tabular}\n")


def t_team5():
    print("% ===== Team-size 5 =====")
    print("\\begin{tabular}{ll cccccc}\\toprule")
    print(" & & \\multicolumn{3}{c}{AlphaNLI} & \\multicolumn{3}{c}{Social} \\\\")
    print("\\textbf{Model} & \\textbf{Condition} & Acc & Lock-In & Explore & Acc & Lock-In & Explore \\\\ \\midrule")
    rows = ["flat", "transactional", "transformational", "situational"]
    rl = {"flat": "Flat", "transactional": "Transactional", "transformational": "Transformational", "situational": "Situational"}
    expected = 5 * 3 * 100  # 5 cond x 3 seeds x 100
    for slug in MODELS:
        Ss = {}
        for r in ["alphanli", "social"]:
            S, d = load(f"sr0_{slug}_team5_{r}")
            Ss[r] = (S, d, len(S) if S else 0)
        for ci, c in enumerate(rows):
            cells = []
            for r in ["alphanli", "social"]:
                S, d, nn = Ss[r]
                if S and nn >= expected:
                    g = gold_for(r); a = acc(S, g, c); lk, ex = lockin_explore(S, d, g, c)
                    cells += [f"{a:.3f}", f"{lk:.3f}", f"{ex:.3f}"]
                else:
                    cells += ["---", "---", "---"]
            head = MLAB[slug] if ci == 0 else ""
            print(f"{head} & {rl[c]} & " + " & ".join(cells) + " \\\\")
        print("\\midrule")
    print("\\bottomrule\\end{tabular}\n")


def t_rb6():
    print("% ===== Round-budget 6 =====")
    print("\\begin{tabular}{ll cc}\\toprule")
    print("\\textbf{Model} & \\textbf{Condition} & AlphaNLI Acc & Social Acc \\\\ \\midrule")
    rows = ["flat", "transactional", "transformational", "situational"]
    rl = {"flat": "Flat", "transactional": "Transactional", "transformational": "Transformational", "situational": "Situational"}
    expected = 5 * 3 * 100
    for slug in MODELS:
        Ss = {r: load(f"sr0_{slug}_rb6_{r}")[0] for r in ["alphanli", "social"]}
        for ci, c in enumerate(rows):
            cells = []
            for r in ["alphanli", "social"]:
                S = Ss[r]
                cells.append(f"{acc(S, gold_for(r), c):.3f}" if S and len(S) >= expected else "---")
            head = MLAB[slug] if ci == 0 else ""
            print(f"{head} & {rl[c]} & " + " & ".join(cells) + " \\\\")
        print("\\midrule")
    print("\\bottomrule\\end{tabular}\n")


if __name__ == "__main__":
    t_madsr(); t_inclusive(); t_rb6(); t_team5()
