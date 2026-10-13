"""Emit LaTeX table bodies for the shared-round-0 paper rewrite, straight from
sr0 outputs (no hand transcription). Tables:

  T1  Main accuracy: per model, 4 regimes, all conditions.
  T2  Per-controller gain over shared R0-vote (F1/F2), with BH q-values.
  T3  Behavioral signatures (lock-in/explore/recovery) on the social regime.
  T4  MATH-500 F3 (revise effect vs round-0 reliability); gpt-oss uses hi/3072.

Run: PYTHONPATH=. python3 scripts/analysis/sr0_make_tables.py
"""
from __future__ import annotations

import sys, json, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.accuracy import exact_match, normalize_prediction  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402
from scripts.analysis.reviewer_followups import round0_majority_answer, paired_perm_p  # noqa: E402
from scripts.analysis.sr0_clean_analysis import analyze, MODELS, bh  # noqa: E402

REG = ["alphanli", "closed", "social", "mixed"]
REGLAB = {"alphanli": "AlphaNLI", "closed": "Closed", "social": "Social", "mixed": "Mixed"}
MLAB = {"gpt-oss-120b": "\\texttt{gpt-oss-120b}", "llama-4-scout": "\\texttt{llama-4-scout}", "gemma-4-31B-it": "\\texttt{gemma-4-31B}"}
CONDS = [("flat", "Flat"), ("theory_free_control", "Theory-free control"),
         ("transactional", "Transactional"), ("transformational", "Transformational"),
         ("situational", "Situational"), ("transactional_accept_only", "\\quad accept-only"),
         ("transformational_directives_only", "\\quad directives-only")]

# cache analyze() results
A = {(m, r): analyze(slug, r) for slug, m in MODELS.items() for r in REG
     for slug2 in [slug]}  # placeholder
A = {}
for slug, m in MODELS.items():
    for r in REG:
        A[(m, r)] = (slug, analyze(slug, r))


def t1_accuracy():
    print("% ===== T1: main accuracy (rows=conditions, cols=regime), per model =====")
    for slug, m in MODELS.items():
        print(f"% --- {m} ---")
        print("\\begin{tabular}{lcccc}\\toprule")
        print("\\textbf{Condition} & " + " & ".join(REGLAB[r] for r in REG) + " \\\\ \\midrule")
        for c, lab in CONDS:
            cells = []
            for r in REG:
                res = A[(m, r)][1]
                v = res["conds"].get(c) if res else None
                cells.append(f"{v['final']:.3f}" if v else "---")
            print(f"{lab} & " + " & ".join(cells) + " \\\\")
        # R0-vote row
        r0 = [f"{A[(m,r)][1]['r0_acc']:.3f}" if A[(m,r)][1] else "---" for r in REG]
        print("\\midrule R0-vote & " + " & ".join(r0) + " \\\\ \\bottomrule")
        print(f"\\end{{tabular}}  % {m}\n")


def t2_gain():
    # BH across leadership combos
    tests = []
    for slug, m in MODELS.items():
        for r in REG:
            res = A[(m, r)][1]
            if not res:
                continue
            for c in ["transactional", "transformational", "situational"]:
                v = res["conds"].get(c)
                if v:
                    tests.append(((m, r, c), v["p"]))
    q = bh(tests)
    print("% ===== T2: per-controller gain over shared R0-vote (F1/F2) =====")
    print("\\begin{tabular}{llccc}\\toprule")
    print("\\textbf{Model} & \\textbf{Regime} & \\textbf{Transac.} & \\textbf{Transform.} & \\textbf{Situa.} \\\\ \\midrule")
    for slug, m in MODELS.items():
        for r in REG:
            res = A[(m, r)][1]
            if not res:
                continue
            cells = []
            for c in ["transactional", "transformational", "situational"]:
                v = res["conds"][c]
                g = v["gain"]
                s = f"{g:+.3f}"
                if g >= 0.05 and q[(m, r, c)] < 0.05:
                    s = f"\\textbf{{{g:+.3f}}}*"
                elif g >= 0.05:
                    s = f"\\textbf{{{g:+.3f}}}"
                cells.append(s)
            print(f"{MLAB[m]} & {REGLAB[r]} & " + " & ".join(cells) + " \\\\")
    print("\\bottomrule\\end{tabular}")
    print("% * = >=5pp gain AND BH q<0.05. R0-vote per (model,regime) in T1.\n")


def t3_behavioral_social():
    print("% ===== T3: behavioral signatures on SOCIAL (lock-in / explore / recovery) =====")
    print("\\begin{tabular}{l" + "ccc" * 3 + "}\\toprule")
    print(" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{MLAB[m]}}}" for m in MODELS.values()) + " \\\\")
    print("\\textbf{Condition} & " + " & ".join(["Lock-In & Explore & Recov."] * 3) + " \\\\ \\midrule")
    for c, lab in CONDS[:5]:
        cells = []
        for m in MODELS.values():
            res = A[(m, "social")][1]
            v = res["conds"].get(c) if res else None
            if v:
                cells += [f"{v['lockin']:.3f}", f"{v['explore']:.3f}", f"{v['recovery']:.3f}"]
            else:
                cells += ["---", "---", "---"]
        print(f"{lab} & " + " & ".join(cells) + " \\\\")
    print("\\bottomrule\\end{tabular}\n")


def t4_math():
    gold = {r["task_id"]: r["gold_answer"] for r in load_jsonl(ROOT / "data/splits/math500_hard_50.jsonl")}
    ng = {t: normalize_prediction(g, g) for t, g in gold.items()}
    DIRS = {"gpt-oss-120b": "sr0_gpt_oss_120b_math500_hi",
            "llama-4-scout": "sr0_llama_4_scout_math500",
            "gemma-4-31B-it": "sr0_gemma_4_31b_math500"}
    print("% ===== T4: MATH-500 F3 (revise vs round-0 reliability) =====")
    print("\\begin{tabular}{lcccccc}\\toprule")
    print("\\textbf{Model} & R0-vote & Flat & Transac. & Accept-only & $\\Delta_{\\text{revise}}$ & $p$ \\\\ \\midrule")
    for m, dn in DIRS.items():
        d = ROOT / "outputs/runs" / dn
        S = load_jsonl(d / "run_summaries.jsonl")
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
        r0acc = sum(int(r0[k] == ng[k[0]]) for k in r0) / len(r0)
        acc = {}
        for c in ["flat", "transactional", "transactional_accept_only"]:
            acc[c] = {(s["task_id"], s["seed"]): int(exact_match(s["final_answer"], gold[s["task_id"]]))
                      for s in S if s["condition"] == c}
        mean = lambda c: sum(acc[c].values()) / len(acc[c])
        k = set(acc["transactional_accept_only"]) & set(acc["transactional"])
        d_rev, p = paired_perm_p([acc["transactional_accept_only"][x] - acc["transactional"][x] for x in k])
        print(f"{MLAB[m]} & {r0acc:.3f} & {mean('flat'):.3f} & {mean('transactional'):.3f} & "
              f"{mean('transactional_accept_only'):.3f} & {d_rev:+.3f} & {p:.3f} \\\\")
    print("\\bottomrule\\end{tabular}")
    print("% Delta_revise = accept-only - transactional (effect of removing the revise action).\n")


if __name__ == "__main__":
    t1_accuracy(); t2_gain(); t3_behavioral_social(); t4_math()
