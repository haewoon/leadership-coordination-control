"""LaTeX tables for the boundary-probe regimes (B salvage): ANLI-R3, Winogrande,
Scruples-contested. Emits (1) the body typology table tab:boundary and (2) per-regime
appendix tables tab:appendix_boundary_*.

Run: PYTHONPATH=. python3 scripts/analysis/sr0_boundary_tables.py
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

CONDS = ["flat", "theory_free_control", "transactional", "transformational", "situational"]
CLAB = {"flat": "Flat", "theory_free_control": "Theory-free", "transactional": "Transactional",
        "transformational": "Transformational", "situational": "Situational"}
MODELS = [("gpt_oss_120b", "gpt-oss-120b"), ("llama_4_scout", "llama-4-scout"), ("gemma_4_31b", "gemma-4-31B")]
REGIMES = {"anli": ("ANLI-R3", "data/splits/anli_r3_100.jsonl"),
           "winogrande": ("Winogrande", "data/splits/winogrande_xl_100.jsonl"),
           "scruples": ("Scruples", "data/splits/scruples_contested_99.jsonl")}


def cell(dirn, split):
    d = ROOT / "outputs/runs" / dirn
    if not (d / "run_summaries.jsonl").exists():
        return None
    S = load_jsonl(d / "run_summaries.jsonl")
    gold = {r["task_id"]: r["gold_answer"] for r in load_jsonl(ROOT / split)}
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
    pwrong = sum(1 for k in r0 if r0[k] != ng[k[0]]) / len(r0)
    per, rec, lock = {}, {}, {}
    for c in CONDS:
        runs, rn, rd, lk, ln = {}, 0, 0, 0, 0
        for s in S:
            if s["condition"] != c:
                continue
            key = (s["task_id"], s["seed"])
            runs[key] = int(exact_match(s["final_answer"], gold[s["task_id"]]))
            if key in r0:
                fn = normalize_prediction(s["final_answer"], gold[s["task_id"]])
                lk += int(fn == r0[key]); ln += 1
                if r0[key] != ng[key[0]]:
                    rd += 1; rn += runs[key]
        per[c] = runs
        rec[c] = rn / rd if rd else float("nan")
        lock[c] = lk / ln if ln else float("nan")
    acc = {c: sum(per[c].values()) / len(per[c]) for c in CONDS}
    k = set(per["situational"]) & set(per["flat"])
    ds, ps = paired_perm_p([per["situational"][x] - per["flat"][x] for x in k])
    return dict(vote=vote, pwrong=pwrong, acc=acc, rec=rec, lock=lock, ds=ds, ps=ps)


def appendix_tables():
    for reg, (lab, split) in REGIMES.items():
        cells = {ms: cell(f"sr0_{ms}_{reg}", split) for ms, _ in MODELS}
        print(f"% ===== appendix: {lab} =====")
        print("\\begin{tabular}{lccc}")
        print("\\toprule")
        print(f"\\textbf{{Condition}} & \\texttt{{gpt-oss-120b}} & \\texttt{{llama-4-scout}} & \\texttt{{gemma-4-31B}} \\\\")
        print("\\midrule")
        print("Round-0 vote & " + " & ".join(f"{cells[ms]['vote']:.3f}" for ms, _ in MODELS) + " \\\\")
        print("\\midrule")
        for c in CONDS:
            print(f"{CLAB[c]} & " + " & ".join(f"{cells[ms]['acc'][c]:.3f}" for ms, _ in MODELS) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}\n")


def body_typology():
    # representative (regime, model) rows; recovery shown to 2dp and Delta_rec from the 2dp values
    rows = [("Social", "llama_4_scout", "social", "data/splits/social_chemistry_broad_bc_100.jsonl"),
            ("Social", "gemma_4_31b", "social", "data/splits/social_chemistry_broad_bc_100.jsonl"),
            ("Scruples", "llama_4_scout", "scruples", "data/splits/scruples_contested_99.jsonl"),
            ("Scruples", "gpt_oss_120b", "scruples", "data/splits/scruples_contested_99.jsonl"),
            ("ANLI-R3", "llama_4_scout", "anli", "data/splits/anli_r3_100.jsonl"),
            ("Winogrande", "gemma_4_31b", "winogrande", "data/splits/winogrande_xl_100.jsonl"),
            ("MATH-500", "llama_4_scout", "math", "data/splits/math500_hard_50.jsonl"),
            ("MATH-500", "gemma_4_31b", "math", "data/splits/math500_hard_50.jsonl")]
    mathdir = {"gpt_oss_120b": "sr0_gpt_oss_120b_math500_hi", "llama_4_scout": "sr0_llama_4_scout_math500",
               "gemma_4_31b": "sr0_gemma_4_31b_math500"}
    mlab = {"gpt_oss_120b": "gpt-oss-120b", "llama_4_scout": "llama-4-scout", "gemma_4_31b": "gemma-4-31B"}
    print("% ===== body: typology table tab:boundary =====")
    print("\\begin{tabular}{llccccc}")
    print("\\toprule")
    print("\\textbf{Regime} & \\textbf{Model} & $P(r_0\\text{ wrong})$ & Recov.\\ flat & Recov.\\ situ. & $\\Delta_{\\text{rec}}$ & $\\Delta$acc \\\\")
    print("\\midrule")
    for lab, ms, reg, split in rows:
        dirn = mathdir[ms] if reg == "math" else f"sr0_{ms}_{reg}"
        r = cell(dirn, split)
        rf, rsv = round(r["rec"]["flat"], 2), round(r["rec"]["situational"], 2)
        drec = rsv - rf
        dacc = r["acc"]["situational"] - r["acc"]["flat"]
        sig = "*" if r["ps"] < 0.05 and dacc >= 0.05 else ""
        print(f"{lab} & \\texttt{{{mlab[ms]}}} & {r['pwrong']:.2f} & {rf:.2f} & {rsv:.2f} & {drec:+.2f} & {dacc:+.3f}{sig} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    print("#" * 30, "BODY TYPOLOGY", "#" * 30)
    body_typology()
    print("\n" + "#" * 30, "APPENDIX TABLES", "#" * 30)
    appendix_tables()
