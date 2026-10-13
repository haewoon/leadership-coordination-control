"""Regenerate the two main-text figures from the shared-round-0 sr0 data:
  lockin_recovery_crossmodel.png : lock-in vs recovery, all (model,regime,condition) points.
  pareto_all.png                 : mean tokens/run vs accuracy, same points.

Run: PYTHONPATH=. python3 scripts/analysis/sr0_make_figures.py
"""
from __future__ import annotations
import sys, collections
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.accuracy import exact_match  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402
from scripts.analysis.sr0_clean_analysis import analyze, MODELS  # noqa: E402

FIGDIR = ROOT / "outputs" / "analysis" / "figs"
FIGDIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42})  # embed TrueType, not Type 3


def save_fig(fig, stem):
    """Write both the PNG (slides/README) and the vector PDF (paper)."""
    fig.savefig(FIGDIR / f"{stem}.png", dpi=300)
    fig.savefig(FIGDIR / f"{stem}.pdf")
    print(f"wrote {stem}.png + {stem}.pdf")
REG = ["alphanli", "closed", "social", "mixed"]
SPLIT = {"alphanli": "alphanli_scale_step_100.jsonl", "closed": "closed_ended_candidate_curated_100.jsonl",
         "social": "social_chemistry_broad_bc_100.jsonl", "mixed": "mixed_benchmark_150.jsonl"}
COND = ["flat", "theory_free_control", "transactional", "transformational", "situational", "transactional_accept_only"]
COL = {"flat": "#888888", "theory_free_control": "#1f77b4", "transactional": "#d62728",
       "transformational": "#2ca02c", "situational": "#9467bd", "transactional_accept_only": "#e377c2"}
MK = {"flat": "o", "theory_free_control": "s", "transactional": "D", "transformational": "^",
      "situational": "*", "transactional_accept_only": "v"}
LAB = {"flat": "Flat", "theory_free_control": "Theory-free", "transactional": "Transactional",
       "transformational": "Transformational", "situational": "Situational", "transactional_accept_only": "Transac. accept-only"}


def fig_lockin_recovery():
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for mslug in MODELS:
        for regime in REG:
            res = analyze(mslug, regime)
            if not res:
                continue
            for c in COND:
                v = res["conds"].get(c)
                if not v:
                    continue
                ax.scatter(v["lockin"], v["recovery"], c=COL[c], marker=MK[c],
                           s=110 if c == "situational" else 60, alpha=0.8, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Majority lock-in"); ax.set_ylabel("Recovery from incorrect round-0 majority")
    ax.set_title("Lock-in vs. recovery across 12 (model, regime) combinations")
    handles = [Line2D([0], [0], marker=MK[c], color="w", markerfacecolor=COL[c], markersize=9, label=LAB[c]) for c in COND]
    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); save_fig(fig, "lockin_recovery_crossmodel"); plt.close(fig)


def mean_tokens(mslug, regime):
    d = ROOT / "outputs" / "runs" / f"sr0_{mslug}_{regime}"
    tot = collections.defaultdict(float); n = collections.defaultdict(int)
    by_run_cond = {}
    for s in load_jsonl(d / "run_summaries.jsonl"):
        by_run_cond[s["run_id"]] = s["condition"]
    run_tok = collections.defaultdict(float)
    for e in load_jsonl(d / "trace.jsonl"):
        if e.get("event_type") == "agent_output":
            run_tok[e["run_id"]] += (e.get("prompt_tokens", 0) + e.get("completion_tokens", 0))
    for rid, tk in run_tok.items():
        c = by_run_cond.get(rid)
        if c:
            tot[c] += tk; n[c] += 1
    return {c: (tot[c] / n[c]) for c in tot if n[c]}


def fig_pareto():
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for mslug in MODELS:
        for regime in REG:
            res = analyze(mslug, regime)
            if not res:
                continue
            tok = mean_tokens(mslug, regime)
            for c in COND:
                v = res["conds"].get(c)
                if not v or c not in tok:
                    continue
                ax.scatter(tok[c], v["final"], c=COL[c], marker=MK[c],
                           s=110 if c == "situational" else 60, alpha=0.8, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Mean tokens per run (prompt + completion)"); ax.set_ylabel("Exact-match accuracy")
    ax.set_title("Cost vs. quality across the 3 x 4 model x regime matrix")
    handles = [Line2D([0], [0], marker=MK[c], color="w", markerfacecolor=COL[c], markersize=9, label=LAB[c]) for c in COND]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); save_fig(fig, "pareto_all"); plt.close(fig)


def fig_predictive():
    """Recovery advantage Delta_rec vs situational gain over flat, one point per
    (model, regime) setting. The predictive relationship the boundary rests on:
    gain appears only where Delta_rec is large and positive. The single significant
    point (llama-4-scout social) is marked; the rest cluster near zero gain."""
    from scripts.analysis.sr0_predictive_analysis import build_table, ols, spearman
    rows = [r for r in build_table() if r["delta_rec"] is not None]
    x = [r["delta_rec"] for r in rows]
    y = [r["gain_situ_over_flat"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
    ax.axvline(0, color="gray", lw=0.8, alpha=0.6)
    # OLS trend
    a, b = ols(x, y)
    xs = [min(x) - 0.02, max(x) + 0.02]
    ax.plot(xs, [a + b * xv for xv in xs], "-", color="#333333", lw=1.2, alpha=0.7,
            label=f"OLS (Spearman $\\rho$={spearman(x, y):+.2f})")
    tstyle = {"main": ("#1f77b4", "o", "main matrix"), "probe": ("#ff7f0e", "^", "boundary probe")}
    seen = set()
    for r in rows:
        col, mk, tl = tstyle[r["tier"]]
        sig = r["gain_situ_over_flat"] >= 0.05 and r["p_flat"] < 0.05
        ax.scatter(r["delta_rec"], r["gain_situ_over_flat"],
                   c=("#9467bd" if sig else col), marker=("*" if sig else mk),
                   s=(320 if sig else 70), alpha=0.85, edgecolors="white", linewidths=0.6,
                   zorder=5 if sig else 3, label=(tl if r["tier"] not in seen else None))
        seen.add(r["tier"])
        if sig:
            ax.annotate(f"{r['model'].replace('_','-')} {r['regime']}",
                        (r["delta_rec"], r["gain_situ_over_flat"]),
                        textcoords="offset points", xytext=(-8, 8), fontsize=8, ha="right")
    ax.set_xlabel("Recovery advantage  $\\Delta_{\\mathrm{rec}} = $ recovery(situational) $-$ recovery(flat)")
    ax.set_ylabel("Situational gain over flat (exact-match accuracy)")
    ax.set_title("Recovery advantage predicts controller gain (24 settings)")
    from matplotlib.lines import Line2D as _L
    handles = [_L([0], [0], color="#333333", lw=1.2, label=f"OLS (Spearman $\\rho$={spearman(x, y):+.2f})"),
               _L([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8, label="main matrix"),
               _L([0], [0], marker="^", color="w", markerfacecolor="#ff7f0e", markersize=8, label="boundary probe"),
               _L([0], [0], marker="*", color="w", markerfacecolor="#9467bd", markersize=13, label="significant gain")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); save_fig(fig, "recovery_advantage_predictive"); plt.close(fig)


def fig_reliability():
    """recovery vs breakage per (controller, cell): transactional at origin (=vote),
    situational/transformational up-left of the y=x line (recover more than they break)."""
    from scripts.analysis.sr0_reliability_characterization import cell, REG
    show = ["flat", "transactional", "transformational", "situational"]
    pts = {c: ([], []) for c in show}
    for mslug in MODELS:
        for regime in REG:
            r = cell(mslug, regime)
            if not r:
                continue
            for c in show:
                cc = r["ctrl"][c]
                if cc["recovery"] == cc["recovery"] and cc["breakage"] == cc["breakage"]:
                    pts[c][0].append(cc["breakage"]); pts[c][1].append(cc["recovery"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    lim = 0.32
    ax.plot([0, lim], [0, lim], "--", color="gray", lw=1.0, alpha=0.7, label="recovery = breakage")
    for c in show:
        ax.scatter(pts[c][0], pts[c][1], c=COL[c], marker=MK[c],
                   s=130 if c == "situational" else 70, alpha=0.85,
                   edgecolors="white", linewidths=0.5, label=LAB[c])
    ax.set_xlim(-0.01, lim); ax.set_ylim(-0.01, lim)
    ax.set_xlabel("Breakage  $P(\\mathrm{final\\ wrong}\\mid \\mathrm{round\\text{-}0\\ correct})$")
    ax.set_ylabel("Recovery  $P(\\mathrm{final\\ correct}\\mid \\mathrm{round\\text{-}0\\ wrong})$")
    ax.set_title("Recovery vs. breakage per (controller, combination)")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9); ax.grid(True, alpha=0.25)
    fig.tight_layout(); save_fig(fig, "recovery_breakage"); plt.close(fig)


if __name__ == "__main__":
    fig_lockin_recovery(); fig_pareto(); fig_reliability(); fig_predictive()
