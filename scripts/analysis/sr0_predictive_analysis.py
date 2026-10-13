"""Formal predictive analysis of the readiness-gap boundary from existing sr0 data.

Tests whether the three readiness-gap predictors
  (a) round-0 majority reliability, (b) recoverability, (c) flat-interaction recovery
predict controller gains across settings, with effect sizes, uncertainty estimates,
and a held-out (leave-one-model-out / leave-one-task-out) evaluation. No new runs:
assembles a tidy per-setting table across all 24 (model, regime) settings
(12 main matrix + 12 boundary probes) and reports:
  - Spearman correlations of each predictor (and the composite Delta_rec) with gain
  - a subset-robustness check (main-matrix-only vs probes-only) against probe selection
  - bootstrap CIs on per-setting gains and on the correlations
  - leave-one-model-out and leave-one-task-out predicted-vs-actual, decomposed into the
    null settings (which generalize) and the single positive setting (which does not)

Outputs the tidy table to outputs/analysis/sr0_predictive_table.{csv,json}.
build_table() is imported by sr0_make_figures.py::fig_predictive.

Run: PYTHONPATH=. python3 scripts/analysis/sr0_predictive_analysis.py
Pure-python (no numpy); reuses the repo scorers + round-0 majority + paired permutation.
The headline reproduces the paper's reported gain CI exactly (pipeline validated).
"""
from __future__ import annotations
import sys, csv, json, random, collections
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.accuracy import exact_match, normalize_prediction  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402
from scripts.analysis.reviewer_followups import round0_majority_answer, paired_perm_p  # noqa: E402

SPLITS = {
    "alphanli": "alphanli_scale_step_100.jsonl",
    "closed": "closed_ended_candidate_curated_100.jsonl",
    "social": "social_chemistry_broad_bc_100.jsonl",
    "mixed": "mixed_benchmark_150.jsonl",
    "anli": "anli_r3_100.jsonl",
    "winogrande": "winogrande_xl_100.jsonl",
    "scruples": "scruples_contested_99.jsonl",
    "math500": "math500_hard_50.jsonl",
}
MODELS = ["gpt_oss_120b", "llama_4_scout", "gemma_4_31b"]
MAIN_REG = ["alphanli", "closed", "social", "mixed"]
PROBE_REG = ["anli", "winogrande", "scruples"]
MATHDIR = {"gpt_oss_120b": "sr0_gpt_oss_120b_math500_hi",
           "llama_4_scout": "sr0_llama_4_scout_math500",
           "gemma_4_31b": "sr0_gemma_4_31b_math500"}
CONDS = ["flat", "theory_free_control", "transactional", "transformational", "situational"]
OUTDIR = ROOT / "outputs" / "analysis"


def build_settings():
    S = []
    for m in MODELS:
        for r in MAIN_REG:
            S.append((m, r, f"sr0_{m}_{r}", SPLITS[r], "main"))
        for r in PROBE_REG:
            S.append((m, r, f"sr0_{m}_{r}", SPLITS[r], "probe"))
        S.append((m, "math500", MATHDIR[m], SPLITS["math500"], "probe"))
    return S


def extract(dirn, split):
    """Per-setting metrics + per-(task,seed) correctness vectors for paired stats."""
    d = ROOT / "outputs/runs" / dirn
    if not (d / "run_summaries.jsonl").exists():
        return None
    S = load_jsonl(d / "run_summaries.jsonl")
    gold = {r["task_id"]: r["gold_answer"] for r in load_jsonl(ROOT / "data/splits" / split)}
    ng = {t: normalize_prediction(g, g) for t, g in gold.items()}
    E = collections.defaultdict(list)
    for e in load_jsonl(d / "trace.jsonl"):
        if e["event_type"] == "agent_output" and e["round_idx"] == 0:
            E[e["run_id"]].append(e)
    r0maj = {}
    for s in S:
        if s["condition"] != "flat":
            continue
        m = round0_majority_answer(E.get(s["run_id"], []), gold[s["task_id"]])
        if m is not None:
            r0maj[(s["task_id"], s["seed"])] = m
    if not r0maj:
        return None
    r0_correct = {k: int(r0maj[k] == ng[k[0]]) for k in r0maj}
    vote_acc = mean(r0_correct.values())
    fc = {c: {} for c in CONDS}
    lock = {c: {} for c in CONDS}
    for s in S:
        c = s["condition"]
        if c not in CONDS:
            continue
        k = (s["task_id"], s["seed"])
        if k not in r0maj:
            continue
        fc[c][k] = int(exact_match(s["final_answer"], gold[s["task_id"]]))
        lock[c][k] = int(normalize_prediction(s["final_answer"], gold[s["task_id"]]) == r0maj[k])
    out = {"vote_acc": vote_acc, "pwrong": 1 - vote_acc, "n_items": len(r0maj),
           "r0_correct": r0_correct, "fc": fc, "cond": {}}
    for c in CONDS:
        keys = list(fc[c])
        if not keys:
            out["cond"][c] = None
            continue
        wrong = [k for k in keys if not r0_correct[k]]
        right = [k for k in keys if r0_correct[k]]
        out["cond"][c] = {
            "acc": mean(fc[c][k] for k in keys),
            "recovery": mean(fc[c][k] for k in wrong) if wrong else float("nan"),
            "breakage": mean(1 - fc[c][k] for k in right) if right else float("nan"),
            "lockin": mean(lock[c].values()),
            "n_wrong": len(wrong), "n_right": len(right),
        }
    return out


def build_table():
    """Return one tidy row per available (model, regime) setting. No file writes."""
    rows = []
    for m, reg, dirn, split, tier in build_settings():
        r = extract(dirn, split)
        if r is None:
            print(f"  [missing] {dirn}", file=sys.stderr)
            continue
        situ, flat, transf = r["cond"]["situational"], r["cond"]["flat"], r["cond"]["transformational"]
        if not (situ and flat):
            print(f"  [no situ/flat] {dirn}", file=sys.stderr)
            continue
        keys_sf = [k for k in r["fc"]["situational"] if k in r["fc"]["flat"]]
        d_flat = [r["fc"]["situational"][k] - r["fc"]["flat"][k] for k in keys_sf]
        keys_sv = list(r["fc"]["situational"])
        d_vote = [r["fc"]["situational"][k] - r["r0_correct"][k] for k in keys_sv]
        _, p_flat = paired_perm_p(d_flat)
        _, p_vote = paired_perm_p(d_vote)
        ci_flat = boot_ci_mean(d_flat)
        ci_vote = boot_ci_mean(d_vote)
        drec = situ["recovery"] - flat["recovery"]
        rows.append({
            "model": m, "regime": reg, "tier": tier, "n_items": r["n_items"],
            "reliability": round(r["vote_acc"], 4), "pwrong": round(r["pwrong"], 4),
            "recovery_flat": round(flat["recovery"], 4) if flat["recovery"] == flat["recovery"] else None,
            "recovery_situ": round(situ["recovery"], 4) if situ["recovery"] == situ["recovery"] else None,
            "delta_rec": round(drec, 4) if drec == drec else None,
            "lockin_situ": round(situ["lockin"], 4), "lockin_flat": round(flat["lockin"], 4),
            "acc_flat": round(flat["acc"], 4), "acc_situ": round(situ["acc"], 4),
            "acc_transf": round(transf["acc"], 4) if transf else None,
            "gain_situ_over_flat": round(mean(d_flat), 4),
            "ci_flat_lo": round(ci_flat[0], 4), "ci_flat_hi": round(ci_flat[1], 4), "p_flat": round(p_flat, 4),
            "gain_situ_over_vote": round(mean(d_vote), 4),
            "ci_vote_lo": round(ci_vote[0], 4), "ci_vote_hi": round(ci_vote[1], 4), "p_vote": round(p_vote, 4),
        })
    return rows


# ---------- stats helpers (pure python) ----------

def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(x, y):
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def spearman(x, y):
    return pearson(_ranks(x), _ranks(y))


def perm_corr_p(x, y, iters=10000, seed=17):
    obs = abs(spearman(x, y))
    rng = random.Random(seed)
    yy = list(y)
    extreme = 0
    for _ in range(iters):
        rng.shuffle(yy)
        if abs(spearman(x, yy)) >= obs - 1e-12:
            extreme += 1
    return extreme / iters


def boot_ci_corr(x, y, iters=10000, seed=17, fn=spearman):
    rng = random.Random(seed)
    n = len(x)
    vals = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        v = fn([x[i] for i in idx], [y[i] for i in idx])
        if v == v:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def boot_ci_mean(diffs, iters=10000, seed=17):
    rng = random.Random(seed)
    n = len(diffs)
    vals = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)]
    vals.sort()
    return vals[int(0.025 * iters)], vals[int(0.975 * iters)]


def ols(x, y):
    mx, my = mean(x), mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    b = sxy / sxx if sxx > 0 else 0.0
    return my - b * mx, b


def main():
    rows = build_table()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(OUTDIR / "sr0_predictive_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (OUTDIR / "sr0_predictive_table.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWrote outputs/analysis/sr0_predictive_table.{{csv,json}}: {len(rows)} settings\n")

    hdr = (f"{'model':14s} {'regime':10s} {'tier':5s} {'relia':>6s} {'rec_flat':>8s} "
           f"{'rec_situ':>8s} {'d_rec':>6s} {'g_flat':>7s} {'p':>6s} {'g_vote':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for x in sorted(rows, key=lambda r: -r["gain_situ_over_flat"]):
        print(f"{x['model']:14s} {x['regime']:10s} {x['tier']:5s} {x['reliability']:>6.2f} "
              f"{(x['recovery_flat'] if x['recovery_flat'] is not None else float('nan')):>8.3f} "
              f"{(x['recovery_situ'] if x['recovery_situ'] is not None else float('nan')):>8.3f} "
              f"{(x['delta_rec'] if x['delta_rec'] is not None else float('nan')):>+6.2f} "
              f"{x['gain_situ_over_flat']:>+7.3f} {x['p_flat']:>6.3f} {x['gain_situ_over_vote']:>+7.3f}")

    valid = [x for x in rows if x["delta_rec"] is not None and x["recovery_flat"] is not None]
    y = [x["gain_situ_over_flat"] for x in valid]
    preds = {
        "reliability (1-pwrong)": [x["reliability"] for x in valid],
        "pwrong": [x["pwrong"] for x in valid],
        "recoverability (recovery_situ)": [x["recovery_situ"] for x in valid],
        "flat recovery": [x["recovery_flat"] for x in valid],
        "Delta_rec (composite)": [x["delta_rec"] for x in valid],
    }
    print("\n" + "=" * 78)
    print(f"PREDICTIVE ANALYSIS  (outcome y = situational gain over flat; N={len(valid)} settings)")
    print("=" * 78)
    print(f"{'predictor':34s} {'Spearman':>9s} {'perm p':>8s} {'boot 95% CI':>20s}")
    for name, xv in preds.items():
        print(f"{name:34s} {spearman(xv, y):>+9.3f} {perm_corr_p(xv, y):>8.3f}   "
              f"[{boot_ci_corr(xv, y)[0]:+.3f}, {boot_ci_corr(xv, y)[1]:+.3f}]")

    dr = preds["Delta_rec (composite)"]
    a, b = ols(dr, y)
    print(f"\nOLS  y = {a:+.4f} + {b:+.4f} * Delta_rec   "
          f"(gain rises {b*0.1*100:+.1f}pp per +0.10 recovery advantage)")

    print("\n--- Subset robustness (Delta_rec -> gain), guarding against diagnostic-probe selection ---")
    for lab, sub in [("all 24", valid),
                     ("main matrix only (not selected on predictors)", [x for x in valid if x["tier"] == "main"]),
                     ("probes only (selected to span predictors)", [x for x in valid if x["tier"] == "probe"])]:
        xv = [x["delta_rec"] for x in sub]
        yv = [x["gain_situ_over_flat"] for x in sub]
        pw = [x["pwrong"] for x in sub]
        print(f"  {lab:48s} N={len(sub):2d}  Spearman(d_rec)={spearman(xv, yv):+.3f} "
              f"(p={perm_corr_p(xv, yv):.3f})  Spearman(pwrong)={spearman(pw, yv):+.3f}")

    def loo(groups_of, label):
        print(f"\n--- Leave-one-{label}-out: fit y~Delta_rec on the rest, predict held-out ---")
        errs_null, errs_pos = [], []
        for g in sorted(set(groups_of(x) for x in valid)):
            tr = [(dr[i], y[i]) for i, x in enumerate(valid) if groups_of(x) != g]
            te = [(dr[i], y[i], valid[i]) for i, x in enumerate(valid) if groups_of(x) == g]
            aa, bb = ols([t[0] for t in tr], [t[1] for t in tr])
            for xv2, yv2, xr in te:
                pred = aa + bb * xv2
                is_pos = yv2 >= 0.05 and xr["p_flat"] < 0.05
                (errs_pos if is_pos else errs_null).append(abs(pred - yv2))
                if is_pos or abs(pred - yv2) > 0.03:
                    tag = "POSITIVE" if is_pos else "null"
                    print(f"  held-out {xr['model']}/{xr['regime']:9s} [{tag}]: "
                          f"actual={yv2:+.3f} pred={pred:+.3f} err={pred-yv2:+.3f}")
        print(f"  mean |LOO err|  null settings: {mean(errs_null):.4f} (n={len(errs_null)})  "
              f"| positive setting: {mean(errs_pos) if errs_pos else float('nan'):.4f} (n={len(errs_pos)})")

    loo(lambda x: x["model"], "model")
    loo(lambda x: x["regime"], "task")

    print("\n" + "=" * 78)
    print("HEADLINE (positive settings, gain>=5pp & p<0.05):")
    for x in [x for x in valid if x["gain_situ_over_flat"] >= 0.05 and x["p_flat"] < 0.05]:
        print(f"  {x['model']}/{x['regime']}: over flat {x['gain_situ_over_flat']:+.3f} "
              f"CI[{x['ci_flat_lo']:+.3f},{x['ci_flat_hi']:+.3f}] p={x['p_flat']:.3f} | "
              f"over vote {x['gain_situ_over_vote']:+.3f} CI[{x['ci_vote_lo']:+.3f},{x['ci_vote_hi']:+.3f}] p={x['p_vote']:.3f}")


if __name__ == "__main__":
    main()
