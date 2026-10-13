"""Shared statistics helpers plus two post-hoc analyses from earlier review feedback.

(1) Paired permutation p-value for `transactional_accept_only` vs `flat`
    on `gpt-oss-120b` MATH-500 Level 5 (also reports `llama-4-scout`).
    Pairs by (task_id, seed); test statistic is mean(AO_correct) - mean(flat_correct).

(2) Round-0 majority vote baseline on the 12 paper-12 cells:
    extracted from existing `flat` traces, take the round-0 agent
    majority answer as the "no controller, no further rounds" null
    and compare against the published `transactional_accept_only` cells.

Usage:
    python3 scripts/analysis/reviewer_followups.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.accuracy import exact_match, normalize_prediction  # noqa: E402
from src.utils.io import load_config, load_jsonl  # noqa: E402

FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*([^\n\r|]+)", re.IGNORECASE)


# ---------------------- shared helpers ----------------------

def crossround_majority_answer(events: list[dict], gold: str | None) -> str | None:
    """Cross-round majority extractor: scan all agent_output events for
    'Final answer: X' patterns across all rounds, normalize, take majority."""
    extracted: list[str] = []
    for ev in events:
        if ev.get("event_type") != "agent_output":
            continue
        payload = ev.get("payload", "")
        if not FINAL_ANSWER_RE.search(payload):
            continue
        norm = normalize_prediction(payload, gold)
        if norm:
            extracted.append(norm)
    if not extracted:
        return None
    return Counter(extracted).most_common(1)[0][0]


def round0_majority_answer(events: list[dict], gold: str | None) -> str | None:
    """Round-0-only majority extractor: each agent's round-0 output is a vote."""
    r0 = [
        e for e in events
        if e.get("event_type") == "agent_output" and int(e.get("round_idx", -1)) == 0
    ]
    if not r0:
        return None
    answers = [normalize_prediction(e["payload"], gold) for e in r0]
    answers = [a for a in answers if a]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def paired_perm_p(diffs: list[int], n_iters: int = 10_000, seed: int = 17) -> tuple[float, float]:
    """Two-sided paired permutation: randomly flip the sign of each pair's diff.
    Returns (observed_mean_diff, two_sided_p)."""
    diffs = list(diffs)
    observed = mean(diffs) if diffs else 0.0
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return (observed, float("nan"))
    extreme = 0
    abs_obs = abs(observed)
    for _ in range(n_iters):
        # Flip each diff's sign with probability 0.5
        s = 0.0
        for d in diffs:
            if rng.random() < 0.5:
                s -= d
            else:
                s += d
        if abs(s / n) >= abs_obs:
            extreme += 1
    return (observed, extreme / n_iters)


# ---------------------- Analysis 1: AO vs flat on MATH-500 ----------------------

MATH500_CONFIG_TRIPLES = {
    "scout": [
        "src/config/selfhosted_llama_4_scout_math500.json",
        "src/config/selfhosted_llama_4_scout_math500_full_seeds12.json",
        "src/config/selfhosted_llama_4_scout_math500_accept_only.json",
    ],
    "oss120b": [
        "src/config/selfhosted_gpt_oss_120b_math500.json",
        "src/config/selfhosted_gpt_oss_120b_math500_full_seeds12.json",
        "src/config/selfhosted_gpt_oss_120b_math500_accept_only.json",
    ],
}


def analysis_1_paired_perm_math500_ao_vs_flat() -> None:
    print("=" * 70)
    print("(1) Paired permutation: transactional_accept_only vs flat on MATH-500 L5")
    print("    Pairing: (task_id, seed). Score: cross-round-majority-extracted accuracy.")
    print("=" * 70)
    for model_slug, paths in MATH500_CONFIG_TRIPLES.items():
        configs = [load_config(ROOT / p) for p in paths]
        gold: dict[str, str] = {}
        for cfg in configs:
            for task_cfg in cfg["tasks"]:
                for row in load_jsonl(ROOT / task_cfg["path"]):
                    gold[row["task_id"]] = row["gold_answer"]
        summaries: list[dict] = []
        events: list[dict] = []
        for cfg in configs:
            out = ROOT / cfg["output_dir"]
            summaries.extend(load_jsonl(out / "run_summaries.jsonl"))
            events.extend(load_jsonl(out / "trace.jsonl"))
        events_by_run: dict[str, list[dict]] = defaultdict(list)
        for ev in events:
            events_by_run[ev["run_id"]].append(ev)

        # Score each (cond, task_id, seed) using cross-round majority
        score: dict[tuple[str, str, int], int] = {}
        for s in summaries:
            cond = s["condition"]
            tid = s["task_id"]
            sd = s["seed"]
            g = gold.get(tid)
            if g is None:
                continue
            evts = events_by_run.get(s["run_id"], [])
            fixed_norm = crossround_majority_answer(evts, g)
            if fixed_norm is None:
                score[(cond, tid, sd)] = int(exact_match(s["final_answer"], g))
            else:
                score[(cond, tid, sd)] = int(exact_match(f"Final answer: {fixed_norm}", g))

        # Build paired diffs: AO - flat over common (task_id, seed)
        ao_keys = {(t, s) for (c, t, s) in score if c == "transactional_accept_only"}
        flat_keys = {(t, s) for (c, t, s) in score if c == "flat"}
        common = sorted(ao_keys & flat_keys)
        diffs = [score[("transactional_accept_only", t, s)] - score[("flat", t, s)] for t, s in common]
        ao_mean = mean([score[("transactional_accept_only", t, s)] for t, s in common])
        flat_mean = mean([score[("flat", t, s)] for t, s in common])
        observed, p = paired_perm_p(diffs, n_iters=10_000, seed=17)
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        ties = sum(1 for d in diffs if d == 0)
        print(f"\n  {model_slug}:")
        print(f"    n_pairs        = {len(common)}")
        print(f"    flat acc       = {flat_mean:.3f}")
        print(f"    AO acc         = {ao_mean:.3f}")
        print(f"    observed delta = {observed:+.4f}  ({wins} AO-wins, {losses} flat-wins, {ties} ties)")
        print(f"    paired perm p  = {p:.4f}  (two-sided, 10000 iters)")


# ---------------------- Analysis 2: round-0 vote baseline on paper-12 ----------------------

PAPER12_FLAT_DIRS = {
    # Maps (model, regime) -> list of flat-condition output dirs.
    # We pull round-0 agent outputs from these and compute majority vote.
    ("qwen3.5", "alphanli"): [
        "alphanli_scale_step_100_core_qwen35",
    ],
    ("qwen3.5", "closed"): [
        "closed_ended_candidate_100_core_qwen35",
    ],
    ("qwen3.5", "social"): [
        "social_chemistry_broad_bc_100_v2_core_qwen35",
    ],
    ("qwen3.5", "mixed"): [
        "mixed_benchmark_150_core_qwen35",
    ],
    ("llama3.1", "alphanli"): [
        "alphanli_scale_step_100_llama31_full_seeds012",
    ],
    ("llama3.1", "closed"): [
        "closed_ended_candidate_100_llama31_full_seeds012",
    ],
    ("llama3.1", "social"): [
        "social_chemistry_broad_bc_100_llama31_full_seeds012",
    ],
    ("llama3.1", "mixed"): [
        "mixed_benchmark_150_llama31_seed0",
        "mixed_benchmark_150_llama31_seeds12",
    ],
    ("gpt-5-mini", "alphanli"): [
        "alphanli_scale_step_100_openai_seed0",
        "alphanli_scale_step_100_openai_seeds12",
    ],
    ("gpt-5-mini", "closed"): [
        "closed_ended_candidate_100_openai_seed0",
        "closed_ended_candidate_100_openai_seeds12",
    ],
    ("gpt-5-mini", "social"): [
        "social_chemistry_broad_bc_100_openai_full_seeds012",
    ],
    ("gpt-5-mini", "mixed"): [
        "mixed_benchmark_150_openai_seeds12_v3",
        "mixed_benchmark_150_openai_spotcheck_seed0_v2",
    ],
}

REGIME_TO_SPLIT = {
    "alphanli": ["alphanli_scale_step_100.jsonl"],
    "closed": ["closed_ended_candidate_curated_100.jsonl"],
    "social": ["social_chemistry_broad_bc_100.jsonl"],
    "mixed": ["mixed_benchmark_150.jsonl"],
}

# Published accept-only (AO) accuracies from notes/bass_two_pillar_paper12.md §3
AO_ACC_PUBLISHED = {
    ("qwen3.5", "alphanli"): 0.880,
    ("qwen3.5", "social"):   0.507,
    ("qwen3.5", "closed"):   0.687,
    ("qwen3.5", "mixed"):    0.671,
    ("llama3.1", "alphanli"): 0.773,
    ("llama3.1", "social"):  0.500,
    ("llama3.1", "closed"):  0.767,
    ("llama3.1", "mixed"):   0.613,
    ("gpt-5-mini", "alphanli"): 0.860,
    ("gpt-5-mini", "social"): 0.433,
    ("gpt-5-mini", "closed"): 0.807,
    ("gpt-5-mini", "mixed"):  0.698,
}


def analysis_2_round0_vote_paper12() -> None:
    print("\n" + "=" * 70)
    print("(2) Round-0 majority vote baseline on paper-12 cells.")
    print("    Extracted from existing flat traces (round-0 agent outputs only).")
    print("    Compared to published transactional_accept_only accuracy.")
    print("=" * 70)
    print(f"\n  {'Model':12s} {'Regime':10s} {'n':>4s} {'R0-vote':>9s} {'AO (pub)':>9s} {'Δ':>7s}")
    diffs_for_summary: list[float] = []
    for (model, regime), dirs in PAPER12_FLAT_DIRS.items():
        # Load gold
        gold: dict[str, str] = {}
        for split in REGIME_TO_SPLIT[regime]:
            for row in load_jsonl(ROOT / "data" / "splits" / split):
                gold[row["task_id"]] = row["gold_answer"]
        # Load summaries + events from flat runs
        summaries: list[dict] = []
        events: list[dict] = []
        for d in dirs:
            p_sum = ROOT / "outputs" / "runs" / d / "run_summaries.jsonl"
            p_trace = ROOT / "outputs" / "runs" / d / "trace.jsonl"
            if p_sum.exists():
                summaries.extend(load_jsonl(p_sum))
            if p_trace.exists():
                events.extend(load_jsonl(p_trace))
        events_by_run: dict[str, list[dict]] = defaultdict(list)
        for ev in events:
            events_by_run[ev["run_id"]].append(ev)
        # Restrict to flat condition and dedupe by (task_id, seed) (use first observed)
        seen: set[tuple[str, int]] = set()
        scores: list[int] = []
        for s in summaries:
            if s.get("condition") != "flat":
                continue
            key = (s["task_id"], s["seed"])
            if key in seen:
                continue
            g = gold.get(s["task_id"])
            if g is None:
                continue
            seen.add(key)
            evts = events_by_run.get(s["run_id"], [])
            r0 = round0_majority_answer(evts, g)
            if r0 is None:
                continue
            scores.append(int(r0 == g))
        if not scores:
            print(f"  {model:12s} {regime:10s} {'(no flat data)':>30s}")
            continue
        r0_acc = sum(scores) / len(scores)
        ao_acc = AO_ACC_PUBLISHED[(model, regime)]
        delta = r0_acc - ao_acc
        diffs_for_summary.append(delta)
        print(f"  {model:12s} {regime:10s} {len(scores):>4d} {r0_acc:>9.3f} {ao_acc:>9.3f} {delta:>+7.3f}")
    if diffs_for_summary:
        print(f"\n  Summary: median |R0 - AO| Δ = {sorted(abs(d) for d in diffs_for_summary)[len(diffs_for_summary)//2]:.3f}pp")
        print(f"           max |R0 - AO| Δ    = {max(abs(d) for d in diffs_for_summary):.3f}pp")
        print(f"           range              = [{min(diffs_for_summary):+.3f}, {max(diffs_for_summary):+.3f}]")


# ---------------------- Analysis 3: round-0 vote vs all controllers ----------------------

def analysis_3_round0_vote_vs_all_controllers() -> None:
    """For each (model, regime, controller) combination, compare the controller's
    final accuracy against the round-0 majority vote baseline computed from FLAT
    runs (controller-free, interaction-less). Tests whether the round-0-vote
    reducibility found for transactional generalizes to other controllers."""
    print("\n" + "=" * 70)
    print("(3) Round-0 vote vs all 5 standard controllers on paper-12.")
    print("    R0-vote is computed once per (model, regime) from FLAT round-0")
    print("    outputs. Each controller's accuracy is its own final answer.")
    print("=" * 70)
    print(f"\n  {'Model':10s} {'Regime':10s} {'n':>4s} {'R0-vote':>8s} "
          f"{'Flat':>9s} {'ThFr':>9s} {'Trans':>9s} {'Tform':>9s} {'Sit':>9s}")

    controllers = ["flat", "theory_free_control", "transactional", "transformational", "situational"]
    summary_diffs = {c: [] for c in controllers}

    for (model, regime), dirs in PAPER12_FLAT_DIRS.items():
        # Load gold
        gold: dict[str, str] = {}
        for split in REGIME_TO_SPLIT[regime]:
            for row in load_jsonl(ROOT / "data" / "splits" / split):
                gold[row["task_id"]] = row["gold_answer"]

        # Load all dirs for this (model, regime) — we may need additional dirs
        # beyond just the flat one for non-flat conditions.
        # Use the canonical GROUPS map.
        try:
            from scripts.compute_statistics import GROUPS, BENCHMARK_TO_SPLIT  # noqa
            cs_key = (model, regime if regime != "closed" else "closed_ended")
            all_dirs = GROUPS.get(cs_key, dirs)
        except Exception:
            all_dirs = dirs

        summaries: list[dict] = []
        events: list[dict] = []
        for d in all_dirs:
            p_sum = ROOT / "outputs" / "runs" / d / "run_summaries.jsonl"
            p_trace = ROOT / "outputs" / "runs" / d / "trace.jsonl"
            if p_sum.exists():
                summaries.extend(load_jsonl(p_sum))
            if p_trace.exists():
                events.extend(load_jsonl(p_trace))
        events_by_run: dict[str, list[dict]] = defaultdict(list)
        for ev in events:
            events_by_run[ev["run_id"]].append(ev)

        # R0-vote: from flat condition only
        seen_flat: set[tuple[str, int]] = set()
        r0_scores: list[int] = []
        for s in summaries:
            if s.get("condition") != "flat":
                continue
            key = (s["task_id"], s["seed"])
            if key in seen_flat:
                continue
            g = gold.get(s["task_id"])
            if g is None:
                continue
            seen_flat.add(key)
            r0 = round0_majority_answer(events_by_run.get(s["run_id"], []), g)
            if r0 is None:
                continue
            r0_scores.append(int(r0 == g))
        n_r0 = len(r0_scores)
        r0_acc = sum(r0_scores) / n_r0 if n_r0 else 0.0

        # Each controller's final accuracy
        cond_scores: dict[str, list[int]] = {c: [] for c in controllers}
        for s in summaries:
            cond = s.get("condition")
            if cond not in cond_scores:
                continue
            g = gold.get(s["task_id"])
            if g is None:
                continue
            cond_scores[cond].append(int(exact_match(s["final_answer"], g)))
        cond_accs = {c: (sum(v) / len(v) if v else float("nan")) for c, v in cond_scores.items()}

        # Print row
        row = f"  {model:10s} {regime:10s} {n_r0:>4d} {r0_acc:>8.3f}"
        for c in controllers:
            v = cond_accs[c]
            if v != v:  # NaN
                row += f" {'n/a':>9s}"
            else:
                delta = v - r0_acc
                summary_diffs[c].append(delta)
                row += f" {v:>5.3f}({delta:>+5.3f})"
        print(row)

    print("\n  Per-controller summary (Δ = controller_acc − R0-vote_acc, paper-12 only):")
    for c in controllers:
        diffs = summary_diffs[c]
        if not diffs:
            continue
        mean_d = sum(diffs) / len(diffs)
        within_4pp = sum(1 for d in diffs if abs(d) <= 0.04)
        med_abs = sorted(abs(d) for d in diffs)[len(diffs)//2]
        print(f"    {c:25s} n={len(diffs):>2d}  mean Δ={mean_d:+.4f}  "
              f"median |Δ|={med_abs:.3f}  within ±4pp: {within_4pp}/{len(diffs)}")


# ---------------------- Analysis 4: per-leadership-controller gain cells ----------------------

LEADERSHIP_CONTROLLERS = ["transactional", "transformational", "situational"]
GAIN_THRESHOLD_PP = 0.05  # >=5pp lift over R0-vote


def per_run_signatures(events: list[dict], gold: str, final_answer_raw: str) -> dict | None:
    """Behavioral signatures for a single run, aligned with paper canonical metrics.

    Returns:
        revise_rate : per-agent fraction whose latest-round answer differs from round 0
        lock_in     : 1 if R0-majority == normalized final answer, else 0 (None if R0 unknown)
        r0_correct  : 1 if R0-majority == gold
    """
    events_out = [e for e in events if e.get("event_type") == "agent_output"]
    if not events_out:
        return None
    per_agent: dict[str, dict[int, str]] = defaultdict(dict)
    rounds: set[int] = set()
    for e in events_out:
        src = e.get("source", "")
        try:
            ridx = int(e.get("round_idx", -1))
        except (TypeError, ValueError):
            continue
        if ridx < 0:
            continue
        ans = normalize_prediction(e.get("payload", ""), gold)
        if ans is None:
            continue
        per_agent[src][ridx] = ans
        rounds.add(ridx)
    if not rounds:
        return None
    r0_answers = [by_r[0] for by_r in per_agent.values() if 0 in by_r]
    if not r0_answers:
        return None
    r0_majority = Counter(r0_answers).most_common(1)[0][0]

    revises = 0
    total = 0
    for by_r in per_agent.values():
        if 0 not in by_r:
            continue
        latest = max(by_r.keys())
        if latest > 0:
            total += 1
            if by_r[latest] != by_r[0]:
                revises += 1
    revise_rate = (revises / total) if total > 0 else 0.0

    final_norm = normalize_prediction(final_answer_raw, gold)
    lock_in = int(r0_majority == final_norm) if final_norm is not None else None

    return {
        "revise_rate": revise_rate,
        "lock_in": lock_in,
        "r0_correct": 1 if r0_majority == gold else 0,
    }


def _aggregate_signatures(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    revise = mean(r["revise_rate"] for r in rows)
    lockins = [r["lock_in"] for r in rows if r["lock_in"] is not None]
    lock_in = mean(lockins) if lockins else float("nan")
    wrong_r0 = [r for r in rows if r["r0_correct"] == 0]
    recovery = mean(r["final_correct"] for r in wrong_r0) if wrong_r0 else float("nan")
    r0_acc = mean(r["r0_correct"] for r in rows)
    final_acc = mean(r["final_correct"] for r in rows)
    return {
        "n": len(rows),
        "revise_rate": revise,
        "lock_in": lock_in,
        "recovery": recovery,
        "r0_acc": r0_acc,
        "final_acc": final_acc,
        "n_wrong_r0": len(wrong_r0),
    }


def analysis_4_gain_cells_leadership_only() -> None:
    """For T, Tf, S only: per-cell gain over R0-vote and behavioral signatures
    at gain cells. flat / theory_free_control omitted (not leadership controllers)."""
    print("\n" + "=" * 78)
    print("(4) Per-leadership-controller gain cells over R0-vote (paper-12).")
    print(f"    Gain cell := controller_acc - R0-vote_acc >= {GAIN_THRESHOLD_PP*100:.0f}pp.")
    print("    Behavioral signatures contrasted: gain cells vs non-gain cells.")
    print("=" * 78)

    # cell_records[(model, regime)] = {
    #   'r0_acc': float,
    #   'controllers': {controller: {'final_acc': float, 'rows': [per-task records]}}
    # }
    try:
        from scripts.compute_statistics import GROUPS  # noqa
    except Exception:
        GROUPS = {}

    cell_records: dict[tuple[str, str], dict] = {}

    for (model, regime), flat_dirs in PAPER12_FLAT_DIRS.items():
        gold: dict[str, str] = {}
        for split in REGIME_TO_SPLIT[regime]:
            for row in load_jsonl(ROOT / "data" / "splits" / split):
                gold[row["task_id"]] = row["gold_answer"]

        cs_key = (model, regime if regime != "closed" else "closed_ended")
        all_dirs = GROUPS.get(cs_key, flat_dirs)

        summaries: list[dict] = []
        events: list[dict] = []
        for d in all_dirs:
            p_sum = ROOT / "outputs" / "runs" / d / "run_summaries.jsonl"
            p_trace = ROOT / "outputs" / "runs" / d / "trace.jsonl"
            if p_sum.exists():
                summaries.extend(load_jsonl(p_sum))
            if p_trace.exists():
                events.extend(load_jsonl(p_trace))
        events_by_run: dict[str, list[dict]] = defaultdict(list)
        for ev in events:
            events_by_run[ev["run_id"]].append(ev)

        # R0-vote accuracy from flat condition (same path as analysis_3)
        seen_flat: set[tuple[str, int]] = set()
        r0_scores: list[int] = []
        for s in summaries:
            if s.get("condition") != "flat":
                continue
            key = (s["task_id"], s["seed"])
            if key in seen_flat:
                continue
            g = gold.get(s["task_id"])
            if g is None:
                continue
            seen_flat.add(key)
            r0 = round0_majority_answer(events_by_run.get(s["run_id"], []), g)
            if r0 is None:
                continue
            r0_scores.append(int(r0 == g))
        r0_acc = (sum(r0_scores) / len(r0_scores)) if r0_scores else float("nan")

        # Per-leadership-controller per-run signatures
        ctrl_data: dict[str, list[dict]] = {c: [] for c in LEADERSHIP_CONTROLLERS}
        for s in summaries:
            cond = s.get("condition")
            if cond not in ctrl_data:
                continue
            g = gold.get(s["task_id"])
            if g is None:
                continue
            sig = per_run_signatures(events_by_run.get(s["run_id"], []), g, s.get("final_answer", ""))
            if sig is None:
                continue
            sig["final_correct"] = int(exact_match(s["final_answer"], g))
            ctrl_data[cond].append(sig)

        ctrl_summary = {
            c: {
                "final_acc": (mean(r["final_correct"] for r in rows) if rows else float("nan")),
                "rows": rows,
            }
            for c, rows in ctrl_data.items()
        }
        cell_records[(model, regime)] = {"r0_acc": r0_acc, "controllers": ctrl_summary}

    # Print per-cell gain table
    print(f"\n  Gain table (controller_acc - R0-vote_acc, paper-12):")
    print(f"  {'Model':10s} {'Regime':10s} {'R0-vote':>8s} "
          f"{'T':>13s} {'Tf':>13s} {'S':>13s}")
    for (model, regime), rec in cell_records.items():
        r0 = rec["r0_acc"]
        row = f"  {model:10s} {regime:10s} {r0:>8.3f}"
        for c in LEADERSHIP_CONTROLLERS:
            acc = rec["controllers"][c]["final_acc"]
            if acc != acc:
                row += f" {'n/a':>13s}"
                continue
            delta = acc - r0
            mark = "*" if delta >= GAIN_THRESHOLD_PP else " "
            row += f" {acc:>6.3f}({delta:>+5.3f}){mark}"
        print(row)
    print(f"  (* = gain cell, delta >= {GAIN_THRESHOLD_PP*100:.0f}pp)")

    # Per-controller behavioral signature contrast: gain cells vs non-gain cells
    print("\n  Behavioral signatures at gain cells vs non-gain cells:")
    print(f"  {'Controller':18s} {'Bucket':12s} {'n_cells':>7s} {'n_runs':>7s} "
          f"{'revise':>7s} {'lock_in':>8s} {'recovery':>9s} "
          f"{'R0_acc':>7s} {'final':>7s}")
    for c in LEADERSHIP_CONTROLLERS:
        gain_rows: list[dict] = []
        nongain_rows: list[dict] = []
        gain_cells = 0
        nongain_cells = 0
        for (model, regime), rec in cell_records.items():
            r0 = rec["r0_acc"]
            acc = rec["controllers"][c]["final_acc"]
            rows = rec["controllers"][c]["rows"]
            if not rows or acc != acc:
                continue
            delta = acc - r0
            if delta >= GAIN_THRESHOLD_PP:
                gain_rows.extend(rows)
                gain_cells += 1
            else:
                nongain_rows.extend(rows)
                nongain_cells += 1
        for label, rows, n_cells in [
            ("gain", gain_rows, gain_cells),
            ("non-gain", nongain_rows, nongain_cells),
        ]:
            agg = _aggregate_signatures(rows)
            if agg["n"] == 0:
                print(f"  {c:18s} {label:12s} {n_cells:>7d} {0:>7d}  (no cells)")
                continue
            print(f"  {c:18s} {label:12s} {n_cells:>7d} {agg['n']:>7d} "
                  f"{agg['revise_rate']:>7.3f} {agg['lock_in']:>8.3f} "
                  f"{agg['recovery']:>9.3f} {agg['r0_acc']:>7.3f} {agg['final_acc']:>7.3f}")

    # Per-(controller, cell) gain-cell breakdown
    print("\n  Gain-cell breakdown by controller (per (model, regime)):")
    for c in LEADERSHIP_CONTROLLERS:
        any_gain = False
        for (model, regime), rec in cell_records.items():
            r0 = rec["r0_acc"]
            acc = rec["controllers"][c]["final_acc"]
            rows = rec["controllers"][c]["rows"]
            if not rows or acc != acc:
                continue
            delta = acc - r0
            if delta < GAIN_THRESHOLD_PP:
                continue
            any_gain = True
            agg = _aggregate_signatures(rows)
            print(f"    [{c}] {model:10s} {regime:10s}: "
                  f"R0={r0:.3f}, ctrl={acc:.3f}, Δ={delta:+.3f} | "
                  f"revise={agg['revise_rate']:.3f}, "
                  f"lock_in={agg['lock_in']:.3f}, "
                  f"recovery={agg['recovery']:.3f} "
                  f"(n_runs={agg['n']}, n_wrong_r0={agg['n_wrong_r0']})")
        if not any_gain:
            print(f"    [{c}] no gain cells at threshold {GAIN_THRESHOLD_PP*100:.0f}pp")


if __name__ == "__main__":
    analysis_1_paired_perm_math500_ao_vs_flat()
    analysis_2_round0_vote_paper12()
    analysis_3_round0_vote_vs_all_controllers()
    analysis_4_gain_cells_leadership_only()
