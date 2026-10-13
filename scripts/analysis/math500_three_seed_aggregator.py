"""Aggregate MATH-500 Level 5 results across (seed 0 + seeds 1-2 + accept_only seed 0)
config triples, returning per-condition accuracy (pre-fix and cross-round-majority-fixed)
and behavioral signature metrics (lock-in, recovery, explore_rate) for paper Section 6.

Usage:
    python3 scripts/analysis/math500_three_seed_aggregator.py <model_slug>

where <model_slug> is "scout" or "oss120b".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.accuracy import exact_match, normalize_prediction  # noqa: E402
from src.utils.io import load_config, load_jsonl  # noqa: E402


FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*([^\n\r|]+)", re.IGNORECASE)


CONFIG_TRIPLES = {
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


def load_gold(configs: list[dict]) -> dict[str, str]:
    gold: dict[str, str] = {}
    for cfg in configs:
        for task_cfg in cfg["tasks"]:
            for row in load_jsonl(ROOT / task_cfg["path"]):
                gold[row["task_id"]] = row["gold_answer"]
    return gold


def crossround_majority_answer(events: list[dict], gold: str | None) -> str | None:
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
    round0 = [e for e in events if e.get("event_type") == "agent_output" and int(e.get("round_idx", -1)) == 0]
    if not round0:
        return None
    answers = [normalize_prediction(e["payload"], gold) for e in round0]
    answers = [a for a in answers if a]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def aggregate(model_slug: str) -> dict:
    config_paths = CONFIG_TRIPLES[model_slug]
    configs = [load_config(ROOT / p) for p in config_paths]
    gold = load_gold(configs)

    summaries: list[dict] = []
    events: list[dict] = []
    for cfg in configs:
        out = ROOT / cfg["output_dir"]
        summaries.extend(load_jsonl(out / "run_summaries.jsonl"))
        events.extend(load_jsonl(out / "trace.jsonl"))

    events_by_run: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        events_by_run[ev["run_id"]].append(ev)

    stats: dict[str, dict[str, list]] = defaultdict(lambda: {
        "orig_correct": [],
        "fixed_correct": [],
        "lock_in_flags": [],
        "recovery_attempts_flags": [],  # incorrect round-0 majority cases
        "recovered_flags": [],          # subset of recovery_attempts that recovered
        "explore_flags": [],
        "stop_rounds": [],
    })

    for s in summaries:
        cond = s["condition"]
        tid = s["task_id"]
        g = gold.get(tid)
        if g is None:
            continue
        run_evts = events_by_run.get(s["run_id"], [])

        orig_correct = int(exact_match(s["final_answer"], g))
        fixed_norm = crossround_majority_answer(run_evts, g)
        if fixed_norm is None:
            fixed_correct = orig_correct
        else:
            fixed_correct = int(exact_match(f"Final answer: {fixed_norm}", g))

        r0 = round0_majority_answer(run_evts, g)
        final_norm = normalize_prediction(s["final_answer"], g)
        if r0 is not None and final_norm is not None:
            stats[cond]["lock_in_flags"].append(int(r0 == final_norm))
            if r0 != g:
                stats[cond]["recovery_attempts_flags"].append(1)
                # use fixed final for recovery, since that's the paper-reported accuracy
                stats[cond]["recovered_flags"].append(int(fixed_correct == 1))

        ctrl = [e for e in run_evts if e.get("event_type") == "control_action"]
        explore = any(e.get("payload") == "explore" for e in ctrl)
        stats[cond]["explore_flags"].append(int(explore))
        stats[cond]["stop_rounds"].append(s.get("rounds_used", 0))

        stats[cond]["orig_correct"].append(orig_correct)
        stats[cond]["fixed_correct"].append(fixed_correct)

    out_rows = {}
    for cond, b in stats.items():
        n = len(b["orig_correct"])
        out_rows[cond] = {
            "n": n,
            "acc_orig": round(sum(b["orig_correct"]) / n, 3),
            "acc_fixed": round(sum(b["fixed_correct"]) / n, 3),
            "delta_extraction_fix": round(
                (sum(b["fixed_correct"]) - sum(b["orig_correct"])) / n, 3
            ),
            "lock_in": (
                round(sum(b["lock_in_flags"]) / len(b["lock_in_flags"]), 3)
                if b["lock_in_flags"]
                else None
            ),
            "recovery_n": len(b["recovery_attempts_flags"]),
            "recovery_rate": (
                round(sum(b["recovered_flags"]) / len(b["recovery_attempts_flags"]), 3)
                if b["recovery_attempts_flags"]
                else None
            ),
            "explore_rate": round(sum(b["explore_flags"]) / n, 3),
            "stop_round_mean": round(sum(b["stop_rounds"]) / n, 3),
        }
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(CONFIG_TRIPLES.keys()))
    parser.add_argument("--json", action="store_true", help="emit JSON instead of plain text")
    args = parser.parse_args()

    rows = aggregate(args.model)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    order = ["flat", "theory_free_control", "transactional", "transactional_accept_only", "transformational", "situational"]
    header = f"{'condition':28s} {'n':>3s} {'orig':>6s} {'fixed':>6s} {'Dfix':>7s} {'lock':>6s} {'rec_n':>5s} {'rec':>6s} {'expl':>6s} {'stopR':>6s}"
    print(f"=== {args.model} ===")
    print(header)
    for c in order:
        if c not in rows:
            continue
        r = rows[c]
        lock = f"{r['lock_in']:.3f}" if r['lock_in'] is not None else "  n/a"
        rec = f"{r['recovery_rate']:.3f}" if r['recovery_rate'] is not None else "  n/a"
        print(
            f"{c:28s} {r['n']:>3d} {r['acc_orig']:>6.3f} {r['acc_fixed']:>6.3f}"
            f" {r['delta_extraction_fix']:>+7.3f} {lock:>6s} {r['recovery_n']:>5d}"
            f" {rec:>6s} {r['explore_rate']:>6.3f} {r['stop_round_mean']:>6.3f}"
        )


if __name__ == "__main__":
    main()
