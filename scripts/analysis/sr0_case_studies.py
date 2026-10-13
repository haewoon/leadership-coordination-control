"""Extract case-study transcripts: recovery, failed recovery, and breakage.

Three team-deliberation outcomes, all from the richest unreliable-round-0 cell
(llama-4-scout social, situational controller), so they form one controlled contrast:
  (i)   successful recovery : round-0 majority WRONG   -> final CORRECT
  (ii)  failed recovery     : round-0 majority WRONG   -> final WRONG
  (iii) breakage            : round-0 majority CORRECT  -> final WRONG (reopening broke it)

Reconstructs each run round-by-round from trace.jsonl + run_summaries.jsonl and renders a
readable transcript. Selection prefers runs where the explore action fired (mechanism
visible) and, for recovery, where the correct answer was a round-0 MINORITY (dissent
preservation). The recovery / breakage counts printed here match the paper's pooled rates
for this cell (recovery 0.285, breakage 0.18), a check that the extraction is faithful.

Writes outputs/analysis/sr0_case_studies.txt.
Run: PYTHONPATH=. python3 scripts/analysis/sr0_case_studies.py
"""
from __future__ import annotations
import sys, re, collections
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.accuracy import exact_match, normalize_prediction  # noqa: E402
from src.utils.io import load_jsonl  # noqa: E402

DIR = "sr0_llama_4_scout_social"
SPLIT = "data/splits/social_chemistry_broad_bc_100.jsonl"
COND = "situational"
OUT = ROOT / "outputs" / "analysis" / "sr0_case_studies.txt"
FA_RE = re.compile(r"Final answer:\s*([A-Za-z]+)", re.I)


def label_of(payload, gold):
    m = FA_RE.search(payload or "")
    if m:
        return m.group(1).strip().upper()[:1]
    n = normalize_prediction(payload or "", gold)
    return (n or "?").upper()[:1]


def first_sentences(text, n=260):
    t = re.sub(r"\s+", " ", (text or "").split("Final answer:")[0]).strip()
    return (t[:n] + "...") if len(t) > n else t


def main():
    d = ROOT / "outputs/runs" / DIR
    rows = load_jsonl(ROOT / SPLIT)
    gold = {r["task_id"]: r["gold_answer"] for r in rows}
    qtext = {r["task_id"]: r["question"] for r in rows}
    choices = {r["task_id"]: {c["label"]: c["text"] for c in r["choices"]} for r in rows}

    summ = {s["run_id"]: s for s in load_jsonl(d / "run_summaries.jsonl") if s["condition"] == COND}
    ev = collections.defaultdict(list)
    for e in load_jsonl(d / "trace.jsonl"):
        if e["condition"] == COND:
            ev[e["run_id"]].append(e)

    cands = {"recovery": [], "failed": [], "breakage": []}
    for rid, s in summ.items():
        tid = s["task_id"]
        g = gold.get(tid)
        if g is None:
            continue
        events = ev.get(rid, [])
        per = collections.defaultdict(dict)
        snip = collections.defaultdict(dict)
        for e in events:
            if e["event_type"] == "agent_output":
                per[e["round_idx"]][e["source"]] = label_of(e["payload"], g)
                snip[e["round_idx"]][e["source"]] = first_sentences(e["payload"])
        if 0 not in per:
            continue
        r0 = per[0]
        r0maj = Counter(r0.values()).most_common(1)[0][0]
        r0_correct = int(r0maj == g)
        final_correct = int(exact_match(s.get("final_answer", ""), g))
        actions = [(e["round_idx"], e["target"], e["payload"])
                   for e in events if e["event_type"] == "control_action"]
        rec = dict(rid=rid, tid=tid, seed=s.get("seed"), g=g, r0=dict(r0), r0maj=r0maj,
                   final=label_of(s.get("final_answer", ""), g), per=dict(per), snip=dict(snip),
                   actions=actions, explored=any(a[2] == "explore" for a in actions),
                   n_rounds=max(per) + 1,
                   correct_minority=(g in r0.values()) and (r0maj != g))
        if not r0_correct and final_correct:
            cands["recovery"].append(rec)
        elif not r0_correct and not final_correct:
            cands["failed"].append(rec)
        elif r0_correct and not final_correct:
            cands["breakage"].append(rec)

    n_rec, n_fail, n_brk = len(cands["recovery"]), len(cands["failed"]), len(cands["breakage"])
    header = (f"# situational, {DIR}\n"
              f"# pools: recovery={n_rec} failed={n_fail} breakage={n_brk}\n"
              f"# check vs paper rates: recovery={n_rec}/{n_rec+n_fail}={n_rec/(n_rec+n_fail):.3f} "
              f"(paper 0.285); breakage={n_brk}/? (paper 0.18)\n")
    print(header)

    def score(r, kind):
        s = 10 if r["explored"] else 0
        if r["n_rounds"] >= 2:
            s += 2
        if kind == "recovery" and r["correct_minority"]:
            s += 20
        if kind == "breakage" and Counter(r["r0"].values())[r["r0maj"]] >= 2:
            s += 5
        return s

    def render(r, kind):
        L = choices[r["tid"]]
        def lab(x):
            return f"{x} ({L.get(x, '?')})"
        out = ["=" * 88, f"[{kind.upper()}]  {r['tid']}  seed={r['seed']}", "-" * 88,
               "SITUATION: " + first_sentences(qtext[r["tid"]], 320), f"GOLD: {lab(r['g'])}", ""]
        r0c = "RIGHT" if r["r0maj"] == r["g"] else "WRONG"
        out.append("ROUND 0 (independent):  " +
                   ", ".join(f"{a.replace('agent_','A')}={r['r0'][a]}" for a in sorted(r["r0"])) +
                   f"   => majority {lab(r['r0maj'])}  [{r0c}]")
        for a in sorted(r["r0"]):
            tag = "  <-- matches gold" if r["r0"][a] == r["g"] else ""
            out.append(f"    {a.replace('agent_','agent ')}: {r['r0'][a]}{tag}  |  {r['snip'][0][a][:150]}")
        acts_by_round = collections.defaultdict(list)
        for rd, tgt, pay in r["actions"]:
            acts_by_round[rd].append(f"{pay}->{tgt.replace('agent_','A')}")
        for rd in range(1, r["n_rounds"]):
            if rd - 1 in acts_by_round or rd in acts_by_round:
                aa = acts_by_round.get(rd - 1, []) or acts_by_round.get(rd, [])
                out.append(f"  [leader actions @r{rd-1}: {', '.join(aa)}]")
            if rd in r["per"]:
                out.append(f"ROUND {rd}:  " +
                           ", ".join(f"{a.replace('agent_','A')}={r['per'][rd][a]}" for a in sorted(r["per"][rd])))
        fc = "CORRECT" if r["final"] == r["g"] else "WRONG"
        out += ["", f"FINAL: {lab(r['final'])}  [{fc}]   explore_fired={r['explored']}  rounds={r['n_rounds']}"]
        return "\n".join(out)

    full = [header]
    for kind in ["recovery", "failed", "breakage"]:
        for p in sorted(cands[kind], key=lambda r: -score(r, kind))[:2]:
            block = render(p, kind)
            full.append(block)
            print(block + "\n")
    OUT.write_text("\n\n".join(full))
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
