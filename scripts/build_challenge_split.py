from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_jsonl


TOKEN_RE = re.compile(r"[a-z0-9]+")
EXACTLY_DUPLICATE_THRESHOLD = 0.8
AMBIGUITY_CUES = {
    "might",
    "could",
    "would",
    "should",
    "best",
    "most",
    "more",
    "less",
    "before",
    "after",
    "than",
    "without",
    "while",
    "related",
    "lawful",
    "valuable",
}
UNSTABLE_STRATEGYQA_CUES = {
    "hypothetically",
}


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def commonsenseqa_score(row: dict) -> tuple[float, dict[str, float]]:
    q_tokens = tokenize(row["question"])
    choices = row["choices"]
    gold_label = row["gold_answer"]
    gold_choice = next(choice for choice in choices if choice["label"] == gold_label)
    gold_tokens = tokenize(gold_choice["text"])

    distractor_sims = []
    question_choice_sims = []
    normalized_choice_texts = []
    for choice in choices:
        choice_tokens = tokenize(choice["text"])
        question_choice_sims.append(jaccard(q_tokens, choice_tokens))
        normalized_choice_texts.append(" ".join(sorted(choice_tokens)))
        if choice["label"] != gold_label:
            distractor_sims.append(jaccard(gold_tokens, choice_tokens))

    plausible_choices = sum(sim > 0 for sim in question_choice_sims)
    duplicate_penalty = len(normalized_choice_texts) - len(set(normalized_choice_texts))
    cue_count = sum(token in AMBIGUITY_CUES for token in q_tokens)

    score = (
        3.0 * max(distractor_sims, default=0.0)
        + 1.5 * plausible_choices
        + 1.0 * duplicate_penalty
        + 0.5 * cue_count
    )
    return score, {
        "top_distractor_similarity": round(max(distractor_sims, default=0.0), 4),
        "plausible_choice_count": plausible_choices,
        "duplicate_penalty": duplicate_penalty,
        "cue_count": cue_count,
    }


def is_stable_commonsenseqa(row: dict) -> bool:
    choice_token_sets = [tokenize(choice["text"]) for choice in row["choices"]]
    normalized_choice_texts = [" ".join(sorted(tokens)) for tokens in choice_token_sets]
    if len(normalized_choice_texts) != len(set(normalized_choice_texts)):
        return False

    for i, left in enumerate(choice_token_sets):
        for j, right in enumerate(choice_token_sets):
            if i >= j:
                continue
            if jaccard(left, right) >= EXACTLY_DUPLICATE_THRESHOLD:
                return False
    return True


def strategyqa_score(row: dict) -> tuple[float, dict[str, float]]:
    question = row["question"]
    q_tokens = tokenize(question)
    cue_count = sum(token in AMBIGUITY_CUES for token in q_tokens)
    capitalized_terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question)
    named_entity_count = len(capitalized_terms)
    comparator_count = sum(token in {"than", "before", "after", "more", "less"} for token in q_tokens)
    negation_count = sum(token in {"not", "without", "never", "no"} for token in q_tokens)

    score = (
        1.5 * cue_count
        + 1.0 * named_entity_count
        + 1.5 * comparator_count
        + 1.0 * negation_count
    )
    return score, {
        "cue_count": cue_count,
        "named_entity_count": named_entity_count,
        "comparator_count": comparator_count,
        "negation_count": negation_count,
    }


def alphanli_score(row: dict) -> tuple[float, dict[str, float]]:
    obs1_tokens = tokenize(row["metadata"]["obs1"])
    obs2_tokens = tokenize(row["metadata"]["obs2"])
    hyp_a_tokens = tokenize(row["choices"][0]["text"])
    hyp_b_tokens = tokenize(row["choices"][1]["text"])

    a_obs_support = 0.5 * jaccard(obs1_tokens, hyp_a_tokens) + 0.5 * jaccard(obs2_tokens, hyp_a_tokens)
    b_obs_support = 0.5 * jaccard(obs1_tokens, hyp_b_tokens) + 0.5 * jaccard(obs2_tokens, hyp_b_tokens)
    support_balance = 1.0 - abs(a_obs_support - b_obs_support)
    hypothesis_similarity = jaccard(hyp_a_tokens, hyp_b_tokens)
    obs_cue_count = sum(token in AMBIGUITY_CUES for token in obs1_tokens | obs2_tokens)
    causal_cues = sum(
        token in {"because", "after", "before", "then", "so", "wanted", "needed", "decided", "learned"}
        for token in obs1_tokens | obs2_tokens | hyp_a_tokens | hyp_b_tokens
    )

    score = (
        4.0 * support_balance
        + 2.0 * hypothesis_similarity
        + 0.5 * obs_cue_count
        + 0.25 * causal_cues
    )
    return score, {
        "a_obs_support": round(a_obs_support, 4),
        "b_obs_support": round(b_obs_support, 4),
        "support_balance": round(support_balance, 4),
        "hypothesis_similarity": round(hypothesis_similarity, 4),
        "obs_cue_count": obs_cue_count,
        "causal_cues": causal_cues,
    }


def is_stable_alphanli(row: dict) -> bool:
    hyp_a = tokenize(row["choices"][0]["text"])
    hyp_b = tokenize(row["choices"][1]["text"])
    similarity = jaccard(hyp_a, hyp_b)
    if similarity < 0.05:
        return False
    if similarity > 0.8:
        return False
    return True


def is_stable_strategyqa(row: dict) -> bool:
    q_tokens = tokenize(row["question"])
    if any(cue in q_tokens for cue in UNSTABLE_STRATEGYQA_CUES):
        return False
    return True


def select_rows(rows: list[dict], dataset: str, limit: int, stable_only: bool = False) -> list[dict]:
    scored = []
    for row in rows:
        if stable_only:
            if dataset == "commonsenseqa" and not is_stable_commonsenseqa(row):
                continue
            if dataset == "strategyqa" and not is_stable_strategyqa(row):
                continue
            if dataset == "alphanli" and not is_stable_alphanli(row):
                continue
        if dataset == "commonsenseqa":
            score, details = commonsenseqa_score(row)
        elif dataset == "strategyqa":
            score, details = strategyqa_score(row)
        elif dataset == "alphanli":
            score, details = alphanli_score(row)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        scored.append((score, row, details))

    scored.sort(key=lambda item: (-item[0], item[1]["task_id"]))
    selected = []
    for score, row, details in scored[:limit]:
        enriched = dict(row)
        metadata = dict(enriched.get("metadata", {}))
        metadata["challenge_score"] = round(score, 4)
        metadata["challenge_details"] = details
        enriched["metadata"] = metadata
        selected.append(enriched)
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["commonsenseqa", "strategyqa", "alphanli"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--stable-only", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    selected = select_rows(rows=rows, dataset=args.dataset, limit=args.limit, stable_only=args.stable_only)
    write_jsonl(Path(args.output), selected)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "input_rows": len(rows),
                "selected_rows": len(selected),
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
