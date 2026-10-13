from __future__ import annotations

import re


FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*([^\n\r|]+)", re.IGNORECASE)
NUMERIC_TOKEN_RE = re.compile(
    r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # 1,500 or 12,345.67
    r"|[-+]?\d+\.\d+"                       # 18.5
    r"|[-+]?\.\d+"                           # .5
    r"|[-+]?\d+"                             # 18
)


def extract_final_answer(prediction: str) -> str:
    match = FINAL_ANSWER_RE.search(prediction)
    if match:
        return match.group(1).strip().strip(".")
    return prediction.strip()


def _is_numeric_str(s: str | None) -> bool:
    if s is None:
        return False
    try:
        float(s.replace(",", "").replace("$", "").strip())
        return True
    except (ValueError, AttributeError):
        return False


def _canonicalize_number(raw: str) -> str | None:
    raw = raw.replace(",", "").replace("$", "").strip()
    try:
        v = float(raw)
        if v.is_integer():
            return str(int(v))
        return f"{v:g}"
    except ValueError:
        return None


def _extract_last_number(text: str) -> str | None:
    matches = NUMERIC_TOKEN_RE.findall(text)
    if not matches:
        return None
    return _canonicalize_number(matches[-1])


def normalize_prediction(prediction: str, gold_answer: str | None) -> str:
    extracted = extract_final_answer(prediction).strip()
    if gold_answer is None:
        return extracted

    # Numeric path: handle open-ended numeric answers (e.g., GSM8K).
    if _is_numeric_str(gold_answer):
        num = _extract_last_number(extracted)
        if num is None:
            num = _extract_last_number(prediction)
        return num if num is not None else extracted

    lowered = extracted.lower()
    gold_lower = gold_answer.strip().lower()

    if gold_lower in {"yes", "no"}:
        if lowered in {"a", "yes"}:
            return "yes"
        if lowered in {"b", "no"}:
            return "no"
        return lowered

    label_match = re.match(r"^([A-Za-z])(?:\b|[\).:].*)", extracted)
    if label_match:
        return label_match.group(1).upper()
    return extracted.upper()


def exact_match(prediction: str, gold_answer: str | None) -> int:
    if gold_answer is None:
        return 0
    normalized_prediction = normalize_prediction(prediction, gold_answer)

    # Numeric path
    if _is_numeric_str(gold_answer):
        if not normalized_prediction:
            return 0
        try:
            return int(
                abs(
                    float(normalized_prediction)
                    - float(gold_answer.replace(",", "").replace("$", "").strip())
                )
                < 1e-6
            )
        except (ValueError, TypeError):
            return 0

    normalized_gold = gold_answer.strip().lower()
    if normalized_gold in {"yes", "no"}:
        return int(normalized_prediction == normalized_gold)
    return int(normalized_prediction == gold_answer.strip().upper())
