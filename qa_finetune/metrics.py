"""Simple, dependency-free answer-quality metrics for QA."""

from __future__ import annotations

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = (text or "").lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def contains_answer(pred: str, gold: str) -> float:
    """1 if the full gold answer appears inside the prediction (useful for chatty models)."""
    g = normalize(gold)
    return float(bool(g) and g in normalize(pred))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common: dict[str, int] = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F1 based on the longest common subsequence of tokens."""
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    prev = [0] * (len(g) + 1)
    for tp in p:
        cur = [0] * (len(g) + 1)
        for j, tg in enumerate(g, 1):
            cur[j] = prev[j - 1] + 1 if tp == tg else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(g)
    return 2 * precision * recall / (precision + recall)


METRICS = {
    "exact_match": exact_match,
    "contains": contains_answer,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
}


def score_all(preds: list[str], golds: list[str]) -> dict[str, float]:
    if not golds:
        return {}
    return {name: sum(fn(p, g) for p, g in zip(preds, golds)) / len(golds)
            for name, fn in METRICS.items()}
