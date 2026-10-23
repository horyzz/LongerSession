"""Topic 召回评估（模糊匹配）。"""
from __future__ import annotations

import re

_NUM_PREFIX = re.compile(r"^\s*(\d+[\.\)、]|[-*•])\s*")

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "its", "it",
    "this", "that", "these", "those", "their", "our", "your", "how", "what",
    "why", "when", "can", "could", "should", "would", "role", "impact",
    "effects", "benefits", "future", "history", "psychology", "culture",
}


def normalize_topic(t: str) -> str:
    t = t.strip().lower()
    t = _NUM_PREFIX.sub("", t).strip()
    t = re.sub(r"\s+", " ", t)
    if t.startswith("the "):
        t = t[4:]
    t = t.strip().strip(".,;:\"'`")
    return t


def _word_set(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", t) if w not in _STOPWORDS}


def topic_matches(gt: str, pred: str) -> bool:
    """判断预测行是否命中某个 ground-truth topic。"""
    g = normalize_topic(gt)
    p = normalize_topic(pred)
    if not g or not p:
        return False
    if g == p:
        return True
    if g in p or p in g:
        return True
    wg = _word_set(g)
    wp = _word_set(p)
    if not wg or not wp:
        return False
    inter = len(wg & wp)
    overlap = inter / min(len(wg), len(wp))
    return overlap >= 0.7 or (inter / len(wg) >= 0.6 and inter / len(wp) >= 0.6)


def topic_recall(ground_truth: list[str], predicted: list[str]) -> dict:
    """计算预测 topic 列表对真值的召回。"""
    matched = []
    used_pred: set[int] = set()
    for gt in ground_truth:
        hit = False
        for i, p in enumerate(predicted):
            if i in used_pred:
                continue
            if topic_matches(gt, p):
                matched.append(gt)
                used_pred.add(i)
                hit = True
                break
        if not hit:
            matched.append(None)
    recall = sum(1 for m in matched if m is not None) / len(ground_truth) if ground_truth else 0.0
    return {
        "recall": round(recall, 4),
        "matched": [m for m in matched if m is not None],
        "missing": [g for g, m in zip(ground_truth, matched) if m is None],
    }
