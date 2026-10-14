"""承接分数计算。"""
from __future__ import annotations

import numpy as np

from longersession.encoder.base import Encoder
from longersession.models.turn import Turn


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_coherence_scores(turns: list[Turn], encoder: Encoder) -> list[float]:
    """计算相邻 turn 之间的承接分数。

    score[i] = cos_sim(enc(turn_i), enc(turn_{i+1}))，长度 = len(turns) - 1。
    分数越低，越可能是话题切换点（切点代价 = score）。
    """
    if len(turns) < 2:
        return []

    texts = [t.combined_text() for t in turns]
    embeddings = encoder.encode(texts)
    arr = np.asarray(embeddings, dtype=float)

    scores: list[float] = []
    for i in range(len(arr) - 1):
        scores.append(_cosine(arr[i], arr[i + 1]))
    return scores
