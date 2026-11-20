"""承接分数计算（长历史版）。

score[i] = cos_sim(enc(history[0..i]), enc(turn_{i+1}))
即：把「第 0..i 个 turn 拼成的历史」和「下一轮 turn」分别 encode，算余弦相似度。
历史越长、下一轮越承接，分数越高；分数低 = 话题切换点。

与训练对齐：训练时 anchor = 长历史（多个 turn 拼接），positive = 下一轮，
in-batch 其它对话做负样本（MultipleNegativesRankingLoss）。
"""
from __future__ import annotations

import numpy as np

from longersession.encoder.base import Encoder
from longersession.models.turn import Turn

SEP = "\n\n"  # turn 分隔符，与训练数据构建一致


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_coherence_scores(
    turns: list[Turn],
    encoder: Encoder,
    use_history: bool = False,
) -> list[float]:
    """计算相邻 turn 之间的承接分数。

    use_history=False（默认）：单 turn 打分（balanced 模型的配方，效果最好）。
        score[i] = cos(enc(turn_i), enc(turn_{i+1}))
    use_history=True：长历史打分（实验性，当前无长历史训练的模型可用）。
        score[i] = cos(enc(join(turn_0..turn_i)), enc(turn_{i+1}))

    返回长度 = len(turns) - 1，分数越低越可能是话题切换点。
    """
    n = len(turns)
    if n < 2:
        return []

    if not use_history:
        texts = [t.combined_text() for t in turns]
        arr = np.asarray(encoder.encode(texts), dtype=float)
        return [_cosine(arr[i], arr[i + 1]) for i in range(n - 1)]

    # 长历史版：一次构建所有 (anchor, candidate) 对，批量 encode
    texts: list[str] = []
    hist_parts: list[str] = []
    for i in range(n - 1):
        hist_parts.append(turns[i].combined_text())
        texts.append(SEP.join(hist_parts))            # anchor = 历史[0..i]
        texts.append(turns[i + 1].combined_text())    # candidate = 下一轮

    arr = np.asarray(encoder.encode(texts), dtype=float)
    return [_cosine(arr[2 * i], arr[2 * i + 1]) for i in range(n - 1)]
