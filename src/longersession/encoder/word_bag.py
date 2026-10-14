"""WordBag encoder：词级词袋，轻量 baseline（适合英文，无需 torch）。"""
from __future__ import annotations

import math
import zlib

from longersession.encoder.base import Encoder


class WordBagEncoder(Encoder):
    """词级词袋 encoder。

    按空白分词，每个词用 zlib.crc32 稳定 hash 到维度，句子向量 = 词频向量（L2 归一化）。
    词级差异能区分话题（比字符级 MockEncoder 强），不依赖任何模型。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _index(self, word: str) -> int:
        return zlib.crc32(word.encode("utf-8")) % self.dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in text.lower().split():
                vec[self._index(word)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm == 0.0:
                result.append(vec)
            else:
                result.append([v / norm for v in vec])
        return result
