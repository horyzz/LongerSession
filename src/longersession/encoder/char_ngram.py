"""CharNGram encoder：字符 n-gram，中英文通用（无需分词，无需 torch）。"""
from __future__ import annotations

import math
import zlib

from longersession.encoder.base import Encoder


class CharNGramEncoder(Encoder):
    """字符 n-gram 词袋 encoder。

    - 英文：字符 trigram 能捕捉词级信息（"deploy" → "dep","epl","plo","loy"）
    - 中文：字符 trigram 能捕捉词级信息（"部署方案" → "部署方","署方案"）
    - 无需分词，中英文通用，零模型依赖
    """

    def __init__(self, n: int = 3, dim: int = 512):
        self.n = n
        self.dim = dim

    def _index(self, gram: str) -> int:
        return zlib.crc32(gram.encode("utf-8")) % self.dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            s = text.lower()
            vec = [0.0] * self.dim
            if len(s) < self.n:
                vec[self._index(s)] += 1.0
            else:
                for i in range(len(s) - self.n + 1):
                    vec[self._index(s[i : i + self.n])] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm == 0.0:
                result.append(vec)
            else:
                result.append([v / norm for v in vec])
        return result
