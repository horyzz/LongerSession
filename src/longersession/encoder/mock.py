"""Mock encoder：无模型，字符级确定性 embedding。"""
from __future__ import annotations

import math

from longersession.encoder.base import Encoder


class MockEncoder(Encoder):
    """基于字符分布的确定性 encoder，用于跑通链路（不依赖 torch）。

    字符分布相近的文本会得到相近的向量（弱语义近似），
    仅用于端到端验证，实际语义由 BGEM3Encoder 提供。
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for ch in text:
                vec[ord(ch) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm == 0.0:
                result.append(vec)
            else:
                result.append([v / norm for v in vec])
        return result
