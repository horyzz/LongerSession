"""增量 tag 聚类：tag 名 = 锚点 topic（派生值，避免重新上 tag）。"""
from __future__ import annotations

import numpy as np

from longersession.encoder.base import Encoder


def _cosine(a: list[float], b: list[float]) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(arr_a) * np.linalg.norm(arr_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / denom)


class TagClusterer:
    """增量地把 topic 聚类成 tag。

    - tag 名 = 锚点 topic（第一个进入簇的 topic），固定不变 → 不存在"重新上 tag"
    - 新 topic 和现有各 tag 的锚点算 embedding 相似度：
      高于阈值 → 归入最近 tag；否则新建 tag（锚点 = 该 topic）
    - 这是离线后学习过程，embedding 只在这里 + 切割时使用
    """

    def __init__(self, encoder: Encoder, threshold: float = 0.7):
        self.encoder = encoder
        self.threshold = threshold
        # tag（锚点）-> 成员 topic 列表
        self.clusters: dict[str, list[str]] = {}
        self._emb_cache: dict[str, list[float]] = {}

    def _emb(self, topic: str) -> list[float]:
        if topic not in self._emb_cache:
            self._emb_cache[topic] = self.encoder.encode([topic])[0]
        return self._emb_cache[topic]

    def add(self, topic: str) -> str:
        """把 topic 归入最近的 tag 或新建 tag，返回 tag 名。"""
        for tag, members in self.clusters.items():
            if topic in members:
                return tag

        emb = self._emb(topic)
        best_tag: str | None = None
        best_sim = -1.0
        for tag in self.clusters:
            sim = _cosine(emb, self._emb(tag))
            if sim > best_sim:
                best_sim = sim
                best_tag = tag

        if best_tag is not None and best_sim >= self.threshold:
            self.clusters[best_tag].append(topic)
            return best_tag

        # 新建 tag，锚点 = 该 topic
        self.clusters[topic] = [topic]
        return topic

    def tag_of(self, topic: str) -> str | None:
        """返回 topic 所属的 tag 名（未聚类返回 None）。"""
        for tag, members in self.clusters.items():
            if topic in members:
                return tag
        return None

    def all_tags(self) -> list[str]:
        return list(self.clusters.keys())
