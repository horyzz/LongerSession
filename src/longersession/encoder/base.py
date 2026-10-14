"""Encoder 抽象基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Encoder(ABC):
    """文本编码器抽象：把文本列表编码成 embedding 向量列表。

    用于计算承接分数（coherence score），可插拔：
    - MockEncoder（无模型，跑通链路）
    - BGEM3Encoder（sentence-transformers 加载 BGE-M3）
    """

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """编码文本列表，返回与输入等长的 embedding 列表。"""
        raise NotImplementedError
