"""Chunk 模型：语义切割结果。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """语义切割产生的一个区间（chunk）。

    一个 chunk 由若干连续 turn（锚点）聚合而成，允许与相邻 chunk 重叠
    （overlap 通过共享边界附近的 turn 实现）。
    """

    chunk_id: str
    session_id: str
    # 本 chunk 包含的 turn id（锚点序列，可能与前/后 chunk 重叠）
    turn_ids: list[str] = Field(default_factory=list)
    # topic：LLM 自由生成的精确标签（开放集合，不归一化），可空
    topic: str | None = None
    # tag：后学习增量聚类得到的归一化标签（可空，非预定义）
    tag: str | None = None
    # 本 chunk 用到的 tool_name 列表（合并各 turn，去重）
    used_tools: list[str] = Field(default_factory=list)
    # 是否与前一 chunk 共享锚点（overlap）
    overlap_prev: bool = False
    # 指向原文的引用（offset 或存储地址）
    content_ref: str = ""
    # token 数（切割约束/统计用）
    token_count: int = 0
