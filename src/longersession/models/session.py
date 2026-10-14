"""Session 索引模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from longersession.models.chunk import Chunk


class LabelRef(BaseModel):
    """标签 -> chunk 引用（L1 粗索引条目，label 是 topic 或 tag 名）。"""

    label: str
    chunk_ids: list[str] = Field(default_factory=list)


class SessionIndex(BaseModel):
    """单个 session 的索引（immutable，process 后不再变）。

    L1: topic_refs / tag_refs（标签 -> chunk 引用，小，可聚合）
    L2: chunks（chunk 本体，大，只在本 session 存）
    原文（turns）由存储层管理，通过 chunk.turn_ids 定位。
    """

    session_id: str
    # 链表指针，指向父 session（null 表示根 session）
    previous: str | None = None
    topic_refs: list[LabelRef] = Field(default_factory=list)
    tag_refs: list[LabelRef] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None
