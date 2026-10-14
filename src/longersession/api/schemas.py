"""API 请求/响应 schema。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from longersession.models.message import Message


class ProcessRequest(BaseModel):
    session_id: str
    previous: str | None = None
    messages: list[Message]


class ProcessResponse(BaseModel):
    session_id: str
    num_turns: int
    num_chunks: int
    topics: list[str]


class QueryRequest(BaseModel):
    session_id: str
    # 可选：指定标签则返回该标签下的 chunk 引用；不指定则只返回标签列表
    label: str | None = None
    kind: str | None = None  # "topic" | "tag"


class QueryResponse(BaseModel):
    # 沿 previous 链聚合到的全部 topic / tag 名（agent 自行选择）
    topics: list[str]
    tags: list[str]
    # 命中标签的 chunk 引用（session_id, chunk_id），未指定 label 时为空
    label: str | None = None
    kind: str | None = None
    chunk_refs: list[tuple[str, str]] = Field(default_factory=list)
