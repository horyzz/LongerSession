"""消息模型。"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    """消息角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """工具调用（assistant 消息里的中间过程）。"""

    id: str
    name: str  # tool_name
    arguments: str = ""  # JSON 字符串


class Message(BaseModel):
    """单条消息。"""

    role: Role
    content: str = ""
    # assistant 消息里的工具调用；其余角色通常为空
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # tool 消息用它关联到对应的 tool_call
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
