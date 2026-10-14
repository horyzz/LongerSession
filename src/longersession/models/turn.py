"""Turn 模型：user + assistant 组合（切割单元）。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from longersession.models.message import Message, Role


class Turn(BaseModel):
    """一个 user turn 及其对应的 assistant 侧消息序列。

    assistant 侧包含 assistant 回复与中间的 tool 结果消息，
    直到下一个 user 消息之前的所有内容。
    """

    turn_id: str
    user: Message  # user 消息（锚点）
    assistant: list[Message] = Field(default_factory=list)

    def user_text(self) -> str:
        """user 输入的纯文本。"""
        return self.user.content

    def assistant_text(self) -> str:
        """assistant 的纯文本，剥离 tool 结果与工具调用。"""
        parts: list[str] = []
        for m in self.assistant:
            if m.role == Role.ASSISTANT and m.content:
                parts.append(m.content)
        return "\n".join(parts)

    def used_tools(self) -> list[str]:
        """本 turn 用到的 tool_name 列表（去重、保序）。"""
        names: list[str] = []
        seen: set[str] = set()
        for m in self.assistant:
            if m.role != Role.ASSISTANT:
                continue
            for tc in m.tool_calls:
                if tc.name not in seen:
                    seen.add(tc.name)
                    names.append(tc.name)
        return names

    def combined_text(self) -> str:
        """user + assistant 文本（切割语义输入，剥离 tool）。"""
        parts = [self.user_text()]
        at = self.assistant_text()
        if at:
            parts.append(at)
        return "\n".join(parts)
