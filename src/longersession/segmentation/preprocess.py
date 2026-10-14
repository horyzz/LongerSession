"""预处理：消息清洗 + turn 分组。"""
from __future__ import annotations

from longersession.models.message import Message, Role
from longersession.models.turn import Turn

# 无语义的短 user 消息（后续迭代可用于合并，当前版本保留为独立 turn）
TRIVIAL_USER_TEXTS = {"继续", "ok", "好的", "嗯", "收到", "continue", "go on"}


def is_empty_message(m: Message) -> bool:
    """消息是否为空（无正文且无工具调用）。"""
    return not m.content.strip() and not m.tool_calls


def clean_messages(messages: list[Message]) -> list[Message]:
    """丢弃空消息，返回清洗后的消息序列。"""
    return [m for m in messages if not is_empty_message(m)]


def messages_to_turns(messages: list[Message]) -> list[Turn]:
    """把消息序列组装成 turn 列表。

    每个 turn 从一条 user 消息开始，直到下一条 user 消息之前的所有
    assistant / tool 消息都归入该 turn。system 消息与空消息被跳过。
    """
    turns: list[Turn] = []
    current_user: Message | None = None
    current_assistant: list[Message] = []
    counter = 0

    for m in messages:
        if m.role == Role.SYSTEM:
            continue
        if is_empty_message(m):
            continue
        if m.role == Role.USER:
            if current_user is not None:
                counter += 1
                turns.append(
                    Turn(
                        turn_id=f"turn_{counter}",
                        user=current_user,
                        assistant=current_assistant,
                    )
                )
            current_user = m
            current_assistant = []
        else:
            current_assistant.append(m)

    # 收尾最后一个 turn
    if current_user is not None:
        counter += 1
        turns.append(
            Turn(
                turn_id=f"turn_{counter}",
                user=current_user,
                assistant=current_assistant,
            )
        )

    return turns
