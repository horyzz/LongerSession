"""preprocess 单元测试。"""
from longersession.models.message import Message, Role, ToolCall
from longersession.segmentation.preprocess import (
    clean_messages,
    is_empty_message,
    messages_to_turns,
)


def test_is_empty_message():
    assert is_empty_message(Message(role=Role.USER, content="")) is True
    assert is_empty_message(Message(role=Role.USER, content="  ")) is True
    assert is_empty_message(Message(role=Role.USER, content="hi")) is False
    # 有 tool_call 的 assistant 消息不算空
    assert (
        is_empty_message(
            Message(role=Role.ASSISTANT, content="", tool_calls=[ToolCall(id="1", name="x")])
        )
        is False
    )


def test_clean_messages():
    msgs = [
        Message(role=Role.USER, content="hi"),
        Message(role=Role.ASSISTANT, content=""),
    ]
    assert len(clean_messages(msgs)) == 1


def test_messages_to_turns_skips_system_and_empty():
    msgs = [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.USER, content="q1"),
        Message(role=Role.ASSISTANT, content="a1"),
        Message(role=Role.ASSISTANT, content=""),  # empty, should be dropped
        Message(role=Role.USER, content="q2"),
        Message(role=Role.ASSISTANT, content="a2"),
    ]
    turns = messages_to_turns(msgs)
    assert len(turns) == 2
    assert turns[0].user.content == "q1"
    assert [m.content for m in turns[0].assistant] == ["a1"]
    assert turns[1].user.content == "q2"


def test_messages_to_turns_keeps_tool_messages():
    msgs = [
        Message(role=Role.USER, content="q"),
        Message(
            role=Role.ASSISTANT,
            content="a",
            tool_calls=[ToolCall(id="1", name="read_file", arguments="{}")],
        ),
        Message(role=Role.TOOL, content="file content", tool_call_id="1"),
        Message(role=Role.ASSISTANT, content="done"),
    ]
    turns = messages_to_turns(msgs)
    assert len(turns) == 1
    assert turns[0].used_tools() == ["read_file"]
    # tool 消息保留在 assistant 侧，不丢失原文
    assert any(m.role == Role.TOOL for m in turns[0].assistant)
