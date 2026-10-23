"""LongEval（LongChat 版）数据加载与多 topic 测试用例生成。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from longersession.models.message import Message, Role

CONVERSATIONS_JSONL = Path(__file__).parent / "data" / "topics" / "conversations.jsonl"

_TURN_RE = re.compile(r"(USER|ASSISTANT):\s*")


@dataclass
class Conversation:
    topic: str
    topic_id: int
    text: str


@dataclass
class TestCase:
    test_id: str
    num_topics: int
    topics: list[str]          # ground-truth topic 名（按顺序）
    conversation: str          # 拼接后的长对话
    conversations: list[Conversation]


def load_conversations(path: Path = CONVERSATIONS_JSONL) -> list[Conversation]:
    convs: list[Conversation] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            convs.append(
                Conversation(
                    topic=obj["topic"],
                    topic_id=int(obj["topic_id"]),
                    text=obj["conversation"],
                )
            )
    return convs


def parse_turns(text: str) -> list[tuple[str, str]]:
    """把 'USER: ... \\n ASSISTANT: ...' 文本解析为 [(role, content), ...]。"""
    # 按换行切块，去掉空白，再按角色前缀归并
    pieces = re.split(r"\s*\n\s*", text.strip())
    turns: list[tuple[str, str]] = []
    cur_role: str | None = None
    cur_buf: list[str] = []
    for p in pieces:
        if not p:
            continue
        m = _TURN_RE.match(p)
        if m:
            if cur_role is not None and cur_buf:
                turns.append((cur_role, " ".join(cur_buf).strip()))
            cur_role = m.group(1)
            cur_buf = [p[m.end():]]
        else:
            # 续行（同一条消息的后续内容）
            if cur_role is not None:
                cur_buf.append(p)
    if cur_role is not None and cur_buf:
        turns.append((cur_role, " ".join(cur_buf).strip()))
    return turns


def text_to_messages(text: str) -> list[Message]:
    """把对话文本解析为 LongerSession Message 列表（用于走真实索引管线）。"""
    messages: list[Message] = []
    for role, content in parse_turns(text):
        r = Role.USER if role == "USER" else Role.ASSISTANT
        messages.append(Message(role=r, content=content))
    return messages


def generate_test_cases(
    num_topics_list: list[int] | None = None,
    num_samples: int = 2,
    seed: int = 0,
) -> list[TestCase]:
    """生成多 topic 测试用例：拼接 num_topics 段单 topic 对话成一段长 session。"""
    import random

    num_topics_list = num_topics_list or [5, 10]
    convs = load_conversations()
    rng = random.Random(seed)
    cases: list[TestCase] = []

    for k in num_topics_list:
        k = min(k, len(convs))
        for s in range(num_samples):
            chosen = rng.sample(convs, k)
            combined = "".join(c.text for c in chosen)
            cases.append(
                TestCase(
                    test_id=f"topics-{k}-sample-{s}",
                    num_topics=k,
                    topics=[c.topic for c in chosen],
                    conversation=combined,
                    conversations=chosen,
                )
            )
    return cases
