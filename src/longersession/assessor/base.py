"""LLM 评估器（Assessor）抽象：给会话打 topic-shift 真值边界。"""
from __future__ import annotations

from abc import ABC, abstractmethod


def count_turns(messages: list[dict]) -> int:
    """统计 user turn 数量（与切割管线的 turn 口径一致）。"""
    return sum(1 for m in messages if m.get("role") == "user")


def sanitize_boundaries(messages: list[dict], boundaries: list[int]) -> list[int]:
    """把 LLM 输出的边界清洗为合法边界列表。

    - 只保留 [1, turns-1] 范围内的整数（最后一段之后切无意义）
    - 升序去重
    """
    turns = count_turns(messages)
    valid: list[int] = []
    for b in boundaries:
        try:
            b = int(b)
        except (TypeError, ValueError):
            continue
        if 0 < b < turns and b not in valid:
            valid.append(b)
    return sorted(valid)


class Assessor(ABC):
    """给会话消息打 topic-shift 真值边界的抽象（可插拔）。

    返回 1-based 边界列表，语义为「在第 p 个 turn 之后切」，与 eval 的
    boundaries 口径、切割管线的 cuts 口径一致。
    """

    @abstractmethod
    def assess(self, messages: list[dict], session_id: str | None = None) -> list[int]:
        """给定原始消息，返回 topic-shift 边界列表。"""
        raise NotImplementedError


class MockAssessor(Assessor):
    """脚本化评估器（测试/离线跑通用，不调 LLM）。

    script 按 session_id 映射到预设边界；未命中时返回 default（默认空）。
    """

    def __init__(
        self,
        script: dict[str, list[int]] | None = None,
        default: list[int] | None = None,
    ):
        self.script = script or {}
        self.default = default or []

    def assess(self, messages: list[dict], session_id: str | None = None) -> list[int]:
        if session_id is not None and session_id in self.script:
            return sanitize_boundaries(messages, self.script[session_id])
        return sanitize_boundaries(messages, self.default)
