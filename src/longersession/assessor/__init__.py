"""LLM 评估器（Assessor）模块（阶段 4）。

用户用自己的 LLM 给 telemetry 收集的会话打 topic-shift 真值边界，
产出带真值的 eval 数据，供阶段 5 回训。
"""
from longersession.assessor.base import (
    Assessor,
    MockAssessor,
    count_turns,
    sanitize_boundaries,
)
from longersession.assessor.prompts import build_assess_messages
from longersession.assessor.factory import create_assessor

__all__ = [
    "Assessor",
    "MockAssessor",
    "count_turns",
    "sanitize_boundaries",
    "build_assess_messages",
    "create_assessor",
]
