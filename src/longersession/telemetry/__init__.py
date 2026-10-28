"""Telemetry 本地收集模块（阶段 3）。

每次 process 自动向本地 JSONL 追加一条记录，含原始会话、预测边界、
topic 与 chunk 摘要，用于阶段 4 的 LLM 评估与阶段 5 的回训。
"""
from longersession.telemetry.telemetry import TelemetryRecorder, TelemetryRecord

__all__ = ["TelemetryRecorder", "TelemetryRecord"]
