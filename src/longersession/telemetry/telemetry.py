"""Telemetry 记录模型与 JSONL 追加存储。"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TelemetryRecord(BaseModel):
    """一次 process 的审计记录（本地 JSONL 的一行）。

    记录原始会话（messages）+ 预测边界（predicted_boundaries）+ topic，
    便于阶段 4 用用户自己的 LLM 补打 topic-shift 真值，阶段 5 回训。
    """

    version: str = "1"  # schema 版本，便于后续演化
    timestamp: str  # ISO 8601（UTC）
    session_id: str
    previous: str | None = None
    encoder: str = ""  # encoder provider 名（审计用）
    num_turns: int = 0
    num_chunks: int = 0
    messages: list[dict[str, Any]] = Field(default_factory=list)  # 原始消息
    predicted_boundaries: list[int] = Field(default_factory=list)  # 1-based 切点
    topics: list[str] = Field(default_factory=list)  # chunk topic 名（去空）
    chunks: list[dict[str, Any]] = Field(default_factory=list)  # chunk 摘要


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TelemetryRecorder:
    """把 TelemetryRecord 追加到本地 JSONL。

    - 目录自动创建
    - 每次写入单行 JSON + flush（append 模式，单行追加对多进程也基本安全）
    - 线程内加锁，避免并发交错
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, record: TelemetryRecord | dict[str, Any]) -> None:
        """追加一条记录。"""
        if isinstance(record, TelemetryRecord):
            record = record.model_dump(mode="json")
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

    def close(self) -> None:
        """JSONL 为 append 即写即落盘，无缓冲；close 仅作占位保持对称。"""
        return None


def now_iso() -> str:
    return _now_iso()
