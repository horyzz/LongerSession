"""telemetry 本地收集测试（阶段 3）。"""
import json

from fastapi.testclient import TestClient

from longersession.api.app import create_app
from longersession.config import Config, StorageConfig, TelemetryConfig
from longersession.telemetry import TelemetryRecorder, TelemetryRecord


def test_recorder_appends_jsonl(tmp_path):
    path = tmp_path / "sub" / "sessions.jsonl"
    recorder = TelemetryRecorder(path)
    recorder.record(
        TelemetryRecord(
            timestamp="2026-10-24T00:00:00+00:00",
            session_id="s1",
            encoder="char_ngram",
            num_turns=4,
            num_chunks=2,
            messages=[{"role": "user", "content": "hi"}],
            predicted_boundaries=[2],
            topics=["部署"],
            chunks=[{"chunk_id": "chunk_0", "turn_ids": ["turn_1"]}],
        )
    )
    recorder.record({"session_id": "s2"})
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    d0 = json.loads(lines[0])
    assert d0["session_id"] == "s1"
    assert d0["predicted_boundaries"] == [2]
    assert d0["topics"] == ["部署"]
    assert json.loads(lines[1])["session_id"] == "s2"


def test_process_writes_telemetry(tmp_path):
    cfg = Config(
        storage=StorageConfig(db_path=tmp_path / "e2e.db"),
        telemetry=TelemetryConfig(enabled=True, path=tmp_path / "sessions.jsonl"),
    )
    app = create_app(cfg)
    messages = [
        {"role": "user", "content": "我想部署服务，Docker 方案怎么样"},
        {"role": "assistant", "content": "Docker 很适合小项目"},
        {"role": "user", "content": "数据备份怎么做"},
        {"role": "assistant", "content": "定期快照"},
        {"role": "user", "content": "备份频率怎么定"},
        {"role": "assistant", "content": "每天增量"},
    ]
    with TestClient(app) as c:
        r = c.post("/process", json={"session_id": "s1", "messages": messages})
        assert r.status_code == 200

    lines = (tmp_path / "sessions.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["session_id"] == "s1"
    assert rec["num_turns"] == 3
    assert len(rec["messages"]) == 6
    assert "predicted_boundaries" in rec
    assert "topics" in rec
    assert rec["encoder"] == "mock"


def test_process_telemetry_disabled(tmp_path):
    cfg = Config(
        storage=StorageConfig(db_path=tmp_path / "e2e.db"),
        telemetry=TelemetryConfig(enabled=False, path=tmp_path / "sessions.jsonl"),
    )
    app = create_app(cfg)
    with TestClient(app) as c:
        r = c.post(
            "/process",
            json={
                "session_id": "s1",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            },
        )
        assert r.status_code == 200
    assert not (tmp_path / "sessions.jsonl").exists()
