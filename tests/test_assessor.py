"""LLM 评估器（Assessor）测试（阶段 4）。"""
from longersession.assessor.base import (
    MockAssessor,
    count_turns,
    sanitize_boundaries,
)
from longersession.assessor.factory import create_assessor
from longersession.assessor.openai_compat import OpenAICompatAssessor, _extract_json
from longersession.assessor.prompts import build_assess_messages
from longersession.assessor.runner import run_assessment
from longersession.config import AssessorConfig

MSGS = [
    {"role": "user", "content": "部署 docker"},
    {"role": "assistant", "content": "用 compose"},
    {"role": "user", "content": "镜像管理"},
    {"role": "assistant", "content": "用 tag"},
    {"role": "user", "content": "数据备份"},
    {"role": "assistant", "content": "快照"},
    {"role": "user", "content": "备份频率"},
    {"role": "assistant", "content": "每天"},
]


def test_count_turns_and_sanitize():
    assert count_turns(MSGS) == 4
    assert sanitize_boundaries(MSGS, [2, 99, 2, 0, "x"]) == [2]


def test_mock_assessor_script():
    a = MockAssessor(script={"s1": [2, 99]}, default=[])
    assert a.assess(MSGS, "s1") == [2]
    assert a.assess(MSGS, "other") == []


def test_build_assess_messages():
    msgs = build_assess_messages(MSGS)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "[1] user:" in msgs[1]["content"]
    assert "boundaries" in msgs[1]["content"]


def test_extract_json():
    assert _extract_json('```json\n{"boundaries": [2, 5]}\n```') == {"boundaries": [2, 5]}
    assert _extract_json('result: {"boundaries": []}') == {"boundaries": []}


def test_openai_compat_assess_parse():
    a = OpenAICompatAssessor(api_key="sk-test", model="deepseek-chat")
    a._chat = lambda messages: '{"boundaries": [2, 99, 0]}'
    assert a.assess(MSGS, "s1") == [2]


def test_create_assessor_factory():
    assert isinstance(create_assessor(AssessorConfig(provider="mock")), MockAssessor)
    oa = create_assessor(AssessorConfig(provider="openai", model="m"))
    assert isinstance(oa, OpenAICompatAssessor)
    assert oa.model == "m"


def test_run_assessment_incremental(tmp_path):
    import json

    tp = tmp_path / "sessions.jsonl"
    records = [
        {"session_id": "s1", "encoder": "x", "messages": MSGS, "predicted_boundaries": [2]},
        {"session_id": "s2", "encoder": "x", "messages": MSGS, "predicted_boundaries": []},
    ]
    tp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )

    assessor = MockAssessor(script={"s1": [3], "s2": []})
    out = tmp_path / "eval"
    stats = run_assessment(tp, assessor, out, resume=True)
    assert stats["n_assessed"] == 2
    case = json.loads((out / "telemetry_s1.json").read_text(encoding="utf-8"))
    assert case["boundaries"] == [3]
    assert case["predicted_boundaries"] == [2]

    stats2 = run_assessment(tp, assessor, out, resume=True)
    assert stats2["n_assessed"] == 0
    assert stats2["n_skipped"] == 2
