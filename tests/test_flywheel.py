"""数据飞轮（回训）测试（阶段 5）。"""
import json

from longersession.config import AssessorConfig
from training.flywheel import build_flywheel_pairs, merge_pairs
from training.telemetry_pairs import eval_case_to_pairs

CASE = {
    "case_id": "telemetry_s1",
    "language": "zh",
    "length": "short",
    "messages": [
        {"role": "user", "content": "部署 docker"},
        {"role": "assistant", "content": "用 compose"},
        {"role": "user", "content": "镜像管理"},
        {"role": "assistant", "content": "用 tag"},
        {"role": "user", "content": "数据备份"},
        {"role": "assistant", "content": "快照"},
        {"role": "user", "content": "备份频率"},
        {"role": "assistant", "content": "每天"},
    ],
    "boundaries": [2],
}


def test_eval_case_to_pairs():
    pairs = eval_case_to_pairs(CASE)
    assert len(pairs) == 3
    assert [p["label"] for p in pairs] == [1, 0, 1]
    assert [p["pair_type"] for p in pairs] == ["same_topic", "cross_topic", "same_topic"]
    assert all(p["source"] == "telemetry" for p in pairs)
    # text 剥离 tool：这里无 tool，但确认 combined_text 语义非空
    assert all(p["text_a"].strip() and p["text_b"].strip() for p in pairs)


def test_flywheel_build_pairs(tmp_path):
    tp = tmp_path / "sessions.jsonl"
    tp.write_text(
        json.dumps(
            {
                "session_id": "s1",
                "encoder": "char_ngram",
                "messages": CASE["messages"],
                "predicted_boundaries": [2],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    eval_dir = tmp_path / "eval"
    pairs_out = tmp_path / "pairs.jsonl"
    stats, n_pairs = build_flywheel_pairs(
        tp, AssessorConfig(provider="mock"), eval_dir, pairs_out
    )
    assert stats["n_assessed"] == 1
    assert n_pairs == 3

    merged = tmp_path / "merged.jsonl"
    total = merge_pairs([pairs_out], merged)
    assert total == 3
