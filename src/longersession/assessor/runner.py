"""评估 runner：把 telemetry JSONL 批量/增量评估成带真值的 eval 数据。

用法：
    from longersession.assessor.factory import create_assessor
    from longersession.assessor.runner import run_assessment
    run_assessment("data/telemetry/sessions.jsonl", assessor, "data/telemetry/eval")

    # CLI
    uv run python -m longersession.assessor.runner \
        --telemetry data/telemetry/sessions.jsonl \
        --out data/telemetry/eval --provider openai --resume
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from longersession.assessor.base import Assessor
from longersession.assessor.factory import create_assessor
from longersession.config import AssessorConfig


def load_telemetry(path: str | Path) -> list[dict]:
    """读 telemetry JSONL，返回记录列表（跳过空行）。"""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def detect_language(messages: list[dict]) -> str:
    """按 CJK 字符占比粗判语言（zh / en）。"""
    text = "".join(m.get("content", "") or "" for m in messages)
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return "zh" if cjk > 0 and cjk >= ascii_letters else "en"


def detect_length(messages: list[dict]) -> str:
    """按 user turn 数粗判长度（short / long）。"""
    turns = sum(1 for m in messages if m.get("role") == "user")
    return "long" if turns >= 10 else "short"


def _safe_id(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)


def to_eval_case(record: dict, boundaries: list[int]) -> dict:
    """把一条 telemetry 记录 + 评估真值转成统一 eval case。"""
    session_id = record.get("session_id", "")
    return {
        "case_id": f"telemetry_{_safe_id(session_id)}",
        "language": detect_language(record.get("messages", [])),
        "length": detect_length(record.get("messages", [])),
        "session_id": session_id,
        "encoder": record.get("encoder", ""),
        "messages": record.get("messages", []),
        "boundaries": boundaries,  # LLM 真值
        "predicted_boundaries": record.get("predicted_boundaries", []),  # 管线预测
    }


def assess_records(records: list[dict], assessor: Assessor) -> list[dict]:
    """对每条记录跑 assessor，返回 eval case 列表。"""
    cases: list[dict] = []
    for rec in records:
        boundaries = assessor.assess(rec.get("messages", []), rec.get("session_id"))
        cases.append(to_eval_case(rec, boundaries))
    return cases


def run_assessment(
    telemetry_path: str | Path,
    assessor: Assessor,
    out_dir: str | Path,
    resume: bool = True,
    seed: int = 0,
    sample: int = 0,
    sample_out: str | Path | None = None,
) -> dict:
    """批量/增量评估，写统一 eval case JSON。

    - resume=True 时跳过已存在的 case（按 session_id 落文件名，增量续跑）。
    - sample>0 时导出 N 条抽检样本到 sample_out（人工复核用）。
    - 返回统计 dict。
    """
    records = load_telemetry(telemetry_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_assessed = 0
    n_skipped = 0
    written_cases: list[dict] = []
    for rec in records:
        case = to_eval_case(rec, [])  # 先算 case_id（不含真值也稳定）
        case_path = out / f"{case['case_id']}.json"
        if resume and case_path.exists():
            n_skipped += 1
            continue
        boundaries = assessor.assess(rec.get("messages", []), rec.get("session_id"))
        case["boundaries"] = boundaries
        case_path.write_text(
            json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written_cases.append(case)
        n_assessed += 1

    stats: dict = {
        "n_total": len(records),
        "n_assessed": n_assessed,
        "n_skipped": n_skipped,
        "out_dir": str(out),
    }

    if sample > 0:
        sampled = written_cases
        if len(sampled) > sample:
            sampled = random.Random(seed).sample(sampled, sample)
        if sample_out is not None:
            sp = Path(sample_out)
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text(
                json.dumps(sampled, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            stats["sample_out"] = str(sp)
            stats["n_sampled"] = len(sampled)

    stats["consistency"] = consistency_summary(records, written_cases)
    return stats


def consistency_summary(records: list[dict], cases: list[dict]) -> dict:
    """预测边界 vs 真值边界的一致性信号（严格微平均 boundary F1）。

    反映当前切割管线与 LLM 真值的贴合度，用于抽检时快速发现异常。
    """
    by_session = {c.get("session_id"): c for c in cases}
    gold: list[list[int]] = []
    pred: list[list[int]] = []
    for rec in records:
        c = by_session.get(rec.get("session_id"))
        if c is None:
            continue
        gold.append(c.get("boundaries", []))
        pred.append(rec.get("predicted_boundaries", []))
    return _micro_boundary_f1(gold, pred)


def _micro_boundary_f1(gold: list[list[int]], pred: list[list[int]]) -> dict:
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        gs, ps = set(g), list(p)
        matched: set[int] = set()
        case_tp = 0
        for b in ps:
            for gb in gs:
                if gb in matched:
                    continue
                if abs(b - gb) == 0:
                    case_tp += 1
                    matched.add(gb)
                    break
        tp += case_tp
        fp += len(ps) - case_tp
        fn += len(gs) - len(matched)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="telemetry → 带真值 eval 数据")
    p.add_argument("--telemetry", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--provider", default="openai")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default="")
    p.add_argument("--model", default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--sample-out", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = AssessorConfig(provider=args.provider)
    if args.base_url:
        cfg.base_url = args.base_url
    if args.api_key:
        cfg.api_key = args.api_key
    if args.model:
        cfg.model = args.model
    assessor = create_assessor(cfg)

    stats = run_assessment(
        args.telemetry, assessor, args.out, resume=args.resume,
        seed=args.seed, sample=args.sample, sample_out=args.sample_out,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
