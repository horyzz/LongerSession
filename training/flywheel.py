"""数据飞轮端到端编排：telemetry → LLM 评估 → 训练样本对 → 回训。

把阶段 3/4/5 串起来的一条命令：

    # 只跑「评估 + 构建样本对」（默认，不训练）
    uv run python -m training.flywheel \
        --telemetry data/telemetry/sessions.jsonl \
        --eval-dir data/telemetry/eval \
        --pairs-out data/training/pairs_telemetry.jsonl

    # 合并已有训练数据 + 触发回训（需 uv sync --extra train）
    uv run python -m training.flywheel \
        --telemetry data/telemetry/sessions.jsonl \
        --merge data/training/pairs_merged.jsonl \
        --train --model BAAI/bge-m3 \
        --output-dir data/training/model_flywheel --epochs 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from longersession.assessor.factory import create_assessor
from longersession.assessor.runner import run_assessment
from longersession.config import AssessorConfig
from training.telemetry_pairs import eval_cases_to_pairs, load_eval_cases


def merge_pairs(files: list[str | Path], out: str | Path) -> int:
    """合并多个训练样本对 JSONL，返回总对数。"""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w", encoding="utf-8") as fo:
        for f in files:
            with open(f, "r", encoding="utf-8") as fi:
                for line in fi:
                    line = line.strip()
                    if line:
                        fo.write(line + "\n")
                        total += 1
    return total


def build_flywheel_pairs(
    telemetry_path: str | Path,
    assessor_config: AssessorConfig,
    eval_dir: str | Path,
    pairs_out: str | Path,
    resume: bool = True,
    sample: int = 0,
    sample_out: str | Path | None = None,
    seed: int = 0,
) -> tuple[dict, int]:
    """评估 + 构建样本对，返回 (评估统计, 样本对数)。"""
    assessor = create_assessor(assessor_config)
    stats = run_assessment(
        telemetry_path, assessor, eval_dir, resume=resume,
        seed=seed, sample=sample, sample_out=sample_out,
    )
    cases = load_eval_cases(eval_dir)
    pairs = eval_cases_to_pairs(cases)
    out = Path(pairs_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return stats, len(pairs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="数据飞轮：telemetry → 评估 → 样本对 → 回训")
    p.add_argument("--telemetry", default="data/telemetry/sessions.jsonl")
    p.add_argument("--eval-dir", default="data/telemetry/eval")
    p.add_argument("--pairs-out", default="data/training/pairs_telemetry.jsonl")

    p.add_argument("--provider", default="openai")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default="")
    p.add_argument("--model", default=None)
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--sample-out", default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--merge", action="append", default=[],
                   help="额外合并的已有训练 JSONL（可多次指定）")
    p.add_argument("--merged-out", default="data/training/pairs_flywheel.jsonl")

    p.add_argument("--train", action="store_true", help="评估后触发回训")
    p.add_argument("--output-dir", default="data/training/model_flywheel")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args(argv)

    cfg = AssessorConfig(provider=args.provider)
    if args.base_url:
        cfg.base_url = args.base_url
    if args.api_key:
        cfg.api_key = args.api_key
    if args.model:
        cfg.model = args.model

    stats, n_pairs = build_flywheel_pairs(
        args.telemetry, cfg, args.eval_dir, args.pairs_out,
        resume=args.resume, sample=args.sample, sample_out=args.sample_out,
        seed=args.seed,
    )
    print(f"[assess] {json.dumps(stats, ensure_ascii=False)}", file=sys.stderr)
    print(f"[pairs] {n_pairs} 对 → {args.pairs_out}", file=sys.stderr)

    train_data = args.pairs_out
    if args.merge:
        merge_pairs([args.pairs_out, *args.merge], args.merged_out)
        train_data = args.merged_out
        print(f"[merge] 合并 {len(args.merge) + 1} 个文件 → {args.merged_out}", file=sys.stderr)

    if args.train:
        # 惰性导入，避免未装 train extra 时 import 失败
        from training.train import main as train_main

        train_argv = [
            "--data", train_data,
            "--model", args.model or "BAAI/bge-m3",
            "--output-dir", args.output_dir,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
        ]
        if args.fp16:
            train_argv.append("--fp16")
        print(f"[train] 回训数据 {train_data} → {args.output_dir}", file=sys.stderr)
        return train_main(train_argv)

    print(f"[next] 回训：uv run python -m training.train --data {train_data} "
          f"--model BAAI/bge-m3 --output-dir {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
