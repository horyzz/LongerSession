"""承接关系 encoder 对比学习训练（SimCSE 式 in-batch negatives）。

用 (text_a, text_b, label=1) 正样本对训练一个 sentence-transformer 模型，
loss = MultipleNegativesRankingLoss（in-batch 其它对自动做负样本）。

用法：
    # 本地（CPU / GPU 自动检测）
    uv run python -m training.train --data data/training/pairs_oasst1.jsonl

    # 指定模型 / 超参
    uv run python -m training.train --data data/training/pairs_oasst1.jsonl \
        --model BAAI/bge-m3 --epochs 3 --batch-size 16 --lr 2e-5 --fp16
"""
from __future__ import annotations

import argparse
import json
import sys

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

try:  # sentence-transformers >= 6
    from sentence_transformers.sentence_transformer.losses import (
        MultipleNegativesRankingLoss,
    )
except ImportError:  # 兼容旧版
    from sentence_transformers.losses import MultipleNegativesRankingLoss


def load_positive_pairs(path: str, max_pairs: int | None = None) -> list[dict]:
    """从 JSONL 读入 label=1 的正样本对，返回 [{anchor, positive}, ...]。"""
    pairs: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("label") == 1 and r.get("text_a") and r.get("text_b"):
                pairs.append({"anchor": r["text_a"], "positive": r["text_b"]})
            if max_pairs is not None and len(pairs) >= max_pairs:
                break
    return pairs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="对比学习训练承接关系 encoder")
    p.add_argument("--data", required=True, help="训练 JSONL（build_dataset 产出）")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                   help="基座 sentence-transformer 模型（bge-small 起步，可换 bge-m3）")
    p.add_argument("--output-dir", default="data/training/model")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--scale", type=float, default=20.0,
                   help="MNRL 温度系数（越小越尖锐）")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="最多用多少正样本对（调试用）")
    p.add_argument("--fp16", action="store_true", help="GPU 上开启混合精度")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    pairs = load_positive_pairs(args.data, args.max_pairs)
    if not pairs:
        print("错误：没有读到正样本对", file=sys.stderr)
        return 1
    print(f"正样本对数量: {len(pairs)}", file=sys.stderr)

    model = SentenceTransformer(args.model)
    dataset = Dataset.from_list(pairs)  # 列：anchor, positive
    loss = MultipleNegativesRankingLoss(model, scale=args.scale)

    train_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=False,
        logging_steps=50,
        save_strategy="epoch",
        seed=args.seed,
        report_to=[],            # 关闭 wandb 等上报
        dataloader_drop_last=True,  # 避免最后 batch 只剩 1 个导致 in-batch 无负样本
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        loss=loss,
    )
    trainer.train()

    # 保存最终模型（sentence-transformers 格式，可直接被 SentenceTransformer 加载）
    model.save_pretrained(args.output_dir)
    print(f"[saved] {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
