"""构建承接关系对比学习训练集。

用法：
    # 只构建 OASST1
    uv run python -m training.build_dataset --source oasst1 --out data/training/pairs.jsonl

    # 只构建 BELLE（限定行数，快速验证）
    uv run python -m training.build_dataset --source belle --belle-max-rows 50000

    # 两个都构建（默认）
    uv run python -m training.build_dataset

下载走 HuggingFace 镜像（默认 hf-mirror.com，国内直连）。
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from training.belle import belle_pairs
from training.oasst1 import oasst1_pairs

HF_ENDPOINT = "https://hf-mirror.com"

OASST1_REPO = "OpenAssistant/oasst1"
OASST1_FILE = "2023-04-12_oasst_all.messages.jsonl.gz"
BELLE_REPO = "BelleGroup/multiturn_chat_0.8M"
BELLE_FILE = "multiturn_chat_0.8M.json"


def _download(repo: str, filename: str) -> str:
    print(f"[download] {repo} :: {filename}", file=sys.stderr)
    return hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        endpoint=HF_ENDPOINT,
    )


def _iter_jsonl_gz(path: str) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _iter_jsonl(path: str, max_rows: int | None = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stats(pairs: list[dict]) -> str:
    from collections import Counter

    by_source = Counter(p["source"] for p in pairs)
    by_label = Counter(p["label"] for p in pairs)
    by_type = Counter(p["pair_type"] for p in pairs)
    by_lang = Counter(p["lang"] for p in pairs)
    lines = [
        f"总样本对: {len(pairs)}",
        f"按 source: {dict(by_source)}",
        f"按 label : {dict(by_label)}",
        f"按 pair_type: {dict(by_type)}",
        f"按 lang  : {dict(by_lang)}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="构建承接关系对比学习训练集")
    p.add_argument("--source", default="oasst1,belle",
                   help="逗号分隔：oasst1,belle（默认都构建）")
    p.add_argument("--out", default="data/training/pairs.jsonl")
    p.add_argument("--langs", default="en,zh", help="OASST1 语言白名单（逗号分隔）")
    p.add_argument("--cross-neg-ratio", type=float, default=1.0,
                   help="OASST1 跨树负样本相对正样本的比例")
    p.add_argument("--belle-max-rows", type=int, default=100000,
                   help="BELLE 最多读入行数（None=全部）")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="全局样本对上限（None=不限制）")
    p.add_argument("--oasst1-local", default=None,
                   help="本地 OASST1 messages.jsonl.gz 路径（跳过下载）")
    p.add_argument("--belle-local", default=None,
                   help="本地 BELLE multiturn .json 路径（跳过下载，适合大文件）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--endpoint", default=HF_ENDPOINT)
    args = p.parse_args(argv)

    langs = tuple(x.strip() for x in args.langs.split(",") if x.strip())
    sources = {x.strip() for x in args.source.split(",") if x.strip()}

    pairs: list[dict] = []
    rng = random.Random(args.seed)

    if "oasst1" in sources:
        path = args.oasst1_local or _download(OASST1_REPO, OASST1_FILE)
        msgs = _iter_jsonl_gz(path)
        print(f"[oasst1] 读入 {len(msgs)} 条原始消息", file=sys.stderr)
        pairs.extend(oasst1_pairs(msgs, langs=langs,
                                  cross_neg_ratio=args.cross_neg_ratio, rng=rng))

    if "belle" in sources:
        path = args.belle_local or _download(BELLE_REPO, BELLE_FILE)
        rows = _iter_jsonl(path, max_rows=args.belle_max_rows)
        print(f"[belle] 读入 {len(rows)} 行", file=sys.stderr)
        pairs.extend(belle_pairs(rows, max_pairs=args.max_pairs))

    if args.max_pairs is not None and len(pairs) > args.max_pairs:
        rng.shuffle(pairs)
        pairs = pairs[: args.max_pairs]

    rng.shuffle(pairs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] 写入 {len(pairs)} 条 → {out}", file=sys.stderr)
    print(_stats(pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
