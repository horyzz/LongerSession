"""telemetry 真值 eval 数据 → 承接样本对（复用 training schema）。

把带真值 boundaries 的 eval case 转成相邻 turn 的 (text_a, text_b, label) 样本对：
- 无边界（同 topic）→ label=1（承接，正样本）
- 有边界（跨 topic）→ label=0（切换，负样本）

text 用 turn.combined_text()（user + assistant，剥离 tool），与切割管线语义输入一致。

用法：
    uv run python -m training.telemetry_pairs \
        --eval-dir data/telemetry/eval \
        --out data/training/pairs_telemetry.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from longersession.models.message import Message
from longersession.segmentation.preprocess import messages_to_turns


def _pair(text_a: str, text_b: str, same: bool, tree_id: str, lang: str) -> dict | None:
    a = text_a.strip()
    b = text_b.strip()
    if not a or not b:
        return None
    return {
        "text_a": a,
        "text_b": b,
        "label": 1 if same else 0,
        "source": "telemetry",
        "pair_type": "same_topic" if same else "cross_topic",
        "lang": lang,
        "tree_id": tree_id,
    }


def eval_case_to_pairs(case: dict) -> list[dict]:
    """单个带真值 eval case → 样本对列表。"""
    messages = case.get("messages", [])
    boundaries = set(case.get("boundaries", []))
    case_id = case.get("case_id", "unknown")
    lang = case.get("language", "en")

    msgs = [Message.model_validate(m) for m in messages]
    turns = messages_to_turns(msgs)
    if len(turns) < 2:
        return []

    pairs: list[dict] = []
    for i in range(len(turns) - 1):
        # boundary 语义：在第 i+1 个 turn（1-based）之后切
        boundary_after = (i + 1) in boundaries
        p = _pair(
            turns[i].combined_text(),
            turns[i + 1].combined_text(),
            same=not boundary_after,
            tree_id=case_id,
            lang=lang,
        )
        if p:
            pairs.append(p)
    return pairs


def eval_cases_to_pairs(cases: list[dict]) -> list[dict]:
    pairs: list[dict] = []
    for c in cases:
        pairs.extend(eval_case_to_pairs(c))
    return pairs


def load_eval_cases(eval_dir: str | Path) -> list[dict]:
    p = Path(eval_dir)
    if not p.exists():
        return []
    return [
        json.load(open(f, "r", encoding="utf-8"))
        for f in sorted(p.glob("*.json"))
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="telemetry 真值 → 训练样本对")
    ap.add_argument("--eval-dir", default="data/telemetry/eval")
    ap.add_argument("--out", default="data/training/pairs_telemetry.jsonl")
    args = ap.parse_args(argv)

    cases = load_eval_cases(args.eval_dir)
    if not cases:
        print("错误：没有读到 eval case（先跑 assessor runner）", file=sys.stderr)
        return 1

    pairs = eval_cases_to_pairs(cases)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter

    print(f"[done] {len(cases)} 个 case → {len(pairs)} 对 → {out}", file=sys.stderr)
    print(f"  按 label: {dict(Counter(r['label'] for r in pairs))}", file=sys.stderr)
    print(f"  按 pair_type: {dict(Counter(r['pair_type'] for r in pairs))}", file=sys.stderr)
    print(f"  按 lang: {dict(Counter(r['lang'] for r in pairs))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
