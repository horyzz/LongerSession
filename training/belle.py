"""BELLE（multiturn_chat_0.8M）多轮对话 → 承接样本对。

BELLE 的多轮版每条记录格式：
    {"instruction": "Human:你好…\nAssistant:当然…\nHuman:…\nAssistant:", "input": "", "output": "…"}

其中 instruction 是 Human/Assistant 交替的多轮历史，末尾 "Assistant:" 为空，
output 是补全最后一轮 assistant 的回答。

解析策略：
- 用正则按行首 "Human:" / "Assistant:" 切分 instruction → turn 序列。
- 末尾空的 assistant turn 用 output 填充。
- 相邻 turn 组成正样本：Human→Assistant 记 "qa"，Assistant→Human 记 "aq"。

注意：BELLE 是 ChatGPT 生成的数据，只有正样本（无人工 rank/质量标签），
负样本交给训练时的 in-batch negatives。
"""
from __future__ import annotations

import re

_SPLIT_RE = re.compile(r"\n(?=(?:Human|Assistant):)")


def parse_turns(instruction: str, output: str) -> list[tuple[str, str]]:
    """把 instruction+output 解析成 [(role, text), ...]，role ∈ {user, assistant}。"""
    parts = _SPLIT_RE.split(instruction.strip())
    turns: list[tuple[str, str]] = []
    for p in parts:
        p = p.strip()
        if p.startswith("Human:"):
            turns.append(("user", p[len("Human:"):].strip()))
        elif p.startswith("Assistant:"):
            turns.append(("assistant", p[len("Assistant:"):].strip()))
        elif p:
            # 无前缀的残留文本，归到上一个 turn（容错）
            if turns:
                turns[-1] = (turns[-1][0], turns[-1][1] + " " + p)
            else:
                turns.append(("user", p))

    out = output.strip()
    if turns and turns[-1][0] == "assistant" and not turns[-1][1]:
        turns[-1] = ("assistant", out)
    elif out:
        turns.append(("assistant", out))
    return turns


def belle_pairs(
    rows: list[dict],
    max_pairs: int | None = None,
) -> list[dict]:
    """把 BELLE 行列表转成样本对列表。"""
    pairs: list[dict] = []
    for idx, row in enumerate(rows):
        instruction = row.get("instruction") or ""
        output = row.get("output") or ""
        turns = parse_turns(instruction, output)
        for i in range(len(turns) - 1):
            ra, ta = turns[i]
            rb, tb = turns[i + 1]
            if not ta or not tb:
                continue
            if ra == "user" and rb == "assistant":
                pair_type = "qa"
            elif ra == "assistant" and rb == "user":
                pair_type = "aq"
            else:
                continue
            pairs.append(
                {
                    "text_a": ta,
                    "text_b": tb,
                    "label": 1,
                    "source": "belle",
                    "pair_type": pair_type,
                    "lang": "zh",
                    "tree_id": str(idx),
                }
            )
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs
