"""OASST1（OpenAssistant）树 → 承接样本对。

OASST1 是「对话树」：一个 prompt 挂多个 assistant 候选回复，回复又继续被追问。
我们把它转成 (text_a, text_b, label) 样本对：

- 正样本（label=1）：树中的父子边。
    prompter→assistant 记 pair_type="qa"（同轮问答承接）
    assistant→prompter 记 pair_type="aq"（跨轮话题延续）
- 硬负样本（label=0，pair_type="hard_neg"）：同一 prompt 的多个 assistant 候选回复里，
    rank>0 的（非最优）回复相对父消息是不承接的。
- 跨树负样本（label=0，pair_type="cross_neg"）：不同树随机配对。

过滤条件：未删除、通过评审（review_result=True）、树状态 ready_for_export、
角色为 prompter/assistant、文本非空、语言在白名单内。
"""
from __future__ import annotations

import random


def _filter_messages(messages: list[dict], langs: tuple[str, ...]) -> list[dict]:
    kept: list[dict] = []
    for m in messages:
        if m.get("deleted"):
            continue
        if not m.get("review_result"):
            continue
        if m.get("tree_state") != "ready_for_export":
            continue
        if m.get("role") not in ("prompter", "assistant"):
            continue
        if m.get("lang") not in langs:
            continue
        text = (m.get("text") or "").strip()
        if not text:
            continue
        m = dict(m)
        m["text"] = text
        kept.append(m)
    return kept


def _record(a: dict, b: dict, label: int, source: str, pair_type: str) -> dict:
    return {
        "text_a": a["text"],
        "text_b": b["text"],
        "label": label,
        "source": source,
        "pair_type": pair_type,
        "lang": b.get("lang") or a.get("lang"),
        "tree_id": a.get("message_tree_id") or b.get("message_tree_id"),
    }


def oasst1_pairs(
    messages: list[dict],
    langs: tuple[str, ...] = ("en", "zh"),
    cross_neg_ratio: float = 1.0,
    rng: random.Random | None = None,
) -> list[dict]:
    """把 OASST1 消息列表转成样本对列表。"""
    rng = rng or random.Random(0)
    msgs = _filter_messages(messages, langs)
    by_id = {m["message_id"]: m for m in msgs}
    children: dict[str, list[dict]] = {}
    for m in msgs:
        pid = m.get("parent_id")
        if pid and pid in by_id:
            children.setdefault(pid, []).append(m)

    pairs: list[dict] = []

    # 1) 正样本：所有父子边
    for m in msgs:
        pid = m.get("parent_id")
        if not pid or pid not in by_id:
            continue
        parent = by_id[pid]
        if parent["role"] == "prompter" and m["role"] == "assistant":
            pairs.append(_record(parent, m, 1, "oasst1", "qa"))
        elif parent["role"] == "assistant" and m["role"] == "prompter":
            pairs.append(_record(parent, m, 1, "oasst1", "aq"))

    # 2) 硬负样本：同 prompt 多个 assistant 候选，rank>0 的做负样本
    for pid, kids in children.items():
        parent = by_id[pid]
        ranked = [k for k in kids if k["role"] == "assistant" and k.get("rank") is not None]
        if len(ranked) < 2:
            continue
        ranked.sort(key=lambda k: float(k["rank"]))
        for k in ranked[1:]:
            pairs.append(_record(parent, k, 0, "oasst1", "hard_neg"))

    # 3) 跨树负样本
    if cross_neg_ratio > 0 and len(msgs) >= 2:
        n = int(len(pairs) * cross_neg_ratio)
        n = min(n, len(msgs) * 5)  # 上限保护
        by_tree: dict[str, list[dict]] = {}
        for m in msgs:
            by_tree.setdefault(m.get("message_tree_id") or m["message_id"], []).append(m)
        tree_ids = list(by_tree.keys())
        if len(tree_ids) >= 2:
            made = 0
            guard = 0
            while made < n and guard < n * 20:
                guard += 1
                ta = rng.choice(tree_ids)
                tb = rng.choice(tree_ids)
                if ta == tb:
                    continue
                a = rng.choice(by_tree[ta])
                b = rng.choice(by_tree[tb])
                pairs.append(_record(a, b, 0, "oasst1", "cross_neg"))
                made += 1

    return pairs
