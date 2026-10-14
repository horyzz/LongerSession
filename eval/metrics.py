"""评估指标：Boundary F1（带容差）/ Pk。"""
from __future__ import annotations


def boundary_f1(
    gold: list[int], pred: list[int], tolerance: int = 0
) -> dict[str, float | int]:
    """边界检测的 precision / recall / F1。

    gold / pred 是边界 turn 序号列表（1-based，在 turn p 之后切）。
    tolerance = 允许的容差窗口（turn 数），0 表示严格匹配（对齐 TIAGE 协议）。
    """
    gold_set = set(gold)
    pred_list = list(pred)

    matched_gold: set[int] = set()
    tp = 0
    for p in pred_list:
        for g in gold_set:
            if g in matched_gold:
                continue
            if abs(p - g) <= tolerance:
                tp += 1
                matched_gold.add(g)
                break

    fp = len(pred_list) - tp
    fn = len(gold_set) - len(matched_gold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_all(
    gold_cases: list[list[int]],
    pred_cases: list[list[int]],
    tolerance: int = 0,
) -> dict[str, float | int]:
    """多 case 汇总（微平均）。"""
    total_tp = total_fp = total_fn = 0
    for g, p in zip(gold_cases, pred_cases):
        r = boundary_f1(g, p, tolerance)
        total_tp += int(r["tp"])
        total_fp += int(r["fp"])
        total_fn += int(r["fn"])
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": total_tp, "fp": total_fp, "fn": total_fn}


def pk(gold: list[int], pred: list[int], n_turns: int, k: int | None = None) -> float:
    """Pk 指标（主题分割标准，值越低越好）。

    gold/pred 是边界 turn 序号（1-based，在 turn p 之后切）。
    """
    if k is None:
        k = max(1, n_turns // 2)
    gold_set = set(gold)
    pred_set = set(pred)

    disagreements = 0
    pairs = 0
    for i in range(0, n_turns - k):
        j = i + k
        # i 和 j 之间的边界（位置 p 表示 turn p 之后切，i<p<=j 表示分隔 i 和 j）
        g_sep = any(i < b <= j for b in gold_set)
        p_sep = any(i < b <= j for b in pred_set)
        if g_sep != p_sep:
            disagreements += 1
        pairs += 1
    return disagreements / pairs if pairs > 0 else 0.0
