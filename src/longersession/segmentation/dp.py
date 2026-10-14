"""DP 切割：长度约束 + 最小化切点代价。"""
from __future__ import annotations


def dp_segment(
    scores: list[float],
    min_len: int = 2,
    max_len: int = 20,
    threshold: float = 0.5,
) -> list[int]:
    """用动态规划找最优切割点。

    Args:
        scores: 边界承接分数，scores[i] = turn_i 与 turn_{i+1} 的相似度，长度 = n-1。
        min_len / max_len: 每个 chunk 的 turn 数约束（闭区间）。
        threshold: 承接分数阈值，切点代价 = score - threshold（低分区切是负代价，鼓励切）。

    Returns:
        切点位置列表（1-based，表示在某个 turn 之后切），升序。
        例如 [3] 表示在第 3 个 turn 结束后切一刀。
    """
    n = len(scores) + 1  # turn 总数
    if n <= min_len:
        return []

    INF = float("inf")
    dp = [INF] * (n + 1)
    prev = [-1] * (n + 1)
    dp[0] = 0.0

    for i in range(1, n + 1):
        # 最后一个 chunk = turn[j .. i-1]（0-based），长度 = i - j
        lo = max(0, i - max_len)
        hi = i - min_len
        for j in range(lo, hi + 1):
            length = i - j
            if length < min_len or length > max_len:
                continue
            cut_cost = (scores[j - 1] - threshold) if j > 0 else 0.0
            cost = dp[j] + cut_cost
            if cost < dp[i]:
                dp[i] = cost
                prev[i] = j

    if dp[n] == INF:
        # 约束无解（例如 max_len 过小），退化为不切割
        return []

    # 回溯切点
    cuts: list[int] = []
    i = n
    while i > 0:
        j = prev[i]
        if j <= 0:
            break
        cuts.append(j)
        i = j
    return sorted(cuts)
