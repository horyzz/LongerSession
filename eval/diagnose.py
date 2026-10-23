"""诊断：encoder 承接分数对 topic 边界的区分能力。

对每个相邻 turn 对，标记其是否为 gold boundary，统计：
- boundary vs 非 boundary 的余弦分数分布（mean/median/std）
- AUROC（边界分数应系统低于非边界，>0.5 说明方向正确，越高分离越好）
- 纯阈值（GenEnc 式）在该信号上能达到的最优 F1（扫描所有可能阈值）

用法：
    uv run python -m eval.diagnose <dataset> <encoder_provider>
"""
from __future__ import annotations

import sys

from eval.datasets import load_dataset
from longersession.config import Config
from longersession.encoder.factory import create_encoder
from longersession.models.message import Message, Role
from longersession.segmentation.coherence import compute_coherence_scores
from longersession.segmentation.preprocess import messages_to_turns


def _auroc(boundary_scores: list[float], non_scores: list[float]) -> float:
    """AUROC：P(随机 boundary 分数 < 随机非 boundary 分数)。

    用 rank 法（Mann-Whitney U）。>0.5 表示边界分数更低（方向正确）。
    """
    bs = sorted(boundary_scores)
    ns = sorted(non_scores)
    nb = len(bs)
    nn = len(ns)
    if nb == 0 or nn == 0:
        return float("nan")
    # 对每个 boundary 分数，数有多少个非 boundary 分数大于它（+0.5 平局）
    u = 0.0
    for s in bs:
        # 用二分统计 ns 中 > s 的数量
        lo, hi = 0, nn
        while lo < hi:
            mid = (lo + hi) // 2
            if ns[mid] <= s:
                lo = mid + 1
            else:
                hi = mid
        greater = nn - lo
        # 平局（== s）计 0.5
        equal = 0
        for t in ns:
            if t == s:
                equal += 1
        u += greater + 0.5 * equal
    return u / (nb * nn)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "tiage"
    encoder_provider = sys.argv[2] if len(sys.argv) > 2 else "bge_m3"

    cfg = Config()
    cfg.encoder.provider = encoder_provider
    encoder = create_encoder(cfg.encoder)

    cases = load_dataset(dataset)

    boundary_scores: list[float] = []
    non_scores: list[float] = []
    all_scores: list[float] = []

    for case in cases:
        msgs = [
            Message(role=Role(m["role"]), content=m["content"])
            for m in case["messages"]
        ]
        turns = messages_to_turns(msgs)
        scores = compute_coherence_scores(turns, encoder)
        gold = set(case["boundaries"])  # 1-based turn index，切在 turn p 之后
        for i, s in enumerate(scores):
            # scores[i] 对应 turn_{i+1} 与 turn_{i+2} 之间，即切点位置 p=i+1
            p = i + 1
            all_scores.append(s)
            if p in gold:
                boundary_scores.append(s)
            else:
                non_scores.append(s)

    auroc = _auroc(boundary_scores, non_scores)
    print(f"dataset={dataset}, encoder={encoder_provider}")
    print(f"n_boundary={len(boundary_scores)}  n_non={len(non_scores)}")
    print(f"boundary  score: mean={_mean(boundary_scores):.4f} "
          f"median={_median(boundary_scores):.4f}")
    print(f"non-bound score: mean={_mean(non_scores):.4f} "
          f"median={_median(non_scores):.4f}")
    print(f"AUROC (boundary 分数应更低): {auroc:.4f}")

    # 纯阈值最优 F1（GenEnc 式，无 DP），作为信号上限参考
    best_f1 = 0.0
    best_th = 0.0
    uniq = sorted(set(round(s, 4) for s in all_scores))
    for th in uniq:
        tp = sum(1 for s in boundary_scores if s < th)
        fp = sum(1 for s in non_scores if s < th)
        fn = len(boundary_scores) - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    print(f"纯阈值最优 F1(tol=0)={best_f1:.4f} @ th={best_th:.4f} "
          f"(GenEnc 式，无 DP 长度约束)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
