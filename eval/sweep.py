"""阈值扫描：对指定 encoder 在指定数据集上扫 threshold，找最优 F1。

用法：
    uv run python -m eval.sweep <dataset> <encoder_provider> [th1 th2 ...]

默认扫描 threshold 0.30~0.60（步长 0.05）。
encoder 只需 encode 一次，然后对每个 threshold 重新跑 DP（DP 很快）。
"""
from __future__ import annotations

import sys

from eval.datasets import load_dataset
from eval.metrics import evaluate_all
from longersession.config import Config
from longersession.encoder.factory import create_encoder
from longersession.models.message import Message, Role
from longersession.segmentation.coherence import compute_coherence_scores
from longersession.segmentation.dp import dp_segment
from longersession.segmentation.preprocess import messages_to_turns


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "tiage"
    encoder_provider = sys.argv[2] if len(sys.argv) > 2 else "bge_m3"
    if len(sys.argv) > 3:
        thresholds = [float(x) for x in sys.argv[3:]]
    else:
        thresholds = [round(0.30 + 0.05 * i, 2) for i in range(7)]  # 0.30..0.60

    cfg = Config()
    cfg.encoder.provider = encoder_provider
    encoder = create_encoder(cfg.encoder)
    seg = cfg.segmentation

    cases = load_dataset(dataset)

    # encode 一次，缓存每 case 的承接分数
    case_scores: list[list[float]] = []
    golds: list[list[int]] = []
    for case in cases:
        msgs = [
            Message(role=Role(m["role"]), content=m["content"])
            for m in case["messages"]
        ]
        turns = messages_to_turns(msgs)
        case_scores.append(compute_coherence_scores(turns, encoder))
        golds.append(case["boundaries"])

    print(f"dataset={dataset}, encoder={encoder_provider}")
    print(f"{'threshold':>9}  {'F1(tol0)':>9}  {'F1(tol1)':>9}  {'P':>6}  {'R':>6}")
    best = None
    for th in thresholds:
        preds = [
            dp_segment(
                s,
                min_len=seg.min_chunk_turns,
                max_len=seg.max_chunk_turns,
                threshold=th,
            )
            for s in case_scores
        ]
        r0 = evaluate_all(golds, preds, tolerance=0)
        r1 = evaluate_all(golds, preds, tolerance=1)
        print(
            f"{th:>9.2f}  {r0['f1']:>9.3f}  {r1['f1']:>9.3f}  "
            f"{r0['precision']:>6.3f}  {r0['recall']:>6.3f}"
        )
        if best is None or r0["f1"] > best[1]:
            best = (th, r0["f1"], r1["f1"], r0["precision"], r0["recall"])
    print(f"\nbest: threshold={best[0]:.2f} F1(tol0)={best[1]:.3f} "
          f"F1(tol1)={best[2]:.3f} P={best[3]:.3f} R={best[4]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
