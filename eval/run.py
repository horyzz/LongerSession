"""统一评估入口：指定 dataset set + encoder，跑切割算 F1。"""
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


def predict_boundaries(messages: list[dict], encoder, seg_config) -> list[int]:
    """跑切割算法（承接分数 + DP），返回预测 boundaries。"""
    msgs = [
        Message(role=Role(m["role"]), content=m["content"]) for m in messages
    ]
    turns = messages_to_turns(msgs)
    scores = compute_coherence_scores(turns, encoder)
    cuts = dp_segment(
        scores,
        min_len=seg_config.min_chunk_turns,
        max_len=seg_config.max_chunk_turns,
        threshold=seg_config.threshold,
    )
    return cuts


def _group_key(case: dict) -> str:
    """分组键：有 language + length 按维度分组，否则 overall。"""
    if "language" in case and "length" in case:
        return f"{case['language']}-{case['length']}"
    return "overall"


def evaluate(dataset: str, encoder_provider: str = "char_ngram") -> dict:
    """跑评估，返回分维度结果字典。

    dataset: "tiage" | "cases"
    """
    cases = load_dataset(dataset)

    cfg = Config()
    cfg.encoder.provider = encoder_provider
    encoder = create_encoder(cfg.encoder)
    seg_config = cfg.segmentation

    groups: dict[str, dict[str, list[list[int]]]] = {}
    for case in cases:
        key = _group_key(case)
        groups.setdefault(key, {"gold": [], "pred": []})
        groups[key]["gold"].append(case["boundaries"])
        groups[key]["pred"].append(
            predict_boundaries(case["messages"], encoder, seg_config)
        )

    results: dict[str, dict] = {}
    all_gold: list[list[int]] = []
    all_pred: list[list[int]] = []
    for key in sorted(groups):
        g = groups[key]
        r0 = evaluate_all(g["gold"], g["pred"], tolerance=0)
        r1 = evaluate_all(g["gold"], g["pred"], tolerance=1)
        results[key] = {
            "f1": r0["f1"],
            "f1_tol1": r1["f1"],
            "precision": r0["precision"],
            "recall": r0["recall"],
            "n_cases": len(g["gold"]),
        }
        all_gold.extend(g["gold"])
        all_pred.extend(g["pred"])

    r0 = evaluate_all(all_gold, all_pred, tolerance=0)
    r1 = evaluate_all(all_gold, all_pred, tolerance=1)
    results["overall"] = {
        "f1": r0["f1"],
        "f1_tol1": r1["f1"],
        "precision": r0["precision"],
        "recall": r0["recall"],
        "n_cases": len(cases),
    }
    return results


def main(
    dataset: str = "cases",
    encoder_provider: str = "char_ngram",
) -> int:
    results = evaluate(dataset, encoder_provider)
    print(f"dataset={dataset}, encoder={encoder_provider}")
    for key in sorted(results):
        r = results[key]
        print(
            f"  {key}: F1(tol0)={r['f1']:.3f} F1(tol1)={r['f1_tol1']:.3f} "
            f"(n={r['n_cases']}, P={r['precision']:.3f} R={r['recall']:.3f})"
        )
    return 0


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "cases"
    enc = sys.argv[2] if len(sys.argv) > 2 else "char_ngram"
    sys.exit(main(ds, enc))
