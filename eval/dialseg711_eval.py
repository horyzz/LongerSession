"""DialSeg711 评测：任务型对话（MultiWOZ+KVRET）topic 分割。

DialSeg711：711 个任务型对话，每个 utterance 有 topic_id + segmentation_label
（1=话题结束）。boundary = segmentation_label==1 的 turn 位置。

用法：
    uv run python -m eval.dialseg711_eval <model_path> [thresholds...]

model_path 可以是 sentence-transformers 模型名或本地路径。
"""
from __future__ import annotations

import json
import sys

import numpy as np
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer


def load_dialseg711() -> list[tuple[list[str], list[int]]]:
    p = hf_hub_download("Coldog2333/dialseg711", "test.json", repo_type="dataset")
    data = json.load(open(p, encoding="utf-8"))
    dials = data["dial_data"]["dialseg711"]
    cases = []
    for d in dials:
        turns = d["turns"]
        utterances = [t["utterance"] for t in turns]
        n = len(utterances)
        boundaries = [
            t["turn_id"] for t in turns
            if t["segmentation_label"] == 1 and 1 <= t["turn_id"] < n
        ]
        cases.append((utterances, boundaries))
    return cases


def dp_segment(scores, min_len=2, max_len=20, threshold=0.5):
    n = len(scores) + 1
    if n <= min_len:
        return []
    INF = float("inf")
    dp = [INF] * (n + 1); prev = [-1] * (n + 1); dp[0] = 0.0
    for i in range(1, n + 1):
        lo = max(0, i - max_len); hi = i - min_len
        for j in range(lo, hi + 1):
            if i - j < min_len or i - j > max_len:
                continue
            cost = dp[j] + ((scores[j - 1] - threshold) if j > 0 else 0.0)
            if cost < dp[i]:
                dp[i] = cost; prev[i] = j
    if dp[n] == INF:
        return []
    cuts = []; i = n
    while i > 0:
        j = prev[i]
        if j <= 0:
            break
        cuts.append(j); i = j
    return sorted(cuts)


def evaluate_all(golds, preds, tolerance=0):
    tp = fp = fn = 0
    for g, p in zip(golds, preds):
        gs = set(g); matched = set(); t = 0
        for x in p:
            for y in gs:
                if y in matched:
                    continue
                if abs(x - y) <= tolerance:
                    t += 1; matched.add(y); break
        tp += t; fp += len(p) - t; fn += len(gs) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def main(argv=None):
    argv = argv or sys.argv[1:]
    model_path = argv[0] if argv else "BAAI/bge-m3"
    thresholds = [float(x) for x in argv[1:]] or [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    cases = load_dialseg711()
    total_boundaries = sum(len(b) for _, b in cases)
    print(f"DialSeg711: {len(cases)} 对话, {total_boundaries} 边界")

    model = SentenceTransformer(model_path)

    # 编码所有 utterance
    all_texts = []
    offsets = []
    for utts, _ in cases:
        offsets.append(len(all_texts))
        all_texts.extend(utts)
    embs = model.encode(all_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    case_scores = []
    for idx, (utts, _) in enumerate(cases):
        s = offsets[idx]
        e = embs[s:s + len(utts)]
        case_scores.append([float(np.dot(e[i], e[i + 1])) for i in range(len(e) - 1)])

    golds = [b for _, b in cases]
    print("\nthreshold   F1(tol0)   F1(tol1)   P       R")
    best = None
    for th in thresholds:
        preds = [dp_segment(s, threshold=th) for s in case_scores]
        f1, p, r = evaluate_all(golds, preds, 0)
        f1t, _, _ = evaluate_all(golds, preds, 1)
        print(f"{th:.2f}       {f1:.3f}      {f1t:.3f}      {p:.3f}   {r:.3f}")
        if best is None or f1 > best[1]:
            best = (th, f1, f1t)
    print(f"\n=== best: F1(tol0)={best[1]:.3f} F1(tol1)={best[2]:.3f} @ th={best[0]:.2f} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
