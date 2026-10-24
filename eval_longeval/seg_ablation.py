"""切割 encoder 消融：char_ngram（弱 baseline） vs 本地 balanced 模型（BGE-M3 微调）。

用 LongEval topics 任务，同样的「切 chunk → 逐 chunk 打 topic → 聚合」流程，
只换 encoder + 阈值，对比 topic 召回。
"""
from __future__ import annotations

import time

from eval_longeval.data import generate_test_cases, text_to_messages
from eval_longeval.metrics import topic_recall
from eval_longeval.strategies import _dedupe, _label_chunk, _split_by_token_cap
from longersession.config import Config
from longersession.encoder.bge_m3 import BGEM3Encoder
from longersession.encoder.factory import create_encoder
from longersession.segmentation.preprocess import messages_to_turns
from longersession.segmentation.segmenter import segment


def seg_recall(case, encoder, threshold: float) -> dict:
    turns = messages_to_turns(text_to_messages(case.conversation))
    cfg = Config()
    cfg.segmentation.threshold = threshold
    cfg.segmentation.overlap_turns = 0
    chunks = segment(turns, encoder, cfg.segmentation, session_id="ablation")
    turn_map = {t.turn_id: t for t in turns}
    chunk_texts = []
    for ch in chunks:
        tt = [turn_map[tid].combined_text() for tid in ch.turn_ids if tid in turn_map]
        chunk_texts.extend(_split_by_token_cap(tt, max_tokens=800))

    labels = []
    prompt_tokens = 0
    for ct in chunk_texts:
        lab, usage = _label_chunk(ct)
        labels.append(lab)
        prompt_tokens += usage.get("prompt_tokens", 0)
    predicted = _dedupe(labels)
    r = topic_recall(case.topics, predicted)
    return {
        "num_chunks": len(chunks),
        "recall": r["recall"],
        "matched": r["matched"],
        "missing": r["missing"],
        "predicted": predicted,
        "prompt_tokens": prompt_tokens,
    }


def run_ablation() -> dict:
    cases = generate_test_cases([5, 10], 2, seed=0)

    cfg = Config()
    cfg.encoder.provider = "char_ngram"
    enc_ngram = create_encoder(cfg.encoder)
    enc_bge = BGEM3Encoder(
        model_name="BAAI/bge-m3",
        device="mps",
        max_seq_length=2048,
        local_dir="data/models/longersession_model_balanced",
    )

    encoders = [
        ("char_ngram (thr=0.5)", enc_ngram, 0.5),
        ("bge_m3-balanced (thr=0.63)", enc_bge, 0.63),
    ]

    per_case = []
    for case in cases:
        row = {"test_id": case.test_id, "num_topics": case.num_topics, "results": {}}
        for name, enc, thr in encoders:
            t0 = time.time()
            r = seg_recall(case, enc, thr)
            r["latency_s"] = round(time.time() - t0, 1)
            row["results"][name] = r
        per_case.append(row)

    agg = {}
    for name, _, _ in encoders:
        vals = [c["results"][name] for c in per_case]
        agg[name] = {
            "avg_recall": round(sum(v["recall"] for v in vals) / len(vals), 3),
            "avg_chunks": round(sum(v["num_chunks"] for v in vals) / len(vals), 1),
            "avg_prompt_tokens": round(
                sum(v["prompt_tokens"] for v in vals) / len(vals), 1
            ),
        }

    return {"encoders": [e[0] for e in encoders], "aggregate": agg, "cases": per_case}


def render_markdown(report: dict) -> str:
    lines = [
        "# 切割 encoder 消融（LongEval topics，DeepSeek 打标）",
        "",
        "| encoder | 平均 topic 召回 | 平均 chunk 数 | 平均 prompt tokens |",
        "|---|---|---|---|",
    ]
    for name in report["encoders"]:
        a = report["aggregate"][name]
        lines.append(
            f"| {name} | {a['avg_recall']:.1%} | {a['avg_chunks']:.1f} | "
            f"{a['avg_prompt_tokens']:.0f} |"
        )
    lines += ["", "## 逐用例", ""]
    for c in report["cases"]:
        lines.append(f"### {c['test_id']}（{c['num_topics']} topics）")
        lines.append("")
        for name in report["encoders"]:
            r = c["results"][name]
            lines.append(
                f"- {name}: recall={r['recall']:.0%}, chunks={r['num_chunks']}, "
                f"missing={r['missing']}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    import json
    from pathlib import Path

    t0 = time.time()
    report = run_ablation()
    out = Path("data/longeval/seg_ablation_report")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), "utf-8"
    )
    (out / "report.md").write_text(render_markdown(report), "utf-8")
    print(render_markdown(report))
    print(f"\n总耗时 {time.time()-t0:.1f}s，报告写入 data/longeval/seg_ablation_report/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
