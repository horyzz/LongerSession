"""LongEval 真实基准运行器：DeepSeek × 四策略，产出报告。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from eval_longeval.data import generate_test_cases
from eval_longeval.strategies import (
    EvalResult,
    compression_strategy,
    full_strategy,
    index_strategy,
    truncation_strategy,
)

DEFAULT_OUTPUT_DIR = Path("data/longeval")


def run_case(case, budget_tokens: int, strategies: list[str]) -> dict:
    """对单个测试用例跑选定策略，返回各策略结果 dict。"""
    results: dict[str, dict] = {}
    runners = {
        "full": lambda: full_strategy(case),
        "truncation": lambda: truncation_strategy(case, budget_tokens),
        "compression": lambda: compression_strategy(case),
        "index-conversation": lambda: index_strategy(case, "conversation"),
        "index-segmentation": lambda: index_strategy(case, "segmentation"),
    }
    for name in strategies:
        try:
            r: EvalResult = runners[name]()
            results[name] = _result_to_dict(r)
        except Exception as e:  # noqa: BLE001
            results[name] = {
                "strategy": name,
                "error": repr(e),
                "recall": None,
            }
    return results


def _result_to_dict(r: EvalResult) -> dict:
    return {
        "strategy": r.strategy,
        "recall": r.recall,
        "predicted_topics": r.predicted_topics,
        "matched": r.matched,
        "missing": r.missing,
        "prompt_tokens": r.prompt_tokens,
        "max_prompt_tokens": r.max_prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "num_llm_calls": r.num_llm_calls,
        "latency_s": r.latency_s,
        "note": r.note,
        "extra": r.extra,
    }


def run_eval(
    num_topics_list: list[int] | None = None,
    num_samples: int = 2,
    budget_tokens: int = 1500,
    seed: int = 0,
    strategies: list[str] | None = None,
    model: str | None = None,
) -> dict:
    """跑完整评估，返回报告 dict。"""
    import os

    if model:
        os.environ["deepseek_model"] = model

    strategies = strategies or [
        "full",
        "truncation",
        "compression",
        "index-conversation",
        "index-segmentation",
    ]
    cases = generate_test_cases(num_topics_list, num_samples, seed)

    per_case = []
    for case in cases:
        per_case.append(
            {
                "test_id": case.test_id,
                "num_topics": case.num_topics,
                "topics": case.topics,
                "conversation_tokens_est": len(case.conversation.split()),
                "results": run_case(case, budget_tokens, strategies),
            }
        )

    # 汇总每个策略的平均 recall / tokens / latency
    agg: dict[str, dict] = {}
    for s in strategies:
        vals = [c["results"][s] for c in per_case]
        recalls = [v["recall"] for v in vals if v.get("recall") is not None]
        ptokens = [v.get("prompt_tokens", 0) for v in vals if v.get("recall") is not None]
        mtokens = [v.get("max_prompt_tokens", 0) for v in vals if v.get("recall") is not None]
        ctokens = [v.get("completion_tokens", 0) for v in vals if v.get("recall") is not None]
        lat = [v.get("latency_s", 0) for v in vals if v.get("recall") is not None]
        errs = [v.get("error") for v in vals if v.get("error")]
        agg[s] = {
            "avg_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "n": len(recalls),
            "avg_prompt_tokens": round(sum(ptokens) / len(ptokens), 1) if ptokens else 0,
            "avg_max_prompt_tokens": round(sum(mtokens) / len(mtokens), 1) if mtokens else 0,
            "avg_completion_tokens": round(sum(ctokens) / len(ctokens), 1) if ctokens else 0,
            "avg_latency_s": round(sum(lat) / len(lat), 2) if lat else 0,
            "errors": errs,
        }

    return {
        "config": {
            "num_topics_list": num_topics_list or [5, 10],
            "num_samples": num_samples,
            "budget_tokens": budget_tokens,
            "seed": seed,
            "strategies": strategies,
            "model": os.environ.get("deepseek_model", "deepseek-flash"),
            "base_url": os.environ.get("deepseek_base_url", "https://api.deepseek.com"),
        },
        "aggregate": agg,
        "cases": per_case,
    }


# ---------------------------------------------------------------- 报告渲染

def render_markdown(report: dict) -> str:
    agg = report["aggregate"]
    cfg = report["config"]
    lines = [
        "# LongEval（LongChat topics）真实基准 · DeepSeek",
        "",
        f"- 模型：`{cfg['model']}`（base_url=`{cfg['base_url']}`）",
        f"- 测试用例：num_topics={cfg['num_topics_list']}，每档 {cfg['num_samples']} 个样本，seed={cfg['seed']}",
        f"- 截断预算：`{cfg['budget_tokens']}` tokens",
        "",
        "## 各策略平均结果",
        "",
        "| 策略 | 平均 topic 召回 | 总 prompt tokens | 单次最大 prompt tokens | 平均 LLM 调用数 | 平均耗时 |",
        "|---|---|---|---|---|---|",
    ]
    for s in cfg["strategies"]:
        a = agg[s]
        recall = f"{a['avg_recall']:.1%}" if a["avg_recall"] is not None else "err"
        # 每个策略的 LLM 调用数取各 case 平均
        calls = [c["results"][s].get("num_llm_calls", 0) for c in report["cases"]]
        avg_calls = sum(calls) / len(calls) if calls else 0
        lines.append(
            f"| {s} | {recall} | {a['avg_prompt_tokens']:.0f} | "
            f"{a['avg_max_prompt_tokens']:.0f} | {avg_calls:.1f} | {a['avg_latency_s']}s |"
        )
    lines += [
        "",
        "## 逐用例明细",
        "",
    ]
    for c in report["cases"]:
        lines.append(f"### {c['test_id']}（{c['num_topics']} topics）")
        lines.append("")
        lines.append("真值 topics：" + " | ".join(c["topics"]))
        lines.append("")
        lines.append("| 策略 | recall | 命中 | 缺失 |")
        lines.append("|---|---|---|---|")
        for s in cfg["strategies"]:
            r = c["results"][s]
            if "error" in r and r["recall"] is None:
                lines.append(f"| {s} | ❌ {r['error'][:60]} | | |")
                continue
            lines.append(
                f"| {s} | {r['recall']:.1%} | {', '.join(r['matched'])[:120]} "
                f"| {', '.join(r['missing'])[:120]} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_report(report: dict, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "longeval_report.json"
    md_path = output_dir / "longeval_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    md_path.write_text(render_markdown(report), "utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    import sys

    args = sys.argv[1:]
    num_topics = [int(x) for x in args[0].split(",")] if args and args[0] else [5, 10]
    num_samples = int(args[1]) if len(args) > 1 else 2
    budget = int(args[2]) if len(args) > 2 else 1500
    t0 = time.time()
    report = run_eval(num_topics, num_samples, budget)
    paths = write_report(report)
    print(render_markdown(report))
    print(f"\n总耗时 {time.time()-t0:.1f}s，报告已写入: {paths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
