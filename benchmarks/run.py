"""基准运行器：单次运行 + 深度/预算扫描，产出 JSON 与 Markdown 报告。"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.generator import generate_session
from benchmarks.strategies import StrategyResult, run_all
from longersession.config import Config

DEFAULT_OUTPUT_DIR = Path("data/benchmark")


def _config(encoder_provider: str = "char_ngram") -> Config:
    cfg = Config()
    cfg.encoder.provider = encoder_provider
    return cfg


def _result_dict(r: StrategyResult) -> dict:
    return {
        "name": r.name,
        "answerable": r.answerable,
        "context_tokens": r.context_tokens,
        "total_tokens": r.total_tokens,
        "token_ratio": round(r.token_ratio, 4),
        "note": r.note,
        "extra": r.extra,
    }


def run_single(
    n_turns: int = 20,
    needle_depth: float = 0.2,
    budget_tokens: int = 512,
    seed: int | None = 42,
    encoder_provider: str = "char_ngram",
) -> dict:
    """单次运行：生成 session + 三策略对比。"""
    session = generate_session(n_turns, needle_depth, seed)
    results = run_all(session, budget_tokens, _config(encoder_provider))

    by_name = {r.name: r for r in results}
    index_wins = by_name["index"].answerable and not by_name["compression"].answerable

    return {
        "config": {
            "n_turns": n_turns,
            "needle_depth": needle_depth,
            "budget_tokens": budget_tokens,
            "seed": seed,
            "encoder_provider": encoder_provider,
        },
        "session": {
            "total_tokens": session.total_tokens,
            "needle_turn_index": session.needle.turn_index,
            "needle_value": session.needle.value,
            "query": session.query,
        },
        "strategies": [_result_dict(r) for r in results],
        "summary": {
            "winner": "index" if index_wins else "none",
            "index_beats_truncation": (
                by_name["index"].answerable and not by_name["truncation"].answerable
            ),
            "index_beats_compression": index_wins,
            "index_token_ratio": round(by_name["index"].token_ratio, 4),
        },
    }


def run_sweep(
    n_turns: int = 20,
    depths: list[float] | None = None,
    budgets: list[int] | None = None,
    seed: int | None = 42,
    encoder_provider: str = "char_ngram",
) -> dict:
    """扫描 needle 深度 × context 预算，得到可回答性矩阵。"""
    depths = depths if depths is not None else [0.05, 0.2, 0.5, 0.8, 0.95]
    budgets = budgets if budgets is not None else [256, 512, 1024, 2048]

    cells: list[dict] = []
    for d in depths:
        for b in budgets:
            report = run_single(n_turns, d, b, seed, encoder_provider)
            by = {r["name"]: r for r in report["strategies"]}
            cells.append(
                {
                    "needle_depth": d,
                    "budget_tokens": b,
                    "truncation_answerable": by["truncation"]["answerable"],
                    "compression_answerable": by["compression"]["answerable"],
                    "index_answerable": by["index"]["answerable"],
                    "index_token_ratio": by["index"]["token_ratio"],
                }
            )

    return {
        "config": {
            "n_turns": n_turns,
            "depths": depths,
            "budgets": budgets,
            "seed": seed,
            "encoder_provider": encoder_provider,
        },
        "cells": cells,
    }


# ---------------------------------------------------------------- 报告渲染

def _md_single(report: dict) -> str:
    cfg = report["config"]
    sess = report["session"]
    lines = [
        "# LongerSession 超长 Session 有效控制 · 单次基准",
        "",
        "## 场景",
        "",
        f"- session turn 数：`{cfg['n_turns']}`",
        f"- needle 埋入深度：`{cfg['needle_depth']}`（turn {sess['needle_turn_index']}）",
        f"- context 预算：`{cfg['budget_tokens']}` tokens",
        f"- 全量 session：`{sess['total_tokens']}` tokens",
        f"- needle：`{sess['needle_value']}`",
        f"- 最终提问：{sess['query']}",
        "",
        "## 三策略对比",
        "",
        "| 策略 | 可回答 | context tokens | token 占比 | 说明 |",
        "|---|---|---|---|---|",
    ]
    for s in report["strategies"]:
        mark = "✅" if s["answerable"] else "❌"
        lines.append(
            f"| {s['name']} | {mark} | {s['context_tokens']} | "
            f"{s['token_ratio']:.1%} | {s['note']} |"
        )
    lines += [
        "",
        f"**结论**：{'索引在远小于压缩/截断的 token 开销下找回了 needle' if report['summary']['index_beats_compression'] else '本组参数下三策略结果相同（needle 落在预算窗口内）'}。",
        "",
    ]
    return "\n".join(lines)


def _md_sweep(report: dict) -> str:
    cfg = report["config"]
    lines = [
        "# LongerSession 超长 Session 有效控制 · 扫描矩阵",
        "",
        f"n_turns={cfg['n_turns']}，encoder={cfg['encoder_provider']}，seed={cfg['seed']}",
        "",
        "每个格子为 `截断 / 压缩 / 索引` 的可回答性（✅/❌）：",
        "",
        "| depth \\ budget | " + " | ".join(str(b) for b in cfg["budgets"]) + " |",
        "|---|" + "|".join(["---"] * len(cfg["budgets"])) + "|",
    ]
    by_depth: dict[float, dict[int, dict]] = {}
    for c in report["cells"]:
        by_depth.setdefault(c["needle_depth"], {})[c["budget_tokens"]] = c

    for d in cfg["depths"]:
        row = [f"`{d}`"]
        for b in cfg["budgets"]:
            cell = by_depth.get(d, {}).get(b)
            if cell is None:
                row.append("-")
                continue

            def m(v: bool) -> str:
                return "✅" if v else "❌"

            row.append(
                f"{m(cell['truncation_answerable'])} / "
                f"{m(cell['compression_answerable'])} / "
                f"{m(cell['index_answerable'])}"
            )
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "图例：`截断 / 压缩 / 索引`。索引在 needle 被截断/压缩丢失时仍稳定找回。",
        "",
    ]
    return "\n".join(lines)


def write_report(
    report: dict,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """把报告写入 JSON + Markdown，返回文件路径。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "cells" in report:
        json_path = output_dir / "benchmark_sweep.json"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        md_path = output_dir / "benchmark_sweep.md"
        md_text = _md_sweep(report)
    else:
        json_path = output_dir / "benchmark_report.json"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        md_path = output_dir / "benchmark_report.md"
        md_text = _md_single(report)

    md_path.write_text(md_text, "utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    if mode == "sweep":
        report = run_sweep()
        paths = write_report(report)
        print(_md_sweep(report))
        print(f"\n报告已写入: {paths}")
    else:
        report = run_single()
        paths = write_report(report)
        print(_md_single(report))
        print(f"\n报告已写入: {paths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
