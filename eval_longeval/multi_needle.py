"""多 needle 验证：10 个 topic 各埋一个具体事实，测「压缩 vs 索引」能保住几个。

这是「信息密度」的直接测试——长 session 里几十上百个具体事实，
压缩（有损摘要）必然丢；索引（chunk + 检索）结构上不丢。

指标：fact survival = 该事实的随机 code 是否仍存在于策略的表示里
（压缩=摘要文本；索引=检索到的 chunk）。不依赖 LLM 答题，无噪声。
"""
from __future__ import annotations

import random
import secrets
import time
from dataclasses import dataclass

from eval_longeval.data import Conversation, load_conversations
from eval_longeval.deepseek import chat
from eval_longeval.needle_qa import _inject_needle, _lexical_score
from eval_longeval.strategies import SYSTEM, _est_tokens

KEYS = [
    "backup password",
    "deployment region",
    "API rate limit",
    "cache TTL",
    "max retry count",
    "budget cap",
    "on-call escalation number",
    "database shard count",
    "feature flag name",
    "rollback trigger",
]

QUESTION_TEMPLATE = "What is our {key}?"


@dataclass
class MultiCase:
    test_id: str
    n_topics: int
    keys: list[str]
    codes: list[str]          # code_i 对应 conversations[i]
    questions: list[str]
    session_text: str
    conversations: list[Conversation]


def _code() -> str:
    return f"CODE-{secrets.token_hex(2).upper()}-{secrets.token_hex(2).upper()}"


def _inject_fact(conv_text: str, key: str, code: str) -> str:
    fact = (
        f"USER: Also remember this: our {key} is {code}. \n"
        f"ASSISTANT: Noted — {key} is {code}. \n"
    )
    marker = "Great, let's talk about the next topic"
    if marker in conv_text:
        return conv_text.replace(marker, fact + "USER: " + marker, 1)
    return conv_text + fact


def generate_multi_case(n_topics: int = 10, seed: int = 0) -> MultiCase:
    convs = load_conversations()
    rng = random.Random(seed)
    chosen = rng.sample(convs, n_topics)
    keys = KEYS[:n_topics]
    codes = [_code() for _ in range(n_topics)]

    injected = []
    for i, c in enumerate(chosen):
        text = _inject_fact(c.text, keys[i], codes[i])
        injected.append(Conversation(topic=c.topic, topic_id=c.topic_id, text=text))

    return MultiCase(
        test_id=f"multi-{n_topics}-s{seed}",
        n_topics=n_topics,
        keys=keys,
        codes=codes,
        questions=[QUESTION_TEMPLATE.format(key=k) for k in keys],
        session_text="".join(c.text for c in injected),
        conversations=injected,
    )


def _survived_indices(text: str, codes: list[str]) -> list[int]:
    low = text.lower()
    return [i for i, c in enumerate(codes) if c.lower() in low]


# ---------------------------------------------------------------- 策略

def full_survival(case: MultiCase) -> dict:
    idx = _survived_indices(case.session_text, case.codes)
    return {
        "survived": idx,
        "survival": len(idx) / len(case.codes),
        "llm_calls": 0,
        "prompt_tokens": 0,
        "note": "原文完整，全部事实都在",
    }


def truncation_survival(case: MultiCase, budget_tokens: int = 1500) -> dict:
    acc = 0
    picked = []
    for c in reversed(case.conversations):
        t = _est_tokens(c.text)
        if acc + t > budget_tokens and picked:
            break
        picked.append(c)
        acc += t
    picked.reverse()
    tail = "".join(c.text for c in picked)
    idx = _survived_indices(tail, case.codes)
    return {
        "survived": idx,
        "survival": len(idx) / len(case.codes),
        "llm_calls": 0,
        "prompt_tokens": 0,
        "kept_conversations": len(picked),
        "note": f"仅保留末尾 {len(picked)}/{case.n_topics} 段",
    }


def compression_survival(case: MultiCase, summary_max_tokens: int = 1024) -> dict:
    """有损摘要：摘要被限制为 3 句话，10 个具体事实塞不下。

    max_tokens 给足是为了容纳推理模型的 reasoning tokens；
    真正的摘要长度由「at most 3 sentences」约束。
    """
    t0 = time.time()
    resp = chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Summarize the following conversation concisely, in at most 3 sentences:\n\n" + case.session_text}],
        max_tokens=summary_max_tokens,
    )
    summ = resp["content"]
    usage = resp["usage"]
    idx = _survived_indices(summ, case.codes)
    return {
        "survived": idx,
        "survival": len(idx) / len(case.codes),
        "llm_calls": 1,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "summary_tokens": usage.get("completion_tokens", 0),
        "latency_s": round(time.time() - t0, 2),
        "note": f"摘要 {usage.get('completion_tokens', 0)} tokens，塞不下 10 个事实",
        "summary": summ,
    }


def index_survival(case: MultiCase) -> dict:
    """索引：每个问题只检索其相关 chunk，检查 code 是否还在该 chunk 里。"""
    hits = []
    for i, q in enumerate(case.questions):
        scored = sorted(
            [(j, _lexical_score(q, c.text)) for j, c in enumerate(case.conversations)],
            key=lambda x: -x[1],
        )
        top_idx = scored[0][0]
        top_text = case.conversations[top_idx].text
        hits.append(case.codes[i].lower() in top_text.lower())

    survived = [i for i, h in enumerate(hits) if h]
    return {
        "survived": survived,
        "survival": len(survived) / len(case.codes),
        "llm_calls": 0,
        "prompt_tokens": 0,
        "note": f"每问只读 1 个 chunk，{len(survived)}/{len(case.codes)} 命中",
    }


# ---------------------------------------------------------------- 运行器

def run_multi_needle(
    n_topics: int = 10,
    seeds: list[int] | None = None,
    budget_tokens: int = 1500,
    summary_max_tokens: int = 1024,
) -> dict:
    seeds = seeds if seeds is not None else [0, 1, 2]
    strategy_names = ["full", "truncation", "compression", "index"]
    per_case = []
    for seed in seeds:
        case = generate_multi_case(n_topics, seed)
        per_case.append(
            {
                "test_id": case.test_id,
                "keys": case.keys,
                "codes": case.codes,
                "results": {
                    "full": full_survival(case),
                    "truncation": truncation_survival(case, budget_tokens),
                    "compression": compression_survival(case, summary_max_tokens),
                    "index": index_survival(case),
                },
            }
        )

    agg = {}
    for s in strategy_names:
        vals = [c["results"][s] for c in per_case]
        agg[s] = {
            "avg_survival": round(sum(v["survival"] for v in vals) / len(vals), 3),
            "avg_survived": round(sum(len(v["survived"]) for v in vals) / len(vals), 1),
            "n_facts": n_topics,
            "avg_prompt_tokens": round(
                sum(v["prompt_tokens"] for v in vals) / len(vals), 1
            ),
            "avg_llm_calls": round(sum(v["llm_calls"] for v in vals) / len(vals), 1),
        }

    return {
        "config": {
            "n_topics": n_topics,
            "seeds": seeds,
            "budget_tokens": budget_tokens,
            "summary_max_tokens": summary_max_tokens,
        },
        "aggregate": agg,
        "cases": per_case,
    }


def render_markdown(report: dict) -> str:
    cfg = report["config"]
    agg = report["aggregate"]
    lines = [
        "# 多 needle 验证 · 压缩一定丢，索引不丢",
        "",
        f"- {cfg['n_topics']} 个 topic，每个埋一个具体事实（唯一 code），共 {cfg['n_topics']} 个",
        f"- 压缩摘要预算：`{cfg['summary_max_tokens']}` tokens；截断预算：`{cfg['budget_tokens']}`",
        f"- seeds={cfg['seeds']}",
        "",
        "## 各策略「事实存活率」（该事实 code 是否仍可被找回）",
        "",
        "| 策略 | 平均存活事实 | 存活率 | LLM 调用 | prompt tokens |",
        "|---|---|---|---|---|",
    ]
    for s in ["full", "truncation", "compression", "index"]:
        a = agg[s]
        lines.append(
            f"| {s} | {a['avg_survived']:.1f}/{a['n_facts']} | {a['avg_survival']:.0%} | "
            f"{a['avg_llm_calls']:.1f} | {a['avg_prompt_tokens']:.0f} |"
        )
    lines += ["", "## 逐 seed 明细", ""]
    for c in report["cases"]:
        lines.append(f"### {c['test_id']}")
        lines.append("")
        lines.append("| 策略 | 存活 | 缺失事实(key) |")
        lines.append("|---|---|---|")
        for s in ["full", "truncation", "compression", "index"]:
            r = c["results"][s]
            survived = set(r["survived"])
            missing = [c["keys"][i] for i in range(cfg["n_topics"]) if i not in survived]
            lines.append(
                f"| {s} | {len(r['survived'])}/{cfg['n_topics']} | "
                f"{', '.join(missing)[:100] if missing else '-'} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    import json
    from pathlib import Path

    t0 = time.time()
    report = run_multi_needle()
    out = Path("data/longeval/multi_needle_report")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), "utf-8"
    )
    (out / "report.md").write_text(render_markdown(report), "utf-8")
    print(render_markdown(report))
    print(f"\n总耗时 {time.time()-t0:.1f}s，报告写入 data/longeval/multi_needle_report/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
