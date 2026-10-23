"""Needle QA 验证：长 session 埋具体事实 → 定向提问，对比四种策略。

这是「索引替代压缩」的主场：问题只指向一个 chunk，而不是要列出所有 topic。
重点看两个量：
  1. 能否答对（needle 是否找回，lossless vs lossy）
  2. query 阶段喂给模型多少 token（答案窗口大小）

索引用 embedding 检索（char_ngram，本项目自带零依赖 encoder），
把问题与每个 chunk 算余弦相似度，只拉 top-1 chunk 喂给模型。
"""
from __future__ import annotations

import math
import random
import re
import secrets
import time
from dataclasses import dataclass, field

from eval_longeval.data import Conversation, load_conversations
from eval_longeval.deepseek import chat
from eval_longeval.strategies import SYSTEM, _est_tokens

from longersession.config import Config
from longersession.encoder.factory import create_encoder

_QUERY_STOPWORDS = {
    "what", "is", "are", "our", "the", "a", "an", "of", "in", "on", "for",
    "to", "and", "or", "do", "does", "did", "was", "were", "my", "your",
    "we", "you", "i", "me", "it", "this", "that", "be", "as", "at", "by",
}

QUESTION_TEMPLATE = "What is our build tag?"


@dataclass
class NeedleCase:
    test_id: str
    n_topics: int
    needle_value: str
    needle_conv_idx: int   # needle 所在 conversation 的下标（0-based）
    session_text: str      # 拼接后的长 session（含 needle）
    conversations: list[Conversation]
    question: str = QUESTION_TEMPLATE


def _inject_needle(conv_text: str, needle_value: str) -> str:
    """在对话结尾前插入一条「请记住 build tag」的 turn。"""
    needle_turn = (
        f"USER: Also, please remember this detail for later: "
        f"our build tag is {needle_value}. \n"
        f"ASSISTANT: Noted — the build tag is {needle_value}. \n"
    )
    marker = "Great, let's talk about the next topic"
    if marker in conv_text:
        return conv_text.replace(marker, needle_turn + "USER: " + marker, 1)
    return conv_text + needle_turn


def generate_needle_case(
    n_topics: int = 10,
    needle_depth: float = 0.3,
    seed: int = 0,
) -> NeedleCase:
    convs = load_conversations()
    rng = random.Random(seed)
    chosen = rng.sample(convs, n_topics)
    needle_value = (
        f"BUILD-{secrets.token_hex(2).upper()}-{secrets.token_hex(2).upper()}"
    )
    needle_conv_idx = int(round(needle_depth * (n_topics - 1)))

    injected = []
    for i, c in enumerate(chosen):
        text = _inject_needle(c.text, needle_value) if i == needle_conv_idx else c.text
        injected.append(
            Conversation(topic=c.topic, topic_id=c.topic_id, text=text)
        )

    return NeedleCase(
        test_id=f"needle-{n_topics}-d{needle_depth}-s{seed}",
        n_topics=n_topics,
        needle_value=needle_value,
        needle_conv_idx=needle_conv_idx,
        session_text="".join(c.text for c in injected),
        conversations=injected,
    )


def _needle_found(response: str, needle: str) -> bool:
    return needle.lower() in response.lower()


def _call_answer(prompt: str, max_tokens: int = 256) -> tuple[str, dict]:
    r = chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return r["content"], r["usage"]


# ---------------------------------------------------------------- strategies

def answer_full(case: NeedleCase) -> dict:
    t0 = time.time()
    prompt = (
        f"Below is our full previous conversation.\n\n{case.session_text}\n\n"
        f"Question: {case.question}\nAnswer concisely."
    )
    content, usage = _call_answer(prompt)
    return {
        "correct": _needle_found(content, case.needle_value),
        "answer": content.strip(),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "llm_calls": 1,
        "latency_s": round(time.time() - t0, 2),
        "answer_context_tokens": _est_tokens(case.session_text),
    }


def answer_truncation(case: NeedleCase, budget_tokens: int = 1500) -> dict:
    t0 = time.time()
    # 从末尾保留对话
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
    prompt = f"Below is the recent part of our conversation.\n\n{tail}\n\nQuestion: {case.question}\nAnswer concisely. If the information is not present, say NOT FOUND."
    content, usage = _call_answer(prompt)
    return {
        "correct": _needle_found(content, case.needle_value),
        "answer": content.strip(),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "llm_calls": 1,
        "latency_s": round(time.time() - t0, 2),
        "answer_context_tokens": _est_tokens(tail),
        "kept_conversations": len(picked),
    }


def answer_compression(case: NeedleCase) -> dict:
    t0 = time.time()
    summ, summ_usage = _call_answer(
        "Summarize the following conversation concisely, in at most 6 sentences:\n\n"
        + case.session_text,
        max_tokens=2048,
    )
    prompt = (
        f"Using only the summary below, answer the question. "
        f"If the answer is not in the summary, say NOT FOUND.\n\n"
        f"Summary:\n{summ}\n\nQuestion: {case.question}"
    )
    content, ans_usage = _call_answer(prompt)
    prompt_tokens = summ_usage.get("prompt_tokens", 0) + ans_usage.get("prompt_tokens", 0)
    return {
        "correct": _needle_found(content, case.needle_value),
        "answer": content.strip(),
        "prompt_tokens": prompt_tokens,
        "llm_calls": 2,
        "latency_s": round(time.time() - t0, 2),
        "answer_context_tokens": _est_tokens(summ),
    }


def _keywords(q: str) -> list[str]:
    return [
        w for w in re.findall(r"[a-z0-9]+", q.lower())
        if w not in _QUERY_STOPWORDS and len(w) >= 2
    ]


def _lexical_score(q: str, chunk_text: str) -> int:
    kws = _keywords(q)
    low = chunk_text.lower()
    phrase = " ".join(kws)
    if phrase and phrase in low:
        return 10 + sum(1 for kw in kws if kw in low)
    return sum(1 for kw in kws if kw in low)


def answer_index(case: NeedleCase, top_k: int = 1) -> dict:
    """索引 + 检索：只拉与问题最相关的 chunk 喂给模型。

    检索用「词法匹配优先 + embedding 兜底」：char_ngram 是弱 encoder，
    小 needle 会淹没在大 chunk 里，纯 embedding 排不准，故先按问题关键词
    命中数排序，再用 embedding 相似度做 tie-break。
    """
    t0 = time.time()
    cfg = Config()
    cfg.encoder.provider = "char_ngram"
    encoder = create_encoder(cfg.encoder)

    chunk_texts = [c.text for c in case.conversations]
    q_emb = encoder.encode([case.question])[0]
    c_embs = encoder.encode(chunk_texts)

    def _cos(a, b):
        na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)

    scored = sorted(
        [
            (
                i,
                _lexical_score(case.question, chunk_texts[i]),
                _cos(q_emb, c_embs[i]),
            )
            for i in range(len(chunk_texts))
        ],
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )
    top = scored[:top_k]
    retrieved = "\n\n".join(chunk_texts[i] for i, _, _ in top)

    prompt = (
        f"Using only the excerpt below, answer the question. "
        f"If the answer is not in the excerpt, say NOT FOUND.\n\n"
        f"Excerpt:\n{retrieved}\n\nQuestion: {case.question}"
    )
    content, usage = _call_answer(prompt)
    return {
        "correct": _needle_found(content, case.needle_value),
        "answer": content.strip(),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "llm_calls": 1,
        "latency_s": round(time.time() - t0, 2),
        "answer_context_tokens": _est_tokens(retrieved),
        "retrieved_chunk_idx": top[0][0] if top else None,
        "needle_conv_idx": case.needle_conv_idx,
        "retrieved_hit": bool(top and top[0][0] == case.needle_conv_idx),
    }


# ---------------------------------------------------------------- 运行器

def run_needle_qa(
    n_topics: int = 10,
    depths: list[float] | None = None,
    budget_tokens: int = 1500,
    seed: int = 0,
) -> dict:
    depths = depths if depths is not None else [0.1, 0.5, 0.9]
    cases = [generate_needle_case(n_topics, d, seed) for d in depths]

    strategy_names = ["full", "truncation", "compression", "index"]
    per_case = []
    for case in cases:
        results = {
            "full": answer_full(case),
            "truncation": answer_truncation(case, budget_tokens),
            "compression": answer_compression(case),
            "index": answer_index(case),
        }
        per_case.append(
            {
                "test_id": case.test_id,
                "needle_value": case.needle_value,
                "needle_conv_idx": case.needle_conv_idx,
                "n_topics": case.n_topics,
                "results": results,
            }
        )

    agg = {}
    for s in strategy_names:
        vals = [c["results"][s] for c in per_case]
        agg[s] = {
            "correct_count": sum(1 for v in vals if v["correct"]),
            "n": len(vals),
            "accuracy": round(sum(1 for v in vals if v["correct"]) / len(vals), 3),
            "avg_answer_context_tokens": round(
                sum(v["answer_context_tokens"] for v in vals) / len(vals), 1
            ),
            "avg_prompt_tokens": round(
                sum(v["prompt_tokens"] for v in vals) / len(vals), 1
            ),
            "avg_llm_calls": round(sum(v["llm_calls"] for v in vals) / len(vals), 1),
        }

    return {
        "config": {
            "n_topics": n_topics,
            "depths": depths,
            "budget_tokens": budget_tokens,
            "seed": seed,
            "question": QUESTION_TEMPLATE,
        },
        "aggregate": agg,
        "cases": per_case,
    }


def render_markdown(report: dict) -> str:
    cfg = report["config"]
    agg = report["aggregate"]
    lines = [
        "# Needle QA · 定向检索验证（DeepSeek）",
        "",
        f"- 长 session：{cfg['n_topics']} topics 拼接，needle 埋在 depths={cfg['depths']} 处",
        f"- 截断预算：{cfg['budget_tokens']} tokens",
        f"- 问题：`{cfg['question']}`（答案=埋入的唯一 build tag）",
        "",
        "## 各策略汇总",
        "",
        "| 策略 | 答对率 | 单次答案窗口 tokens | 总 prompt tokens | LLM 调用 |",
        "|---|---|---|---|---|",
    ]
    for s in ["full", "truncation", "compression", "index"]:
        a = agg[s]
        lines.append(
            f"| {s} | {a['accuracy']:.0%} ({a['correct_count']}/{a['n']}) | "
            f"{a['avg_answer_context_tokens']:.0f} | {a['avg_prompt_tokens']:.0f} | "
            f"{a['avg_llm_calls']:.1f} |"
        )
    lines += ["", "## 逐用例", ""]
    for c in report["cases"]:
        lines.append(f"### {c['test_id']}（needle 在第 {c['needle_conv_idx']} 段对话）")
        lines.append("")
        lines.append("| 策略 | 答对 | 答案 |")
        lines.append("|---|---|---|")
        for s in ["full", "truncation", "compression", "index"]:
            r = c["results"][s]
            lines.append(
                f"| {s} | {'✅' if r['correct'] else '❌'} | {r['answer'][:60]} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    import json
    import sys
    from pathlib import Path

    t0 = time.time()
    report = run_needle_qa()
    out = Path("data/longeval/needle_qa_report")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), "utf-8"
    )
    (out / "report.md").write_text(render_markdown(report), "utf-8")
    print(render_markdown(report))
    print(f"\n总耗时 {time.time()-t0:.1f}s，报告写入 data/longeval/needle_qa_report/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
