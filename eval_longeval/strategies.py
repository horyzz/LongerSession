"""LongEval topics 任务四策略：full / truncation / compression / index。

每个策略都用 DeepSeek 产出「topic 列表」，再与真值做召回匹配。
真实 token 计数取自 API 返回的 usage（权威值）。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from eval_longeval.data import TestCase, text_to_messages
from eval_longeval.deepseek import chat
from eval_longeval.metrics import topic_recall

from longersession.config import Config
from longersession.encoder.factory import create_encoder
from longersession.segmentation.preprocess import messages_to_turns
from longersession.segmentation.segmenter import segment

SYSTEM = "You are a helpful assistant."

QUESTION = (
    "Now the record ends. List all distinct topics we discussed, "
    "one topic name per line. Only output the topic names, no numbering, "
    "no extra explanation."
)


@dataclass
class EvalResult:
    strategy: str
    predicted_topics: list[str]
    recall: float
    matched: list[str]
    missing: list[str]
    prompt_tokens: int
    max_prompt_tokens: int
    completion_tokens: int
    num_llm_calls: int
    latency_s: float
    note: str = ""
    extra: dict = field(default_factory=dict)


def _est_tokens(text: str) -> int:
    words = re.findall(r"\S+", text)
    return max(1, int(round(len(words) * 1.3)))


def _call(messages: list[dict], max_tokens: int) -> tuple[str, dict]:
    r = chat(messages, max_tokens=max_tokens)
    return r["content"], r["usage"]


def parse_topic_lines(text: str) -> list[str]:
    """从 LLM 输出解析 topic 行。"""
    text = text.strip()
    if not text:
        return []
    # JSON 风格 ["a", "b"]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1]
        items = [x.strip().strip("\"'`") for x in inner.split(",")]
        return [x for x in items if x]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) == 1 and "," in lines[0]:
        items = [x.strip() for x in lines[0].split(",")]
        return [x for x in items if x]
    out = []
    for l in lines:
        low = l.lower()
        if (
            low.startswith("here are")
            or low.startswith("the topics")
            or low.startswith("topics:")
            or low in ("topics", "topic", "topics discussed",)
        ):
            continue
        out.append(l)
    return out


def _result(
    strategy: str,
    predicted: list[str],
    case: TestCase,
    prompt_tokens: int,
    max_prompt_tokens: int,
    completion_tokens: int,
    num_calls: int,
    latency: float,
    note: str = "",
    extra: dict | None = None,
) -> EvalResult:
    tr = topic_recall(case.topics, predicted)
    return EvalResult(
        strategy=strategy,
        predicted_topics=predicted,
        recall=tr["recall"],
        matched=tr["matched"],
        missing=tr["missing"],
        prompt_tokens=prompt_tokens,
        max_prompt_tokens=max_prompt_tokens,
        completion_tokens=completion_tokens,
        num_llm_calls=num_calls,
        latency_s=round(latency, 2),
        note=note,
        extra=extra or {},
    )


# ---------------------------------------------------------------- full

def full_strategy(case: TestCase) -> EvalResult:
    t0 = time.time()
    content, usage = _call(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"{case.conversation}\n\n{QUESTION}"}],
        max_tokens=2048,
    )
    predicted = parse_topic_lines(content)
    return _result(
        "full", predicted, case,
        usage.get("prompt_tokens", 0), usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        1, time.time() - t0,
        note="整段对话一次性给模型",
    )


# ---------------------------------------------------------------- truncation

def truncation_strategy(case: TestCase, budget_tokens: int = 1500) -> EvalResult:
    # 从末尾向前保留对话，直到预算用完
    acc = 0
    picked = []
    for c in reversed(case.conversations):
        t = _est_tokens(c.text)
        if acc + t > budget_tokens and picked:
            break
        picked.append(c)
        acc += t
    picked.reverse()
    tail_text = "".join(c.text for c in picked)

    t0 = time.time()
    content, usage = _call(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"{tail_text}\n\n{QUESTION}"}],
        max_tokens=2048,
    )
    predicted = parse_topic_lines(content)
    return _result(
        "truncation", predicted, case,
        usage.get("prompt_tokens", 0), usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        1, time.time() - t0,
        note=f"仅保留末尾 {len(picked)}/{case.num_topics} 段对话",
        extra={"kept_conversations": len(picked), "budget_tokens": budget_tokens},
    )


# ---------------------------------------------------------------- compression

def compression_strategy(case: TestCase) -> EvalResult:
    t0 = time.time()
    # 第一步：有损摘要（丢失精确 topic 名）
    summ_content, summ_usage = _call(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Summarize the following conversation concisely, in at most 5 sentences, covering the main subjects discussed:\n\n" + case.conversation}],
        max_tokens=2048,
    )
    # 第二步：从摘要里提取 topic
    ext_content, ext_usage = _call(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"{summ_content}\n\n{QUESTION}"}],
        max_tokens=2048,
    )
    predicted = parse_topic_lines(ext_content)
    prompt_tokens = summ_usage.get("prompt_tokens", 0) + ext_usage.get("prompt_tokens", 0)
    max_prompt_tokens = max(summ_usage.get("prompt_tokens", 0), ext_usage.get("prompt_tokens", 0))
    completion_tokens = summ_usage.get("completion_tokens", 0) + ext_usage.get("completion_tokens", 0)
    return _result(
        "compression", predicted, case,
        prompt_tokens, max_prompt_tokens, completion_tokens,
        2, time.time() - t0,
        note="先有损摘要再提取 topic",
        extra={"summary_tokens": summ_usage.get("completion_tokens", 0)},
    )


# ---------------------------------------------------------------- index

_LABEL_PROMPT = (
    "Identify the single main topic of the following conversation excerpt. "
    "Reply with only a short topic name (a few words), nothing else.\n\nExcerpt:\n"
)


def _label_chunk(chunk_text: str) -> tuple[str, dict]:
    content, usage = _call(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"{_LABEL_PROMPT}{chunk_text}"}],
        max_tokens=2048,
    )
    label = content.strip().strip("\"'`.,;: ").strip()
    return label, usage


def _split_by_token_cap(turn_texts: list[str], max_tokens: int = 800) -> list[str]:
    """把 turn 文本按 token 上限贪心分组（防超大 chunk）。"""
    out: list[str] = []
    buf: list[str] = []
    acc = 0
    for t in turn_texts:
        tk = _est_tokens(t)
        if buf and acc + tk > max_tokens:
            out.append("\n".join(buf))
            buf = [t]
            acc = tk
        else:
            buf.append(t)
            acc += tk
    if buf:
        out.append("\n".join(buf))
    return out


def _dedupe(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for lab in labels:
        key = lab.lower()
        if lab and key not in seen and lab.lower() not in ("none", "n/a", "unknown"):
            seen.add(key)
            out.append(lab)
    return out


def index_strategy(case: TestCase, chunk_mode: str = "conversation") -> EvalResult:
    """LongerSession 索引：切 chunk → 逐 chunk 打 topic → 聚合标签。

    chunk_mode:
      - "conversation"：oracle 切分（每段单 topic 对话 = 一个 chunk）
      - "segmentation"：真实语义切割管线（char_ngram + DP）
    """
    t0 = time.time()

    if chunk_mode == "conversation":
        chunk_texts = [c.text for c in case.conversations]
        note = f"oracle 切分：{len(chunk_texts)} 个 chunk"
        extra = {"num_chunks": len(chunk_texts), "chunk_mode": chunk_mode}
    else:  # segmentation
        msgs = text_to_messages(case.conversation)
        turns = messages_to_turns(msgs)
        cfg = Config()
        cfg.encoder.provider = "char_ngram"
        encoder = create_encoder(cfg.encoder)
        chunks = segment(turns, encoder, cfg.segmentation, session_id="longeval")
        turn_map = {t.turn_id: t for t in turns}
        chunk_texts = []
        for ch in chunks:
            turn_texts = [
                turn_map[tid].combined_text() for tid in ch.turn_ids if tid in turn_map
            ]
            # 语义切割很弱时会并出超大 chunk；再按 token 上限切小，避免
            # 推理模型把 budget 耗尽在 reasoning 上导致空输出
            chunk_texts.extend(_split_by_token_cap(turn_texts, max_tokens=800))
        note = f"语义切割+尺寸上限：{len(chunk_texts)} 个 chunk"
        extra = {"num_chunks": len(chunk_texts), "chunk_mode": chunk_mode}

    labels: list[str] = []
    prompt_tokens = 0
    max_prompt_tokens = 0
    completion_tokens = 0
    num_calls = 0
    for ct in chunk_texts:
        label, usage = _label_chunk(ct)
        labels.append(label)
        prompt_tokens += usage.get("prompt_tokens", 0)
        max_prompt_tokens = max(max_prompt_tokens, usage.get("prompt_tokens", 0))
        completion_tokens += usage.get("completion_tokens", 0)
        num_calls += 1

    predicted = _dedupe(labels)
    extra["raw_labels"] = labels
    return _result(
        f"index ({chunk_mode})", predicted, case,
        prompt_tokens, max_prompt_tokens, completion_tokens,
        num_calls, time.time() - t0,
        note=note,
        extra=extra,
    )
