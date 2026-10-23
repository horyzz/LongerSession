"""三种策略：截断 / 压缩 / 索引。

每个策略都回答同一个问题：在给定 context 预算下，
能否把埋在长 session 深处的 needle 放回「可回答的 context」中。

判定用确定性的 oracle：needle 的随机密钥字符串是否完整出现在
策略产出的 context 里。这隔离了「检索/索引是否有效」这一核心主张，
不受生成模型抽取能力干扰（与 NIAH 的评分方式一致）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from benchmarks.generator import GeneratedSession
from benchmarks.tokenizer import estimate_tokens

from longersession.config import Config
from longersession.encoder.factory import create_encoder
from longersession.index.aggregate import AggregatedView
from longersession.index.builder import build_session_index
from longersession.models.message import Message, Role, ToolCall
from longersession.segmentation.preprocess import messages_to_turns
from longersession.segmentation.segmenter import segment
from longersession.topic.labeller import MockLabeller


@dataclass
class StrategyResult:
    name: str
    answerable: bool
    context_tokens: int
    total_tokens: int
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def token_ratio(self) -> float:
        return self.context_tokens / self.total_tokens if self.total_tokens else 0.0


# ---------------------------------------------------------------- 基础

def _needle_present(context: str, needle: str) -> bool:
    return needle in context


def _tail_turns(session: GeneratedSession, budget_tokens: int) -> list:
    """从末尾向前取 turn，直到预算用完。返回命中的 TurnRecord 列表（保持原顺序）。"""
    picked: list = []
    acc = 0
    for t in reversed(session.turns):
        if acc + t.tokens > budget_tokens and picked:
            break
        picked.append(t)
        acc += t.tokens
    picked.reverse()
    return picked


# ---------------------------------------------------------------- 截断

def truncation_strategy(session: GeneratedSession, budget_tokens: int) -> StrategyResult:
    """朴素截断：只保留末尾 budget 内的历史。"""
    tail = _tail_turns(session, budget_tokens)
    tail_text = "\n\n".join(t.text for t in tail)
    context = f"[问题] {session.query}\n\n[最近历史]\n{tail_text}"
    return StrategyResult(
        name="truncation",
        answerable=_needle_present(context, session.needle.value),
        context_tokens=estimate_tokens(context),
        total_tokens=session.total_tokens,
        note=f"仅保留末尾 {len(tail)}/{session.n_turns} turn",
        extra={"kept_turns": len(tail), "dropped_turns": session.n_turns - len(tail)},
    )


# ---------------------------------------------------------------- 压缩

def _summarize_turn(session: GeneratedSession, turn_index: int) -> str:
    """有损摘要：只保留 topic + 用户提问（意图），丢弃助手回答里的具体数值。

    这是「压缩会丢信息」的真实写照：摘要保留了「问过什么」，
    却丢掉了「具体答案是什么」（needle 就藏在答案里）。
    """
    t = session.turns[turn_index]
    user_q = t.text.split("\n", 1)[0]
    return f"[{t.topic}] {user_q}"


def compression_strategy(
    session: GeneratedSession, budget_tokens: int
) -> StrategyResult:
    """压缩（有损摘要）：较早的历史被摘要，末尾 budget 内保留原文。"""
    tail = _tail_turns(session, budget_tokens)
    tail_indices = {t.index for t in tail}
    head_indices = [i for i in range(session.n_turns) if i not in tail_indices]

    summary_lines = [_summarize_turn(session, i) for i in head_indices]
    summary_text = "\n".join(summary_lines)
    tail_text = "\n\n".join(t.text for t in tail)

    context = (
        f"[问题] {session.query}\n\n"
        f"[历史摘要]\n{summary_text}\n\n"
        f"[最近历史]\n{tail_text}"
    )
    return StrategyResult(
        name="compression",
        answerable=_needle_present(context, session.needle.value),
        context_tokens=estimate_tokens(context),
        total_tokens=session.total_tokens,
        note=f"摘要 {len(head_indices)} turn + 保留末尾 {len(tail)} turn",
        extra={
            "summarized_turns": len(head_indices),
            "kept_turns": len(tail),
            "summary_tokens": estimate_tokens(summary_text),
        },
    )


# ---------------------------------------------------------------- 索引

def _to_messages(messages: list[dict]) -> list[Message]:
    out: list[Message] = []
    for m in messages:
        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["name"],
                arguments=tc.get("arguments", ""),
            )
            for tc in m.get("tool_calls", [])
        ]
        out.append(
            Message(
                role=Role(m["role"]),
                content=m.get("content", ""),
                tool_calls=tool_calls,
                tool_call_id=m.get("tool_call_id"),
            )
        )
    return out


def _char_overlap(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def best_match_label(query: str, labels: list[str]) -> str | None:
    """在 topic 标签里挑与 query 最匹配的（字符集合 Jaccard）。"""
    best: str | None = None
    best_score = 0.0
    for label in labels:
        score = _char_overlap(label, query)
        if score > best_score:
            best_score = score
            best = label
    return best


def index_strategy(
    session: GeneratedSession,
    config: Config | None = None,
    budget_tokens: int | None = None,
) -> StrategyResult:
    """LongerSession 索引：全量建索引 → 按 topic 定位 → 只拉相关 chunk 原文。

    - 使用「每个 turn 一个 chunk」的 oracle 切割，隔离出「索引/检索」这一核心主张；
      （切割质量由 eval/run.py 的 Boundary F1 单独衡量）
    - 不走 SQLite：单 session 检索直接用 AggregatedView 聚合内存索引即可。
    """
    cfg = config if config is not None else Config()
    # oracle 切割：1 turn = 1 chunk
    cfg.segmentation.min_chunk_turns = 1
    cfg.segmentation.max_chunk_turns = 1
    cfg.segmentation.overlap_turns = 0

    encoder = create_encoder(cfg.encoder)
    labeller = MockLabeller()

    msgs = _to_messages(session.messages)
    turns = messages_to_turns(msgs)
    chunks = segment(turns, encoder, cfg.segmentation, session_id="bench")
    session_index = build_session_index("bench", None, chunks, turns, labeller)

    view = AggregatedView()
    view.add_session(session_index)

    label = best_match_label(session.query, view.topic_labels())

    retrieved_texts: list[str] = []
    refs: list[tuple[str, str]] = []
    if label is not None:
        refs = view.topic_refs.get(label, [])
        turn_map = {t.turn_id: t for t in turns}
        for _, cid in refs:
            chunk = session_index.chunk_by_id(cid)
            if chunk is None:
                continue
            for tid in chunk.turn_ids:
                if tid in turn_map:
                    retrieved_texts.append(turn_map[tid].combined_text())

    retrieved_text = "\n\n".join(retrieved_texts)
    context = (
        f"[问题] {session.query}\n\n"
        f"[按 topic 检索到 {label or '-'}] 的历史原文]\n{retrieved_text}"
    )

    answerable = _needle_present(context, session.needle.value)

    if budget_tokens is not None and estimate_tokens(context) > budget_tokens:
        note = (
            f"检索 {len(refs)} chunk 后仍超预算 "
            f"({estimate_tokens(context)}>{budget_tokens})"
        )
    else:
        note = f"检索 {len(refs)} chunk，命中 topic「{label or '-'}」"

    return StrategyResult(
        name="index",
        answerable=answerable,
        context_tokens=estimate_tokens(context),
        total_tokens=session.total_tokens,
        note=note,
        extra={
            "retrieved_label": label,
            "num_chunks": len(chunks),
            "retrieved_chunks": len(refs),
            "retrieved_tokens": estimate_tokens(retrieved_text),
        },
    )


# ---------------------------------------------------------------- 汇总

def run_all(
    session: GeneratedSession,
    budget_tokens: int = 512,
    config: Config | None = None,
) -> list[StrategyResult]:
    """对同一 session 跑三种策略，返回结果列表。"""
    return [
        truncation_strategy(session, budget_tokens),
        compression_strategy(session, budget_tokens),
        index_strategy(session, config, budget_tokens),
    ]
