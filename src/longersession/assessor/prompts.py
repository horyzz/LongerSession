"""评估 prompt 模板：让 LLM 给会话打 topic-shift 边界真值。"""
from __future__ import annotations

from longersession.models.message import Message
from longersession.segmentation.preprocess import messages_to_turns

SYSTEM_PROMPT = (
    "You are an expert conversation segmenter. You read a multi-turn dialogue and "
    "identify where the topic changes. A topic shift happens when the discussion "
    "moves to a clearly different subject. You output ONLY a JSON object, no prose."
)

OUTPUT_SCHEMA = (
    '{"boundaries": [3, 7]}'
)


def _to_turns(messages: list[dict]):
    msgs = [Message.model_validate(m) for m in messages]
    return messages_to_turns(msgs)


def build_assess_messages(messages: list[dict]) -> list[dict]:
    """把原始消息转成可直接发给 LLM 的 chat messages（system + user）。

    边界语义（与 eval/切割管线一致）：boundaries 里的整数 k 表示「在第 k 个
    turn 之后切」，即第 k 个 turn 与第 k+1 个 turn 属于不同 topic。
    """
    turns = _to_turns(messages)

    lines: list[str] = []
    for i, t in enumerate(turns, start=1):
        lines.append(f"[{i}] user: {t.user_text()}")
        at = t.assistant_text()
        if at:
            lines.append(f"    assistant: {at}")
    transcript = "\n".join(lines)

    user_prompt = (
        "Below is a conversation split into numbered turns (1-based). "
        "Find every topic shift. For each shift, record the turn index k such that "
        "turn k and turn k+1 belong to different topics (i.e. a new topic starts "
        "after turn k).\n\n"
        f"{transcript}\n\n"
        f'Output ONLY a JSON object in this exact shape: {OUTPUT_SCHEMA}. '
        "If there is no topic shift, output {\"boundaries\": []}. "
        "The list must be strictly ascending integers between 1 and "
        f"{len(turns) - 1}."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
