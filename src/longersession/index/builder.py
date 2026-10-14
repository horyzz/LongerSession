"""SessionIndex 构建。"""
from __future__ import annotations

from longersession.models.chunk import Chunk
from longersession.models.session import LabelRef, SessionIndex
from longersession.models.turn import Turn
from longersession.topic.labeller import Labeller


def build_session_index(
    session_id: str,
    previous: str | None,
    chunks: list[Chunk],
    turns: list[Turn],
    labeller: Labeller,
) -> SessionIndex:
    """构建 SessionIndex：给 chunk 打 topic（自由生成），构建 topic_refs。

    - topic 是 LLM 自由生成的字符串，不归一化、无 embedding
    - tag 由离线增量聚类事后填充，构建时为空
    """
    turn_map = {t.turn_id: t for t in turns}
    chunk_texts: list[str] = []
    for c in chunks:
        texts = [turn_map[tid].combined_text() for tid in c.turn_ids if tid in turn_map]
        chunk_texts.append("\n".join(texts))

    topic_names = labeller.label(chunk_texts)

    topic_ref_map: dict[str, list[str]] = {}
    for c, name in zip(chunks, topic_names):
        c.topic = name
        if name is not None:
            topic_ref_map.setdefault(name, []).append(c.chunk_id)

    topic_refs = [
        LabelRef(label=name, chunk_ids=ids)
        for name, ids in sorted(topic_ref_map.items())
    ]

    return SessionIndex(
        session_id=session_id,
        previous=previous,
        topic_refs=topic_refs,
        tag_refs=[],
        chunks=chunks,
    )
