"""AggregatedView：沿 previous 链聚合标签引用。"""
from __future__ import annotations

from longersession.index.store import Store
from longersession.models.session import SessionIndex


class AggregatedView:
    """沿 previous 链聚合的标签视图（L1 粗索引的可全量常驻部分）。

    topic_refs: topic 名 -> [(session_id, chunk_id), ...]
    tag_refs:   tag 名  -> [(session_id, chunk_id), ...]
    只放引用，不复制原文。
    """

    def __init__(self) -> None:
        self.topic_refs: dict[str, list[tuple[str, str]]] = {}
        self.tag_refs: dict[str, list[tuple[str, str]]] = {}

    def add_session(self, session: SessionIndex) -> None:
        for ref in session.topic_refs:
            for cid in ref.chunk_ids:
                self.topic_refs.setdefault(ref.label, []).append(
                    (session.session_id, cid)
                )
        for ref in session.tag_refs:
            for cid in ref.chunk_ids:
                self.tag_refs.setdefault(ref.label, []).append(
                    (session.session_id, cid)
                )

    def topic_labels(self) -> list[str]:
        return list(self.topic_refs.keys())

    def tag_labels(self) -> list[str]:
        return list(self.tag_refs.keys())


def build_aggregated_view(
    store: Store,
    start_session_id: str,
    cache: dict[str, AggregatedView] | None = None,
) -> AggregatedView:
    """沿 previous 链聚合所有祖先 + 自身的 topic/tag 引用。

    session immutable，因此聚合结果永久有效，可用 cache 避免重复遍历。
    """
    if cache is not None and start_session_id in cache:
        return cache[start_session_id]

    view = AggregatedView()
    sid: str | None = start_session_id
    while sid is not None:
        loaded = store.load_session(sid)
        if loaded is None:
            break
        session, _ = loaded
        view.add_session(session)
        sid = session.previous

    if cache is not None:
        cache[start_session_id] = view
    return view
