"""Service 编排：把切割、打标签、索引、存储、查询串起来。"""
from __future__ import annotations

from longersession.api.schemas import (
    ProcessRequest,
    ProcessResponse,
    QueryRequest,
    QueryResponse,
)
from longersession.config import Config
from longersession.encoder.factory import create_encoder
from longersession.index.aggregate import AggregatedView, build_aggregated_view
from longersession.index.builder import build_session_index
from longersession.index.store import Store
from longersession.segmentation.preprocess import messages_to_turns
from longersession.segmentation.segmenter import segment
from longersession.topic.labeller import Labeller, LLMLabeller, MockLabeller


class Service:
    """LongerSession 服务：process（建索引）+ query（结构化定位）。"""

    def __init__(self, config: Config):
        self.config = config
        self.encoder = create_encoder(config.encoder)
        self.labeller = self._create_labeller(config)
        self.store = Store(config.storage.db_path)
        self._view_cache: dict[str, AggregatedView] = {}

    def _create_labeller(self, config: Config) -> Labeller:
        if config.topic.provider == "mock":
            return MockLabeller()
        if config.topic.provider == "llm":
            return LLMLabeller()
        raise ValueError(f"unknown topic provider: {config.topic.provider}")

    def process(self, req: ProcessRequest) -> ProcessResponse:
        """完整链路：消息 → turns → 切割 → 打 topic → 构建索引 → 存储。"""
        turns = messages_to_turns(req.messages)
        chunks = segment(turns, self.encoder, self.config.segmentation, req.session_id)
        session_index = build_session_index(
            req.session_id, req.previous, chunks, turns, self.labeller
        )
        self.store.save_session(session_index, turns)
        topics = sorted({c.topic for c in chunks if c.topic})
        return ProcessResponse(
            session_id=req.session_id,
            num_turns=len(turns),
            num_chunks=len(chunks),
            topics=topics,
        )

    def query(self, req: QueryRequest) -> QueryResponse:
        """沿 previous 链聚合 topic/tag，返回标签列表或按标签取 chunk。

        不指定 label：返回全部 topic/tag 名（agent 自行选择）。
        指定 label：返回该标签下的 chunk 引用（自动区分 topic/tag）。
        """
        view = build_aggregated_view(self.store, req.session_id, self._view_cache)
        topics = sorted(view.topic_labels())
        tags = sorted(view.tag_labels())
        chunk_refs: list[tuple[str, str]] = []
        kind = req.kind

        if req.label is not None:
            if kind is None:
                # 自动判断：优先 tag，其次 topic
                if req.label in view.tag_refs:
                    kind = "tag"
                    chunk_refs = view.tag_refs.get(req.label, [])
                else:
                    kind = "topic"
                    chunk_refs = view.topic_refs.get(req.label, [])
            elif kind == "tag":
                chunk_refs = view.tag_refs.get(req.label, [])
            elif kind == "topic":
                chunk_refs = view.topic_refs.get(req.label, [])

        return QueryResponse(
            topics=topics,
            tags=tags,
            label=req.label,
            kind=kind,
            chunk_refs=chunk_refs,
        )
