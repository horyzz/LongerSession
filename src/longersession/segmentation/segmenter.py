"""切割编排：承接分数 → DP 切割 → chunk 生成（含 overlap）。"""
from __future__ import annotations

from longersession.config import SegmentationConfig
from longersession.encoder.base import Encoder
from longersession.models.chunk import Chunk
from longersession.models.turn import Turn
from longersession.segmentation.coherence import compute_coherence_scores
from longersession.segmentation.dp import dp_segment


def _build_chunks(
    turns: list[Turn],
    cuts: list[int],
    session_id: str,
    overlap_turns: int,
) -> list[Chunk]:
    """根据切点生成 chunk。

    cuts 是 1-based 切点（在 turn p-1 之后切）。overlap 时每个内部切点
    两侧的 chunk 各多包含 overlap_turns 个 turn，实现边界重叠。
    """
    n = len(turns)
    boundaries = [0] + cuts + [n]
    chunks: list[Chunk] = []

    for i in range(len(boundaries) - 1):
        b_start = boundaries[i]
        b_end = boundaries[i + 1]

        # overlap 延伸：左边界向左、右边界向右各延伸 overlap_turns
        start = (b_start - overlap_turns) if i > 0 else 0
        end = (b_end + overlap_turns) if i < len(boundaries) - 2 else n
        start = max(0, start)
        end = min(n, end)

        overlap_prev = start < b_start
        chunk_turns = turns[start:end]

        turn_ids = [t.turn_id for t in chunk_turns]
        used_tools: list[str] = []
        seen: set[str] = set()
        for t in chunk_turns:
            for tool in t.used_tools():
                if tool not in seen:
                    seen.add(tool)
                    used_tools.append(tool)

        token_count = sum(len(t.combined_text()) for t in chunk_turns)

        chunks.append(
            Chunk(
                chunk_id=f"chunk_{i}",
                session_id=session_id,
                turn_ids=turn_ids,
                used_tools=used_tools,
                overlap_prev=overlap_prev,
                content_ref=f"{session_id}:{start}:{end}",
                token_count=token_count,
            )
        )
    return chunks


def segment_with_boundaries(
    turns: list[Turn],
    encoder: Encoder,
    config: SegmentationConfig,
    session_id: str = "session",
) -> tuple[list[Chunk], list[int]]:
    """切割并同时返回预测边界（供 telemetry 记录）。

    返回 (chunks, cuts)：cuts 是 1-based 切点列表（在第 cuts[i] 个 turn 之后切），
    与 eval 的 boundaries 口径一致。
    """
    scores = compute_coherence_scores(turns, encoder)
    cuts = dp_segment(
        scores,
        min_len=config.min_chunk_turns,
        max_len=config.max_chunk_turns,
        threshold=config.threshold,
    )
    return _build_chunks(turns, cuts, session_id, config.overlap_turns), cuts


def segment(
    turns: list[Turn],
    encoder: Encoder,
    config: SegmentationConfig,
    session_id: str = "session",
) -> list[Chunk]:
    """把 turn 列表切割成 chunk 列表（含 overlap 与 used_tools）。"""
    return segment_with_boundaries(turns, encoder, config, session_id)[0]
