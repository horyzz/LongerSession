"""配置模块。"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class EncoderConfig(BaseModel):
    """Encoder 配置（承接分数计算的可插拔 encoder）。"""

    provider: str = "mock"  # mock | word_bag | char_ngram | bge_m3 | api
    model_name: str = "BAAI/bge-m3"
    source: str = "modelscope"  # huggingface | modelscope（bge_m3 模型下载源）
    local_dir: str | None = None  # 本地模型目录（平衡模型），设置后跳过下载
    lazy_load: bool = True
    device: str = "cpu"
    max_seq_length: int = 2048  # 对齐训练口径（平衡模型 seq=2048）
    # API encoder（OpenAI 兼容 embeddings 接口，无需本地模型）
    api_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""  # 留空则从 api_key_env 环境变量读取
    api_key_env: str = "openai_api_key"
    api_model: str = "text-embedding-3-small"
    api_batch_size: int = 16
    api_timeout: int = 60
    api_retries: int = 3


class SegmentationConfig(BaseModel):
    """切割配置（DP 切割 + overlap）。"""

    # chunk 长度约束（按 turn 数）
    min_chunk_turns: int = 2
    max_chunk_turns: int = 20
    # chunk 长度约束（按 token 数，软约束）
    min_chunk_tokens: int = 64
    max_chunk_tokens: int = 4096
    # 承接分数阈值：score 越低越该切，DP 代价 = score - threshold
    threshold: float = 0.5
    # 模糊区（score 接近 threshold）宽度，用于决定 overlap
    boundary_tolerance: float = 0.1
    # overlap 覆盖的 turn 数（模糊区边界共享）
    overlap_turns: int = 1


class TopicConfig(BaseModel):
    """Topic 标签配置。"""

    provider: str = "mock"  # mock | llm
    max_topic_length: int = 10  # topic 名最大字符数
    embedding_dim: int = 1024  # BGE-M3 的 embedding 维度


class StorageConfig(BaseModel):
    """存储配置。"""

    db_path: Path = Path("data/longersession.db")


class TelemetryConfig(BaseModel):
    """Telemetry 本地收集配置（阶段 3）。"""

    enabled: bool = True  # 是否在每次 process 后追加本地 JSONL 记录
    path: Path = Path("data/telemetry/sessions.jsonl")  # JSONL 追加存储路径


class AssessorConfig(BaseModel):
    """LLM 评估器配置（阶段 4，用户自配 LLM）。"""

    provider: str = "openai"  # mock | openai
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""  # 留空则从 api_key_env 环境变量读取
    api_key_env: str = "deepseek_api_key"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 120
    retries: int = 5


class Config(BaseModel):
    """全局配置。"""

    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    topic: TopicConfig = Field(default_factory=TopicConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    assessor: AssessorConfig = Field(default_factory=AssessorConfig)


def get_config() -> Config:
    """返回默认配置实例。"""
    return Config()
