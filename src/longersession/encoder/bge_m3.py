"""BGE-M3 encoder：sentence-transformers 加载，lazy load。

支持两种模型下载源：
- huggingface：默认从 HuggingFace Hub 下载（需代理）。
- modelscope：从 ModelScope（魔搭）下载，国内直连，无需代理。
"""
from __future__ import annotations

from longersession.encoder.base import Encoder


class BGEM3Encoder(Encoder):
    """用 sentence-transformers 加载 BGE-M3。

    模型在首次 encode 时才加载（lazy load），避免 import 时依赖 torch。
    需通过 `uv sync --extra encoder` 安装可选依赖。

    source="modelscope" 时通过 modelscope SDK 下载模型到本地缓存，
    再交给 SentenceTransformer 从本地路径加载（不访问 HuggingFace）。
    """

    # bge-m3 仓库里 onnx/（约 2.1GB）、colbert/sparse 线性权重、imgs/ 等
    # 是冗余文件，dense embedding 用不到，默认跳过以加速下载。
    DEFAULT_ALLOW_PATTERNS: list[str] = [
        "config.json",
        "config_sentence_transformers.json",
        "configuration.json",
        "modules.json",
        "sentence_bert_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "pytorch_model.bin",
        "1_Pooling/*",
        "README.md",
    ]

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        max_seq_length: int = 2048,
        source: str = "modelscope",  # huggingface | modelscope
        local_dir: str | None = None,  # 已下载的本地目录，跳过下载
        allow_patterns: list[str] | None = None,  # modelscope 下载白名单
    ):
        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length
        self.source = source
        self.local_dir = local_dir
        self.allow_patterns = (
            allow_patterns if allow_patterns is not None else self.DEFAULT_ALLOW_PATTERNS
        )
        self._model = None

    def _resolve_model_path(self) -> str:
        """返回交给 SentenceTransformer 的模型路径（或 Hub 名）。"""
        if self.local_dir:
            return self.local_dir
        if self.source == "modelscope":
            from modelscope import snapshot_download  # lazy import

            return snapshot_download(
                self.model_name,
                allow_patterns=self.allow_patterns,
            )
        return self.model_name

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        path = self._resolve_model_path()
        self._model = SentenceTransformer(path, device=self.device)
        # 应用截断长度（与训练对齐；之前漏了这行导致默认 8192）
        self._model.max_seq_length = self.max_seq_length

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            self._load()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=8,
            show_progress_bar=False,
        )
        return embeddings.tolist()
