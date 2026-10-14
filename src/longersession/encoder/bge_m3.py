"""BGE-M3 encoder：sentence-transformers 加载，lazy load。"""
from __future__ import annotations

from longersession.encoder.base import Encoder


class BGEM3Encoder(Encoder):
    """用 sentence-transformers 加载 BGE-M3。

    模型在首次 encode 时才加载（lazy load），避免 import 时依赖 torch。
    需通过 `uv sync --extra encoder` 安装可选依赖。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        max_seq_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length
        self._model = None

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(self.model_name, device=self.device)

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
