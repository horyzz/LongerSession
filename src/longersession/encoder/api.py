"""API encoder：调用 OpenAI 兼容 embeddings 接口，无需本地模型。"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request

from longersession.encoder.base import Encoder


class ApiEncoderError(RuntimeError):
    """API encoder 调用/解析错误。"""


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2 归一化（与其它 encoder 保持一致，供 cosine 相似度使用）。"""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class ApiEncoder(Encoder):
    """调用任意 OpenAI 兼容 /embeddings 接口生成 embedding。

    - 不依赖 torch / sentence-transformers，无需下载本地模型
    - base_url 指向任意 OpenAI 兼容 embeddings 服务
      （OpenAI、DeepSeek、硅基流动、本地 vLLM/Ollama 等）
    - api_key 留空时从 api_key_env 环境变量读取（key 不打印、不落地）
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        api_key_env: str = "openai_api_key",
        model: str = "text-embedding-3-small",
        batch_size: int = 16,
        timeout: int = 60,
        retries: int = 3,
        normalize: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.retries = retries
        self.normalize = normalize

    def _api_key(self) -> str:
        key = (self.api_key or os.environ.get(self.api_key_env, "")).strip()
        if not key:
            raise ApiEncoderError(
                f"未配置 api_key，且环境变量 {self.api_key_env} 未设置"
            )
        return key

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """请求一批文本的 embedding，返回与输入等长、按 index 排序的向量列表。"""
        url = f"{self.base_url}/embeddings"
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        key = self._api_key()

        last_err: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(
                url,
                data=payload,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                data = body["data"]
                data.sort(key=lambda d: d.get("index", 0))
                vecs = [d["embedding"] for d in data]
                if len(vecs) != len(texts):
                    raise ApiEncoderError(
                        f"返回向量数 {len(vecs)} 与请求数 {len(texts)} 不一致"
                    )
                return vecs
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                last_err = ApiEncoderError(f"HTTP {e.code}: {detail}")
                if e.code in (400, 401, 403, 404, 422):
                    raise last_err
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = ApiEncoderError(f"network error: {e}")
            time.sleep(2 ** attempt)

        raise last_err if last_err else ApiEncoderError("unknown error")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """编码文本列表，按 batch_size 分批请求，返回与输入等长的 embedding 列表。"""
        result: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for vec in self._embed_batch(batch):
                result.append(_l2_normalize(vec) if self.normalize else vec)
        return result
