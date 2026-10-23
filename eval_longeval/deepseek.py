"""DeepSeek OpenAI-compatible client（API key 从环境变量读取，绝不打印）。"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

# 读取环境变量（key 不落地、不打印）
API_KEY_ENV = "deepseek_api_key"
BASE_URL_ENV = "deepseek_base_url"
MODEL_ENV = "deepseek_model"

DEFAULT_BASE_URL = "https://api.deepseek.com"
# 官方 API 支持的模型名：deepseek-flash / deepseek-v4-pro
# （用户侧别名 "ds-4.1-flash" 对应 deepseek-flash）
DEFAULT_MODEL = "deepseek-flash"


class DeepSeekError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise DeepSeekError(
            f"环境变量 {API_KEY_ENV} 未设置（请先 export {API_KEY_ENV}=...）"
        )
    return key


def base_url() -> str:
    return os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")


def model_name() -> str:
    return os.environ.get(MODEL_ENV, DEFAULT_MODEL)


def chat(
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 120,
    retries: int = 5,
) -> dict:
    """调用 chat completions，返回 {content, usage}。messages 为 [{role, content}]。"""
    url = f"{base_url()}/chat/completions"
    payload = json.dumps(
        {
            "model": model_name(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode("utf-8")

    key = _api_key()
    last_err: Exception | None = None
    for attempt in range(retries):
        # 每次重试都重建 Request（避免复用带 data 的 Request 导致的异常）
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
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            if content and content.strip():
                return {"content": content, "usage": usage}
            # 空内容（推理 token 吃满预算）→ 重试
            last_err = DeepSeekError("empty content (reasoning-only)")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            last_err = DeepSeekError(f"HTTP {e.code}: {detail}")
            if e.code in (400, 401, 403, 404, 422):
                # 参数/认证/模型不存在错误，重试无意义
                raise last_err
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = DeepSeekError(f"network error: {e}")
        time.sleep(2 ** attempt)

    raise last_err if last_err else DeepSeekError("unknown error")
