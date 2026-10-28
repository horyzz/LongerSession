"""OpenAI 兼容 LLM 评估器（urllib 调用 chat/completions，无第三方依赖）。"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from longersession.assessor.base import Assessor, sanitize_boundaries
from longersession.assessor.prompts import build_assess_messages


class AssessorError(RuntimeError):
    """评估器调用/解析错误。"""


def _extract_json(content: str) -> dict:
    """从 LLM 输出里提取 JSON 对象（容忍 code fence 与前后说明文字）。"""
    text = content.strip()
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 兜底：抓取第一个 {...} 块
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise AssessorError(f"无法解析 LLM 输出的 JSON: {content[:200]!r}")


class OpenAICompatAssessor(Assessor):
    """调用任意 OpenAI 兼容 chat/completions 接口打 topic-shift 真值。

    用户通过 base_url / api_key / model 配置；api_key 留空时从 api_key_env
    指定的环境变量读取（key 绝不打印、不落地）。
    """

    def __init__(
        self,
        base_url: str = "https://api.deepseek.com",
        api_key: str = "",
        api_key_env: str = "deepseek_api_key",
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 120,
        retries: int = 5,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def _api_key(self) -> str:
        key = (self.api_key or os.environ.get(self.api_key_env, "")).strip()
        if not key:
            raise AssessorError(
                f"未配置 api_key，且环境变量 {self.api_key_env} 未设置"
            )
        return key

    def _chat(self, messages: list[dict]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
        ).encode("utf-8")
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
                content = body["choices"][0]["message"]["content"]
                if content and content.strip():
                    return content
                last_err = AssessorError("empty content")
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                last_err = AssessorError(f"HTTP {e.code}: {detail}")
                if e.code in (400, 401, 403, 404, 422):
                    raise last_err
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = AssessorError(f"network error: {e}")
            time.sleep(2 ** attempt)

        raise last_err if last_err else AssessorError("unknown error")

    def assess(self, messages: list[dict], session_id: str | None = None) -> list[int]:
        chat_messages = build_assess_messages(messages)
        content = self._chat(chat_messages)
        data = _extract_json(content)
        raw = data.get("boundaries", [])
        if not isinstance(raw, list):
            raise AssessorError(f"boundaries 不是数组: {raw!r}")
        return sanitize_boundaries(messages, raw)
