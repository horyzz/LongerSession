"""轻量 token 估算。

本基准关注的是三种策略的**相对** token 开销与是否找回 needle，
因此用一个确定性的、可复现的估算器即可（不依赖具体 tokenizer）。

估算规则：
- CJK 字符 ≈ 1.0 token / 字符（中文 tokenizer 常见 0.7~1.5）
- 非 CJK 单词 ≈ 1.3 token / 单词（英文平均 1.2~1.4）
"""
from __future__ import annotations

import re

_CJK = re.compile(
    r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u3040-\u30ff\uac00-\ud7af]"
)


def estimate_tokens(text: str) -> int:
    """估算一段文本的 token 数（确定性、可复现）。"""
    if not text:
        return 0
    cjk_count = len(_CJK.findall(text))
    non_cjk = _CJK.sub(" ", text)
    word_count = len([w for w in non_cjk.split() if w.strip()])
    return max(1, int(round(cjk_count * 1.0 + word_count * 1.3)))
