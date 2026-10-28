"""Assessor 工厂：根据配置创建评估器。"""
from __future__ import annotations

from longersession.assessor.base import Assessor, MockAssessor
from longersession.assessor.openai_compat import OpenAICompatAssessor
from longersession.config import AssessorConfig


def create_assessor(config: AssessorConfig) -> Assessor:
    """根据 config.provider 创建评估器。"""
    if config.provider == "mock":
        return MockAssessor()
    if config.provider == "openai":
        return OpenAICompatAssessor(
            base_url=config.base_url,
            api_key=config.api_key,
            api_key_env=config.api_key_env,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            retries=config.retries,
        )
    raise ValueError(f"unknown assessor provider: {config.provider}")
