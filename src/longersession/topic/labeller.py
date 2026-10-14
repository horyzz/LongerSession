"""Topic 标签器。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Labeller(ABC):
    """给 chunk 打 topic 标签的抽象。"""

    @abstractmethod
    def label(self, chunk_texts: list[str]) -> list[str | None]:
        """给每个 chunk 返回 topic 名，None 表示无 topic。"""
        raise NotImplementedError


class MockLabeller(Labeller):
    """关键词匹配的简单标签器，用于跑通链路（真实标签由 LLM 提供）。"""

    DEFAULT_KEYWORDS: dict[str, list[str]] = {
        "部署": ["部署", "docker", "镜像", "k8s", "deploy", "容器"],
        "数据备份": ["备份", "backup", "数据迁移", "迁移"],
        "性能优化": ["性能", "优化", "索引", "查询", "慢"],
        "监控告警": ["监控", "日志", "告警", "alert", "log"],
        "安全加固": ["安全", "注入", "密码", "security", "sql"],
        "成本优化": ["成本", "cost", "账单", "存储"],
        "需求分析": ["需求", "requirement", "用户", "分析"],
        "技术选型": ["选型", "技术栈", "数据库", "前端", "框架"],
        "测试部署": ["测试", "部署", "上线", "回滚", "test", "deploy"],
    }

    def __init__(self, keyword_topics: dict[str, list[str]] | None = None):
        self.keyword_topics = keyword_topics or self.DEFAULT_KEYWORDS

    def label(self, chunk_texts: list[str]) -> list[str | None]:
        result: list[str | None] = []
        for text in chunk_texts:
            lower = text.lower()
            matched: str | None = None
            for topic, keywords in self.keyword_topics.items():
                if any(kw.lower() in lower for kw in keywords):
                    matched = topic
                    break
            result.append(matched)
        return result


class LLMLabeller(Labeller):
    """调用 LLM 打 topic 标签（占位，具体 provider 待接入）。"""

    def __init__(self, model: str = ""):
        self.model = model

    def label(self, chunk_texts: list[str]) -> list[str | None]:
        raise NotImplementedError("LLM labeller not implemented yet")
