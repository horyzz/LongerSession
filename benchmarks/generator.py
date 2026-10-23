"""合成长 session + 埋入 needle 的生成器。

生成一段 N turn 的运维/研发支持对话，覆盖多个 topic。
在指定深度（needle_index）埋入一条「关键信息（needle）」，
最后追加一个需要该信息才能回答的 query。

needle 是一条随机、唯一的密钥字符串，只出现在 needle turn 中，
用于让 oracle 判定「该策略是否把关键信息放进了 context」。
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field

from benchmarks.tokenizer import estimate_tokens

# 各 topic 的「用户提问 / 助手回答」模板（filler turn）
# needle 固定埋在「数据备份」topic 中。
FILLER_TURNS: dict[str, tuple[str, str]] = {
    "部署方案": (
        "新的服务要怎么部署上线？",
        "建议用 Docker 打包镜像，再通过 K8s 滚动发布，先灰度再全量，出问题能快速回滚。",
    ),
    "性能优化": (
        "最近查询变慢了，怎么排查？",
        "先看慢查询日志，给高频字段加索引，必要时做读写分离和缓存，再压测验证。",
    ),
    "监控告警": (
        "线上告警太多了，怎么收敛？",
        "按告警级别分级收敛，配置聚合和静默窗口，只保留 P0/P1 关键指标，避免告警疲劳。",
    ),
    "安全加固": (
        "安全扫描报了一些漏洞，怎么处理？",
        "优先修复高危漏洞，收紧服务账号权限，加 Web 防火墙规则，并定期轮换密钥。",
    ),
    "成本优化": (
        "这个月账单涨了不少，怎么降本？",
        "清理闲置实例，冷数据转低频存储，购买预留实例，再给大表做生命周期管理。",
    ),
    "技术选型": (
        "消息队列选哪个好？",
        "吞吐优先选 Kafka，简单场景用 Redis Stream 或 RocketMQ，注意消费幂等和积压告警。",
    ),
    "测试上线": (
        "上线前要做哪些测试？",
        "跑一遍回归用例，做冒烟测试和小流量灰度，准备一键回滚，出问题及时止损。",
    ),
}

NEEDLE_TOPIC = "数据备份"

# 非 needle topic（用于填充其它 turn）
FILLER_TOPICS = [t for t in FILLER_TURNS if t != NEEDLE_TOPIC]


@dataclass
class Needle:
    """埋在 session 中的关键信息。"""

    value: str
    turn_index: int          # 0-based，needle 所在 turn 的下标
    topic: str = NEEDLE_TOPIC
    query: str = "生产数据库的备份密钥是多少？"


@dataclass
class TurnRecord:
    """一个 turn 的可观察信息（用于 token 记账与结果分析）。"""

    index: int               # 0-based
    topic: str
    text: str                # user + assistant 组合后的纯文本
    tokens: int = 0
    has_needle: bool = False


@dataclass
class GeneratedSession:
    """生成的 session。"""

    n_turns: int
    needle: Needle
    messages: list[dict] = field(default_factory=list)   # OpenAI 风格消息
    turns: list[TurnRecord] = field(default_factory=list)
    query: str = ""
    total_tokens: int = 0


def _needle_value() -> str:
    """生成唯一随机密钥（needle）。"""
    return f"{secrets.token_hex(2).upper()}-{secrets.token_hex(2).upper()}-{secrets.token_hex(2).upper()}"


def generate_session(
    n_turns: int = 20,
    needle_depth: float = 0.2,
    seed: int | None = None,
) -> GeneratedSession:
    """生成一段 n_turns 的长 session，在 needle_depth 处埋 needle。

    needle_depth 为 [0, 1] 的比例，0.0 表示最开头，1.0 表示最末尾。
    """
    if seed is not None:
        secrets.randbits(0)  # no-op，保证签名一致
        import random

        random.seed(seed)

    n_turns = max(3, n_turns)
    needle_depth = min(1.0, max(0.0, needle_depth))
    needle_index = int(round(needle_depth * (n_turns - 1)))
    needle_value = _needle_value()

    needle = Needle(value=needle_value, turn_index=needle_index)

    messages: list[dict] = []
    turns: list[TurnRecord] = []

    for i in range(n_turns):
        if i == needle_index:
            topic = NEEDLE_TOPIC
            user_text = "生产库的备份策略定好了吗？"
            assistant_text = (
                f"已定好：备份密钥是 `{needle_value}`，"
                f"每天凌晨 2 点做全量备份，保留 30 天，异地再存一份。"
            )
            # 模拟该 turn 用到了工具（query_db），用于演示 used_tools 定位
            turn_messages = [
                {"role": "user", "content": user_text},
                {
                    "role": "assistant",
                    "content": "我先查一下当前的备份配置。",
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "name": "query_db",
                            "arguments": '{"table": "backup_config"}',
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": (
                        f'{{"backup_key": "{needle_value}", "full_time": "02:00", '
                        f'"retention_days": 30}}'
                    ),
                    "tool_call_id": f"call_{i}",
                },
                {"role": "assistant", "content": assistant_text},
            ]
        else:
            # 循环使用 filler topic，保证相邻 topic 尽量不同
            topic = FILLER_TOPICS[i % len(FILLER_TOPICS)]
            user_text, assistant_text = FILLER_TURNS[topic]
            turn_messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]

        messages.extend(turn_messages)
        text = f"{user_text}\n{assistant_text}"
        tokens = estimate_tokens(text)
        turns.append(
            TurnRecord(
                index=i,
                topic=topic,
                text=text,
                tokens=tokens,
                has_needle=(i == needle_index),
            )
        )

    query = needle.query
    total_tokens = sum(t.tokens for t in turns)

    return GeneratedSession(
        n_turns=n_turns,
        needle=needle,
        messages=messages,
        turns=turns,
        query=query,
        total_tokens=total_tokens,
    )
