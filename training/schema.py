"""统一训练样本对 schema。

每条输出记录是一行 JSON（JSONL），字段如下：

- text_a: str    前一个文本
- text_b: str    后一个文本
- label: int     1 = 承接（正样本，语义连贯），0 = 不承接（负样本）
- source: str    数据来源：oasst1 | belle
- pair_type: str 样本类型：
                  qa        user/提问 → assistant/回答（同轮内承接）
                  aq        assistant/回答 → 下一条 user/提问（跨轮承接）
                  hard_neg  同一提问的多个候选回复中，非最优回复（rank>0）
                  cross_neg 跨对话随机负样本
- lang: str      语言
- tree_id: str   来源树/对话 id（oasst1 用 message_tree_id，belle 用行号）

训练时：label=1 的 (text_a, text_b) 是正样本对（in-batch 其它对自动做负样本），
label=0 的是显式硬负样本。
"""

PAIR_TYPES = ("qa", "aq", "hard_neg", "cross_neg")
