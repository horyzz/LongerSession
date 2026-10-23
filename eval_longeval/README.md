# LongEval 真实基准评估（DeepSeek API）

用 **LongEval（LongChat 版，[DachengLi1/LongChat](https://github.com/DachengLi1/LongChat)）的 `topics` 任务**
做真实测评：把多段单 topic 长对话拼成一段「超长 session」，让模型找出所有 topic，
对比四种策略的 **topic 召回** 与 **token 开销**。

## 为什么是 LongEval topics 任务

LongEval 的对话天然是「超长 session」——每段对话以
`USER: I would like to discuss the topic of <TOPIC>` 开头，以
`USER: Great, let's talk about the next topic` 结尾。把 N 段拼接即得到一个
N topic 的长 session，真值就是 N 个 topic 名。这正好对齐 LongerSession 的
「语义切割 + topic 索引」。

## 四种策略（都用 DeepSeek 产出 topic 列表）

| 策略 | 做法 | 预期 |
|---|---|---|
| `full` | 整段对话一次性给模型 | 100% 召回，但需超大单次 context |
| `truncation` | 只给末尾预算内的对话 | 丢头部 topic |
| `compression` | 先有损摘要再提取 | 丢信息，且单次 context 仍很大 |
| `index` | 切 chunk → 逐 chunk 打 topic → 聚合 | lossless，单次 context 极小 |

`index` 两种切分：
- `index-conversation`：oracle 切分（每段单 topic 对话 = 一个 chunk），隔离「索引架构」价值
- `index-segmentation`：真实语义切割管线（char_ngram + DP）+ 尺寸上限，反映当前管线

## 运行

```bash
# 前置：export deepseek_api_key=... （模型默认 deepseek-flash，即用户侧 ds-4.1-flash）
# 可选：export deepseek_base_url=https://api.deepseek.com

# 快速（5 topic × 1 样本）
uv run python -m eval_longeval.run 5 1 1500

# 完整（5/10 topic × 2 样本，截断预算 1500）
uv run python -m eval_longeval.run 5,10 2 1500

# 起服务
uv run uvicorn eval_longeval.service:app --host 0.0.0.0 --port 8001
```

## 结果（deepseek-flash，5/10 topic × 2 样本）

| 策略 | 平均 topic 召回 | 总 prompt tokens | 单次最大 prompt tokens | LLM 调用 | 耗时 |
|---|---|---|---|---|---|
| full | 100.0% | 4148 | **4148** | 1.0 | 1.71s |
| truncation | 30.0% | 1132 | 1132 | 1.0 | 1.22s |
| compression | 75.0% | 4344 | 4140 | 2.0 | 5.0s |
| index-conversation | 100.0% | 4569 | **676** | 7.5 | 6.94s |
| index-segmentation | 82.5% | 4768 | **757** | 8.0 | 19.67s |

**结论**：

1. **索引（oracle 切分）100% 召回，单次最大 context 仅 676 tokens**，是全量（4148）的 **1/6**——这是「有效扩展上下文」的直接证据：用一个很小的窗口，就能无损失处理长 session。
2. **截断只有 30% 召回**（丢失头部 topic），**压缩只有 75% 召回**（有损），且压缩的单次 context（4140）和全量几乎一样大，没省下窗口。
3. **真实语义切割管线（char_ngram）82.5% 召回、757 tokens/次**：略低于 oracle，是因为 char_ngram 是零依赖弱 encoder，会把相邻 topic 并进同一 chunk——这正是项目「对比学习训练专用切割模型」要解决的问题。

## 说明

- **模型名映射**：用户侧别名 `ds-4.1-flash` 在官方 API 中对应 `deepseek-flash`（API 报错提示支持 `deepseek-flash` / `deepseek-v4-pro`）。
- **token 计数**：prompt/completion tokens 取 DeepSeek API `usage` 的权威值。
- **topic 匹配**：归一化 + 去停用词后的词集合重叠系数（见 `metrics.py`），容忍 `Middle Ages history and culture` 这类换序/近义改写。
- **API key**：从环境变量 `deepseek_api_key` 读取，代码不打印、不落地。
