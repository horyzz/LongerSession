# 超长 Session 有效控制基准（NIAH 风格）

验证 LongerSession 的核心主张：**在受限 context 窗口下，索引比压缩、比截断更能有效找回埋在长 session 深处的信息。**

## 思路

对齐学界长上下文评测（[RULER / NIAH, arXiv:2404.06654](https://arxiv.org/abs/2404.06654)、
[Lost in the Middle, arXiv:2307.03172](https://arxiv.org/abs/2307.03172)、LongEval）：

- 生成一段 N turn 的长 session，覆盖多个 topic；
- 在指定深度埋入一条**唯一随机密钥（needle）**，只出现在那一个 turn；
- 末尾追加一个只有靠 needle 才能回答的问题；
- 用确定性的 oracle 判定：策略产出的 context 里**是否完整包含 needle**。

这隔离了「检索/索引是否有效」这一核心主张，不受生成模型抽取能力干扰（与 NIAH 评分方式一致）。

## 三种策略

| 策略 | 做法 | 失败原因 |
|---|---|---|
| `truncation` | 只保留末尾 budget 内的历史 | needle 在窗口外被丢弃 |
| `compression` | 较早历史做有损摘要 + 末尾保留原文 | 摘要保留「问过什么」却丢掉「具体答案」，needle 丢失 |
| `index` | 全量建索引 → 按 topic 定位 → 只拉相关 chunk 原文 | （不失败）lossless，按需取回 |

## 运行

```bash
# 单次运行（默认 20 turn，needle 在 20% 深度，512 token 预算）
uv run python -m benchmarks.run single

# 扫描矩阵（depth × budget）
uv run python -m benchmarks.run sweep

# 起服务
uv run uvicorn benchmarks.service:app --host 0.0.0.0 --port 8000
```

服务端点：

```bash
GET  /health                    # 健康检查
POST /benchmark/run             # 单次三策略对比
POST /benchmark/sweep           # depth × budget 扫描
GET  /benchmark/report          # 最近一次报告
```

`/benchmark/run` 请求体：

```json
{
  "n_turns": 20,
  "needle_depth": 0.2,
  "budget_tokens": 512,
  "seed": 42,
  "encoder_provider": "char_ngram"
}
```

## 结果示例

| 策略 | 可回答 | context tokens | token 占比 | 说明 |
|---|---|---|---|---|
| truncation | ❌ | 493 | 52.6% | 仅保留末尾 10/20 turn |
| compression | ❌ | 690 | 73.6% | 摘要 10 turn + 保留末尾 10 turn |
| index | ✅ | 99 | 10.6% | 检索 1 chunk，命中 topic「数据备份」 |

**结论**：索引用远小于压缩/截断的 token 开销，找回了被它们丢弃的 needle。

## 设计说明

- **oracle 切割**：索引策略默认「每个 turn 一个 chunk」（`min=max=1`），把「检索」这一核心主张从「切割质量」中隔离出来；切割质量由 `eval/run.py` 的 Boundary F1 单独衡量。
- **确定性 token 估算**：CJK ≈ 1 token/字符，非 CJK ≈ 1.3 token/单词，见 `tokenizer.py`。本基准关注相对开销，不依赖具体 tokenizer。
- **LLM-free**：不调用真实 LLM，oracle 只判断 needle 是否在 context 中，保证可复现、零成本。
