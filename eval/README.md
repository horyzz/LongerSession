# Evaluation Cases（第一版）

用于基础验证对话切割（语义分割）算法的 evaluation case 集合。

## 目录结构

```
eval/
├── README.md          # 本文件
├── cases/             # case JSON 文件
│   ├── zh-short-01.json
│   ├── zh-short-02.json
│   ├── zh-long-01.json
│   ├── ...
│   └── en-long-02.json
└── loader.py          # case 加载与校验脚本
```

## Case Schema

每个 case 是一个 JSON 文件：

```json
{
  "case_id": "zh-short-01",
  "language": "zh",
  "length": "short",
  "description": "中文短对话：部署方案 + 数据备份 两个话题",
  "messages": [
    {"role": "user", "content": "帮我看看部署方案"},
    {"role": "assistant", "content": "可以用 Docker 或 K8s", "tool_calls": [{"id": "1", "name": "read_file", "arguments": "{}"}]},
    {"role": "tool", "content": "文件内容...", "tool_call_id": "1"},
    {"role": "user", "content": "选 Docker 吧"}
  ],
  "boundaries": [3]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `case_id` | string | 唯一标识，命名 `{language}-{length}-{n}` |
| `language` | string | `zh`（中文）或 `en`（英文） |
| `length` | string | `short`（短，2-3 话题）或 `long`（长，4-5 话题） |
| `description` | string | 话题组成说明 |
| `messages` | array | 完整对话消息，含 role/content/可选 tool_calls |
| `boundaries` | array[int] | 真值切割点（turn 序号，1-based） |

### messages 结构

每条消息是 OpenAI 风格：

```json
{"role": "user" | "assistant" | "tool" | "system", "content": "...", ...}
```

- `assistant` 消息可含 `tool_calls`（工具调用，`{"id","name","arguments"}`）
- `tool` 消息通过 `tool_call_id` 关联到对应调用，`content` 为工具输出

## 真值定义（boundaries）

- **turn**：一条 `user` 消息 + 其后所有 `assistant`/`tool` 消息，直到下一条 `user` 消息。turn 从 **1** 开始编号。
- **boundary（切割点）**：话题切换的位置，即某个 turn 结束后、下一个 turn 开始前。
- **`boundaries` 数组**：存放切割点之前的最后一个 turn 序号。

例如 `"boundaries": [3]` 表示第 3 个 turn 结束后切一刀，将对话分为 `[turn1..turn3]` 与 `[turn4..]` 两个 chunk。

## 维度覆盖

| 维度 | 取值 | 说明 |
|---|---|---|
| 语言 | `zh` / `en` | 中文 / 英文 |
| 长度 | `short` / `long` | 短：2-3 话题、<10 turn；长：4-5 话题、≥15 turn |

第一版覆盖 4 个组合：`zh-short`、`zh-long`、`en-short`、`en-long`，每个组合至少 2 个 case。

## 使用

```python
from eval.loader import load_cases, validate_case

cases = load_cases("eval/cases")
for c in cases:
    validate_case(c)
```

评估时，将 `messages` 送入切割算法，得到的切割点与 `boundaries` 对比，计算 Boundary F1 / Pk 等指标。

## 真值来源（第一版）

第一版 case 为**人工构造**：对话内容自然、话题边界由构造者明确控制，`boundaries` 即构造时确定的话题切换点。后续迭代计划引入：

1. **拼接法**：把多段已知话题的独立对话拼接，拼接点即真值边界（零成本大规模生成）
2. **真实 agent 日志**：人工标注真实 session 的话题切换点
