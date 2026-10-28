# LongerSession

处理**超长 session** 的索引架构：通过语义切割 + topic 索引，让 agent 在不压缩、不损失信息的前提下快速定位历史内容，实现「有效扩展上下文」。

## 核心思想

传统做法（summary、向量检索、滑窗截断）都会丢失信息。本方案从「压缩」转向「索引」——**保留原文，按需定位**。

```
session 之间组成单链表（previous 指针）
        │
父 session 做语义切割 → 每个区间（chunk）打 topic 标签
        │
子 session 通过 previous 递归拿到所有祖先的 topic 枚举
        │
按 topic 快速定位 → 按需拉取对应 chunk 原文
```

关键设计：

| 层 | 内容 | 特性 |
|---|---|---|
| Session 链表 | `previous` 指针串联父子 session | 允许递归回溯，immutable |
| 语义切割 | 按 user 锚点 + 承接分数切 chunk | embedding 只用于切割，允许 overlap |
| Topic 标签 | LLM 自由生成，细粒度不归一化 | 开放集合，精准描述 |
| Tag 标签 | 增量聚类后学习，归一化 | tag 名 = 锚点 topic，离线演化 |
| Chunk 原文 | 完整对话 + `used_tools` 元数据 | 单点存储，按需拉取 |

**本质**：给 session 做 index，冷热分离——topic 粗索引极小、可全量传递；chunk 原文大、只存一次、按需拉取。

## 架构

```
query embedding
   │  余弦相似度
   ▼
topic 粗索引（枚举 + embedding，小，常驻）
   │  + used_tools 过滤
   ▼
chunk 原文（大，按需拉取）
```

切割流程：`原始消息 → 预处理（剥离 tool/过滤空消息）→ turn 分组 → 承接分数（encoder）→ DP 切割 + overlap → LLM 打 topic`

## 目录结构

```
LongerSession/
├── docs/prd.md              # 产品需求文档
├── pyproject.toml           # 项目定义与依赖
├── src/longersession/       # 核心包
│   ├── config.py            # 配置（encoder/切割/topic/存储）
│   ├── models/              # 数据模型
│   │   ├── message.py       #   Role / ToolCall / Message
│   │   ├── turn.py          #   Turn（user + assistant 组合，切割单元）
│   │   ├── chunk.py         #   Chunk（切割结果，含 topic/tag/used_tools）
│   │   └── session.py       #   LabelRef / SessionIndex（previous 指针）
│   ├── segmentation/        # 切割
│   │   └── preprocess.py    #   消息清洗 + turn 分组
│   ├── encoder/             # 可插拔 encoder（承接分数，baseline BGE-M3）
│   ├── topic/               # topic 标签器（LLM 自由生成）
│   ├── index/               # 索引构建 + AggregatedView 聚合 + 增量 tag 聚类
│   ├── telemetry/           # telemetry 本地收集（阶段 3，JSONL 追加）
│   ├── assessor/            # LLM 评估器（阶段 4，可插拔 + OpenAI 兼容）
│   └── api/                 # FastAPI（POST /process、GET /query）
├── eval/                    # evaluation case（切割质量，Boundary F1）
├── benchmarks/              # 合成 NIAH 基准（截断/压缩/索引）
├── eval_longeval/           # LongEval 真实基准（DeepSeek API）
└── tests/                   # 单元测试
```

## 快速开始

```bash
# 安装依赖（推荐 uv，已默认走清华 PyPI 镜像，无需代理）
uv sync --extra dev

# 安装含 BGE-M3 encoder 的完整依赖
uv sync --extra dev --extra encoder

# 运行单元测试
uv run pytest tests/

# 校验 evaluation case
python3 eval/loader.py
```

> 国内网络说明：`pyproject.toml` 已通过 `[[tool.uv.index]]` 默认指向清华 PyPI 镜像；
> BGE-M3 模型默认从 **ModelScope（魔搭）** 下载（`EncoderConfig.source="modelscope"`），
> 均走国内直连，不依赖代理。

## 使用

规划中的两个 API：

```bash
# 处理一段 session：切割 → 打 topic → 生成索引 → 挂到 previous 链
POST /process

# 查询：沿 previous 链定位相关历史内容
GET /query
```

### MCP 接入（插件化）

除 HTTP 外，也可把 `process` / `query` 作为 MCP 工具接入 pi、OpenAI Agents SDK、Claude 等：

```bash
# 安装 mcp 可选依赖
uv sync --extra mcp

# pi：stdio 方式（本机进程）
pi mcp add longer -- uv run fastmcp run longersession.mcp.server:mcp

# pi：HTTP 方式（远程/团队共享）
uv run fastmcp run longersession.mcp.server:mcp --transport http --port 8002
pi mcp add longer --url http://localhost:8002/mcp

# 验证
pi mcp list --json
```

### API embedding encoder

`encoder.provider` 支持 `api`：调用任意 OpenAI 兼容 `/embeddings` 接口生成 embedding，无需下载本地模型：

```python
from longersession.config import Config, EncoderConfig

cfg = Config(encoder=EncoderConfig(
    provider="api",
    api_base_url="https://api.openai.com/v1",  # 或其它 OpenAI 兼容服务
    api_model="text-embedding-3-small",        # 服务支持的 embedding 模型名
    api_key="",                                 # 留空则从 api_key_env 读取
    api_key_env="openai_api_key",
))
```

## 评估

`eval/cases/` 下是 evaluation case 第一版，覆盖 **中英 × 长短** 四个维度（各 2 个，共 8 个），每个 case 含完整对话与真值切割点（`boundaries`）。详见 [eval/README.md](eval/README.md)。

```bash
python3 eval/loader.py   # 校验 schema、boundaries 合法性、维度覆盖
```

## 有效性验证

核心主张「索引 > 压缩 > 截断」用两套基准、真实模型验证：

### 合成 NIAH 基准（`benchmarks/`）

生成多 topic 长 session、埋入唯一 needle，对比三策略在受限 context 下能否找回：

```bash
uv run python -m benchmarks.run single     # 单次
uv run python -m benchmarks.run sweep      # depth × budget 扫描
uv run uvicorn benchmarks.service:app --port 8000
```

### LongEval 真实基准（`eval_longeval/`，DeepSeek API）

用 LongEval（LongChat 版）topics 任务 + 单/多 needle，真实模型 `deepseek-flash` 对比四策略：

```bash
export deepseek_api_key=...
uv run python -m eval_longeval.run 5,10 2 1500      # topics 任务
uv run python -m eval_longeval.needle_qa            # 单 needle 定向检索
uv run python -m eval_longeval.multi_needle         # 多 needle 信息密度
uv run uvicorn eval_longeval.service:app --port 8001
```

| 场景 | 截断 | 压缩 | 索引 |
|---|---|---|---|
| topics（列出全部 topic） | 30% | 75% | **100%** |
| 单 needle（定向检索） | 33% | 100%（但 10× token） | **100%（1/10 token）** |
| 多 needle（10 个具体事实） | 20% | **0%** | **100%** |

**结论**：索引是唯一「既 lossless 又便宜」的方案——单次 query 只读相关 chunk（约 600 token），
而压缩要先全文摘要（约 6000 token），且随信息密度上升必然丢事实（10 个事实全丢）。
详见 [eval_longeval/README.md](eval_longeval/README.md)。

## 数据飞轮（阶段 3 telemetry → 阶段 4 LLM 评估 → 阶段 5 回训）

一次 process 自动落一条本地 telemetry 记录；用你自己的 LLM 打 topic-shift 真值；再把真值转成训练样本对回训切割模型：

```bash
# 阶段 3：telemetry 默认开启（config.telemetry.enabled），每次 /process 追加到
# data/telemetry/sessions.jsonl，含原始 messages + 预测边界 + topics + chunks 摘要

# 阶段 4：用你的 LLM（OpenAI 兼容接口）给 telemetry 数据打真值边界
#   api_key 从 deepseek_api_key 环境变量读，或 --api-key 传入；默认 DeepSeek，可 --base-url/--model 换
uv run python -m longersession.assessor.runner \
    --telemetry data/telemetry/sessions.jsonl \
    --out data/telemetry/eval --provider openai --resume \
    --sample 5 --sample-out data/telemetry/audit_sample.json

# 阶段 5：真值 → 训练样本对（复用 training schema）
uv run python -m training.telemetry_pairs \
    --eval-dir data/telemetry/eval \
    --out data/training/pairs_telemetry.jsonl

# 一键闭环：telemetry → 评估 → 样本对 → 合并历史数据 → 回训
uv run python -m training.flywheel \
    --telemetry data/telemetry/sessions.jsonl \
    --merge data/training/pairs_merged.jsonl \
    --train --model BAAI/bge-m3 --output-dir data/training/model_flywheel --epochs 2
```

详见 [docs/roadmap.md](docs/roadmap.md) 阶段 3/4/5 与 [docs/milestones.md](docs/milestones.md) M3。

## 当前状态

| 模块 | 状态 |
|---|---|
| 数据模型（message/turn/chunk/topic/session） | ✅ 完成 |
| 预处理（清洗 + turn 分组） | ✅ 完成 |
| 配置模块 | ✅ 完成 |
| evaluation case 第一版 | ✅ 完成 |
| encoder（承接分数，BGE-M3） | ✅ 完成 |
| DP 切割 + overlap | ✅ 完成 |
| topic 标签器 | ✅ 完成 |
| 三层索引 + 聚合 | ✅ 完成 |
| 两个 API | ✅ 完成 |
| 对比学习训练（独立 sub-module + 平衡模型） | ✅ 完成 |
| telemetry 本地收集（阶段 3） | ✅ 完成 |
| LLM 评估器打真值（阶段 4） | ✅ 完成 |
| 数据飞轮回训闭环（阶段 3→4→5） | ✅ 完成 |

## 相关文档

- [产品需求文档](docs/prd.md)
- [架构设计](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Evaluation 说明](eval/README.md)
- [TIAGE 基线结果](eval/tiage/baseline_results.md)
- [合成 NIAH 基准说明](benchmarks/README.md)
- [LongEval 真实基准说明](eval_longeval/README.md)
