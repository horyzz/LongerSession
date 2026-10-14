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
│   └── api/                 # FastAPI（POST /process、GET /query）
├── eval/                    # evaluation case
│   ├── README.md            #   case schema 与真值定义
│   ├── loader.py            #   加载与校验脚本
│   └── cases/               #   8 个 case（中英 × 长短）
└── tests/                   # 单元测试
```

## 快速开始

```bash
# 安装依赖（推荐 uv）
uv sync --extra dev

# 运行单元测试
uv run pytest tests/

# 校验 evaluation case
python3 eval/loader.py
```

## 使用

规划中的两个 API：

```bash
# 处理一段 session：切割 → 打 topic → 生成索引 → 挂到 previous 链
POST /process

# 查询：沿 previous 链定位相关历史内容
GET /query
```

## 评估

`eval/cases/` 下是 evaluation case 第一版，覆盖 **中英 × 长短** 四个维度（各 2 个，共 8 个），每个 case 含完整对话与真值切割点（`boundaries`）。详见 [eval/README.md](eval/README.md)。

```bash
python3 eval/loader.py   # 校验 schema、boundaries 合法性、维度覆盖
```

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
| 对比学习训练（独立 sub-module） | ⏳ 规划中 |

## 相关文档

- [产品需求文档](docs/prd.md)
- [架构设计](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Evaluation 说明](eval/README.md)
- [TIAGE 基线结果](eval/tiage/baseline_results.md)
