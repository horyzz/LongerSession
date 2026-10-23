# LongerSession Roadmap

## 项目目标

处理超长 session：通过语义切割 + topic/tag 索引，让 agent 在不压缩、不损失信息的前提下快速定位历史内容，实现「有效扩展上下文」。

核心思路：**从"压缩"转向"索引"** —— 保留原文，按需定位。

## 当前状态（已完成）

### ✅ 阶段 0：核心架构

- 单链表 session（previous 指针，允许递归回溯）
- 切割管线：`消息 → turns → 承接分数(encoder) → DP 切割 + overlap`
- topic/tag 体系：topic（LLM 自由生成，细粒度不归一化）+ tag（增量聚类后学习）
- 三层索引：session 链表 / 标签粗索引 / chunk 原文（冷热分离）
- 两个 API：`POST /process`、`GET /query`（结构化，无 embedding 检索）
- SQLite 持久化（sessions/turns）
- encoder 可插拔：Mock / WordBag / BGE-M3（lazy load）
- 全量测试 21 passed

### ✅ 阶段 1：TIAGE 初始评估

- 下载 TIAGE 数据（100 对话 / 315 topic-shift 标注）
- 转换脚本：turn 级 0/1 标签 → 统一 eval 格式（boundaries）
- 评估指标：boundary_f1（带容差）+ pk + evaluate_all
- 首个基线：词袋 encoder + DP → **F1=0.329（严格）/ 0.634（容差±1）**，超过 GenEnc 0.250
- 发现并修复 DP 切点代价 bug（score → score - threshold）

### ✅ 阶段 1.5：evaluation case 第一版

- 8 个人工构造 case，覆盖中英 × 长短四维度
- 统一 case schema + 校验脚本（`eval/loader.py`）

### ✅ 阶段 2：接 BGE-M3 真 encoder

- BGE-M3 经 ModelScope 下载（国内直连，`source="modelscope"`），lazy load 正常
- TIAGE：**F1=0.397**（th=0.65，词袋 0.329 → +20.7%），仍低于 RetroTS-T5 0.682
- 自建 cases：**F1=0.625**（th=0.70，char_ngram 0.449 → +39%），en-long 0.105 → 0.421
- 关键发现：BGE-M3 相邻 turn 余弦分布偏高，默认 threshold=0.5 偏低，需按 encoder 调阈值
- 新增阈值扫描工具 `eval/sweep.py`

## Roadmap（未来）

### ⬜ 阶段 3：telemetry 本地收集

| 项 | 内容 |
|---|---|
| 目标 | 收集真实 agent 对话，为评估和训练积累数据 |
| 产出 | Service.process 埋 hook，本地 JSONL 记录原始会话 + 预测边界 + topic |
| 动作 | 加 telemetry hook；JSONL 追加存储；隐私默认本地 |
| 验收 | 每次 process 自动落一条本地记录，可审计、可回溯 |

### ⬜ 阶段 4：LLM 评估器（用户配置）

| 项 | 内容 |
|---|---|
| 目标 | 用用户自己的 LLM 给收集的对话打 topic-shift 真值标签 |
| 产出 | 可插拔 Assessor 抽象 + 默认评估 prompt；telemetry 数据 → 带真值的 eval 数据 |
| 动作 | Assessor 接口；prompt 模板；批量/增量评估；抽检机制 |
| 验收 | 用户配置 LLM 后能产出带真值的 eval 数据，抽检可校验 LLM 标签质量 |

### ✅ 阶段 5：对比学习训练切割模型

| 项 | 内容 |
|---|---|
| 目标 | 训练专用"承接关系"模型，替换/增强 encoder |
| 产出 | `training/` sub-module：OASST1/BELLE/任务型数据 → 对比学习 → 承接分数模型 |
| 动作 | 正样本 (q,a)/(a,q) + 任务型 same_topic，in-batch negatives；bge-m3 MNRL |
| 验收 | 训练模型在 TIAGE 上超过 BGE-M3 零样本，任务型分割大幅提升 |

**结果（详见 `docs/milestones.md`）**：

| 数据集 | 零样本 | OASST1 微调 | 任务型微调 | **平衡模型** |
|---|---|---|---|---|
| DialSeg711 held-out(211) | 0.561 | — | 0.795 | **0.841** |
| TIAGE | 0.397 | **0.464** | 0.428 | 0.458 |
| cases(8) | 0.625 | **0.708** | 0.694 | **0.708** |

平衡模型（OASST1 + 任务型合并，seq=2048）三域占优或并列最优，为最终接入模型。

### ⬜ 阶段 6：数据飞轮

| 项 | 内容 |
|---|---|
| 目标 | 累计数据 → 评估 → 训练 → 再收集，形成正向循环 |
| 产出 | 端到端闭环：telemetry 收集 → LLM 评估 → 训练 → 部署 → 再收集 |
| 动作 | 打通阶段 3/4/5 的链路；tag 聚类完善（增量演化）；评估面板 |
| 验收 | 新数据能持续提升模型，且全程本地化、隐私可控 |

## 依赖关系

```
阶段0(架构) ─┬─▶ 阶段1(TIAGE基线) ──▶ 阶段2(BGE-M3)
             │
             ├─▶ 阶段3(telemetry) ──▶ 阶段4(LLM评估) ─┐
             │                                        ├─▶ 阶段5(训练) ──▶ 阶段6(飞轮)
             └────────────────────────────────────────┘
```

- 阶段 2 与 阶段 3/4 可并行（一个验证质量上限，一个积累数据）
- 阶段 5 依赖阶段 3/4 的数据，也依赖阶段 2 的 baseline 作对照

## 关键设计原则

1. **embedding 只用两处**：切割（承接分数，实时）+ tag 聚类（离线后学习）；query 定位用结构化 GET
2. **topic 自由、tag 聚类**：topic 是 LLM 自由生成的细粒度标签（不归一化）；归一化需求由 tag 的增量聚类满足（tag 名 = 锚点 topic，避免重新上 tag）
3. **冷热分离**：标签粗索引小、可全量常驻；chunk 原文大、单点存储按需拉
4. **session immutable**：索引建成后不变，聚合缓存永久有效
5. **本地优先**：telemetry 收集、LLM 评估、数据存储全部本地，用户自配 LLM，不依赖云端
6. **只搭 pipeline，不碰数据/模型**：框架我们提供，数据和 LLM 由用户配置

## 风险与开放问题

| 风险 | 说明 |
|---|---|
| LLM 标签质量 | TIAGE 人工标注一致性仅 0.479；LLM 打标签可能更飘，需抽检机制 |
| 数据域差异 | TIAGE 是英文闲聊，真实 agent 含 tool 调用，需 telemetry 数据补足 |
| 训练数据冷启动 | 阶段 5 需要足够数据，依赖阶段 3/4 先积累 |
| tag 聚类演化 | 增量聚类中簇分裂/合并的边界情况需持续验证 |
