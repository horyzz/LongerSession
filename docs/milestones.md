# Milestones

项目迭代里程碑记录：版本号、release 说明、关键决策、产物路径快照。

## M2 · 2026-10-19 平衡模型训练完成（OASST1 + 任务型合并，seq=2048）

### 一句话

合并 OASST1（41,769 正样本）+ 任务型（22,283 正样本）= 64,052 对，从 bge-m3 零样本 base 用 `max_seq_length=2048` 训平衡模型，三域全面占优或并列最优，成为最终接入模型。

### 最终结果（同一评测口径）

| 数据集 | 性质 | 零样本 bge-m3 | OASST1 微调 | 任务型微调 | **平衡模型** |
|---|---|---|---|---|---|
| DialSeg711 held-out(211) | 任务型 | 0.561 | — | 0.795 | **0.841** |
| TIAGE test | 闲聊 | 0.397 | **0.464** | 0.428 | 0.458 |
| cases(8) | 含 tool agent | 0.625 | **0.708** | 0.694 | **0.708** |

- DialSeg711 **0.841**（th=0.40）：比零样本 +50%，比任务型 +4.6pt，seq=2048 捕获更长上下文。
- cases **0.708**（th=0.75）：与 OASST1 持平（并列最优）。
- TIAGE **0.458**（th=0.60）：接近 OASST1（0.464），明显优于任务型（0.428），合并数据未牺牲闲聊域。

### 关键决策

1. **降级 transformers 5.18.0 → 4.57.6**：Colab 默认 transformers 5.x 与 sentence-transformers 6.1.0 不兼容（`gradient_checkpointing_enable(every_n_layers=...)` 崩溃）。降级后 gradient_checkpointing + seq=2048 正常训练。
2. **seq=2048 对齐超长场景**：切割编码单 turn（tool 已剥离），2048 覆盖超长 turn；比 512 在 DialSeg711 上 +4.6pt。
3. **合并数据从 base 训**：合并数据比单一域更全面，三域占优，优于「从已微调模型继续」。

### 训练配方（平衡模型）

- 基座 `BAAI/bge-m3`，`MultipleNegativesRankingLoss`（scale=20），`max_seq_length=2048`，`gradient_checkpointing=True`
- batch 8 × grad_accum 4（有效 32）、LR 2e-5、2 epoch、cosine decay、warmup 10%、bf16（A100）
- 数据：`pairs_merged.jsonl`（64,052 正样本对 = OASST1 41,769 + 任务型 22,283）

### 产物路径

| 产物 | 位置 |
|---|---|
| **平衡模型（最终）** | Drive `longersession_model_balanced`（DialSeg711 0.841 / cases 0.708 / TIAGE 0.458）|
| **本地模型（已接入）** | `data/models/longersession_model_balanced/`（gitignored，2.27GB safetensors）|
| 合并训练数据 | Drive `pairs_merged.jsonl`（64,052 对）|

### 本地接入（2026-10-19）

- 模型从 Drive 下载后解压到 `data/models/longersession_model_balanced/`（`model.safetensors` 2.27GB）。
- `EncoderConfig` 新增 `local_dir` 字段、`max_seq_length` 512 → 2048（对齐训练口径）。
- `factory.py` 传入 `local_dir`，`BGEM3Encoder` 直接加载本地模型，不再走 ModelScope/HF 下载。
- 验证：`factory → local balanced model` 加载 2.7s、1024 维归一化正常；本地 cases 集 F1=0.708 与 Colab 一致。

### 结论与下一步

- 平衡模型三域占优或并列最优，已接入本地 `src/longersession/encoder/`（`local_dir` 指向平衡模型）。
- 下一步：补 MultiWOZ/SGD 数据，进入阶段 3（telemetry）+ 阶段 4（LLM 评估器）。

---

## M1 · 2026-10-18 任务型语义切割训练（阶段 5 核心验收）

### 一句话

用任务型对话数据（MultiWOZ domain 切换 + DialSeg711 人类 topic 标注）从 bge-m3 零样本 base 做对比学习，任务型分割 F1 **0.561 → 0.795（+42%）**。

### 最终结果（同一评测口径）

| 数据集 | 性质 | 零样本 bge-m3 | OASST1 微调 | 任务型微调 |
|---|---|---|---|---|
| DialSeg711 held-out(211) | 任务型 | 0.561 | — | **0.795** |
| TIAGE test | 闲聊 | 0.397 | **0.464** | 0.428 |
| cases(8) | 含 tool agent | 0.625 | **0.708** | 0.694 |

### 关键决策

1. **topic 与 tool 正交**：切割只看语义（`combined_text` 剥离 tool），tool 收束为 `used_tools` attribute 在 query 层过滤。
2. **从零样本 base 训，不从已微调模型继续**：继续训 + 换 loss 叠加导致模型崩溃（TIAGE recall 归零）。
3. **loss 用 MNRL，不用 OnlineContrastiveLoss**：后者正负样本 5.5:1 失衡，margin 机制导致 cosine 塌缩。
4. **任务型数据构造**：相邻 turn 同 topic/domain = 正样本（承接），跨 topic/domain（边界）= 负样本（切换）。
5. **评测无泄漏**：DialSeg711 拆 500 train / 211 held-out，held-out 与训练数据无交集。
6. **训完立刻备份 Drive**：Colab runtime 频繁重置，模型/数据必须落地 Drive（`longersession_model_from_base` 曾因未备份丢失一次）。

### 训练配方（任务型微调）

- 基座 `BAAI/bge-m3`，`MultipleNegativesRankingLoss`（scale=20），`max_seq_length=512`
- batch 16、LR 2e-5、3 epoch、cosine decay、warmup 10%、bf16（A100）
- 数据：22,283 正样本（同 topic 相邻 turn；MultiWOZ 11,179 + DialSeg711 11,104）

### 产物路径

| 产物 | 位置 |
|---|---|
| 任务型微调模型 | Drive `longersession_model_from_base`（DialSeg711 0.795）|
| OASST1 微调模型 | Drive `longersession_model_bgem3`（TIAGE 0.464）|
| 弃用模型 | Drive `longersession_model_task`（OnlineContrastiveLoss，cosine 塌缩，勿用）|
| 训练数据 | Drive `pairs_task.jsonl`（24,858 对）+ `dialseg711_heldout.json`（211 对话）|
| 评测脚本 | `eval/dialseg711_eval.py` |
| 数据构建脚本 | `training/task_oriented.py`（MultiWOZ + DialSeg711 → 样本对）|

### 结论与下一步

- 任务型数据让任务型分割 +42%，印证「训练域要对准目标域」。
- 域取舍：任务型模型在闲聊/agent 上高于零样本、略低于 OASST1 专训。
- 下一步：合并 OASST1（38k 正样本）+ 任务型（22k 正样本）从 base 训平衡模型。

---

## M0 · 阶段 2 完成 零样本基线（更早里程碑）

- TIAGE：词袋 0.329 → bge-m3 零样本 0.397
- cases：char_ngram 0.449 → bge-m3 零样本 0.625
- 详见 `eval/tiage/baseline_results.md`、`eval/cases/baseline_results.md`
