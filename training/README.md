# training — 承接关系对比学习训练数据构建

为阶段 5「对比学习训练切割模型」准备训练数据：把开源多轮对话数据集转换成
统一的 `(text_a, text_b, label)` 样本对，供承接关系 encoder 训练。

## 输出格式（JSONL）

每行一条样本对（`training/schema.py` 有完整字段说明）：

```json
{"text_a": "前一个文本", "text_b": "后一个文本", "label": 1, "source": "oasst1",
 "pair_type": "qa", "lang": "en", "tree_id": "..."}
```

| 字段 | 说明 |
|---|---|
| `label` | 1=承接（正样本），0=不承接（负样本）|
| `pair_type` | `qa` 同轮问答 / `aq` 跨轮承接 / `hard_neg` 同 prompt 非最优回复 / `cross_neg` 跨对话负样本 |

训练时：`label=1` 的 `(text_a, text_b)` 是正样本对，in-batch 其它对自动做负样本；
`label=0` 的是显式硬负样本。

## 数据来源与构建方式

### OASST1（OpenAssistant）— 主数据，Apache-2.0

- 13,500+ 志愿者众包的人写、人评**对话树**（161k 消息 / 9,846 棵树）。
- 转换（`training/oasst1.py`）：
  - **正样本**：树中父子边。`prompter→assistant` 记 `qa`，`assistant→prompter` 记 `aq`。
  - **硬负样本**：同一 prompt 的多个 assistant 候选回复里，`rank>0`（非最优）的做 `hard_neg`。
  - **跨树负样本**：不同树随机配对（`--cross-neg-ratio` 控制比例）。
  - 过滤：未删除、通过评审、树状态 ready_for_export、语言白名单。
- 实测（train 全量，lang=en+zh）：**119,866 对**（正 41,769 / hard_neg 18,164 / cross_neg 59,933）。

### BELLE（multiturn_chat_0.8M）— 中文补充，Apache-2.0

- 链家开源、ChatGPT 生成的中文多轮（`instruction` 内是 Human/Assistant 交替历史）。
- 转换（`training/belle.py`）：解析 `instruction` 多轮历史 + `output` 补全末轮，
  相邻 turn 生成 `qa`/`aq` 正样本。无 rank/质量标签，负样本靠 in-batch。
- ⚠️ 文件约 **990MB**，网络不稳时建议先下载到本地再用 `--belle-local` 指定。

### 任务型（MultiWOZ 2.2 + DialSeg711）— 语义 topic 切换，Apache-2.0

- 任务型对话，带 domain/topic 切换标注，用于对齐真实 agent 场景的语义切割。
- 转换（`training/task_oriented.py`）：相邻 turn
  - 同 topic/domain → `same_topic`（label=1，承接）
  - 跨 topic/domain（边界）→ `cross_topic`（label=0，切换）
- MultiWOZ domain 靠 `active_intent` 提取活跃 service；DialSeg711 用人类 `topic_id`（更细）。
- DialSeg711 默认拆 70% train / 30% held-out（`--dialseg711-train-ratio`），held-out 做无泄漏评测。
- 实测：24,858 对（正 22,283 / 负 2,575；MultiWOZ 8,437 对话 + DialSeg711 500 对话）。
- 用法：
  ```bash
  uv run python -m training.task_oriented --out data/training/pairs_task.jsonl
  ```

## 用法

```bash
# 构建 OASST1（约 53MB 下载，快）
uv run python -m training.build_dataset --source oasst1 \
  --out data/training/pairs_oasst1.jsonl

# 构建 BELLE（990MB，用本地文件更稳）
uv run python -m training.build_dataset --source belle \
  --belle-local /path/to/multiturn_chat_0.8M.json \
  --out data/training/pairs_belle.jsonl

# 两个都构建（默认；BELLE 走下载，可能较慢）
uv run python -m training.build_dataset --out data/training/pairs.jsonl
```

常用参数：`--langs en,zh`、`--cross-neg-ratio 1.0`、`--belle-max-rows N`、`--max-pairs N`。

## 许可注意

- OASST1、BELLE 均为 Apache-2.0，但：
  - BELLE 数据由 ChatGPT 生成，可能涉及 OpenAI 服务条款的灰色地带，商用需单独评估。
  - 两者都**没有显式 topic 边界标注**，边界监督需另靠 MultiWOZ/SGD 的 domain 切换
    或研究许可的 TIAGE/SuperDialseg。

## 训练（阶段 5）

用正样本对做 SimCSE 式 in-batch 对比学习，微调 sentence-transformer 模型：

```bash
# 安装训练依赖
uv sync --extra train

# 训练（bge-small 起步，可换 bge-m3）
uv run python -m training.train \
  --data data/training/pairs_oasst1.jsonl \
  --model BAAI/bge-small-en-v1.5 \
  --epochs 3 --batch-size 32 --lr 2e-5 --fp16
```

训练产物是 sentence-transformers 格式（`--output-dir`，默认 `data/training/model`），
可直接 `SentenceTransformer(path)` 加载接入切割管线。

### Colab 训练

上传 `training/colab.ipynb` + 数据 JSONL 到 Colab，选 T4 GPU，按单元格依次运行：
装依赖 → 上传数据 → 训练 → 存到 Drive。详见 notebook 内注释。

### 平衡模型（最终，阶段 5 完成）

合并 OASST1 + 任务型数据（`pairs_merged.jsonl`，64,052 正样本对），从 bge-m3 base 用
`max_seq_length=2048` + gradient_checkpointing 训练，三域占优：

| 数据集 | 零样本 | 任务型微调 | 平衡模型 |
|---|---|---|---|
| DialSeg711 held-out(211) | 0.561 | 0.795 | **0.841** |
| TIAGE | 0.397 | 0.428 | **0.458** |
| cases(8) | 0.625 | 0.694 | **0.708** |

⚠️ Colab 需先降级 transformers：`!pip install -q "transformers<5"`（5.x 与
sentence-transformers 6.1.0 不兼容，`gradient_checkpointing_enable` 会崩）。
训练配方与决策见 `docs/milestones.md` M2。
