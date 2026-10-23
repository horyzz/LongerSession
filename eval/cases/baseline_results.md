# 自建 case 集基线结果

第二个 baseline：用自建 case 集（中英 × 长短，含 tool 调用）验证切割算法。阶段 2 接 BGE-M3 后重跑。

## 数据

- 8 个人工构造 case：zh-short / zh-long / en-short / en-long 各 2 个
- 短对话 5-6 turn，长对话 15 turn
- 含 list_files / search_code / query_db / write_file / run_command 等 tool 调用

## 结果（char_ngram，threshold=0.5）

| 维度 | F1 (tol=0) | F1 (tol=1) | P | R |
|---|---|---|---|---|
| zh-short | 1.000 | 1.000 | 1.000 | 1.000 |
| en-short | 0.800 | 0.800 | 1.000 | 0.667 |
| zh-long | 0.526 | 0.737 | 0.417 | 0.714 |
| en-long | 0.105 | 0.737 | 0.083 | 0.143 |
| **overall** | **0.449** | **0.776** | 0.379 | 0.550 |

## 结果（BGE-M3，threshold=0.70，最优）

| 维度 | F1 (tol=0) | F1 (tol=1) | P | R |
|---|---|---|---|---|
| zh-short | 1.000 | 1.000 | 1.000 | 1.000 |
| en-short | **1.000** | 1.000 | 1.000 | 1.000 |
| zh-long | **0.588** | 0.750 | 0.500 | 0.714 |
| en-long | **0.421** | 0.750 | 0.333 | 0.571 |
| **overall** | **0.625** | **0.833** | 0.536 | 0.750 |

> 注：BGE-M3 默认 threshold=0.5 时 overall 仅 0.154（recall 0.10，切割不足）；
> 扫描到 0.70 后召回恢复。BGE-M3 的相邻 turn 余弦分布偏高，需更高阈值。

### BGE-M3 阈值扫描（overall，tol=0）

| threshold | F1 | P | R |
|---|---|---|---|
| 0.50 | 0.154 | 0.333 | 0.100 |
| 0.55 | 0.364 | 0.462 | 0.300 |
| 0.60 | 0.524 | 0.500 | 0.550 |
| 0.65 | 0.522 | 0.462 | 0.600 |
| **0.70** | **0.625** | **0.536** | **0.750** |
| 0.75 | 0.612 | 0.517 | 0.750 |
| 0.80 | 0.560 | 0.467 | 0.700 |

## 对照（两个 baseline）

| 数据集 | encoder | threshold | F1 (tol=0) |
|---|---|---|---|
| TIAGE | word_bag | 0.35 | 0.329 |
| TIAGE | **bge_m3** | **0.65** | **0.397** |
| 自建 cases | char_ngram | 0.50 | 0.449 |
| 自建 cases | **bge_m3** | **0.70** | **0.625** |

## 分析

1. **BGE-M3 全面优于 char_ngram**：overall 0.449 → 0.625（+39%）。四个维度全升或持平。

2. **en-long 短板被修复**：0.105 → 0.421（4 倍）。char n-gram 对英文长对话失效的问题，被 BGE-M3 真实语义 embedding 解决——这是阶段 2 的核心验证点。

3. **短对话仍满分**：zh-short / en-short 均为 1.0，BGE-M3 没有破坏短对话的确定性。

4. **阈值需按 encoder 调**：BGE-M3 最优 0.70（cases）vs 0.65（TIAGE），与 char_ngram 的 0.50、word_bag 的 0.35 差异明显。默认 threshold 需做成 encoder-aware 或暴露为可调参数。

## 训练模型结果（阶段 5）

| 模型 | threshold | tol=0 F1 | tol=1 F1 |
|---|---|---|---|
| bge-m3 零样本 | 0.70 | 0.625 | 0.833 |
| OASST1 微调（MNRL） | — | **0.708** | — |
| 任务型微调（MultiWOZ + DialSeg711）| 0.70 | 0.694 | — |
| **平衡模型（OASST1+任务型，seq=2048）** | 0.75 | **0.708** | — |

- OASST1 微调与平衡模型并列最优（0.708），任务型微调（0.694）高于零样本 0.625、略低于两者。
- 平衡模型各维度：en-short/zh-short 1.0，en-long 0.588，zh-long 0.706。
- 完整对比与训练配方见 `docs/milestones.md`。

## 复现

```bash
# 统一 eval 入口（指定 set）
uv run python -m eval.run cases char_ngram
uv run python -m eval.run cases bge_m3       # 默认阈值（偏低）
uv run python -m eval.sweep cases bge_m3     # 阈值扫描找最优
```
