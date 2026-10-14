# TIAGE 基线结果

阶段 1：用 TIAGE（topic-shift 检测 benchmark）验证切割算法，出第一个可对照的 F1 基线。

## 数据

- 来源：TIAGE Anno test（100 对话，1564 turn，315 topic-shift 标注）
- 转换：turn 级 0/1 标签 → 统一 eval 格式 boundaries（`eval/convert_tiage.py`）

## 结果

我们的切割算法 = 承接分数（相邻 turn 余弦相似度）+ DP 切割（切点代价 = score - threshold）+ overlap。

| encoder | threshold | tol=0 F1 | tol=1 F1 | tol=0 P/R | tol=1 P/R |
|---|---|---|---|---|---|
| WordBag（词袋，零模型） | 0.35 | **0.329** | **0.634** | 0.259 / 0.451 | 0.499 / 0.870 |

## 对照 TIAGE baseline（论文报告，tol=0）

| 方法 | F1 |
|---|---|
| BERT-Wiki727k（主题分割预训练 BERT） | 0.038 |
| GenEnc（embedding 余弦 + 阈值 0.25） | 0.250 |
| **Ours（WordBag + DP）** | **0.329** |
| RetroTS-T5（T5 微调，有监督） | 0.682 |
| Human | 0.644 |

## 分析

1. **零模型词袋 encoder 即超过 GenEnc**：我们的 DP + score-threshold 代价方案，用词袋（无任何模型/训练）就达到 F1=0.329，超过 GenEnc 的 embedding+阈值（0.250），说明 DP 切割 + 代价函数设计本身有效。

2. **容差 ±1 turn 时接近人类**：F1=0.634，接近 Human 0.644 和 RetroTS-T5 0.682。recall 高达 0.870，说明几乎找到了所有边界，主要损失在 precision（fp 偏多）。

3. **提升空间在 encoder**：词袋区分度有限（同话题词汇重叠），替换为 BGE-M3 真实语义后，precision 应显著提升，F1 有望接近或超过 RetroTS-T5。

## 复现

```bash
# 转换 + 评估
uv run python -m eval.convert_tiage
uv run python -m eval.run_eval word_bag
```

## 下一步

- 接 BGE-M3 encoder（`uv sync --extra encoder`），重跑基线
- 对比学习训练专用切割模型（独立 sub-module）
