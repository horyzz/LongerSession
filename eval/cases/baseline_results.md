# 自建 case 集基线结果

第二个 baseline：用自建 case 集（中英 × 长短，含 tool 调用）验证切割算法。

## 数据

- 8 个人工构造 case：zh-short / zh-long / en-short / en-long 各 2 个
- 短对话 5-6 turn，长对话 15 turn
- 含 list_files / search_code / query_db / write_file / run_command 等 tool 调用

## 结果（char_ngram encoder）

| 维度 | F1 (tol=0) | F1 (tol=1) | P | R |
|---|---|---|---|---|
| zh-short | **1.000** | 1.000 | 1.000 | 1.000 |
| en-short | **0.800** | 0.800 | 1.000 | 0.667 |
| zh-long | **0.526** | 0.737 | 0.417 | 0.714 |
| en-long | **0.105** | 0.737 | 0.083 | 0.143 |
| **overall** | **0.449** | **0.776** | 0.379 | 0.550 |

## 对照 TIAGE 基线

| 数据集 | 性质 | F1 (tol=0) |
|---|---|---|
| TIAGE（word_bag） | 英文真实闲聊，边界模糊 | 0.329 |
| **自建 cases（char_ngram）** | **中英×长短，边界清晰（人工构造）** | **0.449** |

## 分析

1. **短对话接近满分**：zh-short 1.0、en-short 0.8。短对话话题少、边界清晰，切割容易。

2. **长对话显著变差**：zh-long 0.526、en-long 仅 0.105。长对话话题多、边界多，且 en-long 的英文词级信息在 char n-gram 下区分度不足（precison 0.083，几乎全错）。

3. **自建 case 边界比 TIAGE 更"干净"**：人工构造的边界明确，所以 overall 0.449 高于 TIAGE 真实数据 0.329。这印证了「人工 case 是受控验证，真实数据才是硬指标」。

4. **en-long 是最大短板**：char n-gram 对英文长对话失效，需 BGE-M3 真实语义（阶段 2）。

## 复现

```bash
# 统一 eval 入口（指定 set）
uv run python -m eval.run cases char_ngram
```

## 两个 baseline 汇总

| baseline | set | encoder | F1 (tol=0) |
|---|---|---|---|
| TIAGE 初始评估 | `tiage` | word_bag | 0.329 |
| 自建 case 集 | `cases` | char_ngram | 0.449 |

两者互补：TIAGE 是英文真实数据（硬指标），自建 case 覆盖中英×长短 + tool（受控验证）。
