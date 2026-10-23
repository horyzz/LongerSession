"""生成 Colab 完整训练 notebook（colab_full.ipynb）。"""
import json


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


cells = [
    md(
        "# LongerSession 承接关系 encoder 训练（完整版）\n\n"
        "一次跑通：挂载 Drive → 下载数据 → 对比学习训练 → TIAGE 评测。\n\n"
        "**使用步骤**：\n"
        "1. 上传本 notebook 到 Colab，运行时选 **T4 GPU**\n"
        "2. 菜单 → 运行时 → **全部运行**\n"
        "3. 中途会弹 Google Drive 授权，点「允许」\n"
        "4. 训练约 1 小时，模型自动存到 Drive 的 `longersession_model3`\n\n"
        "> 训练期间保持 Colab 标签页打开，避免运行时断连。"
    ),
    code("!pip install -q sentence-transformers datasets accelerate\n"),
    code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "import torch\n"
        "print('GPU available:', torch.cuda.is_available())\n"
        "!nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo 'NO GPU - 请切到 T4 运行时'"
    ),
    code(
        "# ③ 下载 OASST1 并构建正样本对\n"
        "from datasets import load_dataset, concatenate_datasets\n"
        "ds = load_dataset('OpenAssistant/oasst1')\n"
        "all_ds = concatenate_datasets([ds['train'], ds['validation']])\n"
        "df = all_ds.to_pandas()\n\n"
        "m = df[(df['deleted']==False) & (df['review_result']==True) & (df['tree_state']=='ready_for_export') & (df['role'].isin(['prompter','assistant'])) & (df['lang'].isin(['en','zh']))].copy()\n"
        "m['text'] = m['text'].astype(str).str.strip()\n"
        "m = m[m['text'].str.len()>0]\n\n"
        "by_text = dict(zip(m['message_id'], m['text']))\n"
        "by_role = dict(zip(m['message_id'], m['role']))\n\n"
        "pairs = []\n"
        "for mid, pid, role, text in zip(m['message_id'], m['parent_id'], m['role'], m['text']):\n"
        "    if pid in by_text:\n"
        "        pr = by_role[pid]\n"
        "        if pr == 'prompter' and role == 'assistant':\n"
        "            pairs.append({'anchor': by_text[pid], 'positive': text})\n"
        "        elif pr == 'assistant' and role == 'prompter':\n"
        "            pairs.append({'anchor': by_text[pid], 'positive': text})\n"
        "print('positive pairs:', len(pairs))"
    ),
    code(
        "# ④ 对比学习训练（cosine 衰减 + 5 epoch），存到 Drive\n"
        "import os, shutil\n"
        "from datasets import Dataset\n"
        "from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments\n"
        "from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss\n\n"
        "# ===== 可调配置 =====\n"
        "MODEL = \"BAAI/bge-m3\"   # 568M 多语言基座（Colab Pro L4/A100）\n"
        "EPOCHS = 3\n"
        "MAX_SEQ_LENGTH = 2048   # 关键：session 的 turn 可能很长，512 不够；可调 4096/8192\n"
        "BATCH_SIZE = 8\n"
        "GRAD_ACCUM = 4   # 梯度累积，等效 batch = 8*4 = 32\n"
        "LR = 2e-5\n"
        "OUTPUT_DIR = \"/content/model_bgem3\"\n"
        "DRIVE_DIR = \"/content/drive/MyDrive/longersession_model_bgem3\"\n"
        "# ====================\n\n"
        "steps_per_epoch = len(pairs) // BATCH_SIZE\n"
        "total_steps = (steps_per_epoch * EPOCHS) // GRAD_ACCUM   # 优化器步数\n"
        "warmup_steps = int(total_steps * 0.1)\n"
        "print(f'total_steps={total_steps} warmup_steps={warmup_steps}')\n\n"
        "model = SentenceTransformer(MODEL)\n"
        "model.max_seq_length = MAX_SEQ_LENGTH   # 长 turn 需要更长上下文\n"
        "model.gradient_checkpointing_enable()    # 长序列省显存（换一点速度）\n"
        "print('max_seq_length:', model.max_seq_length)\n"
        "dataset = Dataset.from_list(pairs)\n"
        "loss = MultipleNegativesRankingLoss(model, scale=20.0)\n\n"
        "args = SentenceTransformerTrainingArguments(\n"
        "    output_dir=OUTPUT_DIR,\n"
        "    num_train_epochs=EPOCHS,\n"
        "    per_device_train_batch_size=BATCH_SIZE,\n"
        "    gradient_accumulation_steps=GRAD_ACCUM,\n"
        "    learning_rate=LR,\n"
        "    lr_scheduler_type='cosine',\n"
        "    warmup_steps=warmup_steps,\n"
        "    fp16=torch.cuda.is_available(),\n"
        "    bf16=False,\n"
        "    logging_steps=100,\n"
        "    save_strategy='epoch',\n"
        "    report_to=[],\n"
        "    dataloader_drop_last=True,\n"
        ")\n"
        "trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)\n"
        "trainer.train()\n"
        "model.save_pretrained(OUTPUT_DIR)\n\n"
        "os.makedirs(DRIVE_DIR, exist_ok=True)\n"
        "shutil.copytree(OUTPUT_DIR, DRIVE_DIR, dirs_exist_ok=True)\n"
        "print('saved to Drive ->', DRIVE_DIR)"
    ),
    code(
        "# ⑤ TIAGE 评测\n"
        "import requests, json, numpy as np\n"
        "from sentence_transformers import SentenceTransformer\n\n"
        "model = SentenceTransformer('/content/model_bgem3')\n\n"
        "r = requests.get('https://api.github.com/repos/HuiyuanXie/tiage/git/trees/main?recursive=1')\n"
        "paths = [t['path'] for t in r.json()['tree'] if 'anno_test.json' in t['path']]\n"
        "url = 'https://raw.githubusercontent.com/HuiyuanXie/tiage/main/' + paths[0]\n"
        "data = requests.get(url).json()\n\n"
        "cases = []\n"
        "for did, turns in data.items():\n"
        "    utterances = [u for u, l in turns]\n"
        "    boundaries = [i for i, (u, l) in enumerate(turns) if l == '1']\n"
        "    cases.append((utterances, boundaries))\n\n"
        "all_texts = []\n"
        "offsets = []\n"
        "for utts, _ in cases:\n"
        "    offsets.append(len(all_texts))\n"
        "    all_texts.extend(utts)\n\n"
        "embs = model.encode(all_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)\n"
        "case_scores = []\n"
        "for idx, (utts, _) in enumerate(cases):\n"
        "    s = offsets[idx]\n"
        "    e = embs[s:s+len(utts)]\n"
        "    case_scores.append([float(np.dot(e[i], e[i+1])) for i in range(len(e)-1)])\n\n"
        "def dp_segment(scores, min_len=2, max_len=20, threshold=0.5):\n"
        "    n = len(scores) + 1\n"
        "    if n <= min_len:\n"
        "        return []\n"
        "    INF = float('inf')\n"
        "    dp = [INF]*(n+1); prev = [-1]*(n+1); dp[0] = 0.0\n"
        "    for i in range(1, n+1):\n"
        "        lo = max(0, i-max_len); hi = i-min_len\n"
        "        for j in range(lo, hi+1):\n"
        "            if i-j < min_len or i-j > max_len:\n"
        "                continue\n"
        "            cost = dp[j] + ((scores[j-1]-threshold) if j > 0 else 0.0)\n"
        "            if cost < dp[i]:\n"
        "                dp[i] = cost; prev[i] = j\n"
        "    if dp[n] == INF:\n"
        "        return []\n"
        "    cuts = []; i = n\n"
        "    while i > 0:\n"
        "        j = prev[i]\n"
        "        if j <= 0:\n"
        "            break\n"
        "        cuts.append(j); i = j\n"
        "    return sorted(cuts)\n\n"
        "def evaluate_all(golds, preds, tolerance=0):\n"
        "    tp = fp = fn = 0\n"
        "    for g, p in zip(golds, preds):\n"
        "        gs = set(g); matched = set(); t = 0\n"
        "        for x in p:\n"
        "            for y in gs:\n"
        "                if y in matched:\n"
        "                    continue\n"
        "                if abs(x-y) <= tolerance:\n"
        "                    t += 1; matched.add(y); break\n"
        "        tp += t; fp += len(p)-t; fn += len(gs)-len(matched)\n"
        "    precision = tp/(tp+fp) if tp+fp else 0.0\n"
        "    recall = tp/(tp+fn) if tp+fn else 0.0\n"
        "    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0\n"
        "    return f1, precision, recall\n\n"
        "golds = [b for _, b in cases]\n"
        "print('threshold   F1(tol0)   F1(tol1)   P       R')\n"
        "best = None\n"
        "for th in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:\n"
        "    preds = [dp_segment(s, threshold=th) for s in case_scores]\n"
        "    f1, p, r = evaluate_all(golds, preds, 0)\n"
        "    f1t, _, _ = evaluate_all(golds, preds, 1)\n"
        "    print(f'{th:.2f}       {f1:.3f}       {f1t:.3f}       {p:.3f}   {r:.3f}')\n"
        "    if best is None or f1 > best[1]:\n"
        "        best = (th, f1, f1t, p, r)\n"
        "print('=== best: threshold=%.2f F1(tol0)=%.3f F1(tol1)=%.3f ===' % (best[0], best[1], best[2]))\n"
        "print('对照: word_bag 0.329 | bge-m3 零样本 0.397 | bge-small微调 0.429')"
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = "training/colab_full.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("written", out)
