"""生成自建 cases 的 Colab 评测 cell（数据内嵌 + 评测逻辑）。"""
import glob
import json

# 1) 读 case，压缩成只保留 role/content/boundaries
cases = []
for f in sorted(glob.glob("eval/cases/*.json")):
    c = json.load(open(f, encoding="utf-8"))
    cases.append({
        "case_id": c["case_id"],
        "language": c["language"],
        "length": c["length"],
        "messages": [{"role": m["role"], "content": m["content"]} for m in c["messages"]],
        "boundaries": c["boundaries"],
    })

data = json.dumps(cases, ensure_ascii=False)

logic = r'''
import numpy as np
from sentence_transformers import SentenceTransformer

CASES = __CASES__

# 模型路径：训练好的 bge-m3（Colab 上）
model = SentenceTransformer('/content/model_bgem3')

def messages_to_turns(messages):
    turns = []
    cur_user = None
    cur_assistant = []
    for m in messages:
        r = m['role']; c = m.get('content', '')
        if r == 'user':
            if cur_user is not None:
                turns.append((cur_user, '\n'.join(cur_assistant)))
            cur_user = c; cur_assistant = []
        elif r == 'assistant':
            cur_assistant.append(c)
        # tool 消息忽略
    if cur_user is not None:
        turns.append((cur_user, '\n'.join(cur_assistant)))
    return turns

def combined(t):
    u, a = t
    return (u + '\n' + a).strip() if a else u.strip()

def dp_segment(scores, min_len=2, max_len=20, threshold=0.5):
    n = len(scores) + 1
    if n <= min_len:
        return []
    INF = float('inf')
    dp = [INF]*(n+1); prev = [-1]*(n+1); dp[0] = 0.0
    for i in range(1, n+1):
        lo = max(0, i-max_len); hi = i-min_len
        for j in range(lo, hi+1):
            if i-j < min_len or i-j > max_len:
                continue
            cost = dp[j] + ((scores[j-1]-threshold) if j > 0 else 0.0)
            if cost < dp[i]:
                dp[i] = cost; prev[i] = j
    if dp[n] == INF:
        return []
    cuts = []; i = n
    while i > 0:
        j = prev[i]
        if j <= 0:
            break
        cuts.append(j); i = j
    return sorted(cuts)

def evaluate_all(golds, preds, tolerance=0):
    tp = fp = fn = 0
    for g, p in zip(golds, preds):
        gs = set(g); matched = set(); t = 0
        for x in p:
            for y in gs:
                if y in matched:
                    continue
                if abs(x-y) <= tolerance:
                    t += 1; matched.add(y); break
        tp += t; fp += len(p)-t; fn += len(gs)-len(matched)
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return f1, precision, recall

THS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# 每 case 预计算 scores
case_scores = {}
for c in CASES:
    turns = [combined(t) for t in messages_to_turns(c['messages'])]
    texts = [t for t in turns if t]
    embs = model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
    case_scores[c['case_id']] = [float(np.dot(embs[i], embs[i+1])) for i in range(len(embs)-1)]

# 分维度评估
groups = {}
for c in CASES:
    key = f"{c['language']}-{c['length']}"
    groups.setdefault(key, {'g': [], 'scores': []})
    groups[key]['g'].append(c['boundaries'])
    groups[key]['scores'].append(case_scores[c['case_id']])

print('维度        F1(tol0)  F1(tol1)  最佳阈值')
overall_g = []; overall_s = []
for key in sorted(groups):
    g = groups[key]
    best = (0, None, None, None)
    for th in THS:
        preds = [dp_segment(s, threshold=th) for s in g['scores']]
        f1, p, r = evaluate_all(g['g'], preds, 0)
        f1t, _, _ = evaluate_all(g['g'], preds, 1)
        if f1 > best[0]:
            best = (f1, f1t, p, r, th)
    f1, f1t, p, r, th = best[0], best[1], best[2], best[3], best[4]
    print(f'{key:10s} {f1:.3f}     {f1t:.3f}     {th:.2f}  (P={p:.3f} R={r:.3f})')
    overall_g.extend(g['g']); overall_s.extend(g['scores'])

best = (0, None)
for th in THS:
    preds = [dp_segment(s, threshold=th) for s in overall_s]
    f1, p, r = evaluate_all(overall_g, preds, 0)
    if f1 > best[0]:
        best = (f1, th)
print(f'\n=== overall best: F1(tol0)={best[0]:.3f} @ th={best[1]:.2f} ===')
print('对照(cases): char_ngram 0.449 | bge-m3 零样本 0.625')
'''

cell = logic.replace("__CASES__", data)

out = "training/cases_eval_cell.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write(cell)
print(f"written {out}, {len(cell)} chars, {len(data)} chars data")
