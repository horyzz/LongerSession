"""任务型对话 → 承接样本对（MultiWOZ domain 切换 + DialSeg711 topic 标注）。

把任务型对话转成 (text_a, text_b, label) 相邻 turn 样本对：
- 同 topic/domain → label=1（承接，正样本）
- 跨 topic/domain（边界）→ label=0（切换，负样本）

来源：
- MultiWOZ 2.2 train（8,437 对话）：domain = 每 turn 活跃 service（active_intent）
- DialSeg711（711 对话）：人类标注 topic_id（比 domain 更细）

用法：
    uv run python -m training.task_oriented \
        --out data/training/pairs_task.jsonl \
        --dialseg711-train-ratio 0.7 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

MULTIWOZ_BASE = "https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2"
DIALSEG711_REPO = "Coldog2333/dialseg711"
DIALSEG711_FILE = "test.json"


def download_multiwoz_train(max_files: int = 20) -> list[dict]:
    """下载 MultiWOZ 2.2 train 全部 dialogue 文件（GitHub raw，17 个文件）。"""
    def _dl(i: int):
        url = f"{MULTIWOZ_BASE}/train/dialogues_{i:03d}.json"
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception:
            return None

    dialogues: list[dict] = []
    with ThreadPoolExecutor(8) as ex:
        for i, data in enumerate(ex.map(_dl, range(1, max_files + 1)), start=1):
            if data is None:
                break
            dialogues.extend(json.loads(data))
    return dialogues


def active_services(turn: dict) -> set[str]:
    """MultiWOZ turn 的活跃 domain：active_intent != NONE 的 service。

    USER turn 的 frames 含全部 8 个 service（靠 active_intent 区分活跃与否），
    SYSTEM turn 的 frames 只含当前响应 service（active_intent 可能缺失）。
    """
    act: set[str] = set()
    for f in turn.get("frames", []):
        st = f.get("state") or {}
        intent = st.get("active_intent", "NONE")
        if intent and intent != "NONE":
            act.add(f["service"])
    if not act and turn.get("speaker") == "SYSTEM":
        act = {f["service"] for f in turn.get("frames", [])}
    return act


def _pair(a: dict, b: dict, same: bool, source: str, tree_id: str) -> dict | None:
    ua = (a.get("utterance") or "").strip()
    ub = (b.get("utterance") or "").strip()
    if not ua or not ub:
        return None
    return {
        "text_a": ua,
        "text_b": ub,
        "label": 1 if same else 0,
        "source": source,
        "pair_type": "same_topic" if same else "cross_topic",
        "lang": "en",
        "tree_id": tree_id,
    }


def multiwoz_pairs(dialogues: list[dict]) -> list[dict]:
    """MultiWOZ domain 切换 → 相邻 turn 样本对。"""
    pairs: list[dict] = []
    for d in dialogues:
        turns = d["turns"]
        for i in range(len(turns) - 1):
            a, b = turns[i], turns[i + 1]
            sa, sb = active_services(a), active_services(b)
            if not sa or not sb:
                continue
            p = _pair(a, b, bool(sa & sb), "multiwoz", d["dialogue_id"])
            if p:
                pairs.append(p)
    return pairs


def dialseg711_pairs(dialogues: list[dict]) -> list[dict]:
    """DialSeg711 topic_id → 相邻 turn 样本对（人类标注，比 domain 更细）。"""
    pairs: list[dict] = []
    for d in dialogues:
        turns = d["turns"]
        for i in range(len(turns) - 1):
            a, b = turns[i], turns[i + 1]
            p = _pair(a, b, a["topic_id"] == b["topic_id"], "dialseg711", d["dial_id"])
            if p:
                pairs.append(p)
    return pairs


def load_dialseg711() -> list[dict]:
    path = hf_hub_download(DIALSEG711_REPO, DIALSEG711_FILE, repo_type="dataset")
    data = json.load(open(path, encoding="utf-8"))
    return data["dial_data"]["dialseg711"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="构建任务型承接样本对")
    p.add_argument("--out", default="data/training/pairs_task.jsonl")
    p.add_argument("--dialseg711-train-ratio", type=float, default=0.7,
                   help="DialSeg711 用作训练的比例（其余做 held-out 评测）")
    p.add_argument("--heldout-out", default="data/training/dialseg711_heldout.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    pairs: list[dict] = []

    mw = download_multiwoz_train()
    print(f"[multiwoz] {len(mw)} 对话", file=sys.stderr)
    pairs.extend(multiwoz_pairs(mw))

    dialseg = load_dialseg711()
    rng = random.Random(args.seed)
    rng.shuffle(dialseg)
    n_train = int(len(dialseg) * args.dialseg711_train_ratio)
    train_ds, heldout_ds = dialseg[:n_train], dialseg[n_train:]
    print(f"[dialseg711] {len(dialseg)} 对话 → train {len(train_ds)} / heldout {len(heldout_ds)}",
          file=sys.stderr)
    pairs.extend(dialseg711_pairs(train_ds))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    heldout = Path(args.heldout_out)
    heldout.parent.mkdir(parents=True, exist_ok=True)
    json.dump(heldout_ds, open(heldout, "w", encoding="utf-8"), ensure_ascii=False)

    from collections import Counter
    print(f"[done] 写入 {len(pairs)} 对 → {out}", file=sys.stderr)
    print(f"  按 label: {dict(Counter(rec['label'] for rec in pairs))}", file=sys.stderr)
    print(f"  按 source: {dict(Counter(rec['source'] for rec in pairs))}", file=sys.stderr)
    print(f"[heldout] {len(heldout_ds)} 对话 → {heldout}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
