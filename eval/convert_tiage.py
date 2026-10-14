"""TIAGE 数据转换：turn 级 0/1 标签 → 统一 eval 格式。"""
from __future__ import annotations

import json
from pathlib import Path


def convert_tiage_to_eval(tiage_json_path: str | Path) -> list[dict]:
    """把 TIAGE 的 anno_test.json 转成统一 eval 格式。

    TIAGE 格式：dict[dialog_id] = [(utterance, label), ...]
      label "-1" = 首条（N/A），"0" = 无 topic shift，"1" = topic shift。

    映射：
      - 每条 utterance → 一条 user 消息（双人闲聊无 assistant 结构）
      - label[i] == "1" → boundary 包含 i（在第 i 个 turn 之后切，1-based）
    """
    with open(tiage_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases: list[dict] = []
    for dialog_id, turns in data.items():
        messages: list[dict] = []
        boundaries: list[int] = []
        for i, (utterance, label) in enumerate(turns):
            messages.append({"role": "user", "content": utterance})
            if label == "1":
                # utterance[i] 是第 i+1 个 turn（shift），即在第 i 个 turn 之后切
                boundaries.append(i)
        cases.append(
            {
                "case_id": f"tiage_test_{dialog_id}",
                "source": "tiage",
                "language": "en",
                "messages": messages,
                "boundaries": boundaries,
            }
        )
    return cases


def main(tiage_json_path: str = "eval/tiage/anno_test.json",
         out_path: str = "eval/tiage/tiage_eval.json") -> int:
    cases = convert_tiage_to_eval(tiage_json_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    total_boundaries = sum(len(c["boundaries"]) for c in cases)
    print(f"转换完成：{len(cases)} 个 case，{total_boundaries} 个 boundary → {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
