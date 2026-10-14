"""评估数据集统一加载：指定 set 即可。"""
from __future__ import annotations

import json
from pathlib import Path

DATASET_PATHS: dict[str, Path] = {
    "tiage": Path("eval/tiage/tiage_eval.json"),
    "cases": Path("eval/cases"),
}


def load_dataset(name: str) -> list[dict]:
    """加载统一格式的评估 case 列表。

    name: "tiage" | "cases"
    返回 [{case_id, source, language?, length?, messages, boundaries}, ...]
    """
    if name == "tiage":
        with open(DATASET_PATHS["tiage"], "r", encoding="utf-8") as f:
            return json.load(f)
    if name == "cases":
        return [
            json.load(open(f, "r", encoding="utf-8"))
            for f in sorted(DATASET_PATHS["cases"].glob("*.json"))
        ]
    raise ValueError(f"unknown dataset: {name}")


def available_datasets() -> list[str]:
    return sorted(DATASET_PATHS.keys())
