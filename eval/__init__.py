"""Evaluation module：指定 dataset set 即可评估。

用法：
    from eval import evaluate
    evaluate("cases", "char_ngram")

CLI：
    python -m eval.run <dataset> <encoder>
"""
from eval.datasets import available_datasets, load_dataset
from eval.metrics import boundary_f1, evaluate_all, pk

__all__ = [
    "available_datasets",
    "load_dataset",
    "evaluate",
    "boundary_f1",
    "evaluate_all",
    "pk",
]


def __getattr__(name: str):
    # 惰性导入 evaluate，避免 `python -m eval.run` 时 eager import 触发 runpy warning
    if name == "evaluate":
        from eval.run import evaluate

        return evaluate
    raise AttributeError(f"module 'eval' has no attribute {name!r}")
