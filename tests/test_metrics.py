"""metrics 评估指标测试。"""
from eval.metrics import boundary_f1, evaluate_all, pk


def test_perfect_match():
    r = boundary_f1([3, 7], [3, 7], tolerance=0)
    assert r["f1"] == 1.0
    assert r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 0


def test_missing_prediction():
    r = boundary_f1([3, 7], [3], tolerance=0)
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 1
    assert 0 < r["f1"] < 1.0


def test_tolerance_match():
    r = boundary_f1([3], [4], tolerance=1)
    assert r["tp"] == 1 and r["f1"] == 1.0


def test_no_prediction_zero_f1():
    r = boundary_f1([3, 7], [], tolerance=0)
    assert r["f1"] == 0.0


def test_pk_perfect():
    assert pk([3, 7], [3, 7], 10) == 0.0


def test_evaluate_all_micro():
    r = evaluate_all([[3, 7], [5]], [[3, 7], [6]], tolerance=0)
    # case1: tp=2 fp=0 fn=0；case2: tp=0 fp=1 fn=1 → 汇总 tp=2 fp=1 fn=1
    assert r["tp"] == 2 and r["fp"] == 1 and r["fn"] == 1
