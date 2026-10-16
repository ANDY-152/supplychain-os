"""R1 整数刻度金额修复回归测试（方案 2）。

Run: pytest tests/test_allocation_r1_exact.py

契约来源：v4.4（未改）。金额 oracle 用 Fraction(str(x))（精确十进制），
不把被测实现自身的输出当作唯一正确性依据。
说明：随机部分为有限样本（<= 3000 组），不构成“所有输入均无浮点边界”的证明。
"""
import math
import random
from fractions import Fraction

import pandas as pd
import pytest

from app.allocation import (
    BUDGET_EXHAUSTED,
    FULLY_FULFILLED,
    MOQ_INFEASIBLE,
    NO_GAP,
    PARTIALLY_FULFILLED,
    SCALED_TO_ZERO,
    ROLLED_BACK,
    allocate,
)

L2 = ["sku", "procurement_recommended_qty", "current_stock", "daily_demand",
      "lead_time_days", "reason", "procurement_reason"]
R12 = ["allocation_rank", "allocated_qty", "allocation_reason", "unit_cost", "priority",
       "budget_available_before", "budget_spent_today", "budget_remaining_after"]


def frame(rows):
    d = {"sku": [r[0] for r in rows], "procurement_recommended_qty": [float(r[1]) for r in rows],
         "current_stock": [float(r[3]) for r in rows], "daily_demand": [float(r[4]) for r in rows],
         "lead_time_days": [float(r[5]) for r in rows], "reason": ["X"] * len(rows),
         "procurement_reason": ["Y"] * len(rows)}
    return pd.DataFrame(d)[L2]


def cons(*rows):
    return pd.DataFrame(rows, columns=["sku", "moq", "order_multiple"])


def run(rows, costs, priority, constraints, budget, algorithm="R1"):
    return allocate(frame(rows), algorithm=algorithm, budget_available=budget,
                    costs=costs, priority=priority, constraints=constraints)


def alloc_of(out, sku):
    return int(out.loc[out["sku"] == sku, "allocated_qty"].iloc[0])


def reason_of(out, sku):
    return out.loc[out["sku"] == sku, "allocation_reason"].iloc[0]


def f(x):
    return Fraction(str(x))


def _rows(n, q, c, m=1):
    """n SKUs, same cost/q, distinct priorities, moq=0, multiple=m."""
    skus = [f"S{i}" for i in range(n)]
    rows = [(s, q, i + 1, 0, 1.0, 1) for i, s in enumerate(skus)]
    costs = {s: c for s in skus}
    prio = {s: i + 1 for i, s in enumerate(skus)}
    crows = [(s, 0.0, m) for s in skus]
    return skus, rows, costs, prio, crows


def reference_r1(rows, costs, constraints, budget):
    """Exact-decimal R1 reference (Fraction), v4.4 steps; order = ascending priority."""
    q = {r[0]: int(r[1]) for r in rows}
    prio = {r[0]: r[2] for r in rows}
    m, mult = {}, {}
    for s in q:
        moq, mu = constraints.get(s, (0.0, 1))
        mult[s] = int(mu)
        m[s] = max(1, math.ceil(f(moq) / mu)) * int(mu)
    order = sorted(q, key=lambda s: prio[s])
    allocated = {s: 0 for s in q}
    reason = {}
    bud = f(budget)
    if bud == 0:
        for s in q:
            reason[s] = NO_GAP if q[s] == 0 else BUDGET_EXHAUSTED
        return allocated, reason, bud, bud
    e = []
    for s in q:
        if q[s] == 0:
            reason[s] = NO_GAP
        elif f(costs[s]) * m[s] > bud:
            reason[s] = MOQ_INFEASIBLE
        else:
            e.append(s)
    if e:
        T = sum(f(costs[s]) * q[s] for s in e)
        if T <= bud:
            for s in e:
                allocated[s] = q[s]; reason[s] = FULLY_FULFILLED
        else:
            lam = bud / T
            terminal = set()
            for s in e:
                qi = mult[s] * int((q[s] * lam) // mult[s])
                allocated[s] = qi
                if qi == 0:
                    reason[s] = SCALED_TO_ZERO; terminal.add(s)
            for s in e:
                if s in terminal:
                    continue
                if 0 < allocated[s] < m[s]:
                    allocated[s] = 0; reason[s] = SCALED_TO_ZERO; terminal.add(s)
            total = sum(f(costs[s]) * allocated[s] for s in e)
            for s in sorted(e, key=lambda s: prio[s], reverse=True):
                if total <= bud:
                    break
                if allocated[s] > 0 and s not in terminal:
                    total -= f(costs[s]) * allocated[s]
                    allocated[s] = 0; reason[s] = ROLLED_BACK; terminal.add(s)
            for s in e:
                if s not in terminal and allocated[s] > 0:
                    reason[s] = FULLY_FULFILLED if allocated[s] == q[s] else PARTIALLY_FULFILLED
    spent = sum(f(costs[s]) * allocated[s] for s in q)
    return allocated, reason, spent, bud - spent


# ------------------------- R-1..R-5 探针反例（方案2 目标） ------------------

@pytest.mark.parametrize("n,c,q,budget,expected", [
    (2, 0.01, 10, 0.16, {0: 8, 1: 8}),      # R-1 最小反例
    (2, 0.01, 20, 0.30, {0: 15, 1: 15}),    # R-2
    (3, 0.01, 20, 0.57, {0: 19, 1: 19, 2: 19}),  # R-3
    (2, 0.05, 20, 1.90, {0: 19, 1: 19}),    # R-4（旧方案1 误得 S1=0）
    (2, 0.05, 30, 2.90, {0: 29, 1: 29}),    # R-5（旧方案1 误得 S1=0）
])
def test_r1_residual_counterexamples_fixed(n, c, q, budget, expected):
    skus, rows, costs, prio, crows = _rows(n, q, c)
    out = run(rows, costs, prio, cons(*crows), budget)
    for i, s in enumerate(skus):
        assert alloc_of(out, s) == expected[i]
    # independent decimal feasibility
    spent = sum(f(costs[s]) * alloc_of(out, s) for s in skus)
    assert spent <= f(budget)


# ------------------------- 恰好 / ±1 最小金额单位 ---------------------------

def test_r1_exact_equality_full():
    skus, rows, costs, prio, crows = _rows(6, 10, 0.01)
    out = run(rows, costs, prio, cons(*crows), 0.60)
    assert [alloc_of(out, s) for s in skus] == [10] * 6
    assert set(out["allocation_reason"]) == {FULLY_FULFILLED}
    assert out["budget_remaining_after"].iloc[0] == 0.0


def test_r1_one_unit_less():
    skus, rows, costs, prio, crows = _rows(6, 10, 0.01)
    out = run(rows, costs, prio, cons(*crows), 0.59)
    assert [alloc_of(out, s) for s in skus] == [9] * 6
    assert set(out["allocation_reason"]) == {PARTIALLY_FULFILLED}
    assert sum(f(costs[s]) * 9 for s in skus) <= f(0.59)


def test_r1_one_unit_more():
    skus, rows, costs, prio, crows = _rows(6, 10, 0.01)
    out = run(rows, costs, prio, cons(*crows), 0.61)
    assert [alloc_of(out, s) for s in skus] == [10] * 6
    assert out["budget_remaining_after"].iloc[0] == pytest.approx(0.01)


# ------------------------- λ 整包边界 / 略低 --------------------------------

def test_r1_lambda_exactly_on_pack_boundary():
    skus, rows, costs, prio, crows = _rows(2, 10, 1.0)
    out = run(rows, costs, prio, cons(*crows), 10.0)   # lam=0.5, 10*0.5=5
    assert [alloc_of(out, s) for s in skus] == [5, 5]


def test_r1_lambda_just_below_pack_boundary():
    skus, rows, costs, prio, crows = _rows(2, 10, 1.0)
    out = run(rows, costs, prio, cons(*crows), math.nextafter(10.0, 0.0))
    assert [alloc_of(out, s) for s in skus] == [4, 4]


# ------------------------- MOQ / 倍数 / 缩放归零 -----------------------------

def test_r1_moq_boundary():
    rows = [("A", 120, 1, 0, 1.0, 1)]
    costs, prio, c = {"A": 10.0}, {"A": 1}, cons(("A", 100.0, 30))
    full = run(rows, costs, prio, c, 1200.0)
    assert alloc_of(full, "A") == 120 and reason_of(full, "A") == FULLY_FULFILLED
    infeasible = run(rows, costs, prio, c, 1199.99)
    assert alloc_of(infeasible, "A") == 0 and reason_of(infeasible, "A") == MOQ_INFEASIBLE
    over = run(rows, costs, prio, c, 1200.01)
    assert alloc_of(over, "A") == 120


def test_r1_pack_multiple_floor():
    # c=1, M=3, q=9, B=4 -> lam=4/9 -> floor(9*lam/3)=1 pack -> q=3
    skus, rows, costs, prio, crows = _rows(1, 9, 1.0, m=3)
    out = run(rows, costs, prio, cons(*crows), 4.0)
    assert alloc_of(out, "S0") == 3
    assert reason_of(out, "S0") == PARTIALLY_FULFILLED


def test_r1_scaled_to_zero():
    skus, rows, costs, prio, crows = _rows(2, 100, 1.0)
    out = run(rows, costs, prio, cons(*crows), 1.0)
    assert [alloc_of(out, s) for s in skus] == [0, 0]
    assert set(out["allocation_reason"]) == {SCALED_TO_ZERO}


# -------------------- 混合小数精度 / 极小成本 / 大数量 -----------------------

def test_r1_mixed_decimal_precision():
    # cost 3dp, budget 4dp: total=0.003 > 0.0025 -> lam=5/6 -> floor(3*5/6)=2
    out = run([("A", 3, 1, 0, 1.0, 1)], {"A": 0.001}, {"A": 1},
              cons(("A", 0.0, 1)), 0.0025)
    assert alloc_of(out, "A") == 2
    assert reason_of(out, "A") == PARTIALLY_FULFILLED
    assert f(0.001) * 2 <= f(0.0025)


def test_r1_tiny_positive_cost():
    # 5e-324 subnormal cost; budget 1e-320 -> full
    out = run([("A", 1, 1, 0, 1.0, 1)], {"A": 5e-324}, {"A": 1},
              cons(("A", 0.0, 1)), 1e-320)
    assert alloc_of(out, "A") == 1
    assert reason_of(out, "A") == FULLY_FULFILLED
    assert f(5e-324) * 1 <= f(1e-320)
    assert out["budget_remaining_after"].iloc[0] >= 0.0


def test_r1_large_quantity():
    # q=10^12, cost=0.01, B=1e9 -> total=1e10 > B -> lam=0.1 -> q=10^11
    q = 10 ** 12
    out = run([("A", q, 1, 0, 1.0, 1)], {"A": 0.01}, {"A": 1},
              cons(("A", 0.0, 1)), 1e9)
    assert alloc_of(out, "A") == 10 ** 11
    assert reason_of(out, "A") == PARTIALLY_FULFILLED
    assert f(0.01) * (10 ** 11) <= f(1e9)


def test_r1_multi_sku_mixed_decimal_precision():
    # A: 7*0.001, B: 3*0.13, total=0.397 > 0.2 -> A=3, B=1
    out = run([("A", 7, 1, 0, 1.0, 1), ("B", 3, 2, 0, 1.0, 1)],
              {"A": 0.001, "B": 0.13}, {"A": 1, "B": 2},
              cons(("A", 0.0, 1), ("B", 0.0, 1)), 0.2)
    assert alloc_of(out, "A") == 3
    assert alloc_of(out, "B") == 1
    assert reason_of(out, "A") == PARTIALLY_FULFILLED
    assert reason_of(out, "B") == PARTIALLY_FULFILLED
    assert f(0.001) * 3 + f(0.13) * 1 <= f(0.2)
    assert out["budget_remaining_after"].iloc[0] == pytest.approx(0.067)


def test_r1_b_avail_zero_short_circuit():
    rows = [("A", 100, 1, 0, 1.0, 1), ("B", 0, 1, 0, 1.0, 1)]
    out = run(rows, {"A": 10.0, "B": 1.0}, {"A": 1, "B": 1},
              cons(("A", 0.0, 1), ("B", 0.0, 1)), 0.0)
    assert reason_of(out, "A") == BUDGET_EXHAUSTED
    assert reason_of(out, "B") == NO_GAP
    assert out["budget_spent_today"].iloc[0] == 0.0


# ------------------------- R0 / R2 不受影响 --------------------------------

def test_r0_unchanged_passthrough():
    rows = [("A", 10.5, 1, 0, 1.0, 1)]
    out = allocate(frame(rows), algorithm="R0", budget_available=0.0,
                   costs=None, priority=None, constraints=None)
    assert list(out.columns) == L2 + ["allocated_qty", "allocation_reason"]
    assert out["allocated_qty"].iloc[0] == 10.5


def test_r2_unchanged_exact_money():
    out = run([("X", 35, 1, 0, 1.0, 1)], {"X": 0.01}, {"X": 1},
              cons(("X", 0.0, 1)), 0.35, algorithm="R2")
    assert alloc_of(out, "X") == 35
    assert out["budget_remaining_after"].iloc[0] == 0.0


# ------------------------- R0/R1/R2 dtype / 列不变 --------------------------

def test_r1_output_columns_and_dtypes():
    skus, rows, costs, prio, crows = _rows(2, 10, 1.0)
    out = run(rows, costs, prio, cons(*crows), 10.0)
    assert [c for c in out.columns if c in R12] == R12
    assert out["allocation_rank"].dtype == "int64"
    assert out["allocated_qty"].dtype == "int64"
    assert out["priority"].dtype == "int64"
    assert out["budget_remaining_after"].dtype == "float64"


# ------------------------- 有限随机 oracle（<=3000） ------------------------

def test_r1_random_matches_fraction_oracle():
    rng = random.Random(2025)
    checked = 0
    N = 3000
    for _ in range(N):
        n = rng.randint(2, 5)
        rows, costs, prio, crows, cmap = [], {}, {}, [], {}
        for i in range(n):
            s = f"S{i}"
            mu = rng.choice([1, 2, 3, 5, 8])
            moq = 0.0 if rng.random() < 0.5 else float(rng.randint(0, 3) * mu)
            q = mu * rng.randint(1, 20)
            if q < moq:
                q = math.ceil(moq / mu) * mu
            rows.append((s, q, i + 1, 0, 1.0, 1))
            costs[s] = round(rng.uniform(0.01, 20.0), 2)
            prio[s] = i + 1
            crows.append((s, moq, mu))
            cmap[s] = (moq, mu)
        total = sum(f(costs[s]) * r[1] for s, r in zip([x[0] for x in rows], rows))
        mode = rng.random()
        budget = float(total) if mode < 0.5 else (float(total) - 0.01 if mode < 0.8 else float(total) + 0.01)
        budget = max(0.0, budget)
        out = run(rows, costs, prio, cons(*crows), budget)
        ea, er, espent, erem = reference_r1(rows, costs, cmap, budget)
        for s in costs:
            assert alloc_of(out, s) == ea[s], (s, costs, budget, ea, out.to_dict())
            assert reason_of(out, s) == er[s]
        # independent feasibility and spend/remaining (decimal exact)
        spent = sum(f(costs[s]) * ea[s] for s in costs)
        assert spent == espent
        assert spent <= f(budget)
        checked += 1
    assert checked == N
