"""R2 精确金额修复回归测试（R2-only decimal-scale）。

Run: pytest tests/test_allocation_r2_exact.py

契约来源：v4.4（未改）。本文件只断言可观察行为；金额 oracle 用 Fraction，
不把被测实现自身的计算当作唯一正确性依据。
"""
import math
import random
from decimal import Decimal
from fractions import Fraction

import pandas as pd
import pytest

from app.allocation import (
    BUDGET_EXHAUSTED,
    BUDGET_INSUFFICIENT_MIN_PACK,
    FULLY_FULFILLED,
    MOQ_INFEASIBLE,
    NO_GAP,
    PARTIALLY_FULFILLED,
    allocate,
)

L2 = [
    "sku",
    "procurement_recommended_qty",
    "current_stock",
    "daily_demand",
    "lead_time_days",
    "reason",
    "procurement_reason",
]


def frame(rows):
    """rows: list of (sku, q, priority, current_stock, daily_demand, lead_time_days)."""
    data = {
        "sku": [r[0] for r in rows],
        "procurement_recommended_qty": [float(r[1]) for r in rows],
        "current_stock": [float(r[3]) for r in rows],
        "daily_demand": [float(r[4]) for r in rows],
        "lead_time_days": [float(r[5]) for r in rows],
        "reason": ["STOCK_BELOW_REORDER_POINT"] * len(rows),
        "procurement_reason": ["MOQ_APPLIED"] * len(rows),
    }
    return pd.DataFrame(data)[L2]


def cons(*rows):
    return pd.DataFrame(rows, columns=["sku", "moq", "order_multiple"])


def run(rows, costs, priority, constraints, budget, algorithm="R2"):
    return allocate(
        frame(rows),
        algorithm=algorithm,
        budget_available=budget,
        costs=costs,
        priority=priority,
        constraints=constraints,
    )


def alloc_of(out, sku):
    return int(out.loc[out["sku"] == sku, "allocated_qty"].iloc[0])


def reason_of(out, sku):
    return out.loc[out["sku"] == sku, "allocation_reason"].iloc[0]


# ----------------------------- independent Fraction oracle ------------------

def _frac(x):
    return Fraction(str(x))


def reference_r2(rows, costs, constraints, budget):
    """Independent R2 reference in exact rational arithmetic.

    rows: list of (sku, q). constraints: dict sku -> (moq, multiple).
    Returns (allocated, reason, remaining_fraction).
    Processing order = ascending priority (rows carry priority in key).
    """
    q = {r[0]: int(r[1]) for r in rows}
    prio = {r[0]: r[2] for r in rows}
    m, mult = {}, {}
    for sku in q:
        moq, multiple = constraints.get(sku, (0.0, 1))
        moq_f = _frac(moq)
        mult[sku] = int(multiple)
        m[sku] = max(1, math.ceil(moq_f / multiple)) * int(multiple)
    order = sorted(q, key=lambda s: prio[s])

    allocated = {s: 0 for s in q}
    reason = {}
    bud = _frac(budget)
    for s in q:
        if q[s] == 0:
            reason[s] = NO_GAP
        elif _frac(costs[s]) * m[s] > bud:
            reason[s] = MOQ_INFEASIBLE
        else:
            reason[s] = None

    remaining = bud
    for s in order:
        if reason[s] is not None:
            continue
        c = _frac(costs[s])
        if remaining <= 0:
            reason[s] = BUDGET_EXHAUSTED
            continue
        if remaining < c * m[s]:
            reason[s] = BUDGET_INSUFFICIENT_MIN_PACK
            continue
        packs = min(remaining // (c * mult[s]), q[s] // mult[s])
        if packs * mult[s] < m[s]:
            reason[s] = BUDGET_INSUFFICIENT_MIN_PACK
            continue
        qi = packs * mult[s]
        remaining -= c * qi
        allocated[s] = qi
        reason[s] = FULLY_FULFILLED if qi == q[s] else PARTIALLY_FULFILLED
    return allocated, reason, remaining


# --------------------------------- T1-T4 实证缺陷 ---------------------------

def test_t1_single_sku_exact_equality_no_false_failure():
    out = run(
        [("X", 35, 1, 0, 1.0, 1)],
        {"X": 0.01}, {"X": 1}, cons(("X", 0.0, 1)), 0.35,
    )
    assert alloc_of(out, "X") == 35
    assert reason_of(out, "X") == FULLY_FULFILLED
    assert out["budget_spent_today"].iloc[0] == 0.35
    assert out["budget_remaining_after"].iloc[0] == 0.0


def test_t2_multi_sku_cumulative_no_silent_underallocation():
    rows = [(f"B{i}", 10, i + 1, i, 1.0, 1) for i in range(6)]
    costs = {f"B{i}": 0.01 for i in range(6)}
    prio = {f"B{i}": i + 1 for i in range(6)}
    c = cons(*[(f"B{i}", 0.0, 1) for i in range(6)])
    out = run(rows, costs, prio, c, 0.6)
    assert set(out["allocation_reason"]) == {FULLY_FULFILLED}
    assert out["allocated_qty"].tolist() == [10] * 6
    assert out["budget_remaining_after"].iloc[0] == 0.0


def test_t3_c0_multi_sku_exact_equality_full_allocation():
    costs = {"C0_0": 43.09, "C0_1": 4.42, "C0_2": 4.87, "C0_3": 28.01, "C0_4": 16.32}
    prio = {"C0_0": 1, "C0_1": 2, "C0_2": 3, "C0_3": 4, "C0_4": 5}
    qs = {"C0_0": 85, "C0_1": 30, "C0_2": 12, "C0_3": 6, "C0_4": 6}
    mult = {"C0_0": 5, "C0_1": 5, "C0_2": 1, "C0_3": 1, "C0_4": 3}
    rows = [(s, qs[s], prio[s], i, 1.0, 1) for i, s in enumerate(costs)]
    c = cons(*[(s, 0.0, mult[s]) for s in costs])
    out = run(rows, costs, prio, c, 4119.67)
    # 末项此前被浮点少买一包（3 而非 6）
    assert alloc_of(out, "C0_4") == 6
    assert set(out["allocation_reason"]) == {FULLY_FULFILLED}
    assert out["budget_remaining_after"].iloc[0] == 0.0


def test_t4_spurious_min_pack_gone():
    costs = {"C3_0": 26.73, "C3_1": 20.54, "C3_2": 1.07}
    prio = {"C3_0": 1, "C3_1": 2, "C3_2": 3}
    qs = {"C3_0": 21, "C3_1": 30, "C3_2": 1}
    mult = {"C3_0": 3, "C3_1": 3, "C3_2": 1}
    rows = [(s, qs[s], prio[s], i, 1.0, 1) for i, s in enumerate(costs)]
    c = cons(*[(s, 0.0, mult[s]) for s in costs])
    out = run(rows, costs, prio, c, 1178.60)
    assert BUDGET_INSUFFICIENT_MIN_PACK not in set(out["allocation_reason"])
    assert set(out["allocation_reason"]) == {FULLY_FULFILLED}
    assert alloc_of(out, "C3_2") == 1


# ------------------------- T5-T6 exact / ±1 cent ----------------------------

def test_t5_exact_minus_and_plus_one_cent():
    rows = [("A", 10, 1, 0, 1.0, 1), ("B", 10, 2, 0, 1.0, 1)]
    costs = {"A": 0.10, "B": 0.20}
    prio = {"A": 1, "B": 2}
    c = cons(("A", 0.0, 1), ("B", 0.0, 1))

    exact = run(rows, costs, prio, c, 3.00)
    assert exact["allocated_qty"].tolist() == [10, 10]
    assert exact["budget_remaining_after"].iloc[0] == 0.0

    minus = run(rows, costs, prio, c, 2.99)
    assert alloc_of(minus, "A") == 10          # 1.00
    assert alloc_of(minus, "B") == 9           # floor(1.99/0.20)=9
    assert reason_of(minus, "B") == PARTIALLY_FULFILLED
    assert minus["budget_spent_today"].iloc[0] == pytest.approx(2.80)
    assert minus["budget_remaining_after"].iloc[0] == pytest.approx(0.19)

    plus = run(rows, costs, prio, c, 3.01)
    assert plus["allocated_qty"].tolist() == [10, 10]
    assert plus["budget_remaining_after"].iloc[0] == pytest.approx(0.01)


# ---------------- T7 现有契约允许的 >2 位小数 / 次分输入 --------------------

def test_t7_contract_allowed_multidecimal_inputs_are_not_narrowed():
    # 单价 12.345
    out = run([("X", 1, 1, 0, 1.0, 1)], {"X": 12.345}, {"X": 1},
              cons(("X", 0.0, 1)), 100.0)
    assert alloc_of(out, "X") == 1
    assert out["budget_spent_today"].iloc[0] == pytest.approx(12.345)
    assert out["budget_remaining_after"].iloc[0] == pytest.approx(87.655)

    # 次分单价 0.001
    out2 = run([("X", 3, 1, 0, 1.0, 1)], {"X": 0.001}, {"X": 1},
               cons(("X", 0.0, 1)), 1.0)
    assert alloc_of(out2, "X") == 3
    assert out2["budget_spent_today"].iloc[0] == pytest.approx(0.003)

    # 次分预算 0.0049
    out3 = run([("X", 3, 1, 0, 1.0, 1)], {"X": 0.001}, {"X": 1},
               cons(("X", 0.0, 1)), 0.0049)
    assert alloc_of(out3, "X") == 3
    assert out3["budget_remaining_after"].iloc[0] == pytest.approx(0.0019)


# --------------------------- T8-T11 MOQ / 倍数 ------------------------------

def test_t8_moq_infeasible_x_set():
    out = run([("A", 120, 1, 0, 1.0, 1)], {"A": 10.0}, {"A": 1},
              cons(("A", 100.0, 30)), 1100.0)
    assert reason_of(out, "A") == MOQ_INFEASIBLE
    assert alloc_of(out, "A") == 0


def test_t9_r1_scaled_to_zero_unchanged():
    out = run([("A", 100, 1, 0, 1.0, 1), ("B", 100, 2, 0, 1.0, 1)],
              {"A": 1.0, "B": 1.0}, {"A": 1, "B": 2},
              cons(("A", 1.0, 1), ("B", 1.0, 1)), 1.0, algorithm="R1")
    assert set(out["allocation_reason"]) == {"SCALED_TO_ZERO"}
    assert out["allocated_qty"].tolist() == [0, 0]


def test_t10_r2_pack_multiple_and_min_pack_guard():
    # MOQ=100,M=30 -> m=120; B'=1300 -> floor(1300/300)=4 -> q=120 (不是误报 MIN_PACK)
    out = run([("A", 150, 1, 0, 1.0, 1)], {"A": 10.0}, {"A": 1},
              cons(("A", 100.0, 30)), 1300.0)
    assert alloc_of(out, "A") == 120
    assert reason_of(out, "A") == PARTIALLY_FULFILLED
    assert out["budget_remaining_after"].iloc[0] == pytest.approx(100.0)


# ----------------------------- T12 dtypes / rank ----------------------------

def test_t12_dtypes_and_rank_coverage():
    rows = [("A", 10, 1, 5, 1.0, 1), ("B", 10, 2, 4, 1.0, 1), ("C", 10, 1, 3, 1.0, 1)]
    costs = {"A": 1.0, "B": 1.0, "C": 1.0}
    prio = {"A": 1, "B": 2, "C": 1}
    c = cons(("A", 1.0, 1), ("B", 1.0, 1), ("C", 1.0, 1))
    out = run(rows, costs, prio, c, 1000.0)
    assert out["allocation_rank"].dtype == "int64"
    assert out["allocated_qty"].dtype == "int64"
    assert out["priority"].dtype == "int64"
    for col in ("unit_cost", "budget_available_before", "budget_spent_today",
                "budget_remaining_after"):
        assert out[col].dtype == "float64"
    assert sorted(out["allocation_rank"].tolist()) == [1, 2, 3]


# ------------------------------ T13-T16 其他 --------------------------------

def test_t13_b_avail_zero_short_circuit():
    out = run([("A", 100, 1, 0, 1.0, 1), ("B", 0, 1, 0, 1.0, 1)],
              {"A": 10.0, "B": 1.0}, {"A": 1, "B": 1},
              cons(("A", 0.0, 1), ("B", 0.0, 1)), 0.0)
    assert reason_of(out, "A") == BUDGET_EXHAUSTED
    assert reason_of(out, "B") == NO_GAP
    assert out["budget_spent_today"].iloc[0] == 0.0
    assert out["budget_remaining_after"].iloc[0] == 0.0


def test_t14_huge_values():
    out = run([("X", 1000, 1, 0, 1.0, 1)], {"X": 1e9}, {"X": 1},
              cons(("X", 0.0, 1)), 1e12)
    assert alloc_of(out, "X") == 1000
    assert reason_of(out, "X") == FULLY_FULFILLED
    assert out["budget_remaining_after"].iloc[0] == 0.0


def test_t14_scientific_notation_negative_exponent():
    out = run([("X", 10_000_000, 1, 0, 1.0, 1)], {"X": 1e-07}, {"X": 1},
              cons(("X", 0.0, 1)), 1.0)
    assert alloc_of(out, "X") == 10_000_000
    assert reason_of(out, "X") == FULLY_FULFILLED
    assert out["budget_remaining_after"].iloc[0] == 0.0


def test_t14_subnormal_tiny_positive_cost():
    out = run([("X", 1, 1, 0, 1.0, 1)], {"X": 5e-324}, {"X": 1},
              cons(("X", 0.0, 1)), 1e-320)
    assert alloc_of(out, "X") == 1
    assert reason_of(out, "X") == FULLY_FULFILLED
    assert out["budget_remaining_after"].iloc[0] >= 0.0


def test_t16_input_frame_not_mutated():
    rows = [("A", 35, 1, 0, 1.0, 1)]
    f = frame(rows)
    before = f.copy()
    allocate(f, algorithm="R2", budget_available=0.35, costs={"A": 0.01},
             priority={"A": 1}, constraints=cons(("A", 0.0, 1)))
    assert f.equals(before)


# -------------------- T17 随机 oracle 交叉验证（Fraction） -------------------

def test_t17_random_cases_match_fraction_oracle():
    rng = random.Random(20251015)
    checked = 0
    for _ in range(300):
        n = rng.randint(2, 5)
        skus = [f"S{i}" for i in range(n)]
        rows, costs, prio, crows = [], {}, {}, []
        for i, s in enumerate(skus):
            mult = rng.choice([1, 2, 3, 4, 5, 8, 10])
            moq = 0.0 if rng.random() < 0.5 else float(rng.randint(0, 4) * mult)
            q = (mult * rng.randint(1, 20)) if rng.random() > 0.1 else 0
            if q and q < moq:
                q = math.ceil(moq / mult) * mult
            cost = round(rng.uniform(0.01, 20.0), 2)
            rows.append((s, q, i + 1, i, 1.0, 1))
            costs[s] = cost
            prio[s] = i + 1
            crows.append((s, moq, mult))
        total = sum(_frac(costs[s]) * r[1] for s, r in zip(skus, rows))
        mode = rng.random()
        if mode < 0.5:
            budget = float(total)
        elif mode < 0.8:
            budget = float(total) - 0.01
        else:
            budget = float(total) + rng.uniform(0.0, 1.0)
        budget = max(0.0, budget)

        out = run(rows, costs, prio, cons(*crows), budget)
        constraints = {s: (m, M) for s, m, M in crows}
        exp_alloc, exp_reason, exp_remaining = reference_r2(rows, costs, constraints, budget)

        for s in skus:
            assert alloc_of(out, s) == exp_alloc[s], (s, costs, budget, exp_alloc, out.to_dict())
            assert reason_of(out, s) == exp_reason[s]
        assert _frac(out["budget_remaining_after"].iloc[0]) == exp_remaining or \
            abs(float(out["budget_remaining_after"].iloc[0]) - float(exp_remaining)) < 1e-9
        # 独立支出核验：Σ cost*qi == budget - remaining（Fraction 精确）
        spent = sum(_frac(costs[s]) * exp_alloc[s] for s in skus)
        assert spent == _frac(budget) - exp_remaining
        checked += 1
    assert checked == 300
