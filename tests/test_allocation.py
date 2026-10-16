"""Level 3 allocation tests. Run: pytest tests/test_allocation.py

契约来源：`supplychain-os-Level3预算分配-设计方案-v4.4.md`。
本文件为新增测试；不修改任何既有测试。
"""
import pandas as pd
import pytest

from app.allocation import (
    BUDGET_EXHAUSTED,
    BUDGET_INSUFFICIENT_MIN_PACK,
    FULLY_FULFILLED,
    MOQ_INFEASIBLE,
    NO_GAP,
    PARTIALLY_FULFILLED,
    ROLLED_BACK,
    SCALED_TO_ZERO,
    allocate,
)

R12_COLUMNS = [
    "allocation_rank",
    "allocated_qty",
    "allocation_reason",
    "unit_cost",
    "priority",
    "budget_available_before",
    "budget_spent_today",
    "budget_remaining_after",
]

L2_COLUMNS = [
    "sku",
    "procurement_recommended_qty",
    "current_stock",
    "daily_demand",
    "lead_time_days",
    "reason",
    "procurement_reason",
]


def l2_frame(*rows, **extra):
    """rows: (sku, q, current_stock, daily_demand, lead_time_days)."""
    data = {
        "sku": [r[0] for r in rows],
        "procurement_recommended_qty": [float(r[1]) for r in rows],
        "current_stock": [r[2] for r in rows],
        "daily_demand": [float(r[3]) for r in rows],
        "lead_time_days": [r[4] for r in rows],
        "reason": ["STOCK_BELOW_REORDER_POINT"] * len(rows),
        "procurement_reason": ["MOQ_APPLIED"] * len(rows),
    }
    data.update(extra)
    return pd.DataFrame(data)[L2_COLUMNS]


def cons(*rows):
    return pd.DataFrame(rows, columns=["sku", "moq", "order_multiple"])


def reason_of(out, sku):
    return out.loc[out["sku"] == sku, "allocation_reason"].iloc[0]


def alloc_of(out, sku):
    return int(out.loc[out["sku"] == sku, "allocated_qty"].iloc[0])


# --- R0: field invariance (C-3 / A6) ---------------------------------------

def test_r0_appends_only_two_columns_and_preserves_l1_l2():
    frame = l2_frame(("A", 100, 10, 2.0, 3), ("B", 0, 50, 5.0, 1))
    out = allocate(
        frame, algorithm="R0", budget_available=0.0, costs=None, priority=None, constraints=None
    )
    assert list(out.columns) == L2_COLUMNS + ["allocated_qty", "allocation_reason"]
    assert out[L2_COLUMNS].equals(frame)  # 值、dtype、顺序全部不变
    assert alloc_of(out, "A") == 100
    assert alloc_of(out, "B") == 0
    assert reason_of(out, "A") == FULLY_FULFILLED
    assert reason_of(out, "B") == NO_GAP
    # C-3：原值透传，值与 dtype 均与 L2 列一致
    assert out["allocated_qty"].tolist() == frame["procurement_recommended_qty"].tolist()
    assert out["allocated_qty"].dtype == frame["procurement_recommended_qty"].dtype
    for column in R12_COLUMNS:
        if column not in ("allocated_qty", "allocation_reason"):
            assert column not in out.columns


def test_r0_does_not_require_cost_priority_or_constraints():
    frame = l2_frame(("A", 40, 1, 1.0, 1))
    out = allocate(
        frame, algorithm="R0", budget_available=None, costs=None, priority=None, constraints=None
    )
    assert alloc_of(out, "A") == 40


def test_r0_passes_through_non_integer_quantity():
    # C-3：R0 不校验整数，原值透传
    frame = l2_frame(("A", 10.5, 1, 1.0, 1))
    out = allocate(frame, algorithm="R0", budget_available=0.0, costs=None, priority=None, constraints=None)
    assert out["allocated_qty"].iloc[0] == 10.5
    assert float(out["allocated_qty"].iloc[0]) == 10.5
    assert reason_of(out, "A") == FULLY_FULFILLED


def test_r1_r2_reject_non_integer_quantity():
    # R1/R2 仍必须整数校验，不得放宽
    frame = l2_frame(("A", 10.5, 1, 1.0, 1))
    for algorithm in ("R1", "R2"):
        try:
            allocate(
                frame, algorithm=algorithm, budget_available=100.0,
                costs={"A": 1.0}, priority={"A": 1}, constraints=None,
            )
        except ValueError as e:
            assert "integer" in str(e)
        else:
            raise AssertionError(f"{algorithm} must reject non-integer q_i^L2")


# --- R1/R2: priority and cost requirements (C-1) ---------------------------

def test_r1_requires_complete_priority():
    frame = l2_frame(("A", 10, 1, 1.0, 1), ("B", 10, 1, 1.0, 1))
    costs = {"A": 1.0, "B": 1.0}
    try:
        allocate(frame, algorithm="R1", budget_available=100.0, costs=costs, priority={"A": 1}, constraints=None)
    except ValueError as e:
        assert "priority" in str(e)
        return
    raise AssertionError("missing priority must raise")


def test_r2_requires_complete_priority():
    frame = l2_frame(("A", 10, 1, 1.0, 1))
    try:
        allocate(frame, algorithm="R2", budget_available=100.0, costs={"A": 1.0}, priority={}, constraints=None)
    except ValueError as e:
        assert "priority" in str(e)
        return
    raise AssertionError("missing priority must raise")


def test_priority_must_be_integer():
    frame = l2_frame(("A", 10, 1, 1.0, 1))
    try:
        allocate(frame, algorithm="R2", budget_available=100.0, costs={"A": 1.0}, priority={"A": 1.5}, constraints=None)
    except ValueError as e:
        assert "priority" in str(e)
        return
    raise AssertionError("non-integer priority must raise")


@pytest.mark.parametrize("algorithm", ["R1", "R2"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_priority_must_be_finite(algorithm, bad):
    # T14：非有限 priority 必须 fail-loud（契约 v1.3 §13 T14）
    frame = l2_frame(("A", 10, 1, 1.0, 1))
    with pytest.raises(ValueError) as excinfo:
        allocate(
            frame, algorithm=algorithm, budget_available=100.0,
            costs={"A": 1.0}, priority={"A": bad}, constraints=None,
        )
    assert "priority" in str(excinfo.value)


def test_r0_ignores_non_finite_priority():
    # T14：R0 不要求/不读取 priority，非有限值也不得影响结果
    frame = l2_frame(("A", 10, 1, 1.0, 1))
    out = allocate(
        frame, algorithm="R0", budget_available=0.0,
        costs=None, priority={"A": float("inf")}, constraints=None,
    )
    assert alloc_of(out, "A") == 10


def test_cost_must_be_present_and_positive():
    frame = l2_frame(("A", 10, 1, 1.0, 1))
    for costs in ({"B": 1.0}, {"A": 0.0}, {"A": -1.0}, {"A": float("nan")}):
        try:
            allocate(frame, algorithm="R1", budget_available=100.0, costs=costs, priority={"A": 1}, constraints=None)
        except ValueError as e:
            assert "unit_cost" in str(e)
        else:
            raise AssertionError(f"bad costs must raise: {costs}")


# --- MOQ / multiple validation (C1/C6) -------------------------------------

def test_moq_multiple_contract_validation():
    frame = l2_frame(("A", 100, 1, 1.0, 1))  # 100 不是 30 的倍数
    c = cons(("A", 100.0, 30))
    raised = False
    try:
        allocate(frame, algorithm="R1", budget_available=9999.0, costs={"A": 10.0}, priority={"A": 1}, constraints=c)
    except ValueError as e:
        assert "procurement_recommended_qty" in str(e)
        raised = True
    assert raised, "q_i^L2 not multiple of M_i must raise"

    ok = l2_frame(("A", 120, 1, 1.0, 1))
    out = allocate(ok, algorithm="R1", budget_available=9999.0, costs={"A": 10.0}, priority={"A": 1}, constraints=c)
    assert alloc_of(out, "A") == 120


# --- C1 numeric example: MOQ=100, M=30, c=10 -> m_i=120, X threshold=1200 --

def test_c1_x_threshold_uses_c_times_m():
    frame = l2_frame(("A", 120, 1, 1.0, 1))  # 120 = 4*30，满足 MOQ=100 与倍数 30
    c = cons(("A", 100.0, 30))
    out = allocate(frame, algorithm="R1", budget_available=1100.0, costs={"A": 10.0}, priority={"A": 1}, constraints=c)
    assert reason_of(out, "A") == MOQ_INFEASIBLE  # c*m = 10*120 = 1200 > 1100
    assert alloc_of(out, "A") == 0


# --- B_avail == 0 short-circuit (Y4) ---------------------------------------

def test_b_avail_zero_short_circuit():
    frame = l2_frame(("A", 100, 1, 1.0, 1), ("B", 0, 1, 1.0, 1))
    c = cons(("A", 0.0, 1), ("B", 0.0, 1))
    for algorithm in ("R1", "R2"):
        out = allocate(
            frame, algorithm=algorithm, budget_available=0.0,
            costs={"A": 10.0, "B": 1.0}, priority={"A": 1, "B": 1}, constraints=c,
        )
        assert reason_of(out, "A") == BUDGET_EXHAUSTED
        assert reason_of(out, "B") == NO_GAP
        assert alloc_of(out, "A") == 0
        assert out["budget_spent_today"].iloc[0] == 0.0
        assert out["budget_remaining_after"].iloc[0] == 0.0


# --- R1 invariants and reason codes ----------------------------------------

def test_r1_budget_invariant_and_partial():
    frame = l2_frame(("A", 100, 1, 1.0, 1), ("B", 100, 1, 1.0, 1))
    c = cons(("A", 1.0, 1), ("B", 1.0, 1))
    out = allocate(frame, algorithm="R1", budget_available=90.0, costs={"A": 10.0, "B": 10.0}, priority={"A": 1, "B": 2}, constraints=c)
    spent = out["budget_spent_today"].iloc[0]
    assert spent <= 90.0
    # λ = 90 / (10*100 + 10*100) = 0.045 -> floor(100*0.045) = 4
    assert alloc_of(out, "A") == 4 and alloc_of(out, "B") == 4
    assert reason_of(out, "A") == PARTIALLY_FULFILLED
    assert ROLLED_BACK not in set(out["allocation_reason"])


def test_r1_scaled_to_zero():
    frame = l2_frame(("A", 100, 1, 1.0, 1), ("B", 100, 1, 1.0, 1))
    c = cons(("A", 1.0, 1), ("B", 1.0, 1))
    out = allocate(frame, algorithm="R1", budget_available=1.0, costs={"A": 1.0, "B": 1.0}, priority={"A": 1, "B": 2}, constraints=c)
    assert set(out["allocation_reason"]) == {SCALED_TO_ZERO}
    assert out["allocated_qty"].tolist() == [0, 0]
    assert ROLLED_BACK not in set(out["allocation_reason"])


def test_r1_below_min_order_qty_zeroed_not_raised():
    frame = l2_frame(("A", 100, 1, 1.0, 1), ("B", 100, 1, 1.0, 1))
    c = cons(("A", 10.0, 1), ("B", 10.0, 1))
    out = allocate(frame, algorithm="R1", budget_available=10.0, costs={"A": 1.0, "B": 1.0}, priority={"A": 1, "B": 2}, constraints=c)
    # λ=10/200=0.05 -> floor(100*0.05)=5 -> 0<5<10 -> SCALED_TO_ZERO
    assert set(out["allocation_reason"]) == {SCALED_TO_ZERO}


def test_r1_never_emits_min_pack_code():
    frame = l2_frame(("A", 50, 1, 1.0, 1))
    c = cons(("A", 10.0, 5))
    out = allocate(frame, algorithm="R1", budget_available=3.0, costs={"A": 1.0}, priority={"A": 1}, constraints=c)
    assert BUDGET_INSUFFICIENT_MIN_PACK not in set(out["allocation_reason"])


def test_constraints_missing_row_defaults_to_identity():
    frame = l2_frame(("A", 7, 1, 1.0, 1))
    out = allocate(frame, algorithm="R1", budget_available=100.0, costs={"A": 1.0}, priority={"A": 1}, constraints=None)
    assert alloc_of(out, "A") == 7
    assert reason_of(out, "A") == FULLY_FULFILLED


# --- R2 reason codes --------------------------------------------------------

def test_r2_min_pack_and_guard_no_false_positive():
    frame = l2_frame(("A", 150, 1, 1.0, 1))  # 150 = 5*30
    c = cons(("A", 100.0, 30))
    # B'=1300: floor(1300/300)=4, 150//30=5 -> packs=4 -> q=120 (<150); 4*30=120>=120
    out = allocate(frame, algorithm="R2", budget_available=1300.0, costs={"A": 10.0}, priority={"A": 1}, constraints=c)
    assert alloc_of(out, "A") == 120
    assert reason_of(out, "A") == PARTIALLY_FULFILLED  # 旧 k_i<m_i 守卫会误报 MIN_PACK

    # BUDGET_INSUFFICIENT_MIN_PACK 只可能发生在 E 且 B' 已被先前分配压低之后
    frame2 = l2_frame(("A", 150, 1, 1.0, 1), ("B", 150, 1, 1.0, 1))
    c2 = cons(("A", 100.0, 30), ("B", 100.0, 30))
    out2 = allocate(
        frame2, algorithm="R2", budget_available=1300.0,
        costs={"A": 10.0, "B": 10.0}, priority={"A": 1, "B": 2}, constraints=c2,
    )  # A: q=120 cost=1200, B'=100 < c*m=1200 -> MIN_PACK
    assert reason_of(out2, "A") == PARTIALLY_FULFILLED
    assert reason_of(out2, "B") == BUDGET_INSUFFICIENT_MIN_PACK


def test_r2_budget_exhausted_after_spending():
    frame = l2_frame(("A", 100, 1, 1.0, 1), ("B", 100, 1, 1.0, 1))
    c = cons(("A", 1.0, 1), ("B", 1.0, 1))
    out = allocate(frame, algorithm="R2", budget_available=100.0, costs={"A": 1.0, "B": 1.0}, priority={"A": 1, "B": 2}, constraints=c)
    assert reason_of(out, "A") == FULLY_FULFILLED
    assert reason_of(out, "B") == BUDGET_EXHAUSTED


def test_r2_does_not_emit_scaled_or_rolled():
    frame = l2_frame(("A", 100, 1, 1.0, 1), ("B", 100, 1, 1.0, 1))
    c = cons(("A", 1.0, 1), ("B", 1.0, 1))
    out = allocate(frame, algorithm="R2", budget_available=1.0, costs={"A": 1.0, "B": 1.0}, priority={"A": 1, "B": 2}, constraints=c)
    assert SCALED_TO_ZERO not in set(out["allocation_reason"])
    assert ROLLED_BACK not in set(out["allocation_reason"])


# --- rank, determinism, dtype ----------------------------------------------

def test_allocation_rank_covers_all_and_is_unique():
    frame = l2_frame(("A", 10, 1, 1.0, 1), ("B", 10, 1, 1.0, 1), ("C", 10, 1, 1.0, 1))
    c = cons(("A", 1.0, 1), ("B", 1.0, 1), ("C", 1.0, 1))
    out = allocate(
        frame, algorithm="R2", budget_available=1000.0,
        costs={"A": 1.0, "B": 1.0, "C": 1.0}, priority={"A": 1, "B": 2, "C": 1}, constraints=c,
    )
    assert sorted(out["allocation_rank"].tolist()) == [1, 2, 3]
    assert out["allocation_rank"].dtype == "int64"
    assert out["allocated_qty"].dtype == "int64"
    assert R12_COLUMNS == [c for c in out.columns if c in R12_COLUMNS]


def test_row_order_does_not_change_result():
    frame = l2_frame(("A", 10, 5, 1.0, 1), ("B", 10, 4, 1.0, 1), ("C", 10, 3, 1.0, 1))
    c = cons(("A", 1.0, 1), ("B", 1.0, 1), ("C", 1.0, 1))
    kwargs = dict(
        algorithm="R1", budget_available=12.0,
        costs={"A": 1.0, "B": 1.0, "C": 1.0}, priority={"A": 1, "B": 1, "C": 1}, constraints=c,
    )
    a = allocate(frame, **kwargs)
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    b = allocate(shuffled, **kwargs)
    for sku in ("A", "B", "C"):
        assert alloc_of(a, sku) == alloc_of(b, sku)
        assert reason_of(a, sku) == reason_of(b, sku)
        assert int(a.loc[a["sku"] == sku, "allocation_rank"].iloc[0]) == int(b.loc[b["sku"] == sku, "allocation_rank"].iloc[0])


def test_rank_uses_raw_coverage_not_rounded_column(monkeypatch):
    # 两 SKU 的 coverage_days 显示列相同，但原始 current_stock/daily_demand 不同。
    frame = l2_frame(("SKU_A", 10, 100, 3.0, 0), ("SKU_B", 10, 99, 3.0, 0))
    frame["coverage_days"] = [33.33, 33.33]  # 舍入列故意相同
    c = cons(("SKU_A", 1.0, 1), ("SKU_B", 1.0, 1))
    out = allocate(
        frame, algorithm="R2", budget_available=1000.0,
        costs={"SKU_A": 1.0, "SKU_B": 1.0}, priority={"SKU_A": 1, "SKU_B": 1}, constraints=c,
    )
    rank_a = int(out.loc[out["sku"] == "SKU_A", "allocation_rank"].iloc[0])
    rank_b = int(out.loc[out["sku"] == "SKU_B", "allocation_rank"].iloc[0])
    # 原始 CD: B=33.0 < A=33.333 -> B 更紧迫（rank 更小）。若用舍入列则两者相等、退化为 sku 升序。
    assert rank_b < rank_a


def test_negative_budget_rejected():
    frame = l2_frame(("A", 10, 1, 1.0, 1))
    try:
        allocate(frame, algorithm="R1", budget_available=-1.0, costs={"A": 1.0}, priority={"A": 1}, constraints=None)
    except ValueError as e:
        assert "budget_available" in str(e)
        return
    raise AssertionError("negative budget must raise")


def test_unknown_algorithm_rejected():
    frame = l2_frame(("A", 10, 1, 1.0, 1))
    try:
        allocate(frame, algorithm="R9", budget_available=1.0, costs={}, priority={}, constraints=None)
    except ValueError as e:
        assert "algorithm" in str(e)
        return
    raise AssertionError("unknown algorithm must raise")


def test_input_frame_not_mutated():
    frame = l2_frame(("A", 10, 1, 1.0, 1))
    before = frame.copy()
    allocate(frame, algorithm="R1", budget_available=100.0, costs={"A": 1.0}, priority={"A": 1}, constraints=None)
    assert frame.equals(before)


@pytest.mark.parametrize("algorithm", ["R1", "R2"])
def test_empty_frame_returns_structurally_valid_l3_result(algorithm):
    # 空帧（当日全部 SKU 被跳过）：不得因缺 priority 等列而 KeyError
    frame = l2_frame()
    assert len(frame) == 0
    out = allocate(
        frame, algorithm=algorithm, budget_available=100.0,
        costs={}, priority={}, constraints=None,
    )
    assert len(out) == 0
    for column in R12_COLUMNS:
        assert column in out.columns
    assert out["allocated_qty"].dtype == "int64"
    assert out["budget_available_before"].tolist() == []
