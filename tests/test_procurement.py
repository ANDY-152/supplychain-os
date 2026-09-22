"""Run: pytest tests/test_procurement.py  |  python -m tests.test_procurement  (from the repo root)"""
from datetime import date

import pandas as pd

from app.data_loader import (
    load_data_metadata,
    load_in_transit,
    load_inventory,
    load_orders,
    load_procurement_constraints,
)
from app.demand import calculate_demand
from app.inventory import analyze_inventory_frame
from app.procurement import (
    GAP_ONLY,
    MOQ_AND_MULTIPLE_APPLIED,
    MOQ_APPLIED,
    MULTIPLE_APPLIED,
    NO_GAP,
    apply_procurement,
)

AS_OF = date(2026, 9, 20)  # 固定基准日，测试不依赖系统时间


def constraints(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["sku", "moq", "order_multiple"])


def level1(recommended: float, *, shortage: float | None = None, quality: str = "OK", sku: str = "SKU001"):
    """最小 Level 1 结果帧（字段齐全，便于验证未被改写）。"""
    if shortage is None:
        shortage = recommended
    return pd.DataFrame(
        [
            {
                "sku": sku,
                "current_stock": 50.0,
                "reorder_point": 150.0,
                "inventory_position": 50.0,
                "shortage_qty": float(shortage),
                "recommended_order_qty": float(recommended),
                "in_transit_stock": 0.0,
                "status": "REORDER",
                "reason": "STOCK_BELOW_REORDER_POINT",
                "data_quality_status": quality,
            }
        ]
    )


def recommend(recommended, moq=0, multiple=1, *, quality="OK", shortage=None):
    frame = level1(recommended, shortage=shortage, quality=quality)
    return apply_procurement(frame, constraints(("SKU001", moq, multiple))).iloc[0]


# --- 1 / 2 gap <= 0 ---------------------------------------------------------

def test_gap_zero_is_no_gap():
    r = recommend(0)
    assert r.procurement_recommended_qty == 0
    assert r.procurement_reason == NO_GAP


def test_negative_shortage_yields_no_gap():
    r = recommend(0, shortage=-50.0)
    assert r.procurement_recommended_qty == 0
    assert r.procurement_reason == NO_GAP


# --- 3 / 19 / 20 identity ---------------------------------------------------

def test_identity_when_sku_has_no_constraint_row():
    r = apply_procurement(level1(120), constraints()).iloc[0]
    assert r.procurement_recommended_qty == r.recommended_order_qty == 120
    assert r.procurement_reason == GAP_ONLY


def test_missing_constraints_is_identity():
    for cons in (None, constraints()):
        r = apply_procurement(level1(120), cons).iloc[0]
        assert r.procurement_recommended_qty == 120
        assert r.procurement_reason == GAP_ONLY


def test_zero_moq_and_unit_multiple_is_identity():
    r = recommend(120, moq=0, multiple=1)
    assert r.procurement_recommended_qty == 120
    assert r.procurement_reason == GAP_ONLY


# --- 4 / 5 MOQ --------------------------------------------------------------

def test_gap_below_moq_is_raised_to_moq():
    r = recommend(80, moq=100)
    assert r.procurement_recommended_qty == 100
    assert r.procurement_reason == MOQ_APPLIED


def test_gap_above_moq_is_unchanged_and_not_marked_applied():
    r = recommend(150, moq=100)
    assert r.procurement_recommended_qty == 150
    assert r.procurement_reason == GAP_ONLY  # MOQ 存在但未改变数量


def test_moq_equal_to_gap_is_not_marked_applied():
    r = recommend(100, moq=100)
    assert r.procurement_recommended_qty == 100
    assert r.procurement_reason == GAP_ONLY


# --- 6 / 7 / 8 multiple -----------------------------------------------------

def test_multiple_rounds_up():
    r = recommend(120, multiple=50)
    assert r.procurement_recommended_qty == 150
    assert r.procurement_reason == MULTIPLE_APPLIED


def test_moq_then_multiple_applies_both():
    # 80 -> MOQ 90 -> ceil(90/50)*50 = 100
    r = recommend(80, moq=90, multiple=50)
    assert r.procurement_recommended_qty == 100
    assert r.procurement_reason == MOQ_AND_MULTIPLE_APPLIED


def test_multiple_one_is_identity():
    r = recommend(120, multiple=1)
    assert r.procurement_recommended_qty == 120
    assert r.procurement_reason == GAP_ONLY


# --- 15 / 16 provisional ----------------------------------------------------

def test_provisional_false_when_quality_ok():
    r = recommend(80, moq=100, quality="OK")
    assert r.provisional == False  # noqa: E712 - 明确的布林断言
    assert r.procurement_recommended_qty == 100


def test_provisional_true_but_quantity_unchanged():
    ok = recommend(80, moq=100, quality="OK")
    short = recommend(80, moq=100, quality="INSUFFICIENT_HISTORY")
    unknown = recommend(80, moq=100, quality="UNKNOWN_HISTORY")
    assert short.provisional == True and unknown.provisional == True  # noqa: E712
    assert short.procurement_recommended_qty == ok.procurement_recommended_qty
    assert unknown.procurement_recommended_qty == ok.procurement_recommended_qty


# --- 17 / 18 Level 1 不变 + Level 2 可大于 Level 1 ---------------------------

def test_level1_columns_unchanged_and_input_not_mutated():
    frame = level1(80)
    out = apply_procurement(frame, constraints(("SKU001", 100, 50)))
    for column in (
        "reorder_point",
        "inventory_position",
        "shortage_qty",
        "recommended_order_qty",
        "current_stock",
        "in_transit_stock",
        "status",
        "reason",
        "data_quality_status",
    ):
        assert out.iloc[0][column] == frame.iloc[0][column], column
    assert "procurement_recommended_qty" not in frame.columns  # 入参未被改写


def test_procurement_can_exceed_level1():
    r = recommend(80, moq=100)
    assert r.procurement_recommended_qty > r.recommended_order_qty


def test_real_pipeline_level1_numeric_unchanged():
    demand = calculate_demand(load_orders(), AS_OF, coverage_start=load_data_metadata())
    before = analyze_inventory_frame(
        load_inventory(), demand, load_in_transit(), as_of_date=AS_OF
    ).frame
    after = apply_procurement(before, load_procurement_constraints())
    assert len(after) == len(before)
    for column in ("reorder_point", "inventory_position", "shortage_qty", "recommended_order_qty"):
        assert after[column].tolist() == before[column].tolist(), column


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
