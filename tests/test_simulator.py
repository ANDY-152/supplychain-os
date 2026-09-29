"""Run: python -m pytest tests/test_simulator.py  |  python -m tests.test_simulator

验证 app/simulator.py 的冻结语义：每日顺序、到货、需求扣减、决策委托、下单、审计、确定性。
"""
import ast
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from app.audit_log import BUSINESS_COLUMNS
from app.simulator import (
    DEMAND_PROFILE_COLUMNS,
    DailyResult,
    SimulationResult,
    run_simulation,
    simulate_day,
)

START = date(2026, 9, 20)
COVERAGE_START = START - timedelta(days=29)  # 恰好 30 天 -> data_quality OK
NO_CONSTRAINTS = pd.DataFrame(columns=["sku", "moq", "order_multiple"])
APP_DIR = Path(__file__).resolve().parent.parent / "app"

EXPECTED_DECISION_COLUMNS = [
    "sku",
    "current_stock",
    "daily_demand",
    "safety_stock",
    "lead_time_days",
    "coverage_days",
    "reorder_point",
    "status",
    "recommended_order_qty",
    "in_transit_stock",
    "overdue_in_transit",
    "inventory_position",
    "reason",
    "shortage_qty",
    "as_of_date",
    "demand_basis",
    "daily_demand_30d",
    "daily_demand_7d",
    "demand_trend",
    "history_days",
    "order_active_days",
    "data_quality_status",
    "procurement_recommended_qty",
    "provisional",
    "procurement_reason",
]


def orders_frame(*rows):
    return pd.DataFrame(rows, columns=["order_id", "order_date", "sku", "qty"])


def inventory_frame(*rows):
    return pd.DataFrame(rows, columns=["sku", "current_stock", "safety_stock", "lead_time_days"])


def transit_frame(*rows):
    return pd.DataFrame(rows, columns=["po_id", "sku", "qty", "expected_date", "status"])


def profile(*rows):
    return pd.DataFrame(rows, columns=["sku", "simulation_day", "qty"])


def order(order_id, current_day, sku, qty, days_ago=1):
    return (order_id, (current_day - timedelta(days=days_ago)).isoformat(), sku, qty)


def base_inventory(stock=0.0, safety=10.0, lead=5):
    return inventory_frame(("SKU001", stock, safety, lead))


def base_orders(current_day=START, qty=30):
    return orders_frame(order("O1", current_day, "SKU001", qty))


def run_day(
    inv,
    orders,
    transit,
    prof,
    *,
    run_id="R1",
    simulation_day=1,
    current_day=START,
    constraints=NO_CONSTRAINTS,
    coverage_start=COVERAGE_START,
    next_po_seq=0,
):
    return simulate_day(
        run_id=run_id,
        simulation_day=simulation_day,
        current_day=current_day,
        inventory_state=inv,
        transit_state=transit,
        orders_log=orders,
        demand_profile=prof,
        constraints=constraints,
        coverage_start=coverage_start,
        next_po_seq=next_po_seq,
    )


# --- 1. 单日基本流程 / 输出结构 ---------------------------------------------

def test_simulate_day_returns_daily_result():
    inventory, transit, seq, result = run_day(
        base_inventory(0.0), base_orders(), transit_frame(), profile(("SKU001", 1, 0.0))
    )
    assert isinstance(result, DailyResult)
    assert result.run_id == "R1"
    assert result.simulation_day == 1
    assert result.as_of_date == START
    assert isinstance(result.decision_frame, pd.DataFrame)
    assert result.records == [r for r in result.records if r["record_type"] in ("DECISION", "SKIP")]
    assert seq >= 0
    assert "SKU001" in inventory["sku"].tolist()
    assert list(transit.columns) == ["po_id", "sku", "qty", "expected_date", "status"]


def test_run_simulation_single_day_returns_result():
    result = run_simulation(
        run_id="R",
        start_day=START,
        days=1,
        demand_profile=profile(("SKU001", 1, 1.0)),
        orders_log=base_orders(),
        inventory_state=base_inventory(5.0),
        transit_state=transit_frame(),
        constraints=NO_CONSTRAINTS,
        coverage_start=COVERAGE_START,
    )
    assert isinstance(result, SimulationResult)
    assert result.run_id == "R"
    assert result.start_day == START
    assert result.days == 1
    assert result.coverage_start == COVERAGE_START
    assert len(result.daily) == 1
    assert result.records == result.daily[0].records


def test_decision_frame_keeps_existing_25_columns():
    _, _, _, result = run_day(
        base_inventory(0.0), base_orders(), transit_frame(), profile(("SKU001", 1, 0.0))
    )
    assert list(result.decision_frame.columns) == EXPECTED_DECISION_COLUMNS
    assert len(result.decision_frame.columns) == 25


def test_decision_frame_uses_existing_inventory_formulas():
    _, _, _, result = run_day(
        base_inventory(0.0, safety=10.0, lead=5),
        base_orders(),
        transit_frame(),
        profile(("SKU001", 1, 0.0)),
    )
    row = result.decision_frame.iloc[0]
    assert row.reorder_point == row.daily_demand * row.lead_time_days + row.safety_stock
    assert row.inventory_position == row.current_stock + row.in_transit_stock
    assert row.shortage_qty == row.reorder_point - row.inventory_position
    assert row.recommended_order_qty == max(0, row.shortage_qty)


# --- 2/3. 到货规则 ----------------------------------------------------------

def test_arrival_is_processed_before_demand():
    inventory, transit, _, result = run_day(
        base_inventory(5.0),
        base_orders(),
        transit_frame(("PO-1", "SKU001", 100, START.isoformat(), "OPEN")),
        profile(("SKU001", 1, 30.0)),
    )
    demand_row = result.demand[0]
    assert demand_row["stock_before"] == 105.0  # 到货先于需求
    assert demand_row["stock_after"] == 75.0
    assert demand_row["unmet_demand"] == 0.0
    assert result.arrivals == [{"po_id": "PO-1", "sku": "SKU001", "qty": 100.0}]


def test_arrival_boundaries_today_overdue_future():
    transit = transit_frame(
        ("PO-today", "SKU001", 10, START.isoformat(), "OPEN"),
        ("PO-future", "SKU001", 20, (START + timedelta(days=1)).isoformat(), "OPEN"),
        ("PO-overdue", "SKU001", 30, (START - timedelta(days=3)).isoformat(), "OPEN"),
    )
    _, new_transit, _, result = run_day(
        base_inventory(0.0), base_orders(), transit, profile(("SKU001", 1, 0.0))
    )
    status = new_transit.set_index("po_id")["status"].to_dict()
    assert status["PO-today"] == "ARRIVED"  # expected == current_day 当天到货
    assert status["PO-overdue"] == "ARRIVED"  # expected < current_day 也到货
    assert status["PO-future"] == "OPEN"  # expected > current_day 不到货
    assert sorted(a["po_id"] for a in result.arrivals) == ["PO-overdue", "PO-today"]


def test_arrived_po_is_not_counted_as_open_transit():
    _, _, _, result = run_day(
        base_inventory(0.0),
        base_orders(),
        transit_frame(("PO-1", "SKU001", 100, START.isoformat(), "OPEN")),
        profile(("SKU001", 1, 0.0)),
    )
    row = result.decision_frame.iloc[0]
    assert row.current_stock == 100.0
    assert row.in_transit_stock == 0.0  # 已 ARRIVED，不再算 OPEN 在途
    assert row.inventory_position == 100.0  # 只加一次


def test_open_future_po_counted_once_in_position():
    _, _, _, result = run_day(
        base_inventory(0.0),
        base_orders(),
        transit_frame(("PO-1", "SKU001", 100, (START + timedelta(days=5)).isoformat(), "OPEN")),
        profile(("SKU001", 1, 0.0)),
    )
    row = result.decision_frame.iloc[0]
    assert row.in_transit_stock == 100.0
    assert row.inventory_position == 100.0  # 不重复相加


def test_zero_lead_time_new_po_does_not_arrive_same_day():
    inventory, transit, seq, result1 = run_day(
        base_inventory(0.0, lead=0),
        base_orders(),
        transit_frame(),
        profile(("SKU001", 1, 0.0)),
    )
    assert result1.order_placed, "day 1 should place an order"
    po = result1.order_placed[0]
    assert po["expected_date"] == START.isoformat()
    assert transit.set_index("po_id").loc[po["po_id"], "status"] == "OPEN"
    assert result1.arrivals == []

    _, transit2, _, result2 = run_day(
        inventory,
        base_orders(),
        transit,
        profile(("SKU001", 1, 0.0), ("SKU001", 2, 0.0)),
        simulation_day=2,
        current_day=START + timedelta(days=1),
        next_po_seq=seq,
    )
    assert [a["po_id"] for a in result2.arrivals] == [po["po_id"]]
    assert transit2.set_index("po_id").loc[po["po_id"], "status"] == "ARRIVED"


def test_v0_never_creates_cancelled_po():
    _, transit, _, _ = run_day(
        base_inventory(0.0), base_orders(), transit_frame(), profile(("SKU001", 1, 0.0))
    )
    assert "CANCELLED" not in set(transit["status"])


# --- 4. 需求扣减 ------------------------------------------------------------

def test_demand_subtracts_literally_and_records_unmet():
    inventory, _, _, result = run_day(
        base_inventory(10.0), base_orders(), transit_frame(), profile(("SKU001", 1, 30.0))
    )
    demand_row = result.demand[0]
    assert demand_row["stock_before"] == 10.0
    assert demand_row["demand_qty"] == 30.0
    assert demand_row["stock_after"] == -20.0  # 不 clamp 到 0
    assert demand_row["unmet_demand"] == 20.0  # max(0, 30 - 10)
    assert inventory.set_index("sku").loc["SKU001", "current_stock"] == -20.0


def test_unmet_demand_is_not_subtracted_twice():
    inventory, _, _, _ = run_day(
        base_inventory(10.0), base_orders(), transit_frame(), profile(("SKU001", 1, 30.0))
    )
    assert inventory.set_index("sku").loc["SKU001", "current_stock"] == -20.0


def test_missing_sku_day_combination_is_zero():
    inventory, _, _, result = run_day(
        base_inventory(50.0),
        base_orders(),
        transit_frame(),
        profile(("SKU001", 1, 5.0)),  # 没有 day 2 的记录
        simulation_day=2,
        current_day=START + timedelta(days=1),
    )
    assert result.demand[0]["demand_qty"] == 0.0
    assert inventory.set_index("sku").loc["SKU001", "current_stock"] == 50.0


# --- 5. demand_profile 校验 -------------------------------------------------

def test_demand_profile_missing_columns_raises():
    bad = pd.DataFrame([{"sku": "SKU001", "simulation_day": 1}])
    with pytest.raises(ValueError):
        run_simulation(run_id="R", start_day=START, days=1, demand_profile=bad)


def test_demand_profile_null_qty_raises():
    bad = pd.DataFrame([{"sku": "SKU001", "simulation_day": 1, "qty": None}])
    with pytest.raises(ValueError):
        run_simulation(run_id="R", start_day=START, days=1, demand_profile=bad)


def test_demand_profile_negative_qty_raises():
    with pytest.raises(ValueError):
        run_simulation(
            run_id="R", start_day=START, days=1, demand_profile=profile(("SKU001", 1, -1.0))
        )


def test_demand_profile_non_positive_day_raises():
    with pytest.raises(ValueError):
        run_simulation(
            run_id="R", start_day=START, days=1, demand_profile=profile(("SKU001", 0, 1.0))
        )


def test_demand_profile_non_integer_day_raises():
    with pytest.raises(ValueError):
        run_simulation(
            run_id="R", start_day=START, days=1, demand_profile=profile(("SKU001", 1.5, 1.0))
        )


def test_demand_profile_duplicate_combination_raises():
    with pytest.raises(ValueError):
        run_simulation(
            run_id="R",
            start_day=START,
            days=1,
            demand_profile=profile(("SKU001", 1, 1.0), ("SKU001", 1, 2.0)),
        )


def test_run_simulation_rejects_bad_days_and_start_day():
    with pytest.raises(ValueError):
        run_simulation(run_id="R", start_day=START, days=0, demand_profile=profile())
    with pytest.raises(TypeError):
        run_simulation(run_id="R", start_day="2026-09-20", days=1, demand_profile=profile())


# --- 6. orders_log 只读 -----------------------------------------------------

def test_orders_log_is_never_mutated_and_never_receives_simulated_demand():
    orders = base_orders()
    before = orders.copy(deep=True)
    result = run_simulation(
        run_id="R",
        start_day=START,
        days=3,
        demand_profile=profile(*[("SKU001", d, 7.0) for d in (1, 2, 3)]),
        orders_log=orders,
        inventory_state=base_inventory(5.0),
        transit_state=transit_frame(),
        constraints=NO_CONSTRAINTS,
        coverage_start=COVERAGE_START,
    )
    pd.testing.assert_frame_equal(orders, before)
    assert len(result.records) > 0
    assert set(orders["order_id"]) == {"O1"}
    assert len(orders) == 1


# --- 7. 决策链委托（不复制公式） --------------------------------------------

def _calls(source):
    names, attrs = set(), set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                attrs.add(node.func.attr)
    return names, attrs


def test_simulator_delegates_to_existing_modules():
    source = (APP_DIR / "simulator.py").read_text(encoding="utf-8")
    names, attrs = _calls(source)
    assert {"calculate_demand", "analyze_inventory_frame", "apply_procurement"} <= names
    assert "in_transit_metrics" not in names  # 在途由 analyze_inventory_frame 内部处理
    assert "_recommend" not in names  # 不重算 MOQ / order_multiple
    assert "ceil" not in attrs


def test_decision_uses_daily_demand_30d_from_demand_engine():
    _, _, _, result = run_day(
        base_inventory(0.0), base_orders(qty=30), transit_frame(), profile(("SKU001", 1, 0.0))
    )
    row = result.decision_frame.iloc[0]
    assert row.daily_demand == 1.0  # 30 / 30
    assert row.daily_demand == row.daily_demand_30d


# --- 8. 下单 ----------------------------------------------------------------

def test_order_placed_qty_equals_procurement_recommended_qty():
    _, _, _, result = run_day(
        base_inventory(0.0), base_orders(), transit_frame(), profile(("SKU001", 1, 0.0))
    )
    frame_qty = float(result.decision_frame.iloc[0].procurement_recommended_qty)
    assert result.order_placed[0]["qty"] == frame_qty


def test_moq_applied_exactly_once_via_level2():
    constraints = pd.DataFrame([("SKU001", 100.0, 50)], columns=["sku", "moq", "order_multiple"])
    _, _, _, result = run_day(
        base_inventory(0.0, safety=10.0, lead=5),
        base_orders(),
        transit_frame(),
        profile(("SKU001", 1, 0.0)),
        constraints=constraints,
    )
    row = result.decision_frame.iloc[0]
    assert row.recommended_order_qty == 15.0  # L1 gap
    assert row.procurement_recommended_qty == 100.0  # L2: MOQ，再向上取整到 50 的倍数
    assert result.order_placed[0]["qty"] == 100.0  # 直接取 L2，不重复套约束


def test_provisional_true_does_not_block_order():
    _, _, _, result = run_day(
        base_inventory(0.0),
        base_orders(),
        transit_frame(),
        profile(("SKU001", 1, 0.0)),
        coverage_start=START - timedelta(days=5),  # 历史不足 -> provisional True
    )
    row = result.decision_frame.iloc[0]
    assert bool(row.provisional) is True
    assert row.procurement_recommended_qty > 0
    assert result.order_placed, "provisional=True 不得阻止下单"


def test_zero_qty_creates_no_po():
    _, transit, _, result = run_day(
        base_inventory(500.0), base_orders(), transit_frame(), profile(("SKU001", 1, 0.0))
    )
    assert result.decision_frame.iloc[0].procurement_recommended_qty == 0
    assert result.order_placed == []
    assert len(transit) == 0
    decision = [r for r in result.records if r["record_type"] == "DECISION"][0]
    assert decision["order_placed_qty"] == 0
    assert decision["po_id"] is None


def test_at_most_one_po_per_sku_per_day():
    _, _, _, result = run_day(
        base_inventory(0.0), base_orders(), transit_frame(), profile(("SKU001", 1, 0.0))
    )
    skus = [po["sku"] for po in result.order_placed]
    assert len(skus) == len(set(skus))


# --- 9. PO 生命周期 ---------------------------------------------------------

def test_po_id_format_and_expected_date():
    _, _, _, result = run_day(
        base_inventory(0.0, lead=7),
        base_orders(),
        transit_frame(),
        profile(("SKU001", 1, 0.0)),
        run_id="RUNX",
    )
    po = result.order_placed[0]
    assert po["po_id"] == "PO-RUNX-000001"
    assert po["expected_date"] == (START + timedelta(days=7)).isoformat()


def test_po_ids_are_unique_and_deterministic_across_days():
    kwargs = dict(
        run_id="RUNX",
        start_day=START,
        days=4,
        demand_profile=profile(*[("SKU001", d, 0.0) for d in (1, 2, 3, 4)]),
        orders_log=base_orders(),
        inventory_state=base_inventory(0.0, lead=3),
        transit_state=transit_frame(),
        constraints=NO_CONSTRAINTS,
        coverage_start=COVERAGE_START,
    )
    first = run_simulation(**kwargs)
    second = run_simulation(**kwargs)

    ids = [r["po_id"] for r in first.records if r["po_id"]]
    assert len(ids) == len(set(ids))
    assert all(po_id.startswith("PO-RUNX-") for po_id in ids)
    assert [r["po_id"] for r in first.records] == [r["po_id"] for r in second.records]


def test_only_open_to_arrived_transitions():
    inventory, transit, _, _ = run_day(
        base_inventory(0.0),
        base_orders(),
        transit_frame(("PO-1", "SKU001", 10, START.isoformat(), "OPEN")),
        profile(("SKU001", 1, 0.0)),
    )
    assert set(transit["status"]) <= {"OPEN", "ARRIVED", "CANCELLED"}
    assert transit.set_index("po_id").loc["PO-1", "status"] == "ARRIVED"


# --- 10. 零需求 SKU / SKIP --------------------------------------------------

def _zero_demand_state():
    inv = inventory_frame(("SKU001", 5.0, 10.0, 5), ("SKU002", 50.0, 10.0, 3))
    orders = orders_frame(
        order("O1", START, "SKU001", 30),
        ("OLD", (START - timedelta(days=40)).isoformat(), "SKU002", 30),
    )
    return inv, orders


def test_zero_demand_sku_is_skipped_but_kept():
    inv, orders = _zero_demand_state()
    new_inv, transit, _, result = run_day(
        inv, orders, transit_frame(), profile(("SKU001", 1, 1.0), ("SKU002", 1, 1.0))
    )

    decisions = [r for r in result.records if r["record_type"] == "DECISION"]
    skips = [r for r in result.records if r["record_type"] == "SKIP"]
    assert [r["sku"] for r in decisions] == ["SKU001"]
    assert [r["sku"] for r in skips] == ["SKU002"]
    assert skips[0]["skip_field"] == "daily_demand"
    assert "daily_demand" in skips[0]["skip_reason"]

    # SKU002 未被删除，且不产 PO
    assert "SKU002" in new_inv["sku"].tolist()
    assert {d["sku"] for d in result.demand} == {"SKU001", "SKU002"}
    assert all(po["sku"] != "SKU002" for po in result.order_placed)


def test_skipped_sku_continues_next_day():
    inv, orders = _zero_demand_state()
    _, _, seq, _ = run_day(inv, orders, transit_frame(), profile(("SKU001", 1, 1.0)))
    _, _, _, result2 = run_day(
        inv,
        orders,
        transit_frame(),
        profile(),
        simulation_day=2,
        current_day=START + timedelta(days=1),
        next_po_seq=seq,
    )
    skips = [r for r in result2.records if r["record_type"] == "SKIP"]
    assert [r["sku"] for r in skips] == ["SKU002"]


def test_each_sku_has_exactly_one_record_per_day():
    inv, orders = _zero_demand_state()
    _, _, _, result = run_day(inv, orders, transit_frame(), profile(("SKU001", 1, 1.0)))
    skus = [r["sku"] for r in result.records]
    assert sorted(skus) == ["SKU001", "SKU002"]
    assert len(skus) == len(set(skus))


# --- 11. 多日推进 -----------------------------------------------------------

def test_multiday_progression_and_state_handoff():
    result = run_simulation(
        run_id="R",
        start_day=START,
        days=3,
        demand_profile=profile(*[("SKU001", d, 5.0) for d in (1, 2, 3)]),
        orders_log=base_orders(),
        inventory_state=base_inventory(0.0, lead=3),
        transit_state=transit_frame(("PO-1", "SKU001", 100, (START + timedelta(days=1)).isoformat(), "OPEN")),
        constraints=NO_CONSTRAINTS,
        coverage_start=COVERAGE_START,
    )
    assert [d.simulation_day for d in result.daily] == [1, 2, 3]
    assert [d.as_of_date for d in result.daily] == [START, START + timedelta(days=1), START + timedelta(days=2)]
    # state 跨日传递：day2 到货后 day3 期初库存反映出来
    assert result.daily[1].arrivals == [{"po_id": "PO-1", "sku": "SKU001", "qty": 100.0}]
    assert result.records == [r for d in result.daily for r in d.records]


def test_stock_conservation_with_arrival():
    inv0 = base_inventory(0.0, lead=3)
    result = run_simulation(
        run_id="R",
        start_day=START,
        days=3,
        demand_profile=profile(*[("SKU001", d, 5.0) for d in (1, 2, 3)]),
        orders_log=base_orders(),
        inventory_state=inv0,
        transit_state=transit_frame(("PO-1", "SKU001", 100, (START + timedelta(days=1)).isoformat(), "OPEN")),
        constraints=NO_CONSTRAINTS,
        coverage_start=COVERAGE_START,
    )
    opening = float(inv0.set_index("sku").loc["SKU001", "current_stock"])
    arrivals = sum(a["qty"] for d in result.daily for a in d.arrivals)
    demand = sum(r["demand_qty"] for d in result.daily for r in d.demand)
    closing = float(result.daily[-1].decision_frame.set_index("sku").loc["SKU001", "current_stock"])
    assert closing == opening + arrivals - demand


# --- 12. 确定性 / 无系统时钟 ------------------------------------------------

def _run_kwargs():
    return dict(
        run_id="R",
        start_day=START,
        days=3,
        demand_profile=profile(*[("SKU001", d, 5.0) for d in (1, 2, 3)]),
        orders_log=base_orders(),
        inventory_state=base_inventory(0.0, lead=3),
        transit_state=transit_frame(),
        constraints=NO_CONSTRAINTS,
        coverage_start=COVERAGE_START,
    )


def test_run_simulation_is_deterministic():
    first = run_simulation(**_run_kwargs())
    second = run_simulation(**_run_kwargs())
    assert first.records == second.records
    for a, b in zip(first.daily, second.daily):
        pd.testing.assert_frame_equal(a.decision_frame, b.decision_frame)
        assert a.demand == b.demand
        assert a.arrivals == b.arrivals
        assert a.order_placed == b.order_placed


def test_simulator_does_not_read_the_system_clock():
    source = (APP_DIR / "simulator.py").read_text(encoding="utf-8")
    for forbidden in ("datetime.now", "date.today", "time.time", "random"):
        assert forbidden not in source, forbidden


# --- 13. 输入不被修改 -------------------------------------------------------

def test_simulate_day_does_not_mutate_inputs():
    inv = base_inventory(5.0)
    transit = transit_frame(("PO-1", "SKU001", 100, START.isoformat(), "OPEN"))
    orders = base_orders()
    prof = profile(("SKU001", 1, 10.0))
    inv_before, transit_before = inv.copy(deep=True), transit.copy(deep=True)
    orders_before, prof_before = orders.copy(deep=True), prof.copy(deep=True)

    run_day(inv, orders, transit, prof)

    pd.testing.assert_frame_equal(inv, inv_before)
    pd.testing.assert_frame_equal(transit, transit_before)
    pd.testing.assert_frame_equal(orders, orders_before)
    pd.testing.assert_frame_equal(prof, prof_before)


# --- 14. 审计一致性 ---------------------------------------------------------

def test_decision_records_match_decision_frame():
    _, _, _, result = run_day(
        base_inventory(0.0), base_orders(), transit_frame(), profile(("SKU001", 1, 0.0))
    )
    frame = result.decision_frame.set_index("sku")
    for record in [r for r in result.records if r["record_type"] == "DECISION"]:
        row = frame.loc[record["sku"]]
        for column in BUSINESS_COLUMNS:
            assert record[column] == row[column], column


def test_action_fields_match_created_po():
    _, _, _, result = run_day(
        base_inventory(0.0), base_orders(), transit_frame(), profile(("SKU001", 1, 0.0))
    )
    po = result.order_placed[0]
    decision = [r for r in result.records if r["record_type"] == "DECISION"][0]
    assert decision["order_placed_qty"] == po["qty"]
    assert decision["po_id"] == po["po_id"]
    assert decision["expected_date"] == po["expected_date"]


def test_skip_record_matches_warning():
    inv, orders = _zero_demand_state()
    _, _, _, result = run_day(inv, orders, transit_frame(), profile(("SKU001", 1, 1.0)))
    skip = [r for r in result.records if r["record_type"] == "SKIP"][0]
    warning = [w for w in result.skipped if w["sku"] == skip["sku"]][0]
    assert skip["skip_field"] == warning["field"]
    assert skip["skip_reason"] == warning["message"]


def test_skip_is_not_a_decision():
    inv, orders = _zero_demand_state()
    _, _, _, result = run_day(inv, orders, transit_frame(), profile(("SKU001", 1, 1.0)))
    skip = [r for r in result.records if r["record_type"] == "SKIP"][0]
    assert skip["record_type"] == "SKIP"
    assert skip["skipped"] is True
    assert "procurement_recommended_qty" not in skip


def test_audit_jsonl_appends_across_runs(tmp_path):
    path = tmp_path / "audit.jsonl"
    kwargs = _run_kwargs()
    first = run_simulation(audit_path=path, **kwargs)
    lines_first = path.read_text(encoding="utf-8").splitlines()
    assert len(lines_first) == len(first.records)

    run_simulation(audit_path=path, **kwargs)
    lines_second = path.read_text(encoding="utf-8").splitlines()
    assert len(lines_second) == 2 * len(first.records)
    assert lines_second[0] == lines_first[0]
    for line in lines_second:
        json.loads(line)


def test_no_timestamp_in_any_record():
    result = run_simulation(**_run_kwargs())
    for record in result.records:
        assert "timestamp" not in record


# --- 集成：真实数据可跑 -----------------------------------------------------

def test_run_simulation_with_default_data():
    result = run_simulation(
        run_id="REAL",
        start_day=START,
        days=1,
        demand_profile=profile(("SKU001", 1, 1.0)),
    )
    assert isinstance(result, SimulationResult)
    assert len(result.daily) == 1
    day = result.daily[0]
    assert len(day.records) == len(day.decision_frame) + len(day.skipped)
    assert list(day.decision_frame.columns) == EXPECTED_DECISION_COLUMNS
