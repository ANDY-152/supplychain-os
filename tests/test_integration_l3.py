"""Level 3 集成测试（持久化）。

Run: pytest tests/test_integration_l3.py -q -p no:cacheprovider

只通过真实公开接口（`run_simulation` / `allocate` / `build_audit_records`）验证行为；
spy 仅记录真实调用参数，计算仍由被测代码完成，不复制生产实现。
"""
import dataclasses
import json
from datetime import timedelta
from decimal import Decimal
from fractions import Fraction

import pandas as pd
import pytest

from app.audit_log import ACTION_FIELDS, L3_COLUMNS
from app.simulator import DailyResult, _BudgetState, run_simulation, simulate_day
from tests.test_simulator import (
    COVERAGE_START,
    NO_CONSTRAINTS,
    START,
    _run_kwargs,
    base_inventory,
    base_orders,
    inventory_frame,
    order,
    orders_frame,
    profile,
    transit_frame,
)

COSTS = {"SKU001": 10.0}
PRIORITY = {"SKU001": 1}
ENVELOPE_KEYS = ["run_id", "simulation_day", "as_of_date", "sku", "record_type"]
R0_L3_ORDER = ["algorithm", "period_index", "allocated_qty", "allocation_reason"]


def _frac(x):
    return Fraction(str(x))


def decision_records(day_result):
    return [r for r in day_result.records if r["record_type"] == "DECISION"]


def two_sku_kwargs():
    inv = inventory_frame(("SKU001", 0.0, 10.0, 3), ("SKU002", 0.0, 10.0, 3))
    ords = orders_frame(
        order("O1", START, "SKU001", 30),
        order("O2", START, "SKU002", 30),
    )
    prof = profile(("SKU001", 1, 5.0), ("SKU002", 1, 5.0))
    return dict(
        run_id="M",
        start_day=START,
        days=1,
        demand_profile=prof,
        orders_log=ords,
        inventory_state=inv,
        transit_state=transit_frame(),
        constraints=NO_CONSTRAINTS,
        coverage_start=COVERAGE_START,
    )


# --- 1. L3 关闭兼容性 -------------------------------------------------------

def test_l3_off_does_not_call_allocate_and_keeps_25_columns(monkeypatch):
    import app.simulator as sim

    called = []
    real_allocate = sim.allocate

    def spy(frame, **kwargs):
        called.append(kwargs)
        return real_allocate(frame, **kwargs)

    monkeypatch.setattr(sim, "allocate", spy)

    result = run_simulation(**_run_kwargs())

    assert called == []
    for day in result.daily:
        assert len(day.decision_frame.columns) == 25
        assert "allocated_qty" not in day.decision_frame.columns
        assert day.budget_available_before is None
        for placed in day.order_placed:
            expected = float(
                day.decision_frame.loc[
                    day.decision_frame["sku"] == placed["sku"], "procurement_recommended_qty"
                ].iloc[0]
            )
            assert placed["qty"] == expected
        for record in day.records:
            assert not any(key in record for key in L3_COLUMNS)


# --- 2. R0 -----------------------------------------------------------------

def test_r0_uses_l2_quantity_and_needs_no_budget_params():
    result = run_simulation(**_run_kwargs(), algorithm="R0")

    for day in result.daily:
        assert day.algorithm == "R0"
        assert day.budget_available_before is None
        assert day.budget_spent_today == 0.0
        assert day.decision_frame["allocated_qty"].equals(
            day.decision_frame["procurement_recommended_qty"]
        )
        for placed in day.order_placed:
            qty = float(
                day.decision_frame.loc[
                    day.decision_frame["sku"] == placed["sku"], "allocated_qty"
                ].iloc[0]
            )
            assert placed["qty"] == qty
        record = decision_records(day)[0]
        assert [key for key in record if key in L3_COLUMNS] == R0_L3_ORDER


# --- 3. R1/R2 实际接入 -------------------------------------------------------

def test_l3_off_with_l3_param_raises():
    with pytest.raises(ValueError, match=r"\[costs\]"):
        run_simulation(**_run_kwargs(), algorithm=None, costs=COSTS)


# --- U-2a：algorithm=None 门禁参数集合与错误契约 -----------------------------

RUN_GATE_PARAMS = (
    "costs", "priority", "budget_limit", "period_start", "cycle_days", "initial_budget_spent"
)
DAY_GATE_PARAMS = ("costs", "priority", "budget_available", "period_start")
RUN_GATE_VALUES = {
    "costs": COSTS,
    "priority": PRIORITY,
    "budget_limit": 1.0,
    "period_start": START,
    "cycle_days": 30,
    "initial_budget_spent": 0.0,
}
DAY_GATE_VALUES = {
    "costs": COSTS,
    "priority": PRIORITY,
    "budget_available": 1.0,
    "period_start": START,
}


def _gate_message(names):
    return (
        "algorithm=None disables L3 but L3 parameter(s) provided: ["
        + ", ".join(names)
        + "]"
    )


def _day_kwargs():
    return dict(
        run_id="R",
        simulation_day=1,
        current_day=START,
        inventory_state=base_inventory(0.0, lead=3),
        transit_state=transit_frame(),
        orders_log=base_orders(),
        demand_profile=profile(("SKU001", 1, 5.0)),
        constraints=NO_CONSTRAINTS,
        coverage_start=COVERAGE_START,
    )


@pytest.mark.parametrize("name", RUN_GATE_PARAMS)
def test_u2a_run_side_each_gate_param_triggers(name):
    with pytest.raises(ValueError) as excinfo:
        run_simulation(**_run_kwargs(), algorithm=None, **{name: RUN_GATE_VALUES[name]})
    assert str(excinfo.value) == _gate_message([name])


@pytest.mark.parametrize(
    "provided, expected",
    [
        (("cycle_days", "budget_limit", "costs"), ["costs", "budget_limit", "cycle_days"]),
        (("period_start", "priority", "costs"), ["costs", "priority", "period_start"]),
        (("initial_budget_spent", "priority"), ["priority", "initial_budget_spent"]),
    ],
)
def test_u2a_run_side_multi_param_fixed_order(provided, expected):
    kwargs = {name: RUN_GATE_VALUES[name] for name in provided}
    with pytest.raises(ValueError) as excinfo:
        run_simulation(**_run_kwargs(), algorithm=None, **kwargs)
    assert str(excinfo.value) == _gate_message(expected)


@pytest.mark.parametrize("name", DAY_GATE_PARAMS)
def test_u2a_day_side_each_gate_param_triggers(name):
    with pytest.raises(ValueError) as excinfo:
        simulate_day(**_day_kwargs(), algorithm=None, **{name: DAY_GATE_VALUES[name]})
    assert str(excinfo.value) == _gate_message([name])


def test_u2a_day_side_multi_param_fixed_order():
    with pytest.raises(ValueError) as excinfo:
        simulate_day(
            **_day_kwargs(), algorithm=None,
            period_start=START, budget_available=1.0, costs=COSTS,
        )
    assert str(excinfo.value) == _gate_message(["costs", "budget_available", "period_start"])


def test_u2a_run_side_excluded_params_do_not_trigger(tmp_path):
    # constraints / coverage_start 已在 _run_kwargs() 中；再附加 audit_path 核对排除项
    kwargs = _run_kwargs()
    kwargs["audit_path"] = str(tmp_path / "audit.jsonl")
    result = run_simulation(**kwargs, algorithm=None)
    assert result.days == 3


def test_u2a_day_side_excluded_params_do_not_trigger():
    inventory, transit, seq, day = simulate_day(**_day_kwargs(), algorithm=None)
    assert day.algorithm is None


def test_r1_requires_budget_limit():
    with pytest.raises(ValueError, match="budget_limit"):
        run_simulation(**_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY)


def test_r2_requires_budget_limit():
    # T10：R2 缺少预算上限同样必须报错
    with pytest.raises(ValueError, match="budget_limit"):
        run_simulation(**_run_kwargs(), algorithm="R2", costs=COSTS, priority=PRIORITY)


def test_negative_budget_limit_rejected():
    with pytest.raises(ValueError):
        run_simulation(
            **_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY, budget_limit=-1.0
        )

@pytest.mark.parametrize("algorithm", ["R1", "R2"])
def test_r1_r2_order_quantity_equals_allocated_and_spend_matches(monkeypatch, algorithm):
    import app.simulator as sim

    seen = []
    real_allocate = sim.allocate

    def spy(frame, **kwargs):
        seen.append(kwargs)
        return real_allocate(frame, **kwargs)

    monkeypatch.setattr(sim, "allocate", spy)

    result = run_simulation(
        **_run_kwargs(), algorithm=algorithm, costs=COSTS, priority=PRIORITY,
        budget_limit=1_000_000.0,
    )

    assert seen
    assert all(call["algorithm"] == algorithm for call in seen)
    assert seen[0]["costs"] == COSTS
    assert seen[0]["priority"] == PRIORITY
    assert seen[0]["budget_available"] == 1_000_000.0
    assert list(seen[0]["constraints"].columns) == ["sku", "moq", "order_multiple"]

    for day in result.daily:
        allocated = day.decision_frame["allocated_qty"]
        assert (allocated <= day.decision_frame["procurement_recommended_qty"]).all()
        for placed in day.order_placed:
            qty = int(
                day.decision_frame.loc[
                    day.decision_frame["sku"] == placed["sku"], "allocated_qty"
                ].iloc[0]
            )
            assert placed["qty"] == float(qty)
        expected_spent = sum(
            (_frac(COSTS[sku]) * int(qty) for sku, qty in zip(day.decision_frame["sku"], allocated)),
            _frac(0),
        )
        assert _frac(day.budget_spent_today) == expected_spent
        assert _frac(day.budget_remaining_after) == _frac(day.budget_available_before) - expected_spent


# --- 3b. 成本精度前置校验 -----------------------------------------------------

@pytest.mark.parametrize("algorithm", ["R1", "R2"])
def test_r1_r2_reject_three_decimal_cost_before_placing_orders(monkeypatch, algorithm):
    import app.simulator as sim

    placed_calls = []
    real_place = sim._place_orders

    def spy(*args, **kwargs):
        placed_calls.append(kwargs)
        return real_place(*args, **kwargs)

    monkeypatch.setattr(sim, "_place_orders", spy)

    with pytest.raises(ValueError) as excinfo:
        run_simulation(
            **_run_kwargs(), algorithm=algorithm, costs={"SKU001": 12.345},
            priority=PRIORITY, budget_limit=1_000.0,
        )

    message = str(excinfo.value)
    assert "SKU001" in message and "unit_cost" in message
    assert placed_calls == []


def test_direct_simulate_day_rejects_three_decimal_cost(monkeypatch):
    import app.simulator as sim

    placed_calls = []
    real_place = sim._place_orders

    def spy(*args, **kwargs):
        placed_calls.append(kwargs)
        return real_place(*args, **kwargs)

    monkeypatch.setattr(sim, "_place_orders", spy)

    with pytest.raises(ValueError) as excinfo:
        simulate_day(
            run_id="R", simulation_day=1, current_day=START,
            inventory_state=base_inventory(0.0, lead=3), transit_state=transit_frame(),
            orders_log=base_orders(), demand_profile=profile(("SKU001", 1, 5.0)),
            constraints=NO_CONSTRAINTS, coverage_start=COVERAGE_START,
            algorithm="R1", costs={"SKU001": 12.345}, priority=PRIORITY, budget_available=1_000.0,
        )

    assert "SKU001" in str(excinfo.value) and "unit_cost" in str(excinfo.value)
    assert placed_calls == []


# --- F-1：直接 simulate_day 的预算整数分口径 ---------------------------------

@pytest.mark.parametrize("algorithm", ["R1", "R2"])
def test_direct_simulate_day_rejects_excess_precision_budget(monkeypatch, algorithm):
    calls = _spy_place_orders(monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        simulate_day(
            run_id="R", simulation_day=1, current_day=START,
            inventory_state=base_inventory(0.0, lead=3), transit_state=transit_frame(),
            orders_log=base_orders(), demand_profile=profile(("SKU001", 1, 5.0)),
            constraints=NO_CONSTRAINTS, coverage_start=COVERAGE_START,
            algorithm=algorithm, costs=COSTS, priority=PRIORITY, budget_available=10.005,
        )
    message = str(excinfo.value)
    assert "budget_available" in message and "cents" in message
    assert calls == []


@pytest.mark.parametrize("algorithm", ["R1", "R2"])
@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_direct_simulate_day_rejects_invalid_budget(monkeypatch, algorithm, bad):
    calls = _spy_place_orders(monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        simulate_day(
            run_id="R", simulation_day=1, current_day=START,
            inventory_state=base_inventory(0.0, lead=3), transit_state=transit_frame(),
            orders_log=base_orders(), demand_profile=profile(("SKU001", 1, 5.0)),
            constraints=NO_CONSTRAINTS, coverage_start=COVERAGE_START,
            algorithm=algorithm, costs=COSTS, priority=PRIORITY, budget_available=bad,
        )
    assert "budget_available" in str(excinfo.value)
    assert calls == []


def test_direct_simulate_day_two_decimal_budget_matches_cents():
    _inv, _transit, _seq, day = simulate_day(
        run_id="R", simulation_day=1, current_day=START,
        inventory_state=base_inventory(0.0, lead=3), transit_state=transit_frame(),
        orders_log=base_orders(), demand_profile=profile(("SKU001", 1, 5.0)),
        constraints=NO_CONSTRAINTS, coverage_start=COVERAGE_START,
        algorithm="R1", costs={"SKU001": 12.50}, priority=PRIORITY, budget_available=100.0,
    )
    allocated = day.decision_frame["allocated_qty"]
    assert allocated.dtype == "int64"
    spent = sum((_frac(12.50) * int(qty) for qty in allocated), _frac(0))
    assert day.budget_available_before == 100.0
    assert _frac(day.budget_spent_today) == spent
    assert _frac(day.budget_remaining_after) == _frac(100.0) - spent
    assert spent > 0


def test_two_decimal_cost_runs_normally():
    result = run_simulation(
        **_run_kwargs(), algorithm="R1", costs={"SKU001": 12.50},
        priority=PRIORITY, budget_limit=10_000.0,
    )
    day = result.daily[0]
    spent = sum((_frac(12.50) * int(qty) for qty in day.decision_frame["allocated_qty"]), _frac(0))
    assert _frac(day.budget_spent_today) == spent
    for placed in day.order_placed:
        qty = int(day.decision_frame.loc[day.decision_frame["sku"] == placed["sku"], "allocated_qty"].iloc[0])
        assert placed["qty"] == float(qty)


def test_r0_ignores_cost_precision():
    # R0 不启用预算分配，不因未启用 L3 而额外依赖/校验成本配置
    result = run_simulation(**_run_kwargs(), algorithm="R0", costs={"SKU001": 12.345})
    day = result.daily[0]
    assert day.algorithm == "R0"
    assert day.decision_frame["allocated_qty"].equals(day.decision_frame["procurement_recommended_qty"])
    assert day.budget_available_before is None


# --- 3c. S-2 参照集校验 -------------------------------------------------------

def _spy_place_orders(monkeypatch):
    import app.simulator as sim

    calls = []
    real = sim._place_orders

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(sim, "_place_orders", spy)
    return calls


def test_unknown_cost_sku_rejected_before_ordering(monkeypatch):
    calls = _spy_place_orders(monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        run_simulation(
            **_run_kwargs(), algorithm="R1",
            costs={"SKU999": 10.0}, priority=PRIORITY, budget_limit=1_000.0,
        )
    message = str(excinfo.value)
    assert "unknown SKU" in message and "SKU999" in message
    assert calls == []


def test_unknown_priority_sku_rejected_before_ordering(monkeypatch):
    calls = _spy_place_orders(monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        run_simulation(
            **_run_kwargs(), algorithm="R2",
            costs=COSTS, priority={"SKU001": 1, "SKU999": 2}, budget_limit=1_000.0,
        )
    message = str(excinfo.value)
    assert "unknown SKU" in message and "SKU999" in message
    assert calls == []


def test_legal_reference_config_enters_normal_flow():
    result = run_simulation(
        **_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY, budget_limit=1_000.0,
    )
    day = result.daily[0]
    assert day.algorithm == "R1"
    assert day.order_placed
    for placed in day.order_placed:
        qty = int(day.decision_frame.loc[day.decision_frame["sku"] == placed["sku"], "allocated_qty"].iloc[0])
        assert placed["qty"] == float(qty)


# --- 4. 共享预算与预算耗尽 ----------------------------------------------------

def test_shared_budget_total_stays_within_limit():
    result = run_simulation(
        **two_sku_kwargs(), algorithm="R1",
        costs={"SKU001": 10.0, "SKU002": 20.0},
        priority={"SKU001": 1, "SKU002": 2}, budget_limit=100.0,
    )
    day = result.daily[0]
    price = {"SKU001": 10.0, "SKU002": 20.0}
    spent = sum(
        (_frac(price[sku]) * int(qty) for sku, qty in zip(day.decision_frame["sku"], day.decision_frame["allocated_qty"])),
        _frac(0),
    )
    assert spent > 0
    assert spent <= _frac(100.0)
    assert _frac(day.budget_spent_today) == spent
    assert _frac(day.budget_remaining_after) == _frac(100.0) - spent
    for placed in day.order_placed:
        qty = int(day.decision_frame.loc[day.decision_frame["sku"] == placed["sku"], "allocated_qty"].iloc[0])
        assert placed["qty"] == float(qty)


def test_zero_budget_places_no_orders():
    result = run_simulation(**_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY, budget_limit=0.0)
    for day in result.daily:
        assert (day.decision_frame["allocated_qty"] == 0).all()
        assert day.order_placed == []
        assert day.budget_spent_today == 0.0


def test_moq_infeasible_places_no_order():
    kwargs = _run_kwargs()
    kwargs["constraints"] = pd.DataFrame([("SKU001", 100.0, 30)], columns=["sku", "moq", "order_multiple"])
    result = run_simulation(**kwargs, algorithm="R1", costs=COSTS, priority=PRIORITY, budget_limit=1.0)
    day = result.daily[0]
    assert day.decision_frame["allocation_reason"].iloc[0] == "MOQ_INFEASIBLE"
    assert day.decision_frame["allocated_qty"].iloc[0] == 0
    assert day.order_placed == []


# --- 5. 跨日与周期 -----------------------------------------------------------

def test_previous_day_spending_reduces_next_available():
    limit = 100_000.0
    result = run_simulation(**_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY, budget_limit=limit)
    cumulative = _frac(0)
    for day in result.daily:
        assert _frac(day.budget_available_before) == _frac(limit) - cumulative
        cumulative += _frac(day.budget_spent_today)


def test_period_boundary_resets_budget():
    result = run_simulation(
        **_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY,
        budget_limit=10_000.0, period_start=START - timedelta(days=29), initial_budget_spent=100.0,
    )
    days = result.daily
    assert _frac(days[0].budget_available_before) == _frac(10_000.0) - _frac(100.0)
    assert _frac(days[1].budget_available_before) == _frac(10_000.0)  # 边界重置、不结转
    assert days[1].period_index == 1
    assert _frac(days[2].budget_available_before) == _frac(10_000.0) - _frac(days[1].budget_spent_today)


def test_mid_cycle_start_requires_initial_budget_spent():
    with pytest.raises(ValueError, match="initial_budget_spent"):
        run_simulation(
            **_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY,
            budget_limit=1_000.0, period_start=START - timedelta(days=15),
        )


def test_open_in_transit_not_charged_again():
    kwargs = _run_kwargs()
    kwargs.pop("transit_state")
    open_po = pd.DataFrame(
        [("PO-OLD", "SKU001", 100.0, (START + timedelta(days=5)).isoformat(), "OPEN")],
        columns=["po_id", "sku", "qty", "expected_date", "status"],
    )
    limit = 1_000.0
    result = run_simulation(
        **kwargs, transit_state=open_po,
        algorithm="R1", costs=COSTS, priority=PRIORITY, budget_limit=limit,
    )
    day = result.daily[0]
    # D-2：首日可用预算不被历史 OPEN 在途 PO（100 × 10 = 1000 元）预先扣减
    assert day.budget_available_before == limit
    # 当日支出只等于实际分配量金额，而非在途承诺金额
    spent = sum(
        (_frac(COSTS[sku]) * int(qty) for sku, qty in zip(day.decision_frame["sku"], day.decision_frame["allocated_qty"])),
        _frac(0),
    )
    assert _frac(day.budget_spent_today) == spent
    assert _frac(day.budget_remaining_after) == _frac(limit) - spent


# --- T12：跨周期到货不占新周期额度 ------------------------------------------

def test_cross_period_arrival_not_charged_in_new_period():
    kwargs = _run_kwargs()
    kwargs.pop("transit_state")
    open_po = pd.DataFrame(
        [("PO-OLD", "SKU001", 1.0, (START + timedelta(days=2)).isoformat(), "OPEN")],
        columns=["po_id", "sku", "qty", "expected_date", "status"],
    )
    limit = 100_000.0
    result = run_simulation(
        **kwargs, transit_state=open_po,
        algorithm="R1", costs=COSTS, priority=PRIORITY,
        budget_limit=limit, cycle_days=2, period_start=START,
    )
    days = result.daily
    assert days[2].period_index == 1                        # START+2 为第 2 周期首个工作日
    assert _frac(days[2].budget_available_before) == _frac(limit)  # 重置、不结转
    assert any(a["po_id"] == "PO-OLD" for a in days[2].arrivals)   # 该日确实到货
    spent = sum(
        (_frac(COSTS[sku]) * int(qty) for sku, qty in zip(days[2].decision_frame["sku"], days[2].decision_frame["allocated_qty"])),
        _frac(0),
    )
    # 到货不是扣减事件：新周期支出 == 当日实际分配量金额
    assert _frac(days[2].budget_spent_today) == spent
    assert _frac(days[2].budget_remaining_after) == _frac(limit) - spent


# --- 6. 异常不回退 -----------------------------------------------------------

def test_allocate_error_propagates_without_placing_orders(monkeypatch):
    import app.simulator as sim

    placed_calls = []
    real_place = sim._place_orders

    def spy(*args, **kwargs):
        placed_calls.append(kwargs)
        return real_place(*args, **kwargs)

    monkeypatch.setattr(sim, "_place_orders", spy)

    with pytest.raises(ValueError):
        run_simulation(
            **_run_kwargs(), algorithm="R1",
            costs={"SKU002": 10.0},   # 合法参照 SKU，但缺帧内 SKU001 的成本
            priority=PRIORITY, budget_limit=1_000.0,
        )

    assert placed_calls == []


# --- 7. 审计契约 -------------------------------------------------------------

def test_r1_audit_l3_columns_order_and_values():
    result = run_simulation(**_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY, budget_limit=1_000_000.0)
    day = result.daily[0]
    record = decision_records(day)[0]
    assert [key for key in record if key in L3_COLUMNS] == list(L3_COLUMNS)
    row = day.decision_frame[day.decision_frame["sku"] == record["sku"]].iloc[0]
    assert record["algorithm"] == "R1"
    assert record["period_index"] == day.period_index
    assert record["allocated_qty"] == int(row["allocated_qty"])
    assert record["unit_cost"] == float(row["unit_cost"])
    assert record["budget_spent_today"] == day.budget_spent_today
    assert record["budget_remaining_after"] == day.budget_remaining_after


def test_l3_off_audit_key_contract_unchanged():
    result = run_simulation(**_run_kwargs())
    record = decision_records(result.daily[0])[0]
    keys = list(record.keys())
    assert keys[:5] == ENVELOPE_KEYS
    assert keys[-3:] == list(ACTION_FIELDS)
    assert not any(key in record for key in L3_COLUMNS)


# --- V-4：L3 审计 JSONL 往返与序列化类型 -------------------------------------

def test_l3_audit_jsonl_round_trip(tmp_path):
    path = tmp_path / "audit.jsonl"
    result = run_simulation(
        **_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY,
        budget_limit=1_000_000.0, audit_path=str(path),
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(result.records) > 1
    parsed = [json.loads(line) for line in lines]

    # 内存记录中不得出现 Decimal（_to_native / json.dumps 均不支持 Decimal）
    for record in result.records:
        assert not any(isinstance(value, Decimal) for value in record.values())

    decisions = [record for record in parsed if record["record_type"] == "DECISION"]
    assert decisions
    for record in decisions:
        assert [key for key in record if key in L3_COLUMNS] == list(L3_COLUMNS)
        for key in ("allocation_rank", "allocated_qty", "priority", "period_index"):
            assert type(record[key]) is int
        for key in (
            "unit_cost", "budget_available_before",
            "budget_spent_today", "budget_remaining_after",
        ):
            assert type(record[key]) is float


def test_l3_off_audit_jsonl_keeps_key_contract(tmp_path):
    path = tmp_path / "audit.jsonl"
    result = run_simulation(**_run_kwargs(), audit_path=str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(result.records) > 1
    parsed = [json.loads(line) for line in lines]
    for record in parsed:
        assert not any(key in record for key in L3_COLUMNS)
    decisions = [record for record in parsed if record["record_type"] == "DECISION"]
    assert decisions
    for record in decisions:
        keys = list(record.keys())
        assert keys[:5] == ENVELOPE_KEYS
        assert keys[-3:] == list(ACTION_FIELDS)


# --- F-5：接口边界（cycle_days / period_start 类型 / 非有限金额） ------------

@pytest.mark.parametrize("bad", [0, -1, 1.5, "30", True])
def test_f5_cycle_days_invalid_rejected(bad):
    with pytest.raises(ValueError, match="cycle_days"):
        run_simulation(
            **_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY,
            budget_limit=1_000.0, cycle_days=bad,
        )


@pytest.mark.parametrize("bad", ["2026-01-01", 20260101, 1.0])
def test_f5_period_start_wrong_type_rejected(bad):
    with pytest.raises(TypeError, match="period_start"):
        run_simulation(
            **_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY,
            budget_limit=1_000.0, period_start=bad,
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_f5_budget_limit_non_finite_rejected(bad):
    with pytest.raises(ValueError, match="budget_limit"):
        run_simulation(
            **_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY,
            budget_limit=bad,
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_f5_unit_cost_non_finite_rejected(bad):
    with pytest.raises(ValueError, match="unit_cost"):
        run_simulation(
            **_run_kwargs(), algorithm="R1", costs={"SKU001": bad}, priority=PRIORITY,
            budget_limit=1_000.0,
        )


# --- T16：L3 开关下 DailyResult 字段集恒定 -----------------------------------

def test_daily_result_fields_identical_l3_on_off():
    off = run_simulation(**_run_kwargs())
    on = run_simulation(
        **_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY, budget_limit=1_000.0,
    )
    off_day, on_day = off.daily[0], on.daily[0]
    assert type(off_day) is DailyResult and type(on_day) is DailyResult
    assert [f.name for f in dataclasses.fields(off_day)] == [f.name for f in dataclasses.fields(on_day)]
    # L3 关闭时 L3 字段取默认值；字段集不因开关而改变
    assert off_day.algorithm is None
    assert off_day.budget_spent_today == 0.0
    assert off_day.budget_available_before is None
    assert off_day.budget_remaining_after is None
    assert off_day.period_index == 0
    assert on_day.algorithm == "R1"


# --- T5：charged 守卫、提交顺序与失败语义 ------------------------------------

def test_budget_state_plan_resets_on_period_first_day():
    state = _BudgetState(limit_cents=10_000, period_start=START, cycle_days=2, spent_cents=500)
    state.charged["PO-X"] = 0
    plan = state.plan(START + timedelta(days=2))  # 第 2 周期首日
    assert plan.period_index == 1
    assert plan.spent_cents == 0
    assert plan.charged == {}
    assert plan.available == 100.0
    # plan 为纯函数：不修改 state（与旧 advance() 的原地重置不同）
    assert state.spent_cents == 500 and state.charged == {"PO-X": 0}


def test_budget_state_plan_preserves_mid_period():
    state = _BudgetState(limit_cents=10_000, period_start=START, cycle_days=2, spent_cents=500)
    state.charged["PO-X"] = 0
    plan = state.plan(START + timedelta(days=1))
    assert plan.period_index == 0
    assert plan.spent_cents == 500
    assert plan.charged == {"PO-X": 0}
    assert plan.available == 95.0


def test_budget_state_charge_records_and_rejects_duplicate():
    state = _BudgetState(limit_cents=10_000, period_start=START, cycle_days=30, spent_cents=0)
    target = {}
    state.charge("PO-A", 0, target=target)
    state.charge("PO-B", 0, target=target)
    assert target == {"PO-A": 0, "PO-B": 0}
    with pytest.raises(ValueError, match="duplicate charge"):
        state.charge("PO-A", 0, target=target)
    assert target == {"PO-A": 0, "PO-B": 0}  # 失败不改副本


def test_budget_state_commit_day_applies_plan_and_delta():
    state = _BudgetState(limit_cents=10_000, period_start=START, cycle_days=30, spent_cents=100)
    plan = state.plan(START)
    state.commit_day(plan, 250, {"PO-A": 0})
    assert state.spent_cents == 250 and state.charged == {"PO-A": 0}


def _spy_charge(monkeypatch):
    import app.simulator as sim

    calls = []
    real = sim._BudgetState.charge

    def spy(self, po_id, period_index, *, target):
        calls.append((po_id, period_index))
        return real(self, po_id, period_index, target=target)

    monkeypatch.setattr(sim._BudgetState, "charge", spy)
    return calls


def test_charged_is_used_in_production_path_for_r1(monkeypatch):
    calls = _spy_charge(monkeypatch)
    result = run_simulation(
        **_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY, budget_limit=1_000_000.0,
    )
    placed = [p["po_id"] for day in result.daily for p in day.order_placed]
    assert placed  # 确认确有下单
    assert [po_id for po_id, _ in calls] == placed
    assert {period for _, period in calls} == {0}


@pytest.mark.parametrize("extra", [{}, {"algorithm": "R0"}])
def test_charged_not_used_for_l3_off_or_r0(monkeypatch, extra):
    calls = _spy_charge(monkeypatch)
    run_simulation(**_run_kwargs(), **extra)
    assert calls == []


def test_duplicate_po_id_in_day_raises_without_commit(monkeypatch):
    import app.simulator as sim

    committed = []
    real_commit = sim._BudgetState.commit_day

    def spy_commit(self, *args, **kwargs):
        committed.append(True)
        return real_commit(self, *args, **kwargs)

    real_place = sim._place_orders

    def dup(*args, **kwargs):
        transit, seq, action, placed = real_place(*args, **kwargs)
        if placed:
            placed = placed + [dict(placed[0])]  # 同日重复 po_id
        return transit, seq, action, placed

    monkeypatch.setattr(sim, "_place_orders", dup)
    monkeypatch.setattr(sim._BudgetState, "commit_day", spy_commit)

    with pytest.raises(ValueError, match="duplicate charge"):
        run_simulation(
            **_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY, budget_limit=1_000_000.0,
        )
    assert committed == []


def test_bad_budget_precision_raises_before_commit(monkeypatch):
    import app.simulator as sim

    committed = []
    real_commit = sim._BudgetState.commit_day

    def spy_commit(self, *args, **kwargs):
        committed.append(True)
        return real_commit(self, *args, **kwargs)

    real_day = sim.simulate_day

    def wrap(**kwargs):
        inv, transit, seq, day = real_day(**kwargs)
        day.budget_spent_today = 1.005  # 三位小数，无法按整数分表示
        return inv, transit, seq, day

    monkeypatch.setattr(sim, "simulate_day", wrap)
    monkeypatch.setattr(sim._BudgetState, "commit_day", spy_commit)

    with pytest.raises(ValueError, match="budget_spent_today"):
        run_simulation(
            **_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY, budget_limit=1_000_000.0,
        )
    assert committed == []


def test_audit_write_failure_aborts_before_memory_commit(tmp_path, monkeypatch):
    import app.simulator as sim

    committed = []
    real_commit = sim._BudgetState.commit_day

    def spy_commit(self, *args, **kwargs):
        committed.append(True)
        return real_commit(self, *args, **kwargs)

    monkeypatch.setattr(sim._BudgetState, "commit_day", spy_commit)
    bad_path = tmp_path / "missing_dir" / "audit.jsonl"

    with pytest.raises(OSError):
        run_simulation(
            **_run_kwargs(), algorithm="R1", costs=COSTS, priority=PRIORITY,
            budget_limit=1_000_000.0, audit_path=str(bad_path),
        )
    assert committed == []  # 写盘失败 -> 当日内存不提交（此前天数亦未提交）


# --- 阻断 A：R1/R2 空决策帧端到端不崩溃 -------------------------------------

@pytest.mark.parametrize("algorithm", ["R1", "R2"])
def test_r1_r2_empty_frame_day_end_to_end(algorithm):
    # 订单全部落在 30 天窗口之外 -> 所有 SKU daily_demand_30d=0 -> 决策帧为空
    kwargs = dict(
        run_id="E",
        start_day=START,
        days=2,
        demand_profile=profile(("SKU001", 1, 5.0), ("SKU001", 2, 5.0)),
        orders_log=orders_frame(order("O1", START, "SKU001", 30, days_ago=60)),
        inventory_state=base_inventory(0.0, lead=3),
        transit_state=transit_frame(),
        constraints=NO_CONSTRAINTS,
        coverage_start=COVERAGE_START,
    )
    result = run_simulation(
        **kwargs, algorithm=algorithm, costs={}, priority={}, budget_limit=100.0,
    )
    assert len(result.daily) == 2
    for day in result.daily:
        assert len(day.decision_frame) == 0
        assert day.budget_spent_today == 0.0
        assert day.budget_available_before == 100.0
        assert day.budget_remaining_after == 100.0
        assert day.order_placed == []


# --- daily_demand_override（实验用补货决策需求覆盖） -------------------------

def test_daily_demand_override_drives_rop_and_recommendation():
    kwargs = dict(
        run_id="O", start_day=START, days=1,
        demand_profile=profile(("SKU001", 1, 5.0)),
        orders_log=base_orders(), inventory_state=base_inventory(0.0, lead=3),
        transit_state=transit_frame(), constraints=NO_CONSTRAINTS, coverage_start=COVERAGE_START,
    )
    base = run_simulation(**kwargs)  # algorithm=None
    over = run_simulation(**kwargs, daily_demand_override={"SKU001": 100.0})
    b0 = base.daily[0].decision_frame.iloc[0]
    o0 = over.daily[0].decision_frame.iloc[0]
    assert o0["daily_demand"] == 100.0
    assert b0["daily_demand"] != 100.0
    assert o0["procurement_recommended_qty"] > b0["procurement_recommended_qty"]
    # 诊断字段不被覆盖
    assert o0["daily_demand_7d"] == b0["daily_demand_7d"]


def test_daily_demand_override_none_matches_default():
    a = run_simulation(**_run_kwargs())
    b = run_simulation(**_run_kwargs(), daily_demand_override=None)
    for da, db in zip(a.daily, b.daily):
        assert da.decision_frame.equals(db.decision_frame)


@pytest.mark.parametrize("bad", [
    {},                                  # 缺失 SKU001
    {"SKU999": 5.0},                     # 缺失 SKU001（含未知 SKU）
    {"SKU001": -1.0},                    # 负数
    {"SKU001": float("nan")},            # NaN
    {"SKU001": float("inf")},            # +inf
    {"SKU001": float("-inf")},           # -inf
])
def test_daily_demand_override_invalid_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        run_simulation(**_run_kwargs(), daily_demand_override=bad)
