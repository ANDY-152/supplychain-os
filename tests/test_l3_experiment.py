"""Level 3 实验工具测试：B1 需求、优先级、B_full/档位、校验、指标、收敛。"""
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import pytest

from tools.l3_experiment import (
    DEMAND_DENOM,
    FIXED_B1_DAILY,
    PRIORITY_RULES,
    SCENARIOS,
    TIERS,
    aggregate_metrics,
    apply_scenario,
    b1_daily_qty,
    build_demand_profile,
    calibrate_bfull_cents,
    convergence_report,
    priority_map,
    tier_limit_cents,
    validate_inputs,
)

START = date(2026, 9, 22)
SKUS = ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"]


def _orders(rows):
    return pd.DataFrame(rows, columns=["order_id", "order_date", "sku", "qty"])


# --- B1 ---------------------------------------------------------------------

def test_b1_half_up_and_constant():
    orders = _orders([
        ("O1", "2026-08-24", "SKU001", 385),   # 窗口内
        ("O2", "2026-09-22", "SKU002", 390),   # 边界（含）
        ("O3", "2026-08-23", "SKU001", 999),   # 窗口外（start-30）
        ("O4", "2026-09-23", "SKU002", 999),   # 窗口外（未来）
    ])
    q = b1_daily_qty(START, orders, SKUS)
    assert q["SKU001"] == (385 + 15) // DEMAND_DENOM  # 13
    assert q["SKU002"] == (390 + 15) // DEMAND_DENOM   # 13
    assert q["SKU003"] == 0 and q["SKU004"] == 0 and q["SKU005"] == 0


def test_build_demand_profile_shape_and_determinism():
    orders = _orders([("O1", "2026-09-01", "SKU001", 100)])
    p1, q1 = build_demand_profile(START, 5, orders, SKUS)
    p2, q2 = build_demand_profile(START, 5, orders, SKUS)
    assert p1.equals(p2) and q1 == q2
    assert len(p1) == 5 * len(SKUS)
    assert set(p1.columns) == {"sku", "simulation_day", "qty"}
    assert not p1.duplicated(subset=["sku", "simulation_day"]).any()
    assert (p1["qty"] >= 0).all() and p1["qty"].dtype.kind == "i"


# --- 优先级 -----------------------------------------------------------------

PRODUCTS = pd.DataFrame(
    {"sku": SKUS, "category": ["A", "A", "B", "A", "B"]}
)
INVENTORY = pd.DataFrame(
    {"sku": SKUS, "lead_time_days": [5, 7, 10, 5, 3]}
)


def test_apply_scenario_s1c_overrides_memory_copy_only():
    inv = pd.DataFrame({
        "sku": SKUS, "current_stock": [1200.0, 500.0, 5000.0, 200.0, 1500.0],
        "safety_stock": [500, 300, 500, 400, 400], "lead_time_days": [5, 7, 10, 5, 3],
    })
    before = inv.copy()
    out, changes = apply_scenario(inv, "S1-C")
    # 原 DataFrame 不被修改
    assert inv.equals(before)
    # 仅三个 SKU 被覆盖为 0
    got = dict(zip(out["sku"], out["current_stock"]))
    assert got["SKU001"] == 0.0 and got["SKU003"] == 0.0 and got["SKU005"] == 0.0
    assert got["SKU002"] == 500.0 and got["SKU004"] == 200.0
    # changes 记录原始值/覆盖值/原因
    assert {c["sku"] for c in changes} == {"SKU001", "SKU003", "SKU005"}
    by_sku = {c["sku"]: c for c in changes}
    assert by_sku["SKU001"]["original_current_stock"] == 1200.0
    assert by_sku["SKU001"]["override_current_stock"] == 0.0
    assert all(c["reason"] for c in changes)


def test_apply_scenario_baseline_and_unknown():
    inv = pd.DataFrame({"sku": SKUS, "current_stock": [1.0] * len(SKUS)})
    out, changes = apply_scenario(inv, "BASELINE")
    assert out.equals(inv) and changes == []
    with pytest.raises(ValueError, match="unknown scenario"):
        apply_scenario(inv, "S9")


def test_scenarios_cover_expected_skus():
    assert set(SCENARIOS) >= {"BASELINE", "S1-C"}
    assert SCENARIOS["BASELINE"] == {}
    assert SCENARIOS["S1-C"] == {"SKU001": 0.0, "SKU003": 0.0, "SKU005": 0.0}


def test_priority_rules_are_distinct_and_deterministic():
    maps = {r: priority_map(r, SKUS, PRODUCTS, INVENTORY) for r in PRIORITY_RULES}
    assert maps["P_cat"] == {"SKU001": 1, "SKU002": 1, "SKU003": 2, "SKU004": 1, "SKU005": 2}
    # distinct lead 降序密集编号：10->1, 7->2, 5->3, 3->4
    assert maps["P_lead"]["SKU003"] == 1 and maps["P_lead"]["SKU005"] == 4
    assert maps["P_lead"]["SKU002"] == 2
    assert maps["P_rev"] == {"SKU001": 2, "SKU002": 2, "SKU003": 1, "SKU004": 2, "SKU005": 1}
    # 主排序（不依赖二级键）三规则互不相同
    primary = {r: tuple(sorted(m, key=lambda s: m[s])) for r, m in maps.items()}
    assert len({primary["P_cat"], primary["P_lead"], primary["P_rev"]}) >= 2


# --- B_full / 档位 -----------------------------------------------------------

@dataclass
class _Day:
    decision_frame: object
    simulation_day: int = 1
    demand: list = field(default_factory=list)
    order_placed: list = field(default_factory=list)
    budget_spent_today: float = 0.0
    budget_remaining_after: float = None


@dataclass
class _Res:
    daily: list


def test_tier_limit_exact_and_floor():
    b = 12345
    assert tier_limit_cents(b, 1, 1) == b
    assert tier_limit_cents(b, 1, 4) == b // 4
    assert tier_limit_cents(b, 3, 2) == (3 * b) // 2
    # 所有档位为整数分
    for n, d in TIERS:
        assert isinstance(tier_limit_cents(b, n, d), int)


def test_calibrate_bfull_cents():
    frame = pd.DataFrame(
        {"sku": ["SKU001", "SKU002"], "procurement_recommended_qty": [10.0, 5.0]}
    )
    res = _Res([_Day(decision_frame=frame), _Day(decision_frame=frame)])
    costs = {"SKU001": 12.50, "SKU002": 8.75}
    # (1250*10 + 875*5) * 2 days = (12500 + 4375)*2 = 33750
    assert calibrate_bfull_cents(res, costs) == 33750


def test_calibrate_rejects_non_integer_qty():
    frame = pd.DataFrame({"sku": ["SKU001"], "procurement_recommended_qty": [1.5]})
    with pytest.raises(ValueError, match="非非负整数"):
        calibrate_bfull_cents(_Res([_Day(decision_frame=frame)]), {"SKU001": 1.0})


# --- 校验 -------------------------------------------------------------------

_GOOD_CONS = pd.DataFrame({"sku": SKUS, "moq": [100, 200, 50, 100, 50],
                           "order_multiple": [50, 100, 10, 50, 25]})
_GOOD_COSTS = {"SKU001": 12.5, "SKU002": 8.75, "SKU003": 3.2, "SKU004": 25.0, "SKU005": 6.4}


def test_validate_inputs_ok():
    assert validate_inputs(SKUS, _GOOD_COSTS, _GOOD_CONS, PRODUCTS) == []


def test_validate_inputs_reports_problems():
    assert validate_inputs(SKUS, {"SKU001": 1.0}, _GOOD_CONS, PRODUCTS)
    bad_moq = _GOOD_CONS.copy()
    bad_moq.loc[0, "moq"] = -1
    assert validate_inputs(SKUS, _GOOD_COSTS, bad_moq, PRODUCTS)
    bad_mult = _GOOD_CONS.copy()
    bad_mult.loc[0, "order_multiple"] = 0
    assert validate_inputs(SKUS, _GOOD_COSTS, bad_mult, PRODUCTS)


# --- 指标 -------------------------------------------------------------------

def _metrics_result(demand, frame=None, spent=0.0, remaining=None, orders=0):
    if frame is None:
        frame = pd.DataFrame({"sku": [], "procurement_recommended_qty": [], "allocated_qty": []})
    day = _Day(decision_frame=frame, demand=demand, order_placed=[{}] * orders,
               budget_spent_today=spent, budget_remaining_after=remaining)
    return _Res([day])


def test_metrics_zero_demand_returns_null():
    res = _metrics_result([{"sku": "SKU001", "demand_qty": 0.0, "unmet_demand": 0.0,
                           "stock_before": 0.0, "stock_after": 0.0}])
    m = aggregate_metrics(res, algorithm="R0", priority_rule="P_cat", tier=None,
                          budget_limit_cents=None, constraints=None)
    assert m["service_rate"] is None and m["stockout_rate"] is None
    assert m["zero_demand_denominator"] is True


def test_metrics_service_and_stockout_incidence():
    demand = [
        {"sku": "SKU001", "demand_qty": 10.0, "unmet_demand": 2.0,
         "stock_before": 8.0, "stock_after": -2.0},                    # stockout
        {"sku": "SKU002", "demand_qty": 5.0, "unmet_demand": 0.0,
         "stock_before": 5.0, "stock_after": 0.0},                     # ok
        {"sku": "SKU003", "demand_qty": 0.0, "unmet_demand": 0.0,
         "stock_before": 0.0, "stock_after": 0.0},                     # zero demand day
    ]
    m = aggregate_metrics(_metrics_result(demand), algorithm="R0", priority_rule="P_cat",
                          tier=None, budget_limit_cents=None, constraints=None)
    assert m["sum_demand_qty"] == 15.0 and m["sum_unmet_demand"] == 2.0
    assert m["service_rate"] == pytest.approx(1 - 2 / 15)
    assert m["stockout_rate"] == 0.5 and m["stockout_rate_denominator"] == 2


def test_metrics_pack_compliance_detects_violation():
    frame = pd.DataFrame({
        "sku": ["SKU001"], "procurement_recommended_qty": [100.0], "allocated_qty": [75.0],
    })  # 75 不是 order_multiple=50 的倍数
    m = aggregate_metrics(_metrics_result([], frame=frame), algorithm="R2",
                          priority_rule="P_cat", tier=(1, 1), budget_limit_cents=100000,
                          constraints=_GOOD_CONS)
    assert m["pack_violation_count"] == 1


# --- 收敛 -------------------------------------------------------------------

def test_fixed_b1_daily_values():
    assert FIXED_B1_DAILY == {
        "SKU001": 13.0, "SKU002": 13.0, "SKU003": 4.0, "SKU004": 8.0, "SKU005": 4.0,
    }
    assert sum(FIXED_B1_DAILY.values()) == 42.0


def test_ending_backlog_and_new_unmet_metrics():
    demand = [{"sku": "A", "demand_qty": 4.0, "unmet_demand": 104.0,
               "stock_before": -100.0, "stock_after": -104.0}]
    m = aggregate_metrics(_metrics_result(demand), algorithm="R0", priority_rule="P_cat",
                          tier=None, budget_limit_cents=None, constraints=None)
    # unmet 记录含既有积压（-100）→ 104；新增未满足只有当日 4
    assert m["ending_backlog_qty"] == 104.0
    assert m["sum_new_unmet_demand"] == 4.0
    assert m["sum_unmet_demand"] == 104.0


def test_daily_rows_complete_with_has_decision():
    from datetime import date as _date
    from tools.run_l3_experiment import _daily_rows

    class _CsvDay:
        simulation_day = 1
        as_of_date = _date(2026, 9, 22)
        period_index = 0
        arrivals = []
        order_placed = []
        budget_available_before = None
        budget_spent_today = 0.0
        budget_remaining_after = None

        def __init__(self, demand, frame):
            self.demand = demand
            self.decision_frame = frame

    frame = pd.DataFrame({
        "sku": ["SKU001"], "daily_demand_30d": [13.0], "current_stock": [100.0],
        "in_transit_stock": [0.0], "procurement_recommended_qty": [0.0], "allocated_qty": [0.0],
        "allocation_reason": ["NO_GAP"], "priority": [1], "allocation_rank": [1],
    })
    day = _CsvDay(
        demand=[
            {"sku": "SKU001", "demand_qty": 13.0, "stock_before": 100.0, "stock_after": 87.0, "unmet_demand": 0.0},
            {"sku": "SKU002", "demand_qty": 13.0, "stock_before": 0.0, "stock_after": -13.0, "unmet_demand": 13.0},
        ],
        frame=frame,
    )
    rows = list(_daily_rows("R", "R0", None, "P_cat", _Res([day])))
    by = {r["sku"]: r for r in rows}
    assert set(by) == {"SKU001", "SKU002"}          # 含未进入决策帧的 SKU
    assert by["SKU001"]["has_decision"] is True
    assert by["SKU002"]["has_decision"] is False
    assert by["SKU002"]["procurement_recommended_qty"] is None  # 空值而非 0
    assert by["SKU002"]["demand_qty"] == 13.0 and by["SKU002"]["stock_after"] == -13.0
    assert by["SKU001"]["budget_spent_today_cents"] == 0


def test_convergence_report_detects_diffs():
    r0 = _Res([_Day(pd.DataFrame({"sku": ["A"], "procurement_recommended_qty": [10.0]}))])
    good = _Res([_Day(pd.DataFrame(
        {"sku": ["A"], "procurement_recommended_qty": [10.0], "allocated_qty": [10.0]}))])
    bad = _Res([_Day(pd.DataFrame(
        {"sku": ["A"], "procurement_recommended_qty": [10.0], "allocated_qty": [5.0]}))])
    assert convergence_report(good, r0, "R1")["passed"] is True
    conf = convergence_report(bad, r0, "R1")
    assert conf["passed"] is False and len(conf["allocation_diffs"]) == 1
