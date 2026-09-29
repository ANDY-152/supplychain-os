"""Deterministic multi-day simulator (V0).

职责：持有状态、推进日期、模拟需求发生、处理 PO 到货、创建模拟 PO、调用现有决策模块、保存结果、调用 audit_log。

所有业务公式仍然只在 app/demand.py、app/inventory.py、app/procurement.py 中，本模块绝不复制：
- 不重算 30D/7D 需求、ROP、inventory_position、shortage、coverage
- 不重算 MOQ / order_multiple
- 不重算 in_transit（交给 analyze_inventory_frame 内部的 in_transit_metrics）
- 不使用系统时间、不生成 timestamp
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from app.audit_log import append_audit_records, build_audit_records
from app.data_loader import (
    load_data_metadata,
    load_in_transit,
    load_inventory,
    load_orders,
    load_procurement_constraints,
)
from app.demand import calculate_demand
from app.inventory import analyze_inventory_frame
from app.procurement import apply_procurement

DEMAND_PROFILE_COLUMNS = ("sku", "simulation_day", "qty")
TRANSIT_COLUMNS = ("po_id", "sku", "qty", "expected_date", "status")
SKIP_RECORD_TYPE = "SKIP"

__all__ = [
    "DEMAND_PROFILE_COLUMNS",
    "TRANSIT_COLUMNS",
    "SKIP_RECORD_TYPE",
    "DailyResult",
    "SimulationResult",
    "simulate_day",
    "run_simulation",
]


@dataclass
class DailyResult:
    run_id: str
    simulation_day: int
    as_of_date: date
    arrivals: list
    demand: list
    decision_frame: pd.DataFrame
    skipped: list
    data_warnings: list
    order_placed: list
    records: list


@dataclass
class SimulationResult:
    run_id: str
    start_day: date
    days: int
    coverage_start: date
    daily: list
    records: list


def simulate_day(
    *,
    run_id: str,
    simulation_day: int,
    current_day: date,
    inventory_state: pd.DataFrame,
    transit_state: pd.DataFrame,
    orders_log: pd.DataFrame,
    demand_profile: pd.DataFrame,
    constraints: pd.DataFrame | None = None,
    coverage_start: date | None = None,
    next_po_seq: int = 0,
):
    """推进一天。顺序（冻结）：到货 -> 需求扣减 -> 决策 -> 下单 -> 构造审计记录。

    返回 (inventory_state', transit_state', next_po_seq', DailyResult)。
    不修改任何入参 DataFrame。
    """
    if not isinstance(current_day, date):
        raise TypeError("current_day must be a datetime.date")

    # 1) 到货（必须在需求扣减与下单之前：当天新建 PO 因此不会当天到货）
    transit, arrivals = _receive_open_pos(transit_state, current_day)

    # 2) 到货入账
    inventory = _apply_arrivals(inventory_state, arrivals)

    # 3) 模拟需求 -> 字面扣减 + unmet_demand
    inventory, demand_records = _apply_demand(inventory, demand_profile, simulation_day)

    # 4) 决策链：严格调用现有模块，不复制公式
    demand = calculate_demand(orders_log, current_day, coverage_start=coverage_start)
    batch = analyze_inventory_frame(inventory, demand, transit, as_of_date=current_day)
    frame = apply_procurement(batch.frame, constraints)

    # 5) 下单：数量唯一来源 = procurement_recommended_qty；provisional 不阻断
    transit, next_po_seq, action, order_placed = _place_orders(
        frame, transit, run_id, current_day, next_po_seq
    )

    # 6) 审计记录：DECISION（投影 + action） + SKIP（零需求等被 skip 的 SKU）
    records = _audit_records(run_id, simulation_day, current_day, frame, action, batch)

    result = DailyResult(
        run_id=run_id,
        simulation_day=simulation_day,
        as_of_date=current_day,
        arrivals=arrivals,
        demand=demand_records,
        decision_frame=frame,
        skipped=[dict(w) for w in batch.warnings],
        data_warnings=[dict(w) for w in batch.data_warnings],
        order_placed=order_placed,
        records=records,
    )
    return inventory, transit, next_po_seq, result


def run_simulation(
    *,
    run_id: str,
    start_day: date,
    days: int,
    demand_profile: pd.DataFrame,
    orders_log: pd.DataFrame | None = None,
    inventory_state: pd.DataFrame | None = None,
    transit_state: pd.DataFrame | None = None,
    constraints: pd.DataFrame | None = None,
    coverage_start: date | None = None,
    audit_path=None,
) -> SimulationResult:
    """逐日推进 days 天。状态从 data_loader 播种一次（可显式覆盖，便于测试）。

    audit_path 非 None 时，每天追加写 JSONL；否则只累积在返回值里。
    """
    if not isinstance(start_day, date):
        raise TypeError("start_day must be a datetime.date")
    if not isinstance(days, int) or days < 1:
        raise ValueError("days must be an integer >= 1")

    _validate_demand_profile(demand_profile)

    if orders_log is None:
        orders_log = load_orders()
    if inventory_state is None:
        inventory_state = load_inventory()
    if transit_state is None:
        transit_state = load_in_transit()
    if constraints is None:
        constraints = load_procurement_constraints()
    if coverage_start is None:
        coverage_start = load_data_metadata()

    inventory = inventory_state.copy()
    transit = transit_state.copy()
    next_po_seq = 0
    daily, records = [], []
    current_day = start_day

    for simulation_day in range(1, days + 1):
        inventory, transit, next_po_seq, day_result = simulate_day(
            run_id=run_id,
            simulation_day=simulation_day,
            current_day=current_day,
            inventory_state=inventory,
            transit_state=transit,
            orders_log=orders_log,
            demand_profile=demand_profile,
            constraints=constraints,
            coverage_start=coverage_start,
            next_po_seq=next_po_seq,
        )
        daily.append(day_result)
        records.extend(day_result.records)
        if audit_path is not None:
            append_audit_records(audit_path, day_result.records)
        current_day = current_day + timedelta(days=1)

    return SimulationResult(
        run_id=run_id,
        start_day=start_day,
        days=days,
        coverage_start=coverage_start,
        daily=daily,
        records=records,
    )


# --- internal steps ---------------------------------------------------------

def _receive_open_pos(transit_state, current_day):
    transit = transit_state.copy()
    if len(transit) == 0:
        return transit, []
    expected = pd.to_datetime(transit["expected_date"], format="%Y-%m-%d")
    arrive_mask = (transit["status"] == "OPEN") & (expected <= pd.Timestamp(current_day))
    arriving = transit.loc[arrive_mask]
    transit.loc[arrive_mask, "status"] = "ARRIVED"
    arrivals = [
        {"po_id": row.po_id, "sku": row.sku, "qty": _number(row.qty)}
        for row in arriving.itertuples()
    ]
    return transit, arrivals


def _apply_arrivals(inventory_state, arrivals):
    inventory = inventory_state.copy()
    if not arrivals:
        return inventory
    arrival_totals = pd.DataFrame(arrivals).groupby("sku")["qty"].sum()
    inventory["current_stock"] = (
        inventory["current_stock"] + inventory["sku"].map(arrival_totals).fillna(0.0)
    )
    return inventory


def _apply_demand(inventory, demand_profile, simulation_day):
    today = demand_profile[demand_profile["simulation_day"] == simulation_day]
    today = today[["sku", "qty"]].rename(columns={"qty": "demand_qty"})
    demand_today = inventory[["sku", "current_stock"]].merge(today, on="sku", how="left")
    demand_today["demand_qty"] = demand_today["demand_qty"].fillna(0.0)
    demand_today["stock_before"] = demand_today["current_stock"]
    demand_today["stock_after"] = demand_today["stock_before"] - demand_today["demand_qty"]
    demand_today["unmet_demand"] = (
        demand_today["demand_qty"] - demand_today["stock_before"]
    ).clip(lower=0.0)

    updated = inventory.copy()
    updated["current_stock"] = (
        demand_today.set_index("sku")["stock_after"].reindex(inventory["sku"]).to_numpy()
    )
    demand_records = [
        {
            "sku": row.sku,
            "demand_qty": _number(row.demand_qty),
            "stock_before": _number(row.stock_before),
            "stock_after": _number(row.stock_after),
            "unmet_demand": _number(row.unmet_demand),
        }
        for row in demand_today.itertuples()
    ]
    return updated, demand_records


def _place_orders(frame, transit, run_id, current_day, next_po_seq):
    action = {}
    new_rows = []
    order_placed = []
    seq = next_po_seq
    for row in frame.itertuples():
        qty = float(row.procurement_recommended_qty)
        action[row.sku] = {"order_placed_qty": qty, "po_id": None, "expected_date": None}
        if qty <= 0:
            continue
        seq += 1
        po_id = f"PO-{run_id}-{seq:06d}"
        expected_date = (current_day + timedelta(days=int(row.lead_time_days))).isoformat()
        new_rows.append(
            {
                "po_id": po_id,
                "sku": row.sku,
                "qty": qty,
                "expected_date": expected_date,
                "status": "OPEN",
            }
        )
        action[row.sku] = {"order_placed_qty": qty, "po_id": po_id, "expected_date": expected_date}
        order_placed.append(
            {
                "po_id": po_id,
                "sku": row.sku,
                "qty": qty,
                "expected_date": expected_date,
                "procurement_reason": row.procurement_reason,
                "provisional": bool(row.provisional),
            }
        )
    if new_rows:
        transit = pd.concat(
            [transit, pd.DataFrame(new_rows, columns=list(TRANSIT_COLUMNS))],
            ignore_index=True,
        )
    return transit, seq, action, order_placed


def _audit_records(run_id, simulation_day, current_day, frame, action, batch):
    audit_frame = frame.copy()
    audit_frame["run_id"] = run_id
    audit_frame["simulation_day"] = simulation_day
    decision_records = build_audit_records(audit_frame, action)
    skip_records = [
        {
            "run_id": run_id,
            "simulation_day": simulation_day,
            "as_of_date": current_day.isoformat(),
            "sku": warning["sku"],
            "record_type": SKIP_RECORD_TYPE,
            "skipped": True,
            "skip_field": warning["field"],
            "skip_reason": warning["message"],
        }
        for warning in batch.warnings
    ]
    return decision_records + skip_records


def _validate_demand_profile(demand_profile):
    missing = [column for column in DEMAND_PROFILE_COLUMNS if column not in demand_profile.columns]
    if missing:
        raise ValueError(f"demand_profile missing columns: {missing}")

    profile = demand_profile[list(DEMAND_PROFILE_COLUMNS)]
    for column in DEMAND_PROFILE_COLUMNS:
        if profile[column].isna().any():
            raise ValueError(f"null {column} in demand_profile")

    try:
        days = pd.to_numeric(profile["simulation_day"])
        qty = pd.to_numeric(profile["qty"])
    except (ValueError, TypeError) as e:
        raise ValueError(f"invalid demand_profile numeric value: {e}") from e

    if (days % 1 != 0).any() or (days < 1).any():
        raise ValueError("demand_profile simulation_day must be an integer >= 1")
    if (qty < 0).any():
        raise ValueError("demand_profile qty must be >= 0")

    duplicated = profile.duplicated(subset=["sku", "simulation_day"])
    if duplicated.any():
        raise ValueError("duplicate (sku, simulation_day) in demand_profile")


def _number(value):
    return float(value)
