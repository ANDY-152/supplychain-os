"""Deterministic multi-day simulator (V0).

职责：持有状态、推进日期、模拟需求发生、处理 PO 到货、创建模拟 PO、调用现有决策模块、保存结果、调用 audit_log。

所有业务公式仍然只在 app/demand.py、app/inventory.py、app/procurement.py 中，本模块绝不复制：
- 不重算 30D/7D 需求、ROP、inventory_position、shortage、coverage
- 不重算 MOQ / order_multiple
- 不重算 in_transit（交给 analyze_inventory_frame 内部的 in_transit_metrics）
- 不使用系统时间、不生成 timestamp
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
import math

import pandas as pd

from app.allocation import allocate
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

# L3 opt-in 门禁参数集合（顺序冻结；`None` = 未提供）
_L3_GATE_RUN = ("costs", "priority", "budget_limit", "period_start", "cycle_days", "initial_budget_spent")
_L3_GATE_DAY = ("costs", "priority", "budget_available", "period_start")
_L3_ALGORITHMS = ("R0", "R1", "R2")
_DEFAULT_CYCLE_DAYS = 30

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
    # Level 3 字段（全部带默认值；L3 关闭时取默认，现有 4 元组解包/构造零破坏）
    budget_spent_today: float = 0.0
    budget_available_before: float | None = None
    budget_remaining_after: float | None = None
    period_index: int = 0
    algorithm: str | None = None


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
    # Level 3 可选接口（已接线：day 门禁 + R1/R2 allocate/预算；L3 关闭走旧行为）
    algorithm: str | None = None,
    costs: dict | None = None,
    priority: dict | None = None,
    budget_available: float | None = None,
    period_start: date | None = None,
    period_index: int = 0,
    # 实验用可选覆盖：仅替换补货决策的 daily_demand_30d（默认 None = 原行为）
    daily_demand_override: dict | None = None,
):
    """推进一天。顺序（冻结）：到货 -> 需求扣减 -> 决策 -> 下单 -> 构造审计记录。

    返回 (inventory_state', transit_state', next_po_seq', DailyResult)。
    不修改任何入参 DataFrame。
    """
    if not isinstance(current_day, date):
        raise TypeError("current_day must be a datetime.date")

    _check_opt_in_gate(
        algorithm,
        {
            "costs": costs,
            "priority": priority,
            "budget_available": budget_available,
            "period_start": period_start,
        },
        _L3_GATE_DAY,
    )

    if algorithm in ("R1", "R2"):
        for name, value in (("costs", costs), ("priority", priority), ("budget_available", budget_available)):
            if value is None:
                raise ValueError(f"{name} is required when algorithm={algorithm!r}")
        # F-1：直接调用路径与 run_simulation 一致，预算须可按整数分精确表示
        _to_cents("budget_available", budget_available)

    # 1) 到货（必须在需求扣减与下单之前：当天新建 PO 因此不会当天到货）
    transit, arrivals = _receive_open_pos(transit_state, current_day)

    # 2) 到货入账
    inventory = _apply_arrivals(inventory_state, arrivals)

    # 3) 模拟需求 -> 字面扣减 + unmet_demand
    inventory, demand_records = _apply_demand(inventory, demand_profile, simulation_day)

    # 4) 决策链：严格调用现有模块，不复制公式
    demand = calculate_demand(orders_log, current_day, coverage_start=coverage_start)
    if daily_demand_override is not None:
        demand = _apply_daily_demand_override(demand, daily_demand_override)
    batch = analyze_inventory_frame(inventory, demand, transit, as_of_date=current_day)
    frame = apply_procurement(batch.frame, constraints)

    # 4.4) R1/R2：本帧参与的 unit_cost 必须可按整数分精确表示（在分配/下单前 fail-loud）
    if algorithm in ("R1", "R2"):
        _require_unit_costs_in_cents(costs, frame["sku"])

    # 4.5) L3：allocate 在 L2 之后、下单之前；R0/R1/R2 用 allocated_qty，L3 关闭用 L2 数量
    if algorithm is not None:
        frame = allocate(
            frame,
            algorithm=algorithm,
            budget_available=budget_available,
            costs=costs,
            priority=priority,
            constraints=constraints,
        )
        qty_column = "allocated_qty"
    else:
        qty_column = "procurement_recommended_qty"

    # 5) 下单：数量来源由 L3 开关决定；provisional 不阻断
    transit, next_po_seq, action, order_placed = _place_orders(
        frame, transit, run_id, current_day, next_po_seq, qty_column=qty_column
    )

    # 6) 审计记录：DECISION（投影 + action） + SKIP（零需求等被 skip 的 SKU）
    records = _audit_records(
        run_id, simulation_day, current_day, frame, action, batch,
        algorithm=algorithm, period_index=period_index,
    )

    if algorithm in ("R1", "R2"):
        if len(frame) == 0:
            # 空决策帧（当日全部 SKU 被跳过）：无 SKU 可分配，当日无支出。
            # 字段类型与 R1/R2 非空帧保持一致（float）。
            spent_today = 0.0
            available_before = float(budget_available)
            remaining_after = float(budget_available)
        else:
            spent_today = float(frame["budget_spent_today"].iloc[0])
            remaining_after = float(frame["budget_remaining_after"].iloc[0])
            available_before = float(frame["budget_available_before"].iloc[0])
    else:
        spent_today = 0.0
        remaining_after = None
        available_before = None

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
        budget_spent_today=spent_today,
        budget_available_before=available_before,
        budget_remaining_after=remaining_after,
        period_index=period_index,
        algorithm=algorithm,
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
    # Level 3 可选接口（已接线：门禁/BudgetState/allocate；默认 None 保持旧调用兼容）
    algorithm: str | None = None,
    costs: dict | None = None,
    priority: dict | None = None,
    budget_limit: float | None = None,
    cycle_days: int | None = None,
    period_start: date | None = None,
    initial_budget_spent: float | None = None,
    audit_path=None,
    # 实验用可选覆盖：仅替换补货决策的 daily_demand_30d（默认 None = 原行为）
    daily_demand_override: dict | None = None,
) -> SimulationResult:
    """逐日推进 days 天。状态从 data_loader 播种一次（可显式覆盖，便于测试）。

    audit_path 非 None 时，每天追加写 JSONL；否则只累积在返回值里。
    """
    if not isinstance(start_day, date):
        raise TypeError("start_day must be a datetime.date")
    if not isinstance(days, int) or days < 1:
        raise ValueError("days must be an integer >= 1")

    _validate_demand_profile(demand_profile)

    _check_opt_in_gate(
        algorithm,
        {
            "costs": costs,
            "priority": priority,
            "budget_limit": budget_limit,
            "period_start": period_start,
            "cycle_days": cycle_days,
            "initial_budget_spent": initial_budget_spent,
        },
        _L3_GATE_RUN,
    )
    if algorithm is not None and algorithm not in _L3_ALGORITHMS:
        raise ValueError(
            f"unknown algorithm: {algorithm!r} (expected one of {list(_L3_ALGORITHMS)})"
        )

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

    # L3：仅在 R1/R2 下创建局部预算状态（algorithm=None/R0 不创建）
    budget_state = None
    effective_period_start = None
    if algorithm in ("R1", "R2"):
        for name, value in (("costs", costs), ("priority", priority), ("budget_limit", budget_limit)):
            if value is None:
                raise ValueError(f"{name} is required when algorithm={algorithm!r}")
        # S-2：配置键须属于 data/inventory.csv 的 SKU 参照集（非认证主数据）
        reference_skus = set(load_inventory()["sku"])
        unknown = sorted((set(costs) | set(priority)) - reference_skus)
        if unknown:
            raise ValueError(
                f"unknown SKU in L3 config (not in inventory reference set): {unknown}"
            )
        limit_cents = _to_cents("budget_limit", budget_limit)
        cycle = _resolve_cycle_days(cycle_days)
        pstart = start_day if period_start is None else period_start
        if not isinstance(pstart, date):
            raise TypeError("period_start must be a datetime.date")
        offset = (start_day - pstart).days
        if offset < 0:
            raise ValueError(f"start_day {start_day} is before period_start {pstart}")
        initial_cents = _resolve_initial_budget_spent(
            initial_budget_spent, first_day=(offset % cycle == 0), limit_cents=limit_cents
        )
        budget_state = _BudgetState(
            limit_cents=limit_cents, period_start=pstart, cycle_days=cycle, spent_cents=initial_cents
        )
        effective_period_start = pstart

    for simulation_day in range(1, days + 1):
        # 阶段 0：候选 —— 周期计划（纯函数） + 当日计算（simulate_day 不修改调用方状态）
        if budget_state is not None:
            plan = budget_state.plan(current_day)
            budget_available = plan.available
            period_index = plan.period_index
        else:
            plan = None
            budget_available = None
            period_index = 0
        inv_candidate, transit_candidate, seq_candidate, day_result = simulate_day(
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
            algorithm=algorithm,
            costs=costs,
            priority=priority,
            budget_available=budget_available,
            period_start=effective_period_start,
            period_index=period_index,
            daily_demand_override=daily_demand_override,
        )

        # 阶段 1：全量预校验（零副作用；任一失败即抛错，正式状态不变）
        spent_delta = 0
        charged_commit = None
        if budget_state is not None:
            spent_delta = _to_cents("budget_spent_today", day_result.budget_spent_today)
            if plan.spent_cents + spent_delta > plan.limit_cents:
                raise ValueError(
                    "budget_spent_today exceeds remaining budget: "
                    f"{plan.spent_cents + spent_delta} > {plan.limit_cents} (cents)"
                )
            charged_commit = dict(plan.charged)
            for placed in day_result.order_placed:
                budget_state.charge(placed["po_id"], plan.period_index, target=charged_commit)

        # 阶段 2：审计持久化（先于内存提交；失败则当日不提交）
        if audit_path is not None:
            append_audit_records(audit_path, day_result.records)

        # 阶段 3：内存提交（仅赋值/追加，不做校验、转换、序列化或磁盘 I/O）
        inventory, transit, next_po_seq = inv_candidate, transit_candidate, seq_candidate
        daily.append(day_result)
        records.extend(day_result.records)
        if budget_state is not None:
            budget_state.commit_day(plan, spent_delta, charged_commit)
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


def _place_orders(frame, transit, run_id, current_day, next_po_seq, *, qty_column="procurement_recommended_qty"):
    action = {}
    new_rows = []
    order_placed = []
    seq = next_po_seq
    for row in frame.itertuples():
        qty = float(getattr(row, qty_column))
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


def _audit_records(run_id, simulation_day, current_day, frame, action, batch, *, algorithm=None, period_index=0):
    audit_frame = frame.copy()
    audit_frame["run_id"] = run_id
    audit_frame["simulation_day"] = simulation_day
    if algorithm is not None:  # L3 审计字段仅 L3 启用时附加
        audit_frame["algorithm"] = algorithm
        audit_frame["period_index"] = period_index
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


def _apply_daily_demand_override(demand, override):
    """仅替换补货决策使用的 `daily_demand_30d`；诊断列（7d/trend）保持不变。

    override: {sku: value}，必须覆盖 demand 中全部 SKU；值必须有限且 >= 0。
    缺失 SKU / 负数 / NaN / 无穷值均 fail-loud，不静默填零。
    """
    if not isinstance(override, dict):
        raise TypeError("daily_demand_override must be a dict")
    missing = [str(sku) for sku in demand["sku"] if sku not in override]
    if missing:
        raise ValueError(f"daily_demand_override missing SKU(s): {missing}")
    values = []
    for sku in demand["sku"]:
        raw = override[sku]
        try:
            value = float(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"daily_demand_override[{sku!r}] must be a finite number: {raw!r}"
            ) from e
        if not math.isfinite(value):
            raise ValueError(f"daily_demand_override[{sku!r}] must be finite: {raw!r}")
        if value < 0:
            raise ValueError(f"daily_demand_override[{sku!r}] must be >= 0: {value}")
        values.append(value)
    out = demand.copy()
    out["daily_demand_30d"] = values
    return out


def _number(value):
    return float(value)


# --- Level 3 opt-in 门禁 / 预算状态（P4） ----------------------------------

def _check_opt_in_gate(algorithm, provided, gate):
    """`algorithm=None` 时，门禁集合中任一 L3 参数非 None 即报错（None=未提供）。"""
    if algorithm is not None:
        return
    offending = [name for name in gate if provided.get(name) is not None]
    if offending:
        raise ValueError(
            "algorithm=None disables L3 but L3 parameter(s) provided: ["
            + ", ".join(offending)
            + "]"
        )


def _require_finite_non_negative(label, value):
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{label} must be a finite number: {value!r}") from e
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite: {value!r}")
    if number < 0:
        raise ValueError(f"{label} must be >= 0: {number}")
    return number


def _resolve_cycle_days(cycle_days):
    if cycle_days is None:
        return _DEFAULT_CYCLE_DAYS
    if isinstance(cycle_days, bool) or not isinstance(cycle_days, int) or cycle_days < 1:
        raise ValueError(f"cycle_days must be an integer >= 1: {cycle_days!r}")
    return cycle_days


def _require_unit_costs_in_cents(costs, skus):
    """R1/R2：帧内 SKU 的成本必须可按整数分精确表示；缺失成本留给 allocate 报错。"""
    for sku in skus:
        value = costs.get(sku)
        if value is None:
            continue
        _to_cents(f"unit_cost for SKU {sku!r}", value)


def _to_cents(label, value):
    """金额 -> 整数分（严格）：必须有限、>=0 且可按 <=2 位小数精确表示。"""
    number = _require_finite_non_negative(label, value)
    scaled = Decimal(str(number)) * 100
    if scaled != scaled.to_integral_value():
        raise ValueError(
            f"{label} must be representable in integer cents (<= 2 decimals): {value!r}"
        )
    return int(scaled)


def _resolve_initial_budget_spent(value, *, first_day, limit_cents):
    if value is None:
        if first_day:
            return 0
        raise ValueError(
            "initial_budget_spent is required when simulation starts mid-cycle"
        )
    number_cents = _to_cents("initial_budget_spent", value)
    if first_day:
        if number_cents != 0:
            raise ValueError("initial_budget_spent must be 0.0 on the period first day")
        return 0
    if number_cents > limit_cents:
        raise ValueError(
            "initial_budget_spent must not exceed budget_limit: "
            f"{number_cents / 100.0} > {limit_cents / 100.0}"
        )
    return number_cents


@dataclass(frozen=True)
class _BudgetPlan:
    """当日预算视图（纯计算产物，不持有对 `_BudgetState` 的引用）。

    `spent_cents` 为周期首日重置后的值；`charged` 为该日开始时已登记的映射。
    """

    limit_cents: int
    period_index: int
    spent_cents: int
    charged: dict

    @property
    def available(self):
        # 初始化时已拒绝非法支出，故不使用 max(0, ...) 隐藏超预算
        return (self.limit_cents - self.spent_cents) / 100.0


@dataclass
class _BudgetState:
    """最小预算状态（仅 run_simulation 局部持有；L3 关闭/R0 不创建）。

    - `spent_cents` 是**唯一权威金额账本**（整数分，与 allocate 换算一致）。
    - `charged` 仅记录 `po_id -> period_index`（Q1=A，无逐 PO 金额）；
      周期首日清空（Q3=A）。
    """

    limit_cents: int
    period_start: date
    cycle_days: int
    spent_cents: int
    period_index: int = 0
    charged: dict = field(default_factory=dict)

    def plan(self, current_day):
        """纯计划：计算当日预算视图，**不修改本对象**。

        周期首日（`offset % cycle_days == 0`）重置 `spent_cents=0` 且清空 `charged`（不结转）。
        """
        offset = (current_day - self.period_start).days
        if offset < 0:
            raise ValueError(
                f"current_day {current_day} is before period_start {self.period_start}"
            )
        period_index = offset // self.cycle_days
        if offset % self.cycle_days == 0:  # 周期首日：重置、不结转
            return _BudgetPlan(
                limit_cents=self.limit_cents,
                period_index=period_index,
                spent_cents=0,
                charged={},
            )
        return _BudgetPlan(
            limit_cents=self.limit_cents,
            period_index=period_index,
            spent_cents=self.spent_cents,
            charged=dict(self.charged),
        )

    def charge(self, po_id, period_index, *, target):
        """把 `po_id` 登记到**副本** `target`；同周期重复登记抛 `ValueError`。

        生产调用点：`run_simulation` 的预校验阶段（先写入副本，全部通过后才整体提交）。
        """
        if po_id in target:
            raise ValueError(
                f"duplicate charge for po_id {po_id!r} in period {period_index}"
            )
        target[po_id] = period_index

    def commit_day(self, plan, spent_delta, charged):
        """提交当日预算状态：仅赋值，无校验/转换/序列化/磁盘 I/O。"""
        self.spent_cents = plan.spent_cents + spent_delta
        self.period_index = plan.period_index
        self.charged = charged
