"""Level 3 实验辅助（纯函数）：数据校验 / B1 需求 / 优先级 / B_full 标定 / 指标。

冻结口径（见实验计划修订）：
- B1：窗口 [start_day-29, start_day]，分母 30，qty=(total+15)//30，恒定不滚动。
- 优先级：P_cat / P_lead / P_rev；平局由 app.allocation._allocation_rank 二级键处理。
- B_full：R0 逐日建议量 × 整数分单价；tier limit = (num*B_full)//den。
- 指标：服务率 = 1 - Σunmet/Σdemand；零需求返回 null 并置标志。

本模块不做 IO、不写文件、不修改核心算法。
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pandas as pd

from app.simulator import _to_cents  # 与核心一致的整数分转换（实验工具复用私有函数）

WINDOW_DAYS = 30
DEMAND_DENOM = 30
HALF = 15  # DEMAND_DENOM // 2，用于 half-up

# 档位以整数分数表示：1/4, 1/2, 3/4, 1, 3/2
TIERS = ((1, 4), (1, 2), (3, 4), (1, 1), (3, 2))
L3_ALGORITHMS = ("R1", "R2")
PRIORITY_RULES = ("P_cat", "P_lead", "P_rev")
_DEFAULT_CONSTRAINT = (0.0, 1)

# 库存情景（仅内存覆盖 current_stock；不修改数据文件）
# BASELINE = 原始数据（零补货基准）；S1-C = 库存压力情景。
SCENARIOS = {
    "BASELINE": {},
    "S1-C": {"SKU001": 0.0, "SKU003": 0.0, "SKU005": 0.0},
}
SCENARIO_ORDER = ("BASELINE", "S1-C")

# 主实验固定 B1 补货决策需求（研究自定义模拟参数，非海尔真实数据）
FIXED_B1_DAILY = {
    "SKU001": 13.0,
    "SKU002": 13.0,
    "SKU003": 4.0,
    "SKU004": 8.0,
    "SKU005": 4.0,
}


def apply_scenario(inventory, scenario):
    """返回 (inventory_copy, changes)；原 `inventory` 不被修改。

    changes 逐条记录被覆盖 SKU 的原始值、覆盖值与原因，供 manifest 使用。
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario!r}")
    overrides = SCENARIOS[scenario]
    inv = inventory.copy()
    current = dict(zip(inventory["sku"].astype(str), inventory["current_stock"]))
    changes = []
    for sku, value in overrides.items():
        if sku not in current:
            raise ValueError(f"scenario {scenario!r} overrides unknown sku: {sku!r}")
        changes.append(
            {
                "scenario": scenario,
                "sku": sku,
                "original_current_stock": float(current[sku]),
                "override_current_stock": float(value),
                "reason": "S1 stress scenario: force initial stock below ROP (in-memory only)",
            }
        )
        inv.loc[inv["sku"].astype(str) == sku, "current_stock"] = float(value)
    return inv, changes


# --- 数据校验 ---------------------------------------------------------------

def validate_inputs(skus, costs, constraints, products, orders=None, in_transit=None):
    """返回问题字符串列表；空列表表示通过。不抛异常，便于一次性报告全部问题。"""
    problems = []
    sku_list = [str(s) for s in skus]
    sku_set = set(sku_list)
    if len(sku_set) != len(sku_list):
        problems.append("inventory 存在重复 sku")
    if not sku_set:
        problems.append("inventory 无 sku")

    cost_skus = set(str(s) for s in costs)
    missing_costs = sorted(sku_set - cost_skus)
    if missing_costs:
        problems.append(f"缺少成本配置的 sku: {missing_costs}")
    for sku in sorted(sku_set & cost_skus):
        try:
            _to_cents(f"unit_cost[{sku}]", costs[sku])
        except ValueError as exc:
            problems.append(str(exc))

    if constraints is not None and len(constraints):
        cons_skus = set(str(s) for s in constraints["sku"])
        missing_cons = sorted(sku_set - cons_skus)
        if missing_cons:
            problems.append(f"缺少采购约束的 sku: {missing_cons}")
        for row in constraints.itertuples():
            if float(row.moq) < 0:
                problems.append(f"moq 为负: {row.sku!r}={row.moq}")
            if float(row.order_multiple) != int(row.order_multiple) or int(row.order_multiple) < 1:
                problems.append(f"order_multiple 非正整数: {row.sku!r}={row.order_multiple}")
    else:
        problems.append("采购约束为空")

    prod_skus = set(str(s) for s in products["sku"]) if products is not None else set()
    missing_prod = sorted(sku_set - prod_skus)
    if missing_prod:
        problems.append(f"缺少 products 记录的 sku: {missing_prod}")

    if orders is not None:
        order_skus = set(str(s) for s in orders["sku"])
        extra = sorted(order_skus - sku_set)
        if extra:
            problems.append(f"orders 出现 inventory 之外的 sku: {extra}")

    if in_transit is not None and len(in_transit):
        transit_skus = set(str(s) for s in in_transit["sku"])
        extra = sorted(transit_skus - sku_set)
        if extra:
            problems.append(f"in_transit 出现 inventory 之外的 sku: {extra}")

    return problems


# --- B1 需求生成 -------------------------------------------------------------

def b1_daily_qty(start_day, orders, skus):
    """每 SKU 恒定日需求：窗口 [start_day-29, start_day] 总量 half-up / 30。"""
    dates = pd.to_datetime(orders["order_date"], format="%Y-%m-%d")
    lo = pd.Timestamp(start_day) - pd.Timedelta(days=WINDOW_DAYS - 1)
    hi = pd.Timestamp(start_day)
    window = orders[(dates >= lo) & (dates <= hi)]
    totals = window.groupby("sku")["qty"].sum()
    out = {}
    for sku in skus:
        total = int(totals.get(sku, 0))
        out[str(sku)] = (total + HALF) // DEMAND_DENOM
    return out


def build_demand_profile(start_day, days, orders, skus):
    """返回 (demand_profile DataFrame[sku,simulation_day,qty], {sku: qty})。恒定、无滚动。"""
    if not isinstance(days, int) or days < 1:
        raise ValueError("days must be an integer >= 1")
    per_sku = b1_daily_qty(start_day, orders, skus)
    rows = [
        {"sku": sku, "simulation_day": day, "qty": per_sku[sku]}
        for sku in skus
        for day in range(1, days + 1)
    ]
    profile = pd.DataFrame(rows, columns=["sku", "simulation_day", "qty"])
    profile["simulation_day"] = profile["simulation_day"].astype(int)
    profile["qty"] = profile["qty"].astype(int)
    return profile, per_sku


# --- 优先级 -----------------------------------------------------------------

def priority_map_cat(skus, products):
    """类别升序编号（A->1, B->2 ...），模拟关键度（研究自定义，非企业真实优先级）。"""
    cat = dict(zip(products["sku"].astype(str), products["category"]))
    levels = {c: i + 1 for i, c in enumerate(sorted({cat[str(s)] for s in skus}))}
    return {str(s): levels[cat[str(s)]] for s in skus}


def priority_map_lead(skus, inventory):
    """交期越长优先级越高（distinct lead 降序编号）。"""
    lead = dict(zip(inventory["sku"].astype(str), inventory["lead_time_days"]))
    levels = {v: i + 1 for i, v in enumerate(sorted({lead[str(s)] for s in skus}, reverse=True))}
    return {str(s): levels[lead[str(s)]] for s in skus}


def priority_map_rev(skus, products):
    """P_cat 的类别顺序取反。"""
    p = priority_map_cat(skus, products)
    k = max(p.values())
    return {s: k + 1 - v for s, v in p.items()}


def priority_map(rule, skus, products, inventory):
    if rule == "P_cat":
        return priority_map_cat(skus, products)
    if rule == "P_lead":
        return priority_map_lead(skus, inventory)
    if rule == "P_rev":
        return priority_map_rev(skus, products)
    raise ValueError(f"unknown priority rule: {rule!r}")


# --- B_full 标定 / 档位 ------------------------------------------------------

def calibrate_bfull_cents(r0_result, costs):
    """R0 逐日建议量 × 整数分单价，聚合为周期累计（整数分）。"""
    total = 0
    for day in r0_result.daily:
        for row in day.decision_frame.itertuples():
            q = float(row.procurement_recommended_qty)
            if q != int(q) or q < 0:
                raise ValueError(
                    f"procurement_recommended_qty 非非负整数: sku={row.sku!r}, q={q!r}"
                )
            total += _to_cents(f"unit_cost[{row.sku}]", costs[row.sku]) * int(q)
    return total


def tier_limit_cents(bfull_cents, num, den):
    """整数分向下取整；tier=1/1 时精确等于 B_full_cents。"""
    if bfull_cents < 0:
        raise ValueError("bfull_cents must be >= 0")
    return (num * bfull_cents) // den


def tier_label(num, den):
    return f"{num}_{den}"


# --- 指标 -------------------------------------------------------------------

def constraint_lookup(constraints):
    if constraints is None or len(constraints) == 0:
        return {}
    return {
        str(row.sku): (float(row.moq), int(row.order_multiple))
        for row in constraints.itertuples()
    }


def aggregate_metrics(result, *, algorithm, priority_rule, tier, budget_limit_cents, constraints):
    """逐算例指标。零需求/空集合返回 null 并置标志，不伪造 0/1。"""
    lookup = constraint_lookup(constraints)
    is_l3 = algorithm in ("R0",) + L3_ALGORITHMS

    sum_demand = 0.0
    sum_unmet = 0.0          # 含积压的每日“欠交到零”量之和（backorder-inclusive）
    sum_new_unmet = 0.0      # 仅当日新增未满足量（lost-sales 口径，cutoff 前库存不为负）
    sku_days_with_demand = 0
    sku_days_stockout = 0
    total_spent_cents = 0
    pack_violations = 0
    budget_violations = 0
    order_count = 0
    per_sku = {}

    for day in result.daily:
        if algorithm in L3_ALGORITHMS:
            total_spent_cents += _to_cents("budget_spent_today", day.budget_spent_today)
        if day.budget_remaining_after is not None and day.budget_remaining_after < 0:
            budget_violations += 1
        order_count += len(day.order_placed)

        for item in day.demand:
            sku = str(item["sku"])
            d = float(item["demand_qty"])
            u = float(item["unmet_demand"])
            sum_demand += d
            sum_unmet += u
            sum_new_unmet += max(0.0, d - max(0.0, float(item["stock_before"])))
            bucket = per_sku.setdefault(
                sku, {"demand_qty": 0.0, "unmet_demand": 0.0,
                      "procurement_recommended_qty": 0.0, "allocated_qty": 0.0}
            )
            bucket["demand_qty"] += d
            bucket["unmet_demand"] += u
            if d > 0:
                sku_days_with_demand += 1
                if u > 0:
                    sku_days_stockout += 1

        if is_l3:
            for row in day.decision_frame.itertuples():
                sku = str(row.sku)
                rec = float(row.procurement_recommended_qty)
                alloc = float(row.allocated_qty)
                bucket = per_sku.setdefault(
                    sku, {"demand_qty": 0.0, "unmet_demand": 0.0,
                          "procurement_recommended_qty": 0.0, "allocated_qty": 0.0}
                )
                bucket["procurement_recommended_qty"] += rec
                bucket["allocated_qty"] += alloc
                moq, multiple = lookup.get(sku, _DEFAULT_CONSTRAINT)
                if rec == 0:
                    ok = alloc == 0
                else:
                    ok = alloc == 0 or (alloc >= moq and alloc % multiple == 0)
                if not ok:
                    pack_violations += 1

    if sum_demand > 0:
        service_rate = 1.0 - sum_unmet / sum_demand
        unfilled_demand_ratio = sum_unmet / sum_demand
    else:
        service_rate = None
        unfilled_demand_ratio = None

    stockout_rate = (
        sku_days_stockout / sku_days_with_demand if sku_days_with_demand > 0 else None
    )

    # 期末积压：当前库存语义下 stock_after<0 的绝对值之和。
    # 注意：这是“负库存结余”，不是逐订单追踪的欠交；负值仅表示未满足需求未回补。
    ending_backlog_qty = 0.0
    if result.daily:
        ending_backlog_qty = sum(
            max(0.0, -float(item["stock_after"])) for item in result.daily[-1].demand
        )

    if algorithm in L3_ALGORITHMS and budget_limit_cents:
        budget_utilization = total_spent_cents / budget_limit_cents
        remaining_cents = budget_limit_cents - total_spent_cents
    else:
        budget_utilization = None
        remaining_cents = None

    return {
        "algorithm": algorithm,
        "priority_rule": priority_rule,
        "tier": None if tier is None else {"num": tier[0], "den": tier[1]},
        "sum_demand_qty": sum_demand,
        "sum_unmet_demand": sum_unmet,
        "sum_new_unmet_demand": sum_new_unmet,
        "ending_backlog_qty": ending_backlog_qty,
        "service_rate": service_rate,
        "unfilled_demand_ratio": unfilled_demand_ratio,
        "stockout_rate": stockout_rate,
        "stockout_rate_denominator": sku_days_with_demand,
        "zero_demand_denominator": sum_demand == 0,
        "budget_limit_cents": budget_limit_cents,
        "total_spent_cents": total_spent_cents if algorithm in L3_ALGORITHMS else None,
        "remaining_cents": remaining_cents,
        "budget_utilization": budget_utilization,
        "pack_violation_count": pack_violations,
        "budget_violation_count": budget_violations,
        "order_count": order_count,
        "per_sku": per_sku,
    }


# --- 收敛检查 ---------------------------------------------------------------

def convergence_report(result, r0_result, algorithm):
    """tier=1.0：A 建议量一致 vs R0；B 分配满额。返回 {passed, diffs}。"""
    r0 = {}
    for day in r0_result.daily:
        for row in day.decision_frame.itertuples():
            r0[(day.simulation_day, str(row.sku))] = float(row.procurement_recommended_qty)

    a_diffs = []
    b_diffs = []
    for day in result.daily:
        for row in day.decision_frame.itertuples():
            key = (day.simulation_day, str(row.sku))
            rec = float(row.procurement_recommended_qty)
            alloc = float(row.allocated_qty)
            if key not in r0:
                a_diffs.append({"day": key[0], "sku": key[1], "reason": "missing_in_r0"})
            elif rec != r0[key]:
                a_diffs.append({"day": key[0], "sku": key[1], "r1r2": rec, "r0": r0[key]})
            if alloc != rec:
                b_diffs.append({"day": key[0], "sku": key[1], "allocated": alloc, "recommended": rec})

    return {
        "algorithm": algorithm,
        "recommended_matches_r0": not a_diffs,
        "allocated_equals_recommended": not b_diffs,
        "passed": not a_diffs and not b_diffs,
        "recommended_diffs": a_diffs,
        "allocation_diffs": b_diffs,
    }


# --- manifest ---------------------------------------------------------------

def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_snapshot(repo):
    def _run(args):
        try:
            return subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # pragma: no cover - 非 git 环境
            return None

    return {
        "head": _run(["rev-parse", "HEAD"]),
        "status_short": _run(["status", "--short"]),
    }
