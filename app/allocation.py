"""Level 3 budget allocation (R0 / R1 / R2). Pure functions.

Contract source: `supplychain-os-Level3预算分配-设计方案-v4.4.md` (§1.1, §5, §6, §9).

职责（且仅此）：在 L1/L2 结果帧上按冻结契约追加 L3 列。
- 无文件 IO、无系统时钟、无模块级可变状态。
- 绝不修改输入 `frame` 或注入映射。
- 只追加列；L1/L2 既有列、值、dtype、顺序不变。
- R0  -> allocated_qty, allocation_reason
- R1/R2 -> allocation_rank, allocated_qty, allocation_reason, unit_cost,
          priority, budget_available_before, budget_spent_today,
          budget_remaining_after

调用方在装配层负责把 `costs`/`priority` 映射与 `constraints` 帧装配好；
本模块不依赖任何配置文件的物理路径（A2 未决）。
"""
from __future__ import annotations

import math
from decimal import Decimal

import pandas as pd

ALGORITHMS = ("R0", "R1", "R2")

# 8 个原因码（v4.4 §6）
NO_GAP = "NO_GAP"
MOQ_INFEASIBLE = "MOQ_INFEASIBLE"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
BUDGET_INSUFFICIENT_MIN_PACK = "BUDGET_INSUFFICIENT_MIN_PACK"
SCALED_TO_ZERO = "SCALED_TO_ZERO"
ROLLED_BACK = "ROLLED_BACK"
FULLY_FULFILLED = "FULLY_FULFILLED"
PARTIALLY_FULFILLED = "PARTIALLY_FULFILLED"

REASON_CODES = (
    NO_GAP,
    MOQ_INFEASIBLE,
    BUDGET_EXHAUSTED,
    BUDGET_INSUFFICIENT_MIN_PACK,
    SCALED_TO_ZERO,
    ROLLED_BACK,
    FULLY_FULFILLED,
    PARTIALLY_FULFILLED,
)

R0_COLUMNS = ("allocated_qty", "allocation_reason")
R12_COLUMNS = (
    "allocation_rank",
    "allocated_qty",
    "allocation_reason",
    "unit_cost",
    "priority",
    "budget_available_before",
    "budget_spent_today",
    "budget_remaining_after",
)

_FRAME_BASE = ("sku", "procurement_recommended_qty")
_FRAME_RANK = ("current_stock", "daily_demand", "lead_time_days")
_CONSTRAINT_COLUMNS = ("sku", "moq", "order_multiple")
_DEFAULT_CONSTRAINT = (0.0, 1)  # 未配置 = identity（与 apply_procurement 一致）

_NEGATIVE_REMAINING = "budget_remaining_after must not be negative (allocation invariant broken)"

__all__ = [
    "ALGORITHMS",
    "REASON_CODES",
    "R0_COLUMNS",
    "R12_COLUMNS",
    "NO_GAP",
    "MOQ_INFEASIBLE",
    "BUDGET_EXHAUSTED",
    "BUDGET_INSUFFICIENT_MIN_PACK",
    "SCALED_TO_ZERO",
    "ROLLED_BACK",
    "FULLY_FULFILLED",
    "PARTIALLY_FULFILLED",
    "allocate",
]


def allocate(frame, *, algorithm, budget_available, costs, priority, constraints):
    """L1/L2 结果帧 -> 追加 L3 列的副本（纯函数，不修改入参）。

    algorithm: "R0" / "R1" / "R2"。
    budget_available: B_avail = max(0, B_limit - B_spent)，R1/R2 使用。
    costs: {sku: unit_cost > 0}；priority: {sku: int}；constraints: DataFrame[sku,moq,order_multiple]。
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm: {algorithm!r} (expected one of {list(ALGORITHMS)})")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    _require_columns(frame, _FRAME_BASE)
    _require_unique_sku(frame)

    if algorithm == "R0":
        # C-3：R0 不校验（含整数校验），L2 建议量原值透传
        return _allocate_r0(frame)

    # R1/R2 仍保留整数校验，不得放宽
    q = {sku: _require_int_quantity(frame, sku) for sku in frame["sku"]}

    _require_columns(frame, _FRAME_RANK)
    budget = _require_non_negative_scalar("budget_available", budget_available)
    lookup = _constraint_lookup(constraints)
    meta = _build_meta(frame, q, costs, priority, lookup)
    rank = _allocation_rank(frame, meta)

    if algorithm == "R1":
        allocated, reason, spent_int, remaining_int, money_scale = _allocate_r1(
            meta, rank, budget
        )
        if remaining_int < 0:
            raise ValueError(_NEGATIVE_REMAINING)
        spent = spent_int / money_scale
        remaining = remaining_int / money_scale
    else:
        # R2：金额在统一整数刻度上核算（见 `_allocate_r2`），与 R1 的浮点路径隔离。
        allocated, reason, spent_int, remaining_int, money_scale = _allocate_r2(
            meta, rank, budget
        )
        if remaining_int < 0:
            raise ValueError(_NEGATIVE_REMAINING)
        spent = spent_int / money_scale
        remaining = remaining_int / money_scale

    return _write_budgeted(frame, allocated, reason, rank, meta, budget, spent, remaining)


def _allocate_r0(frame):
    """无预算基准组：`allocated_qty` = L2 `procurement_recommended_qty` 原值透传。

    C-3：R0 不校验、不做整数化，也不读预算/成本/优先级/约束；只追加两列。
    """
    quantities = frame["procurement_recommended_qty"]
    out = frame.copy()
    out["allocated_qty"] = quantities.copy()
    out["allocation_reason"] = [
        NO_GAP if value == 0 else FULLY_FULFILLED for value in quantities
    ]
    return out


def _allocate_r1(meta, rank, budget):
    """R1：金额在统一整数刻度上核算（与 R2 同法，消除浮点边界）。

    金额（可用预算与单价）按 `Decimal(str(value))` 解析为 (系数, 指数)，
    用所有金额的最大小数位确定公共刻度，换算为 Python 整数后完成：
    步骤 2 完整订货可行性、步骤 4 λ 缩放的整包向下取整、步骤 6 回退。
    λ = B / Σ c·q，故 M·floor(q·λ/M) = M · ((q·budget_int) // (total_int·M))。
    R0/R2 不受影响。
    返回 (allocated, reason, spent_int, remaining_int, money_scale)。
    """
    skus = list(meta)
    budget_coeff, budget_exp = _decimal_parts(budget)
    cost_parts = {sku: _decimal_parts(meta[sku]["cost"]) for sku in skus}
    decimals = max(
        [max(0, -budget_exp)] + [max(0, -exp) for _, exp in cost_parts.values()]
    )
    money_scale = 10 ** decimals
    budget_int = _scaled_int(budget_coeff, budget_exp, decimals)
    cost_int = {
        sku: _scaled_int(coeff, exp, decimals) for sku, (coeff, exp) in cost_parts.items()
    }

    allocated = {sku: 0 for sku in skus}
    reason = {}

    if budget_int == 0:
        for sku in skus:
            reason[sku] = NO_GAP if meta[sku]["q"] == 0 else BUDGET_EXHAUSTED
        return allocated, reason, 0, 0, money_scale

    n_set, x_set, e_set = _split_sets(meta, budget_int, cost_int)
    for sku in n_set:
        reason[sku] = NO_GAP
    for sku in x_set:
        reason[sku] = MOQ_INFEASIBLE

    if e_set:
        total_int = sum(cost_int[sku] * meta[sku]["q"] for sku in e_set)
        if total_int <= budget_int:
            # 步骤 2：完整订货金额（整数刻度）不超预算 -> 全额
            for sku in e_set:
                allocated[sku] = meta[sku]["q"]
                reason[sku] = FULLY_FULFILLED
        else:
            # 步骤 4：整数刻度 λ 缩放与整包向下取整
            terminal = set()
            for sku in e_set:
                item = meta[sku]
                packs = (item["q"] * budget_int) // (total_int * item["multiple"])
                qi = packs * item["multiple"]
                allocated[sku] = qi
                if qi == 0:
                    reason[sku] = SCALED_TO_ZERO
                    terminal.add(sku)

            # 步骤 5′（Y1）：0 < q_i < m_i -> 归零，不再抬升
            for sku in e_set:
                if sku in terminal:
                    continue
                item = meta[sku]
                if 0 < allocated[sku] < item["m"]:
                    allocated[sku] = 0
                    reason[sku] = SCALED_TO_ZERO
                    terminal.add(sku)

            # 步骤 6（C-5）：严格 `>`（整数刻度），防御性死代码
            total = sum(cost_int[sku] * allocated[sku] for sku in e_set)
            for sku in sorted(e_set, key=lambda s: rank[s], reverse=True):
                if total <= budget_int:
                    break
                if allocated[sku] > 0 and sku not in terminal:
                    total -= cost_int[sku] * allocated[sku]
                    allocated[sku] = 0
                    reason[sku] = ROLLED_BACK
                    terminal.add(sku)

            for sku in e_set:
                if sku not in terminal and allocated[sku] > 0:
                    reason[sku] = (
                        FULLY_FULFILLED
                        if allocated[sku] == meta[sku]["q"]
                        else PARTIALLY_FULFILLED
                    )

    spent_int = sum(cost_int[sku] * allocated[sku] for sku in skus)
    remaining_int = budget_int - spent_int
    return allocated, reason, spent_int, remaining_int, money_scale


def _decimal_parts(value):
    """float -> (coeff, exponent)，满足 `value == coeff * 10**exponent`（精确）。

    使用 `Decimal(str(float(value)))`，即该 float 的**最短十进制往返表示**。
    这不恢复上游 float64 已经丢失的原始十进制精度，只忠实表达当前浮点值。
    """
    sign, digits, exponent = Decimal(str(float(value))).as_tuple()
    coeff = 0
    for digit in digits:
        coeff = coeff * 10 + digit
    return (-coeff if sign else coeff), exponent


def _scaled_int(coeff, exponent, decimals):
    """把 coeff*10**exponent 换算到 10**decimals 刻度的精确整数（decimals >= -exponent）。"""
    return coeff * (10 ** (exponent + decimals))


def _allocate_r2(meta, rank, budget):
    """R2：十进制语义解析 + 统一整数刻度金额核算。

    金额（可用预算与单价）先按 `Decimal(str(value))` 解析为 (系数, 指数)，
    再用所有金额的最大小数位确定统一刻度，换算为 Python 整数后完成：
    可用预算与 MOQ 最小可行金额比较、每包可购数（严格向下取整）、
    剩余预算更新、原因码判定与末尾预算不变量。

    不使用浮点除法，也不使用 `Decimal` 除法/round 决定包数（避免混合精度
    与上下文精度（`DivisionImpossible` / 四舍五入）问题）。
    R2 专用整数 N/X/E 划分（R1 亦使用整数刻度）。
    返回 (allocated, reason, spent_int, remaining_int, money_scale)。
    """
    skus = list(meta)
    budget_coeff, budget_exp = _decimal_parts(budget)
    cost_parts = {sku: _decimal_parts(meta[sku]["cost"]) for sku in skus}
    decimals = max(
        [max(0, -budget_exp)] + [max(0, -exp) for _, exp in cost_parts.values()]
    )
    money_scale = 10 ** decimals
    budget_int = _scaled_int(budget_coeff, budget_exp, decimals)
    cost_int = {
        sku: _scaled_int(coeff, exp, decimals) for sku, (coeff, exp) in cost_parts.items()
    }

    allocated = {sku: 0 for sku in skus}
    reason = {}

    if budget_int == 0:
        for sku in skus:
            reason[sku] = NO_GAP if meta[sku]["q"] == 0 else BUDGET_EXHAUSTED
        return allocated, reason, 0, 0, money_scale

    # 整数刻度 N/X/E 划分。
    n_set, x_set, e_set = [], [], []
    for sku in skus:
        item = meta[sku]
        if item["q"] == 0:
            n_set.append(sku)
        elif cost_int[sku] * item["m"] > budget_int:
            x_set.append(sku)
        else:
            e_set.append(sku)
    for sku in n_set:
        reason[sku] = NO_GAP
    for sku in x_set:
        reason[sku] = MOQ_INFEASIBLE

    remaining_int = budget_int
    for sku in sorted(e_set, key=lambda s: rank[s]):
        item = meta[sku]
        unit = cost_int[sku]
        if remaining_int <= 0:
            reason[sku] = BUDGET_EXHAUSTED
            continue
        if remaining_int < unit * item["m"]:
            reason[sku] = BUDGET_INSUFFICIENT_MIN_PACK
            continue
        packs = min(
            remaining_int // (unit * item["multiple"]),
            item["q"] // item["multiple"],
        )
        if packs * item["multiple"] < item["m"]:
            reason[sku] = BUDGET_INSUFFICIENT_MIN_PACK
            continue
        qi = packs * item["multiple"]
        remaining_int -= unit * qi
        allocated[sku] = qi
        reason[sku] = FULLY_FULFILLED if qi == item["q"] else PARTIALLY_FULFILLED

    spent_int = sum(cost_int[sku] * allocated[sku] for sku in skus)
    return allocated, reason, spent_int, remaining_int, money_scale


def _split_sets(meta, budget_int, cost_int):
    """N/X/E 划分（整数刻度）；X 阈值：c_i·m_i > B_avail。"""
    n_set, x_set, e_set = [], [], []
    for sku, item in meta.items():
        if item["q"] == 0:
            n_set.append(sku)
        elif cost_int[sku] * item["m"] > budget_int:
            x_set.append(sku)
        else:
            e_set.append(sku)
    return n_set, x_set, e_set


def _build_meta(frame, q, costs, priority, lookup):
    meta = {}
    for sku in frame["sku"]:
        cost = _require_positive("unit_cost", sku, costs.get(sku))
        prio = _require_int("priority", sku, priority.get(sku))
        moq, multiple = lookup.get(sku, _DEFAULT_CONSTRAINT)
        if q[sku] > 0 and (q[sku] < moq or q[sku] % multiple != 0):
            raise ValueError(
                f"procurement_recommended_qty {q[sku]} violates MOQ/multiple contract "
                f"for SKU {sku!r}: moq={moq}, order_multiple={multiple}"
            )
        meta[sku] = {
            "q": q[sku],
            "cost": cost,
            "priority": prio,
            "moq": moq,
            "multiple": multiple,
            "m": _min_order_qty(moq, multiple),
        }
    return meta


def _allocation_rank(frame, meta):
    """全体 SKU 按 π_i 升序编号，从 1 起；mergesort + sku 全序。

    空帧（当日全部 SKU 被跳过）无排序对象，直接返回空 rank，
    由 `_write_budgeted` 生成结构正确的空 L3 结果。
    """
    if len(frame) == 0:
        return {}
    rows = []
    for sku in frame["sku"]:
        item = meta[sku]
        row = frame.loc[frame["sku"] == sku].iloc[0]
        daily = _require_positive("daily_demand", sku, row["daily_demand"])
        current = _require_finite("current_stock", sku, row["current_stock"])
        lead = _require_finite("lead_time_days", sku, row["lead_time_days"])
        coverage = current / daily  # C3：未舍入原始值
        rows.append(
            {
                "sku": sku,
                "priority": item["priority"],
                "cd_minus_l": coverage - lead,
                "neg_q": -item["q"],
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        ["priority", "cd_minus_l", "neg_q", "sku"], kind="mergesort"
    )
    return {row.sku: rank for rank, row in enumerate(ranked.itertuples(), start=1)}


def _write_budgeted(frame, allocated, reason, rank, meta, budget, spent, remaining):
    out = frame.copy()
    order = list(frame["sku"])
    out["allocation_rank"] = _int_column(frame, [rank[sku] for sku in order])
    out["allocated_qty"] = _int_column(frame, [allocated[sku] for sku in order])
    out["allocation_reason"] = [reason[sku] for sku in order]
    out["unit_cost"] = [float(meta[sku]["cost"]) for sku in order]
    out["priority"] = _int_column(frame, [meta[sku]["priority"] for sku in order])
    out["budget_available_before"] = [float(budget)] * len(order)
    out["budget_spent_today"] = [float(spent)] * len(order)
    out["budget_remaining_after"] = [float(remaining)] * len(order)
    return out


def _min_order_qty(moq, multiple):
    return max(1, math.ceil(moq / multiple)) * multiple


def _constraint_lookup(constraints):
    if constraints is None or len(constraints) == 0:
        return {}
    _require_columns(constraints, _CONSTRAINT_COLUMNS)
    lookup = {}
    for row in constraints.itertuples():
        moq = _require_non_negative("moq", row.sku, row.moq)
        multiple = _require_int("order_multiple", row.sku, row.order_multiple)
        if multiple < 1:
            raise ValueError(f"order_multiple must be >= 1: {row.sku!r}")
        lookup[row.sku] = (moq, multiple)
    return lookup


# --- validation helpers -----------------------------------------------------

def _require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")


def _require_unique_sku(frame):
    duplicated = frame["sku"][frame["sku"].duplicated()]
    if len(duplicated):
        raise ValueError(f"duplicate SKU in frame: {sorted(set(duplicated.astype(str)))}")


def _require_int_quantity(frame, sku):
    value = _require_finite("procurement_recommended_qty", sku, _cell(frame, sku, "procurement_recommended_qty"))
    if value < 0:
        raise ValueError(f"procurement_recommended_qty must be >= 0 for SKU {sku!r}: {value}")
    if value != math.floor(value):
        raise ValueError(f"procurement_recommended_qty must be an integer for SKU {sku!r}: {value}")
    return int(value)


def _cell(frame, sku, column):
    return frame.loc[frame["sku"] == sku, column].iloc[0]


def _require_finite(label, sku, value):
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid {label} for SKU {sku!r}: {value!r}") from e
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite for SKU {sku!r}: {value!r}")
    return number


def _require_positive(label, sku, value):
    number = _require_finite(label, sku, value)
    if number <= 0:
        raise ValueError(f"{label} must be > 0 for SKU {sku!r}: {number}")
    return number


def _require_non_negative(label, sku, value):
    number = _require_finite(label, sku, value)
    if number < 0:
        raise ValueError(f"{label} must be >= 0 for SKU {sku!r}: {number}")
    return number


def _require_non_negative_scalar(label, value):
    return _require_non_negative(label, "<scalar>", value)


def _require_int(label, sku, value):
    number = _require_finite(label, sku, value)
    if number != math.floor(number):
        raise ValueError(f"{label} must be an integer for SKU {sku!r}: {number}")
    return int(number)


def _int_column(frame, values):
    return pd.Series(values, index=frame.index, dtype="int64")
