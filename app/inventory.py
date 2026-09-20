"""Core inventory calculations.

输入契约（STEP 4 起）：
    库存事实（inventory.csv）+ Demand Engine 输出（app/demand.py）+ 在途事实（in_transit.csv）
    `daily_demand` 一律来自 Demand Engine，**不再来自 inventory.csv**。
    在途只参与 `inventory_position` / `recommended_order_qty`，不参与 `coverage_days` / `status`。
"""
from dataclasses import asdict, dataclass, fields, replace
from datetime import date

import pandas as pd

from app.demand import COLUMNS as _DEMAND_COLUMNS

# analyze_frame() 的输入契约；在途两列为可选（缺列按 0，见 OPTIONAL_FIELDS）
FIELDS = (
    "sku",
    "current_stock",
    "daily_demand",
    "safety_stock",
    "lead_time_days",
    "in_transit_stock",
    "overdue_in_transit",
)

# 可选输入列：缺失时按 0。仅为 V0.1/V0.2 单表入口兼容；
# 批量入口 analyze_inventory_frame() 始终传真实值，不用该默认值掩盖缺失数据。
OPTIONAL_FIELDS = {"in_transit_stock": 0.0, "overdue_in_transit": 0.0}

# Demand Engine 的输出列（除 sku 外），随决策结果一起传递，不改写、不覆盖
DEMAND_COLUMNS = tuple(c for c in _DEMAND_COLUMNS if c != "sku")

# in_transit.csv 契约（STEP 4）
IN_TRANSIT_FIELDS = ("po_id", "sku", "qty", "expected_date", "status")
IN_TRANSIT_STATUSES = ("OPEN", "ARRIVED", "CANCELLED")


@dataclass
class InventoryResult:
    sku: str
    current_stock: float
    daily_demand: float
    safety_stock: float
    lead_time_days: int
    coverage_days: float
    reorder_point: float
    status: str
    recommended_order_qty: float
    in_transit_stock: float
    overdue_in_transit: float
    inventory_position: float
    reason: str


def analyze_inventory(
    sku: str,
    current_stock: float,
    daily_demand: float,
    safety_stock: float,
    lead_time_days: int,
    in_transit_stock: float = 0.0,
    overdue_in_transit: float = 0.0,
) -> InventoryResult:
    values = {
        "current_stock": current_stock,
        "daily_demand": daily_demand,
        "safety_stock": safety_stock,
        "lead_time_days": lead_time_days,
        "in_transit_stock": in_transit_stock,
        "overdue_in_transit": overdue_in_transit,
    }
    null = next((name for name, v in values.items() if pd.isna(v)), None)
    if null is not None:
        raise ValueError(f"{null} must not be null")

    if daily_demand <= 0:
        raise ValueError("daily_demand must be greater than 0")
    if in_transit_stock < 0 or overdue_in_transit < 0:
        raise ValueError("in_transit_stock and overdue_in_transit must not be negative")
    if overdue_in_transit > in_transit_stock:
        raise ValueError("overdue_in_transit must not exceed in_transit_stock (it is a subset)")

    # 现货覆盖天数：只用**现货**。在途未到货，不代表今天能卖
    coverage_days = current_stock / daily_demand
    # ✅核心业务公式：补货点 = 交期内消耗 + 安全库存（需求侧策略参数，不含供应流水）
    reorder_point = daily_demand * lead_time_days + safety_stock
    # 库存位置 = 现货 + OPEN 在途（已下单未到货的供应）
    inventory_position = current_stock + in_transit_stock

    # 判断库存状态：仍看**现货**（物理库存状态），不看库存位置
    if current_stock < safety_stock:
        status = "CRITICAL"    # 危急
    elif current_stock < reorder_point:
        status = "REORDER"     # 需要补货
    elif coverage_days > 180:
        status = "OVERSTOCK"   # 库存过剩
    else:
        status = "NORMAL"      # 正常

    # 需要采购多少：按**库存位置**算，避免已下单未到货时重复采购；不足才补，不为负
    recommended_order_qty = max(
        0,
        reorder_point - inventory_position
    )

    return InventoryResult(
        sku=sku,
        current_stock=current_stock,
        daily_demand=daily_demand,
        safety_stock=safety_stock,
        lead_time_days=lead_time_days,
        coverage_days=round(coverage_days, 2),
        reorder_point=round(reorder_point, 2),
        status=status,
        recommended_order_qty=round(recommended_order_qty, 2),
        in_transit_stock=in_transit_stock,
        overdue_in_transit=overdue_in_transit,
        inventory_position=round(inventory_position, 2),
        reason=_reason(current_stock, reorder_point, in_transit_stock, recommended_order_qty),
    )


def _reason(current_stock: float, reorder_point: float, in_transit_stock: float, recommended_order_qty: float) -> str:
    """短标签解释采购建议的成因（不堆砌长句）。"""
    if recommended_order_qty > 0:
        return "IN_TRANSIT_INSUFFICIENT" if in_transit_stock > 0 else "STOCK_BELOW_REORDER_POINT"
    if current_stock < reorder_point and in_transit_stock > 0:
        return "IN_TRANSIT_COVERS_SHORTAGE"
    return "SUPPLY_SUFFICIENT"


# --- thin bridge: inventory.csv -> analyze_inventory ---------------------

@dataclass
class BatchResult:
    """Batch output: the analyzable rows plus one warning per skipped row."""
    frame: pd.DataFrame
    analyzed_count: int
    skipped_count: int
    warnings: list[dict]


def analyze_frame(df: pd.DataFrame) -> BatchResult:
    """One row per analyzable SKU, worst coverage first. Bad rows are skipped with a warning.

    在途两列为可选输入：缺列按 0（仅为单表入口兼容）。批量入口 analyze_inventory_frame()
    始终传入真实的 in_transit_stock / overdue_in_transit。
    """
    rows, warnings = [], []
    for record in _snapshot_records(df):
        try:
            rows.append(asdict(analyze_inventory(**record)))
        except ValueError as e:
            message = str(e)
            warnings.append(
                {
                    "sku": record["sku"],
                    "field": message.split()[0],  # analyze_inventory 的消息以字段名开头
                    "message": message,
                }
            )

    frame = pd.DataFrame(rows, columns=[f.name for f in fields(InventoryResult)])
    return BatchResult(
        frame=frame.sort_values("coverage_days"),
        analyzed_count=len(rows),
        skipped_count=len(warnings),
        warnings=warnings,
    )


# --- assembly: inventory facts + Demand Engine -> analyze_frame ----------

def analyze_inventory_frame(
    inventory_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    in_transit_df: pd.DataFrame | None = None,
    as_of_date: date | None = None,
) -> BatchResult:
    """装配入口：库存事实 + Demand Engine 输出 + 在途事实 -> BatchResult。

    按 sku 做显式对齐（不 drop、不 fill、不猜），对齐规则见 _aligned_snapshot()。
    demand 与在途的口径、质量结论原样传递到结果帧，不做任何改写。

    in_transit_df=None 表示本次决策**未接入在途数据**（不等于“没有在途”），
    此时 in_transit_stock = 0；传入 DataFrame 时必须同时给 as_of_date，
    否则抛 TypeError（禁止在核心计算里读系统时间）。
    """
    metrics = None
    if in_transit_df is not None:
        if as_of_date is None:
            raise TypeError("as_of_date is required when in_transit_df is provided")
        metrics = in_transit_metrics(in_transit_df, as_of_date)

    aligned = _aligned_snapshot(inventory_df, demand_df, metrics)
    batch = analyze_frame(aligned)
    return replace(
        batch,
        frame=batch.frame.merge(aligned[["sku", *DEMAND_COLUMNS]], on="sku", how="left"),
    )


def in_transit_metrics(in_transit_df: pd.DataFrame, as_of_date: date) -> pd.DataFrame:
    """在途事实 -> 一行一个 SKU：`in_transit_stock`（OPEN 合计）/ `overdue_in_transit`（其中已逾期）。

    口径（STEP 4 冻结）：
    - 只有 status == "OPEN" 计入；ARRIVED / CANCELLED **不计入**。
    - OPEN 且 expected_date < as_of_date 仍**计入** in_transit_stock，同时计入 overdue_in_transit。
      overdue 是风险信息，不是自动扣减供应的理由：当前数据无法证明“逾期 OPEN”一定无效。
    - overdue_in_transit 是 in_transit_stock 的**子集**，绝不与本数重复相加
      （inventory_position = current_stock + in_transit_stock，不再加 overdue）。
    脏数据一律 fail-fast：不做 abs()、不 fillna、不静默转成 as_of_date。
    """
    if not isinstance(as_of_date, date):
        raise TypeError("as_of_date must be a datetime.date")

    missing = [c for c in IN_TRANSIT_FIELDS if c not in in_transit_df.columns]
    if missing:
        raise ValueError(f"in_transit missing columns: {missing}")

    duplicated = in_transit_df["po_id"].duplicated()
    if duplicated.any():
        raise ValueError(
            f"duplicate po_id: {sorted(in_transit_df['po_id'][duplicated].unique().tolist())}"
        )

    for column in IN_TRANSIT_FIELDS:
        nulls = in_transit_df[column].isna()
        if nulls.any():
            raise ValueError(
                f"null {column} in {int(nulls.sum())} in_transit row(s): "
                f"{_keys(in_transit_df, nulls, 'po_id')}"
            )

    unknown = sorted(set(in_transit_df["status"]) - set(IN_TRANSIT_STATUSES))
    if unknown:
        raise ValueError(f"invalid status: {unknown} (allowed: {list(IN_TRANSIT_STATUSES)})")

    try:
        qty = pd.to_numeric(in_transit_df["qty"])
    except (ValueError, TypeError) as e:
        raise ValueError(f"invalid in_transit qty: {e}") from e
    bad_qty = qty <= 0
    if bad_qty.any():
        raise ValueError(
            f"in_transit qty must be > 0: {_keys(in_transit_df, bad_qty, 'po_id')}"
        )

    try:
        expected = pd.to_datetime(in_transit_df["expected_date"], format="%Y-%m-%d")
    except (ValueError, TypeError) as e:
        raise ValueError(f"invalid expected_date (expected YYYY-MM-DD): {e}") from e

    data = in_transit_df.assign(qty=qty, expected_date=expected)
    open_rows = data[data["status"] == "OPEN"]
    overdue_rows = open_rows[open_rows["expected_date"] < pd.Timestamp(as_of_date)]

    skus = sorted(data["sku"].unique())
    # 没有 OPEN 记录的 SKU = 0（结构性补零；脏数据已在上面抛错）
    total = open_rows.groupby("sku")["qty"].sum().reindex(skus, fill_value=0)
    overdue = overdue_rows.groupby("sku")["qty"].sum().reindex(skus, fill_value=0)
    return pd.DataFrame(
        {
            "sku": skus,
            "in_transit_stock": total.to_numpy().astype(float),
            "overdue_in_transit": overdue.to_numpy().astype(float),
        }
    )


def _aligned_snapshot(
    inventory_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    transit_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """把 Demand Engine 输出（+ 在途聚合）接到库存事实上，产出 analyze_frame() 能吃的单表快照。

    `daily_demand` = Demand Engine 的 `daily_demand_30d`（STEP 3 口径，
    不用 7D、不做加权——那属于后续需求策略迭代）。
    """
    if "daily_demand" in inventory_df.columns:
        raise ValueError(
            "inventory contains daily_demand: demand must come from the Demand Engine, "
            "not from the inventory table"
        )

    _check_unique_sku(inventory_df, "inventory")
    _check_unique_sku(demand_df, "demand")

    inventory_skus = pd.Index(inventory_df["sku"])
    demand_skus = pd.Index(demand_df["sku"])
    missing = sorted(inventory_skus.difference(demand_skus).tolist())
    if missing:
        raise ValueError(f"missing demand for SKU: {missing}")
    extra = sorted(demand_skus.difference(inventory_skus).tolist())
    if extra:
        raise ValueError(f"extra demand SKU: {extra}")

    aligned = inventory_df.merge(demand_df[["sku", *DEMAND_COLUMNS]], on="sku", how="left")
    aligned["daily_demand"] = aligned["daily_demand_30d"]

    if transit_metrics is None:
        # 本次未接入在途数据：没有在途事实可用，按 0（与“没有 OPEN 记录”不同，见 in_transit_metrics）
        aligned["in_transit_stock"] = 0.0
        aligned["overdue_in_transit"] = 0.0
    else:
        transit_skus = pd.Index(transit_metrics["sku"])
        unknown = sorted(transit_skus.difference(inventory_skus).tolist())
        if unknown:
            raise ValueError(f"extra in_transit SKU: {unknown}")
        # inventory 有 SKU 而 in_transit 无该 SKU 是合法的：**无 OPEN 在途记录 = 0**，
        # 这是“没有在途”的业务事实，不是缺失数据错误。用 reindex 结构性补零，不做 fillna。
        metrics = (
            transit_metrics.set_index("sku")
            .reindex(inventory_skus, fill_value=0)
            .reset_index()
        )
        aligned = aligned.merge(metrics, on="sku", how="left")
    return aligned


def _keys(df: pd.DataFrame, mask, column: str) -> list:
    if column in df.columns:
        return df.loc[mask, column].tolist()
    return list(df.index[mask])


def _snapshot_records(df: pd.DataFrame) -> list[dict]:
    records = df[[c for c in FIELDS if c in df.columns]].to_dict("records")
    for record in records:
        for column, default in OPTIONAL_FIELDS.items():
            record.setdefault(column, default)
    return records


def _check_unique_sku(df: pd.DataFrame, label: str) -> None:
    duplicates = sorted(df["sku"][df["sku"].duplicated()].unique().tolist())
    if duplicates:
        raise ValueError(f"duplicate {label} SKU: {duplicates}")
