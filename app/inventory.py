"""Core inventory calculations.

输入契约（STEP 3 起）：
    库存事实（inventory.csv）+ Demand Engine 输出（app/demand.py）
    `daily_demand` 一律来自 Demand Engine，**不再来自 inventory.csv**。
"""
from dataclasses import asdict, dataclass, fields, replace

import pandas as pd

from app.demand import COLUMNS as _DEMAND_COLUMNS

# fields analyze_inventory() needs; other CSV columns are ignored
FIELDS = ("sku", "current_stock", "daily_demand", "safety_stock", "lead_time_days")

# Demand Engine 的输出列（除 sku 外），随决策结果一起传递，不改写、不覆盖
DEMAND_COLUMNS = tuple(c for c in _DEMAND_COLUMNS if c != "sku")


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


def analyze_inventory(
    sku: str,
    current_stock: float,
    daily_demand: float,
    safety_stock: float,
    lead_time_days: int,
) -> InventoryResult:
    values = {
        "current_stock": current_stock,
        "daily_demand": daily_demand,
        "safety_stock": safety_stock,
        "lead_time_days": lead_time_days,
    }
    null = next((name for name, v in values.items() if pd.isna(v)), None)
    if null is not None:
        raise ValueError(f"{null} must not be null")

    if daily_demand <= 0:
        raise ValueError("daily_demand must be greater than 0")

    # 库存可以撑多少天
    coverage_days = current_stock / daily_demand
    # ✅核心业务公式：补货点 = 交期内消耗 + 安全库存
    reorder_point = daily_demand * lead_time_days + safety_stock

    # 判断库存状态
    if current_stock < safety_stock:
        status = "CRITICAL"    # 危急
    elif current_stock < reorder_point:
        status = "REORDER"     # 需要补货
    elif coverage_days > 180:
        status = "OVERSTOCK"   # 库存过剩
    else:
        status = "NORMAL"      # 正常

    # 需要采购多少，不足才补，不会出现负数
    recommended_order_qty = max(
        0,
        reorder_point - current_stock
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
    )


# --- thin bridge: inventory.csv -> analyze_inventory ---------------------

@dataclass
class BatchResult:
    """Batch output: the analyzable rows plus one warning per skipped row."""
    frame: pd.DataFrame
    analyzed_count: int
    skipped_count: int
    warnings: list[dict]


def analyze_frame(df: pd.DataFrame) -> BatchResult:
    """One row per analyzable SKU, worst coverage first. Bad rows are skipped with a warning."""
    rows, warnings = [], []
    for record in df[list(FIELDS)].to_dict("records"):
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

def analyze_inventory_frame(inventory_df: pd.DataFrame, demand_df: pd.DataFrame) -> BatchResult:
    """装配入口：库存事实 + Demand Engine 输出 -> BatchResult。

    按 sku 做显式 1:1 对齐（不 drop、不 fill、不猜），对齐规则见 _aligned_snapshot()。
    demand 的口径、质量结论原样传递到结果帧，不做任何改写。
    """
    aligned = _aligned_snapshot(inventory_df, demand_df)
    batch = analyze_frame(aligned)
    return replace(
        batch,
        frame=batch.frame.merge(aligned[["sku", *DEMAND_COLUMNS]], on="sku", how="left"),
    )


def _aligned_snapshot(inventory_df: pd.DataFrame, demand_df: pd.DataFrame) -> pd.DataFrame:
    """把 Demand Engine 输出接到库存事实上，产出 analyze_frame() 能吃的单表快照。

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
    return aligned


def _check_unique_sku(df: pd.DataFrame, label: str) -> None:
    duplicates = sorted(df["sku"][df["sku"].duplicated()].unique().tolist())
    if duplicates:
        raise ValueError(f"duplicate {label} SKU: {duplicates}")
