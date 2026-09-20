"""Core inventory calculations."""
from dataclasses import asdict, dataclass, fields

import pandas as pd

# fields analyze_inventory() needs; other CSV columns are ignored
FIELDS = ("sku", "current_stock", "daily_demand", "safety_stock", "lead_time_days")


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
