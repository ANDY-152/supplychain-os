"""Level 2: procurement recommendation.

只读 Level 1 结果帧 + per-SKU 采购约束（moq / order_multiple），追加三个 Level 2 字段：

    procurement_recommended_qty   数值结果
    provisional                   质量 / 置信状态（不改变数量）
    procurement_reason            解释：实际改变了最终数量的约束

Level 1 完全冻结：本模块**绝不**修改
reorder_point / inventory_position / shortage_qty / recommended_order_qty，
也不反向读取或改写 ROP / position / shortage。缺省（未配置约束）时保持 identity。
"""
import math

import pandas as pd

# Level 2 输出列（不加入 InventoryResult，作为结果帧列追加）
LEVEL2_COLUMNS = ("procurement_recommended_qty", "provisional", "procurement_reason")

# procurement_reason：只标记“实际改变了最终数量”的约束
NO_GAP = "NO_GAP"
GAP_ONLY = "GAP_ONLY"
MOQ_APPLIED = "MOQ_APPLIED"
MULTIPLE_APPLIED = "MULTIPLE_APPLIED"
MOQ_AND_MULTIPLE_APPLIED = "MOQ_AND_MULTIPLE_APPLIED"

_CONSTRAINT_COLUMNS = ("sku", "moq", "order_multiple")


def _recommend(gap: float, moq: float, multiple: int) -> tuple[float, str]:
    """单个 SKU 的 Level 2 约束链：gap -> MOQ(下限) -> multiple(向上取整)。"""
    if gap <= 0:
        return 0.0, NO_GAP

    q = gap
    moq_applied = moq > 0 and moq > q
    if moq_applied:
        q = moq

    multiple_applied = False
    if multiple > 1:
        rounded = math.ceil(q / multiple) * multiple
        if rounded != q:
            q = rounded
            multiple_applied = True

    if moq_applied and multiple_applied:
        reason = MOQ_AND_MULTIPLE_APPLIED
    elif moq_applied:
        reason = MOQ_APPLIED
    elif multiple_applied:
        reason = MULTIPLE_APPLIED
    else:
        reason = GAP_ONLY
    return float(q), reason


def _constraint_lookup(constraints: pd.DataFrame | None) -> dict:
    if constraints is None or len(constraints) == 0:
        return {}
    for column in _CONSTRAINT_COLUMNS:
        if column not in constraints.columns:
            raise ValueError(f"constraints missing column: {column}")
    return {
        row.sku: (float(row.moq), int(row.order_multiple)) for row in constraints.itertuples()
    }


def apply_procurement(frame: pd.DataFrame, constraints: pd.DataFrame | None) -> pd.DataFrame:
    """返回 `frame` 的副本，并追加 Level 2 三列。绝不修改 Level 1 列，也不改入参。"""
    for column in ("recommended_order_qty", "data_quality_status"):
        if column not in frame.columns:
            raise ValueError(f"frame is missing {column} (expected a Level 1 result frame)")

    lookup = _constraint_lookup(constraints)
    quantity, reason, provisional = [], [], []
    for row in frame.itertuples():
        moq, multiple = lookup.get(row.sku, (0.0, 1))
        q, code = _recommend(row.recommended_order_qty, moq, multiple)
        quantity.append(q)
        reason.append(code)
        provisional.append(row.data_quality_status != "OK")

    out = frame.copy()
    out["procurement_recommended_qty"] = quantity
    out["provisional"] = provisional
    out["procurement_reason"] = reason
    return out
