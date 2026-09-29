"""Audit log: pure projection of the decision frame + simulator action -> JSONL.

职责（且仅此）：
- build_audit_records(): 把结果帧投影成 DECISION 审计记录，合并 simulator 的 action 字段
- append_audit_records(): 追加写 JSONL（UTF-8 / ensure_ascii=False）

不做：业务计算、状态推进、时间戳、系统时钟读取、数据库。
SKIP 记录由 simulator 生成，本模块不产出。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# 信封：由 simulator 在传给 build_audit_records 之前附加到结果帧上
ENVELOPE_COLUMNS = ("run_id", "simulation_day", "as_of_date", "sku")

# 结果帧业务字段（冻结稿 E），顺序即记录内字段顺序
BUSINESS_COLUMNS = (
    "demand_basis",
    "current_stock",
    "daily_demand",
    "safety_stock",
    "lead_time_days",
    "in_transit_stock",
    "overdue_in_transit",
    "inventory_position",
    "coverage_days",
    "reorder_point",
    "shortage_qty",
    "status",
    "recommended_order_qty",
    "reason",
    "procurement_recommended_qty",
    "procurement_reason",
    "provisional",
    "daily_demand_30d",
    "daily_demand_7d",
    "demand_trend",
    "history_days",
    "order_active_days",
    "data_quality_status",
)

ACTION_FIELDS = ("order_placed_qty", "po_id", "expected_date")
RECORD_TYPE = "DECISION"

__all__ = [
    "ENVELOPE_COLUMNS",
    "BUSINESS_COLUMNS",
    "ACTION_FIELDS",
    "RECORD_TYPE",
    "build_audit_records",
    "append_audit_records",
]


def build_audit_records(frame, action=None) -> list[dict]:
    """结果帧 -> DECISION 审计记录（每个 SKU 一条）。

    frame 必须携带信封列 run_id / simulation_day / as_of_date / sku。
    action: {sku: {"order_placed_qty": float, "po_id": str | None, "expected_date": str | None}}。
    不修改 frame / action，不重算任何业务指标。
    """
    missing = [column for column in ENVELOPE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"audit frame missing envelope columns: {missing}")

    present = [column for column in BUSINESS_COLUMNS if column in frame.columns]
    lookup = action or {}

    records = []
    for row in frame.to_dict("records"):
        sku = _to_native(row["sku"])
        record = {
            "run_id": _to_native(row["run_id"]),
            "simulation_day": _to_native(row["simulation_day"]),
            "as_of_date": _iso(row["as_of_date"]),
            "sku": sku,
            "record_type": RECORD_TYPE,
        }
        for column in present:
            record[column] = _to_native(row[column])
        sku_action = lookup.get(sku) or {}
        for field in ACTION_FIELDS:
            record[field] = _to_native(sku_action.get(field))
        records.append(record)
    return records


def append_audit_records(path, records) -> None:
    """append 一行一个 JSON object；UTF-8；ensure_ascii=False；不覆盖；失败抛异常。"""
    with Path(path).open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _iso(value):
    value = _to_native(value)
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _to_native(value):
    """numpy 标量 / NaN / NaT -> 原生 Python，便于 json.dumps 且不改语义。"""
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            return value
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if value != value:  # NaN / NaT
            return None
    except (TypeError, ValueError):
        pass
    return value
