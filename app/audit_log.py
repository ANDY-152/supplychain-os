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

# Level 3 审计字段（仅当结果帧携带这些列时条件投影；顺序冻结，见裁决清单 A-1）
L3_COLUMNS = (
    "algorithm",
    "period_index",
    "allocation_rank",
    "allocated_qty",
    "allocation_reason",
    "unit_cost",
    "priority",
    "budget_available_before",
    "budget_spent_today",
    "budget_remaining_after",
)
RECORD_TYPE = "DECISION"

__all__ = [
    "ENVELOPE_COLUMNS",
    "BUSINESS_COLUMNS",
    "ACTION_FIELDS",
    "L3_COLUMNS",
    "RECORD_TYPE",
    "build_audit_records",
    "append_audit_records",
]


def build_audit_records(frame, action=None) -> list[dict]:
    """结果帧 -> DECISION 审计记录（每个 SKU 一条）。

    frame 必须携带信封列 run_id / simulation_day / as_of_date / sku。
    action: {sku: {"order_placed_qty": float, "po_id": str | None, "expected_date": str | None}}。
    不修改 frame / action，不重算任何业务指标。
    L3 字段（`L3_COLUMNS`）：仅当结果帧携带对应列时，于 ACTION_FIELDS 之后按固定顺序
    逐列条件投影；帧内无 L3 列时输出与集成前完全一致。
    """
    missing = [column for column in ENVELOPE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"audit frame missing envelope columns: {missing}")

    present = [column for column in BUSINESS_COLUMNS if column in frame.columns]
    l3_present = [column for column in L3_COLUMNS if column in frame.columns]
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
        for column in l3_present:
            record[column] = _to_native(row[column])
        records.append(record)
    return records


def append_audit_records(path, records) -> None:
    """append 一行一个 JSON object；UTF-8；ensure_ascii=False；不覆盖；失败抛异常。

    先整体序列化为一个 payload（任一条目序列化失败时不打开文件），再单次写入。
    单次写入只缩小部分写入窗口，**不提供跨资源原子性或磁盘事务保证**。
    """
    payload = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(payload)


def _iso(value):
    value = _to_native(value)
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _to_native(value):
    """numpy 标量 / 非有限浮点（NaN/±inf）/ NaT -> 原生 Python，便于 json.dumps 且不改语义。

    JSON（RFC 8259）不允许 NaN/Infinity；故非有限 float 一律归一为 None（JSON null），
    避免 json.dumps 默认产生非严格 JSON 的 `NaN`/`Infinity`/`-Infinity`。
    """
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if value != value:  # NaN / NaT
            return None
    except (TypeError, ValueError):
        pass
    return value
