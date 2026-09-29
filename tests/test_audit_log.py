"""Run: python -m pytest tests/test_audit_log.py  |  python -m tests.test_audit_log

验证 app/audit_log.py 的冻结语义：纯投影、action 合并、JSONL append、类型归一、无 timestamp、无系统时钟。
"""
import ast
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.audit_log import (
    ACTION_FIELDS,
    BUSINESS_COLUMNS,
    ENVELOPE_COLUMNS,
    RECORD_TYPE,
    append_audit_records,
    build_audit_records,
)

APP_DIR = Path(__file__).resolve().parent.parent / "app"


def full_row(**overrides):
    """一个字段齐全的 DECISION 结果帧行（信封 + 全部业务字段）。"""
    row = {
        "run_id": "RUN-1",
        "simulation_day": 1,
        "as_of_date": date(2026, 9, 20),
        "sku": "SKU001",
        "demand_basis": "daily_demand_30d",
        "current_stock": 50.0,
        "daily_demand": 10.0,
        "safety_stock": 20.0,
        "lead_time_days": 5,
        "in_transit_stock": 0.0,
        "overdue_in_transit": 0.0,
        "inventory_position": 50.0,
        "coverage_days": 5.0,
        "reorder_point": 70.0,
        "shortage_qty": 20.0,
        "status": "REORDER",
        "recommended_order_qty": 20.0,
        "reason": "STOCK_BELOW_REORDER_POINT",
        "procurement_recommended_qty": 20.0,
        "procurement_reason": "GAP_ONLY",
        "provisional": False,
        "daily_demand_30d": 10.0,
        "daily_demand_7d": 10.0,
        "demand_trend": "STABLE",
        "history_days": 30,
        "order_active_days": 10,
        "data_quality_status": "OK",
    }
    row.update(overrides)
    return row


def frame(*rows):
    return pd.DataFrame(rows)


# --- 1. build_audit_records: 投影 -----------------------------------------

def test_build_creates_one_decision_record_per_sku():
    records = build_audit_records(frame(full_row(), full_row(sku="SKU002")))
    assert len(records) == 2
    assert [r["sku"] for r in records] == ["SKU001", "SKU002"]
    assert all(r["record_type"] == RECORD_TYPE == "DECISION" for r in records)


def test_envelope_fields_are_correct():
    records = build_audit_records(frame(full_row()))
    record = records[0]
    assert record["run_id"] == "RUN-1"
    assert record["simulation_day"] == 1
    assert record["as_of_date"] == "2026-09-20"
    assert record["sku"] == "SKU001"


def test_business_fields_are_projected_verbatim():
    row = full_row()
    record = build_audit_records(frame(row))[0]
    for column in BUSINESS_COLUMNS:
        assert record[column] == row[column], column


def test_projection_does_not_recompute():
    # daily_demand 与 reorder_point 故意不自洽；投影必须原样带出，不得重算
    row = full_row(daily_demand=7.0, reorder_point=999.0)
    record = build_audit_records(frame(row))[0]
    assert record["daily_demand"] == 7.0
    assert record["reorder_point"] == 999.0


def test_action_fields_are_merged():
    action = {
        "SKU001": {
            "order_placed_qty": 100.0,
            "po_id": "PO-RUN-1-000001",
            "expected_date": "2026-09-25",
        }
    }
    record = build_audit_records(frame(full_row()), action)[0]
    for field in ACTION_FIELDS:
        assert record[field] == action["SKU001"][field]


def test_action_defaults_to_none_without_action():
    record = build_audit_records(frame(full_row()))[0]
    assert record["order_placed_qty"] is None
    assert record["po_id"] is None
    assert record["expected_date"] is None


def test_action_missing_sku_falls_back_to_none():
    record = build_audit_records(frame(full_row()), {"SKU999": {"order_placed_qty": 1}})[0]
    assert record["order_placed_qty"] is None


def test_build_does_not_mutate_inputs():
    df = frame(full_row(), full_row(sku="SKU002"))
    before = df.copy(deep=True)
    action = {"SKU001": {"order_placed_qty": 5.0, "po_id": "P", "expected_date": "2026-09-25"}}
    action_before = {k: dict(v) for k, v in action.items()}

    build_audit_records(df, action)

    pd.testing.assert_frame_equal(df, before)
    assert action == action_before


def test_missing_envelope_columns_raise():
    for column in ENVELOPE_COLUMNS:
        df = frame(full_row()).drop(columns=[column])
        with pytest.raises(ValueError):
            build_audit_records(df)


def test_same_input_is_deterministic():
    df = frame(full_row(), full_row(sku="SKU002"))
    action = {"SKU001": {"order_placed_qty": 5.0, "po_id": "P", "expected_date": "2026-09-25"}}
    assert build_audit_records(df, action) == build_audit_records(df, action)


# --- 2. JSON 类型归一 -------------------------------------------------------

def test_numpy_scalars_are_json_serializable():
    row = full_row(current_stock=np.float64(50.5))
    df = frame(row)
    df["current_stock"] = df["current_stock"].astype(object)
    record = build_audit_records(df)[0]
    text = json.dumps(record)  # 不得抛 TypeError
    assert isinstance(text, str)
    assert json.loads(text)["current_stock"] == 50.5


def test_nan_is_normalized_to_none():
    record = build_audit_records(frame(full_row(coverage_days=float("nan"))))[0]
    assert record["coverage_days"] is None


def test_nat_is_normalized_to_none():
    record = build_audit_records(frame(full_row(as_of_date=pd.NaT)))[0]
    assert record["as_of_date"] is None


def test_no_timestamp_field_in_records():
    record = build_audit_records(frame(full_row()))[0]
    assert "timestamp" not in record


# --- 3. append_audit_records ------------------------------------------------

def test_append_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "audit.jsonl"
    records = build_audit_records(frame(full_row(), full_row(sku="SKU002")))
    append_audit_records(path, records)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == records


def test_append_is_utf8_without_ascii_escaping(tmp_path):
    path = tmp_path / "audit.jsonl"
    records = [
        {
            "run_id": "R",
            "simulation_day": 1,
            "as_of_date": "2026-09-20",
            "sku": "SKU002",
            "record_type": "SKIP",
            "skipped": True,
            "skip_field": "daily_demand",
            "skip_reason": "daily_demand must be greater than 0",
        }
    ]
    append_audit_records(path, records)
    text = path.read_text(encoding="utf-8")
    assert "daily_demand must be greater than 0" in text
    assert "\\u" not in text
    assert json.loads(text.strip()) == records[0]


def test_append_does_not_overwrite_and_keeps_order(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = build_audit_records(frame(full_row(sku="SKU001")))
    second = build_audit_records(frame(full_row(sku="SKU002")))

    append_audit_records(path, first)
    append_audit_records(path, second)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["sku"] == "SKU001"
    assert json.loads(lines[1])["sku"] == "SKU002"


def test_append_failure_propagates(tmp_path):
    path = tmp_path / "missing_dir" / "audit.jsonl"
    with pytest.raises(OSError):
        append_audit_records(path, [{"sku": "SKU001"}])


def test_append_empty_records_writes_nothing(tmp_path):
    path = tmp_path / "audit.jsonl"
    append_audit_records(path, [])
    assert path.read_text(encoding="utf-8") == ""


# --- 4. 无系统时钟 ----------------------------------------------------------

def test_audit_log_does_not_read_the_system_clock():
    source = (APP_DIR / "audit_log.py").read_text(encoding="utf-8")
    for forbidden in ("datetime.now", "date.today", "time.time"):
        assert forbidden not in source, forbidden


def test_audit_log_does_not_call_business_modules():
    source = (APP_DIR / "audit_log.py").read_text(encoding="utf-8")
    called = {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called.isdisjoint(
        {"calculate_demand", "analyze_inventory_frame", "analyze_inventory", "apply_procurement"}
    )
