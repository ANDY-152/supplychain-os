"""Run: python -m tests.test_inventory  (from the repo root)"""
import io
from contextlib import redirect_stdout

import pandas as pd

from app import demo
from app.data_loader import load_csv, load_inventory
from app.inventory import analyze_frame, analyze_inventory


def test_normal_inventory():
    result = analyze_inventory(
        sku="SKU001",
        current_stock=1200,
        daily_demand=80,
        safety_stock=500,
        lead_time_days=5,
    )
    assert result.coverage_days == 15
    assert result.reorder_point == 900
    assert result.status == "NORMAL"
    assert result.recommended_order_qty == 0


def test_reorder_inventory():
    result = analyze_inventory(
        sku="SKU002",
        current_stock=500,
        daily_demand=100,
        safety_stock=300,
        lead_time_days=7,
    )
    assert result.status == "REORDER"
    assert result.recommended_order_qty == 500


def test_critical_inventory():
    result = analyze_inventory(
        sku="SKU004",
        current_stock=200,
        daily_demand=80,
        safety_stock=400,
        lead_time_days=5,
    )
    assert result.status == "CRITICAL"
    assert result.recommended_order_qty == 600


def test_overstock_inventory():
    result = analyze_inventory(
        sku="SKU003",
        current_stock=5000,
        daily_demand=20,
        safety_stock=500,
        lead_time_days=10,
    )
    assert result.status == "OVERSTOCK"
    assert result.recommended_order_qty == 0


def test_never_recommends_negative_qty():
    assert all(
        analyze_inventory(f"S{i}", stock, 10, 100, 5).recommended_order_qty >= 0
        for i, stock in enumerate([0, 100, 500, 5000])
    )


def test_invalid_demand():
    try:
        analyze_inventory(
            sku="SKU001",
            current_stock=1000,
            daily_demand=0,
            safety_stock=300,
            lead_time_days=5,
        )
    except ValueError:
        assert True
    else:
        assert False


def test_null_values_rejected():
    for field in ("current_stock", "daily_demand", "safety_stock", "lead_time_days"):
        args = {
            "sku": "SKU001",
            "current_stock": 1000,
            "daily_demand": 80,
            "safety_stock": 300,
            "lead_time_days": 5,
        }
        args[field] = float("nan")
        try:
            analyze_inventory(**args)
        except ValueError as e:
            assert field in str(e), e
        else:
            raise AssertionError(f"{field}=NaN should raise ValueError")


def test_extra_columns_ignored():
    df = load_inventory().assign(supplier="Acme", unit_cost=1.5, category="A", warehouse="WH1")
    batch = analyze_frame(df)
    assert batch.analyzed_count == 5
    assert batch.skipped_count == 0
    assert batch.frame.iloc[0].sku == "SKU004"


def test_batch_skips_bad_rows():
    df = pd.DataFrame(
        {
            "sku": ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"],
            "current_stock": [1200, 500, 5000, 200, 1500],
            "daily_demand": [80, 100, float("nan"), 80, 50],  # SKU003 daily_demand 为空
            "safety_stock": [500, 300, 500, 400, 400],
            "lead_time_days": [5, 7, 10, 5, 3],
        }
    )
    batch = analyze_frame(df)

    assert batch.analyzed_count == 4
    assert batch.skipped_count == 1
    assert len(batch.warnings) == 1
    warning = batch.warnings[0]
    assert warning["sku"] == "SKU003"
    assert warning["field"] == "daily_demand"
    assert "must not be null" in warning["message"]

    # 其余 4 个 SKU 仍然得到正常分析
    assert set(batch.frame.sku) == {"SKU001", "SKU002", "SKU004", "SKU005"}
    assert batch.frame.iloc[0].sku == "SKU004"  # 覆盖天数最低，排最前
    assert batch.frame[batch.frame.sku == "SKU002"].recommended_order_qty.iloc[0] == 500


def test_all_rows_bad_returns_empty_with_warnings():
    df = pd.DataFrame(
        {
            "sku": ["SKU001", "SKU002"],
            "current_stock": [1200, float("nan")],
            "daily_demand": [float("nan"), 100],
            "safety_stock": [500, 300],
            "lead_time_days": [5, 7],
        }
    )
    batch = analyze_frame(df)

    assert batch.analyzed_count == 0
    assert batch.skipped_count == 2
    assert len(batch.frame) == 0
    assert [w["sku"] for w in batch.warnings] == ["SKU001", "SKU002"]
    assert [w["field"] for w in batch.warnings] == ["daily_demand", "current_stock"]


def test_real_file_has_all_statuses():
    out = analyze_frame(load_inventory()).frame
    assert set(out.status) == {"CRITICAL", "REORDER", "OVERSTOCK", "NORMAL"}
    assert out.iloc[0].sku == "SKU004"  # sorted by coverage, worse first


def test_real_files_load():
    # loader 契约测试：inventory.csv 不再包含 daily_demand（V0.2 STEP 1）
    inventory = load_inventory()
    assert inventory.shape == (5, 4)
    assert "daily_demand" not in inventory.columns
    assert list(load_csv("products.csv").columns) == ["sku", "product_name", "category"]


def test_missing_columns_rejected():
    try:
        load_inventory("products.csv")  # lacks the stock columns
    except ValueError as e:
        assert "current_stock" in str(e)
        return
    raise AssertionError("products.csv should not pass as inventory")


def test_format_number():
    assert demo.format_number(600.0) == "600"
    assert demo.format_number(500.0) == "500"
    assert demo.format_number(5.0) == "5"
    assert demo.format_number(2.5) == "2.5"
    assert demo.format_number(12.75) == "12.75"
    # 小数点后的有效数字必须保留（不能用 int() 截断）
    assert demo.format_number(12.5) == "12.5"
    assert demo.format_number(0) == "0"
    # 异常值不能被静默当成 0
    assert demo.format_number(float("nan")) != "0"
    assert demo.format_number(None) != "0"


def test_report():
    buf = io.StringIO()
    with redirect_stdout(buf):
        demo.main()
    out = buf.getvalue()
    assert "Inventory Decision Engine" in out
    assert "SKU004" in out and "CRITICAL" in out and "建议采购量: 600" in out
    assert "分析完成: 5 个 SKU, 跳过 0 个" in out
    # 整数不再显示成 15.0 / 30.0
    assert "库存覆盖天数: 15" in out
    assert "库存覆盖天数: 30" in out
    assert ".0\n" not in out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
