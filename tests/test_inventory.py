"""Run: python -m tests.test_inventory  (from the repo root)"""
import io
from contextlib import redirect_stdout

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
    out = analyze_frame(df)
    assert len(out) == 5
    assert out.iloc[0].sku == "SKU004"


def test_real_file_has_all_statuses():
    out = analyze_frame(load_inventory())
    assert set(out.status) == {"CRITICAL", "REORDER", "OVERSTOCK", "NORMAL"}
    assert out.iloc[0].sku == "SKU004"  # sorted by coverage, worse first


def test_real_files_load():
    assert load_inventory().shape == (5, 5)
    assert list(load_csv("products.csv").columns) == ["sku", "product_name", "category"]


def test_missing_columns_rejected():
    try:
        load_inventory("products.csv")  # lacks the stock columns
    except ValueError as e:
        assert "daily_demand" in str(e)
        return
    raise AssertionError("products.csv should not pass as inventory")


def test_report():
    buf = io.StringIO()
    with redirect_stdout(buf):
        demo.main()
    out = buf.getvalue()
    assert "Inventory Decision Engine" in out
    assert "SKU004" in out and "CRITICAL" in out and "建议采购量: 600" in out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
