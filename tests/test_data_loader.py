"""Run: pytest  |  python -m tests.test_data_loader  (from the repo root)"""
import tempfile
from pathlib import Path

from app.data_loader import (
    INVENTORY_COLUMNS,
    IN_TRANSIT_COLUMNS,
    ORDERS_COLUMNS,
    PRODUCTS_COLUMNS,
    load_in_transit,
    load_inventory,
    load_orders,
    load_products,
)


def test_inventory_no_longer_requires_daily_demand():
    df = load_inventory()
    assert "daily_demand" not in df.columns
    assert INVENTORY_COLUMNS == {"sku", "current_stock", "safety_stock", "lead_time_days"}


def test_inventory_missing_required_column_fails():
    try:
        load_inventory("products.csv")  # 缺 current_stock 等库存字段
    except ValueError as e:
        assert "current_stock" in str(e)
        return
    raise AssertionError("products.csv should not pass as inventory")


def test_products_loads():
    df = load_products()
    assert list(df.columns) == ["sku", "product_name", "category"]
    assert len(df) == 5


def test_products_missing_required_column_fails():
    try:
        load_products("inventory.csv")  # 缺 product_name / category
    except ValueError as e:
        assert "product_name" in str(e)
        return
    raise AssertionError("inventory.csv should not pass as products")


def test_orders_loads():
    df = load_orders()
    assert list(df.columns) == ["order_id", "order_date", "sku", "qty"]
    assert len(df) == 14
    assert ORDERS_COLUMNS == {"order_id", "order_date", "sku", "qty"}


def test_orders_missing_required_columns_fail():
    try:
        load_orders("inventory.csv")  # 缺 order_id / order_date / qty
    except ValueError as e:
        assert "order_date" in str(e)
        return
    raise AssertionError("inventory.csv should not pass as orders")


def test_in_transit_loads():
    df = load_in_transit()
    assert list(df.columns) == ["po_id", "sku", "qty", "expected_date", "status"]
    assert len(df) == 6
    assert IN_TRANSIT_COLUMNS == {"po_id", "sku", "qty", "expected_date", "status"}


def test_in_transit_missing_required_columns_fail():
    try:
        load_in_transit("inventory.csv")  # 缺 po_id / expected_date / status
    except ValueError as e:
        assert "expected_date" in str(e)
        return
    raise AssertionError("inventory.csv should not pass as in_transit")


def test_extra_columns_do_not_break_loading():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "with_extra.csv"
        path.write_text(
            "sku,current_stock,safety_stock,lead_time_days,supplier,unit_cost,warehouse\n"
            "SKU001,1200,500,5,Acme,1.5,WH1\n",
            encoding="utf-8",
        )
        df = load_inventory(path)  # 绝对路径直接用
        assert set(INVENTORY_COLUMNS) <= set(df.columns)
        for extra in ("supplier", "unit_cost", "warehouse"):
            assert extra in df.columns  # 额外列保留，不报错
        assert len(df) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
