"""Run: python -m tests.test_inventory  (from the repo root)"""
import io
from contextlib import redirect_stdout
from datetime import date

import pandas as pd

from app import demo
from app.data_loader import load_csv, load_inventory, load_orders
from app.demand import calculate_demand
from app.inventory import (
    DEMAND_COLUMNS,
    analyze_frame,
    analyze_inventory,
    analyze_inventory_frame,
)

AS_OF = date(2026, 9, 20)  # 固定基准日，测试不依赖系统时间


def real_demand() -> pd.DataFrame:
    """真实 orders.csv 经 Demand Engine 的需求结果。"""
    return calculate_demand(load_orders(), AS_OF)


def inventory_rows(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["sku", "current_stock", "safety_stock", "lead_time_days"])


def demand_rows(*rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "sku",
            "daily_demand_30d",
            "daily_demand_7d",
            "demand_trend",
            "history_days",
            "order_active_days",
            "data_quality_status",
        ],
    )


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
    inventory = load_inventory().assign(supplier="Acme", unit_cost=1.5, category="A", warehouse="WH1")
    batch = analyze_inventory_frame(inventory, real_demand())
    assert batch.analyzed_count == 5
    assert batch.skipped_count == 0
    assert batch.frame.iloc[0].sku == "SKU004"


# --- STEP 3: Demand Engine -> Inventory Decision Engine 接线 ---------------

def test_wiring_uses_daily_demand_30d_from_demand_engine():
    # A: 库存决策使用的是 Demand Engine 的 daily_demand_30d
    inventory = inventory_rows(("SKU001", 1200, 500, 5))
    demand = demand_rows(("SKU001", 12.5, 20.0, "INCREASING", 30, 10, "OK"))
    row = analyze_inventory_frame(inventory, demand).frame.iloc[0]

    assert row.daily_demand == 12.5
    assert row.daily_demand_30d == 12.5
    assert row.reorder_point == 12.5 * 5 + 500  # V0.1 公式
    assert row.coverage_days == round(1200 / 12.5, 2)


def test_inventory_frame_needs_only_stock_columns():
    # B: 库存表只有 4 列（无 daily_demand）就能完成决策
    inventory = load_inventory()
    assert list(inventory.columns) == ["sku", "current_stock", "safety_stock", "lead_time_days"]
    batch = analyze_inventory_frame(inventory, real_demand())
    assert batch.analyzed_count == 5
    assert batch.skipped_count == 0
    assert "daily_demand" not in inventory.columns


def test_inventory_daily_demand_is_rejected():
    # B(反): 库存表不得自带 daily_demand，需求只能来自 Demand Engine
    inventory = inventory_rows(("SKU001", 1200, 500, 5)).assign(daily_demand=999)
    demand = demand_rows(("SKU001", 12.5, 12.5, "STABLE", 30, 10, "OK"))
    try:
        analyze_inventory_frame(inventory, demand)
    except ValueError as e:
        assert "daily_demand" in str(e) and "Demand Engine" in str(e)
        return
    raise AssertionError("inventory carrying daily_demand must be rejected")


def test_missing_demand_for_sku_fails():
    # C: inventory 有 SKU002，demand 没有 → 必须失败并指出 SKU002
    inventory = inventory_rows(("SKU001", 1200, 500, 5), ("SKU002", 500, 300, 7))
    demand = demand_rows(("SKU001", 10.0, 10.0, "STABLE", 30, 10, "OK"))
    try:
        analyze_inventory_frame(inventory, demand)
    except ValueError as e:
        assert "missing demand" in str(e) and "SKU002" in str(e)
        return
    raise AssertionError("missing demand must raise ValueError")


def test_extra_demand_sku_fails():
    # D: demand 多了 inventory 没有的 SKU999 → 必须失败并指出 SKU999
    inventory = inventory_rows(("SKU001", 1200, 500, 5))
    demand = demand_rows(
        ("SKU001", 10.0, 10.0, "STABLE", 30, 10, "OK"),
        ("SKU999", 5.0, 5.0, "STABLE", 30, 5, "OK"),
    )
    try:
        analyze_inventory_frame(inventory, demand)
    except ValueError as e:
        assert "extra demand SKU" in str(e) and "SKU999" in str(e)
        return
    raise AssertionError("extra demand SKU must raise ValueError")


def test_duplicate_inventory_sku_fails():
    # E
    inventory = inventory_rows(("SKU001", 1200, 500, 5), ("SKU001", 900, 500, 5))
    demand = demand_rows(("SKU001", 10.0, 10.0, "STABLE", 30, 10, "OK"))
    try:
        analyze_inventory_frame(inventory, demand)
    except ValueError as e:
        assert "duplicate inventory SKU" in str(e) and "SKU001" in str(e)
        return
    raise AssertionError("duplicate inventory SKU must raise ValueError")


def test_duplicate_demand_sku_fails():
    # F
    inventory = inventory_rows(("SKU001", 1200, 500, 5))
    demand = demand_rows(
        ("SKU001", 10.0, 10.0, "STABLE", 30, 10, "OK"),
        ("SKU001", 11.0, 11.0, "STABLE", 30, 10, "OK"),
    )
    try:
        analyze_inventory_frame(inventory, demand)
    except ValueError as e:
        assert "duplicate demand SKU" in str(e) and "SKU001" in str(e)
        return
    raise AssertionError("duplicate demand SKU must raise ValueError")


def test_unknown_history_is_passed_through_unchanged():
    # G: UNKNOWN_HISTORY / history_days=None 必须原样传到决策结果
    inventory = inventory_rows(("SKU001", 1200, 500, 5))
    demand = demand_rows(("SKU001", 10.0, 12.0, "INCREASING", None, 4, "UNKNOWN_HISTORY"))
    row = analyze_inventory_frame(inventory, demand).frame.iloc[0]

    assert row.data_quality_status == "UNKNOWN_HISTORY"
    assert row.data_quality_status != "INSUFFICIENT_HISTORY"
    assert row.history_days is None
    assert row.order_active_days == 4
    assert row.demand_trend == "INCREASING"


def test_demand_columns_are_kept_in_the_decision_output():
    inventory = load_inventory()
    frame = analyze_inventory_frame(inventory, real_demand()).frame
    assert set(DEMAND_COLUMNS) <= set(frame.columns)
    assert set(frame.sku) == set(inventory.sku)  # 无 drop、无新增
    assert len(frame) == len(inventory)


def test_trend_and_7d_do_not_change_the_demand_input():
    # H: 7D 与 trend 不参与库存决策，daily_demand 仍是 30D
    inventory = inventory_rows(("SKU001", 1200, 500, 5))
    demand = demand_rows(("SKU001", 10.0, 20.0, "INCREASING", 30, 10, "OK"))
    row = analyze_inventory_frame(inventory, demand).frame.iloc[0]

    assert row.daily_demand == 10.0  # 不是 20.0，也不是 0.5*30D+0.5*7D
    assert row.reorder_point == 10.0 * 5 + 500
    assert row.recommended_order_qty == 0  # 1200 >= 550


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
    inventory = load_inventory()
    # 输入调整（不改断言）：把 SKU002 的现货调到 [safety, reorder_point) 区间，
    # 让 4 种状态在真实数据上同时出现（300 <= 350 < 13.0*7+300 = 391）
    inventory.loc[inventory.sku == "SKU002", "current_stock"] = 350
    out = analyze_inventory_frame(inventory, real_demand()).frame
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
        demo.main(as_of_date=AS_OF)  # 固定基准日：报告内容不依赖系统时间
    out = buf.getvalue()
    assert "Inventory Decision Engine" in out
    assert "SKU004" in out and "CRITICAL" in out
    # 需求来自 Demand Engine 的 daily_demand_30d，不是 inventory.csv 里手填的 80
    assert "日均需求: 7.8333333333" in out
    assert "建议采购量: 239.17" in out
    assert "分析完成: 5 个 SKU, 跳过 0 个" in out
    # 整数仍然不显示成 13.0 / 0.0
    assert "当前库存: 1200" in out
    assert "日均需求: 13" in out
    assert "建议采购量: 0" in out
    assert ".0\n" not in out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
