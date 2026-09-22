"""Program entry point. Run: python -X utf8 -m app.demo

数据流（STEP 3 起）：
    inventory.csv（库存事实） + orders.csv（订单事实）
        → app.demand.calculate_demand()   (Demand Engine)
        → app.inventory.analyze_inventory_frame()   (库存决策)
        → 打印
本模块只负责「加载 → 调用 → 展示」，不复制任何计算。
"""
from datetime import date

import pandas as pd

from app.data_loader import load_inventory, load_orders
from app.demand import calculate_demand
from app.inventory import analyze_inventory_frame


def format_number(value) -> str:
    """展示层格式化：整数不带小数点，非整数保留小数，缺值不当作 0。

    计算层保留真实数值（如 12.5），这里只决定怎么显示。
    """
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.10f}".rstrip("0").rstrip(".")


def main(as_of_date: date | None = None):
    # as_of_date 默认今天：边界层唯一允许读取系统时间的地方。
    # 测试传入固定日期（如 date(2026, 9, 20)）即可得到确定性输出。
    if as_of_date is None:
        as_of_date = date.today()

    # 日均需求一律来自 Demand Engine（daily_demand_30d），不再来自 inventory.csv
    demand = calculate_demand(load_orders(), as_of_date)
    results = analyze_inventory_frame(
        load_inventory(),
        demand,
        as_of_date=as_of_date,
    )
    print("=" * 40)
    print("SupplyChain OS")
    print("Inventory Decision Engine")
    print("=" * 40)

    for row in results.frame.itertuples():
        print(f"\nSKU: {row.sku}")
        print(f"当前库存: {format_number(row.current_stock)}")
        print(f"日均需求: {format_number(row.daily_demand)}")
        print(f"库存覆盖天数: {format_number(row.coverage_days)}")
        print(f"补货点: {format_number(row.reorder_point)}")
        print(f"库存状态: {row.status}")
        print(f"建议采购量: {format_number(row.recommended_order_qty)}")

    print(f"\n分析完成: {results.analyzed_count} 个 SKU, 跳过 {results.skipped_count} 个")
    for w in results.warnings:
        print(f"[WARNING] {w['sku']} / {w['field']}: {w['message']}")


if __name__ == "__main__":
    main()
