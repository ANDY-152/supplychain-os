"""Program entry point. Run: python -X utf8 -m app.demo"""
import pandas as pd

from app.data_loader import load_inventory
from app.inventory import analyze_frame


def format_number(value) -> str:
    """展示层格式化：整数不带小数点，非整数保留小数，缺值不当作 0。

    计算层保留真实数值（如 12.5），这里只决定怎么显示。
    """
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.10f}".rstrip("0").rstrip(".")


def main():
    results = analyze_frame(load_inventory())
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
