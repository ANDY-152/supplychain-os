"""Program entry point. Run: python -X utf8 -m app.demo"""
from app.data_loader import load_inventory
from app.inventory import analyze_frame


def main():
    results = analyze_frame(load_inventory())
    print("=" * 40)
    print("SupplyChain OS")
    print("Inventory Decision Engine")
    print("=" * 40)

    for row in results.itertuples():
        print(f"\nSKU: {row.sku}")
        print(f"当前库存: {row.current_stock}")
        print(f"日均需求: {row.daily_demand}")
        print(f"库存覆盖天数: {row.coverage_days}")
        print(f"补货点: {row.reorder_point}")
        print(f"库存状态: {row.status}")
        print(f"建议采购量: {row.recommended_order_qty}")


if __name__ == "__main__":
    main()
