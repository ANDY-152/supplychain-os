"""Program entry point. Run: python -X utf8 -m app.demo

数据流（STEP 3 起，STEP 6 接入在途，STEP 8 接入 Level 2 采购建议）：
    inventory.csv（库存事实） + orders.csv（订单事实） + in_transit.csv（在途事实）
        + data_metadata.csv（覆盖起点） + procurement_constraints.csv（采购约束）
        → app.demand.calculate_demand()          (Demand Engine)
        → app.inventory.analyze_inventory_frame() (Level 1 库存决策)
        → app.procurement.apply_procurement()     (Level 2 采购建议)
        → 打印
本模块只负责「加载 → 调用 → 展示」，不复制任何计算。
展示的每个数值都直接取自结果帧，绝不在此重新计算。
"""
from datetime import date

import pandas as pd

from app.data_loader import (
    load_data_metadata,
    load_in_transit,
    load_inventory,
    load_orders,
    load_procurement_constraints,
)
from app.demand import calculate_demand
from app.inventory import analyze_inventory_frame
from app.procurement import apply_procurement


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
    # 数据覆盖起点由 data_metadata.csv 声明，只负责加载与传递，不在展示层重算历史长度
    coverage_start = load_data_metadata()
    demand = calculate_demand(load_orders(), as_of_date, coverage_start=coverage_start)
    # STEP 6：接入 in_transit.csv，使 demo 与库存决策引擎使用完全一致的在途口径
    results = analyze_inventory_frame(
        load_inventory(),
        demand,
        in_transit_df=load_in_transit(),
        as_of_date=as_of_date,
    )
    frame = apply_procurement(results.frame, load_procurement_constraints())

    print("=" * 40)
    print("SupplyChain OS")
    print("Inventory Decision Engine")
    print("=" * 40)
    print(f"决策基准日: {as_of_date}")
    # demand_basis 直接读自结果帧，证明它随决策一起透传，不在展示层写死
    print(f"需求口径: {frame.demand_basis.iloc[0] if len(frame) else 'N/A'}")

    # --- 核心决策：直接展示已算好的字段，展示层不做任何计算 ---
    print("\n" + "-" * 40)
    print("核心决策")
    print("-" * 40)
    for row in frame.itertuples():
        print(f"\nSKU: {row.sku}")
        print(f"  状态: {row.status}")
        print(f"  现货: {format_number(row.current_stock)}")
        print(f"  日需求: {format_number(row.daily_demand)}")
        print(f"  在途库存: {format_number(row.in_transit_stock)}")
        print(f"  库存位置: {format_number(row.inventory_position)}")
        print(f"  覆盖天数: {format_number(row.coverage_days)}")
        print(f"  再订货点: {format_number(row.reorder_point)}")
        print(f"  原始缺口: {format_number(row.shortage_qty)}")
        print(f"  建议采购量: {format_number(row.recommended_order_qty)}")
        print(f"  建议原因: {row.reason}")
        # Level 2：只读展示，不在展示层重新计算
        print(f"  采购建议量(L2): {format_number(row.procurement_recommended_qty)}")
        print(f"  采购置信: {'暂定' if row.provisional else '正常'}")
        print(f"  采购原因: {row.procurement_reason}")

    # --- 需求证据 / 置信度：只读透传，绝不进入核心公式 ---
    print("\n" + "-" * 40)
    print("需求证据 / 置信度")
    print("-" * 40)
    for row in frame.itertuples():
        print(f"\nSKU: {row.sku}")
        print(f"  30D日均需求: {format_number(row.daily_demand_30d)}")
        print(f"  7D日均需求: {format_number(row.daily_demand_7d)}")
        print(f"  需求趋势: {row.demand_trend}")
        print(f"  历史覆盖天数: {format_number(row.history_days)}")
        print(f"  有订单天数: {format_number(row.order_active_days)}")
        print(f"  数据质量: {row.data_quality_status}")

    # --- 风险：逾期在途是风险信息，不扣减供应；data_warnings 是提示，不是失败 ---
    print("\n" + "-" * 40)
    print("风险")
    print("-" * 40)
    for row in frame.itertuples():
        print(f"SKU: {row.sku}  逾期在途: {format_number(row.overdue_in_transit)}")

    print(f"\n分析完成: {results.analyzed_count} 个 SKU, 跳过 {results.skipped_count} 个")

    print("\n【硬失败 / 跳过】")
    if results.warnings:
        for w in results.warnings:
            print(f"[WARNING] {w['sku']} / {w['field']}: {w['message']}")
    else:
        print("无")

    print("\n【风险 / 置信度提示】")
    if results.data_warnings:
        for w in results.data_warnings:
            print(f"[{w['code']}] {w['sku']}: {w['message']}")
    else:
        print("无")


if __name__ == "__main__":
    main()
