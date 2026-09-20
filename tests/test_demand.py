"""Run: pytest tests/test_demand.py  |  python -m tests.test_demand  (from the repo root)

所有用例都固定 as_of_date = 2026-09-20，绝不依赖系统时间。
"""
from datetime import date, timedelta

import pandas as pd

from app.demand import COLUMNS, _demand_trend, calculate_demand

AS_OF = date(2026, 9, 20)
COVERAGE_START = AS_OF - timedelta(days=29)  # 恰好 30 天覆盖


def d(days_ago: int) -> str:
    """AS_OF 往前 days_ago 天（负数 = 未来）。"""
    return (AS_OF - timedelta(days=days_ago)).isoformat()


def orders(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["order_id", "order_date", "sku", "qty"])


def one(row) -> pd.Series:
    return calculate_demand(orders(row), AS_OF).iloc[0]


# --- 1 / 2 正常计算 ---------------------------------------------------------

def test_daily_demand_30d_normal():
    out = calculate_demand(orders(("O1", d(1), "SKU001", 30), ("O2", d(5), "SKU001", 30)), AS_OF)
    assert list(out.columns) == list(COLUMNS)
    assert out.daily_demand_30d.iloc[0] == 2.0  # 60 / 30
    # 计算层保持数值类型，不提前格式化成字符串
    assert pd.api.types.is_float_dtype(out.daily_demand_30d)
    assert pd.api.types.is_integer_dtype(out.order_active_days)


def test_daily_demand_7d_normal():
    out = calculate_demand(orders(("O1", d(1), "SKU001", 21), ("O2", d(6), "SKU001", 7)), AS_OF)
    assert out.daily_demand_7d.iloc[0] == 4.0  # 28 / 7
    assert out.daily_demand_30d.iloc[0] == 28 / 30


# --- 3 / 4 窗口边界 ---------------------------------------------------------

def test_30d_window_boundary():
    # d(29) 是窗口第一天（含）；d(30) 已出窗口（不含）
    out = one(("O1", d(29), "SKU001", 30))
    assert out.daily_demand_30d == 1.0
    out = one(("O1", d(30), "SKU001", 3000))
    assert out.daily_demand_30d == 0.0


def test_7d_window_boundary():
    out = one(("O1", d(6), "SKU001", 7))  # 窗口第一天（含）
    assert out.daily_demand_7d == 1.0
    out = one(("O1", d(7), "SKU001", 700))  # 已出 7D 窗口
    assert out.daily_demand_7d == 0.0
    assert out.daily_demand_30d == 700 / 30  # 但仍在 30D 窗口内


def test_as_of_date_itself_is_inside_window():
    assert one(("O1", d(0), "SKU001", 7)).daily_demand_7d == 1.0


def test_future_orders_are_outside_every_window():
    out = one(("O1", d(-1), "SKU001", 500))
    assert out.daily_demand_30d == 0.0
    assert out.daily_demand_7d == 0.0
    assert out.demand_trend == "NO_DEMAND"


# --- 5 没有订单的日子仍计入分母 --------------------------------------------

def test_days_without_orders_still_count_in_denominator():
    out = one(("O1", d(1), "SKU001", 30))
    assert out.daily_demand_30d == 1.0  # 30/30，而不是 30/1
    assert out.daily_demand_7d == 30 / 7
    assert out.order_active_days == 1


# --- 6 / 7 / 8 趋势 ---------------------------------------------------------

def test_demand_trend_increasing():
    # 30D 内、7D 外：70；7D 内：35
    out = calculate_demand(
        orders(("O1", d(25), "SKU001", 70), ("O2", d(1), "SKU001", 35)), AS_OF
    ).iloc[0]
    assert out.daily_demand_30d == 105 / 30
    assert out.daily_demand_7d == 5.0
    assert out.demand_trend == "INCREASING"


def test_demand_trend_decreasing():
    out = calculate_demand(
        orders(("O1", d(25), "SKU001", 300), ("O2", d(1), "SKU001", 7)), AS_OF
    ).iloc[0]
    assert out.daily_demand_30d == 307 / 30
    assert out.daily_demand_7d == 1.0
    assert out.demand_trend == "DECREASING"


def test_demand_trend_stable():
    # 7D = 7 → 1.0/天；30D = 7 + 23 = 30 → 1.0/天，两者相等
    out = calculate_demand(
        orders(("O1", d(10), "SKU001", 23), ("O2", d(1), "SKU001", 7)), AS_OF
    ).iloc[0]
    assert out.daily_demand_30d == 1.0
    assert out.daily_demand_7d == 1.0
    assert out.demand_trend == "STABLE"


# --- 9 / 10 dd30 == 0 的两种情形 -------------------------------------------

def test_no_demand_when_30d_is_zero():
    out = one(("O1", d(40), "SKU001", 500))  # 30 天前的订单，窗口内为 0
    assert out.daily_demand_30d == 0.0
    assert out.daily_demand_7d == 0.0
    assert out.demand_trend == "NO_DEMAND"
    assert out.order_active_days == 0


def test_trend_rule_for_zero_30d():
    # 该分支在 calculate_demand 中构造不出来：7D 窗口是 30D 的子集，
    # 有 7D 订单就必然有 30D 订单。这里按规则直接验证函数本身。
    assert _demand_trend(5.0, 0.0) == "INCREASING"
    assert _demand_trend(0.0, 0.0) == "NO_DEMAND"


def test_zero_30d_implies_zero_7d():
    out = calculate_demand(
        orders(("O1", d(40), "SKU001", 500), ("O2", d(45), "SKU001", 100)), AS_OF
    )
    zero_30d = out.daily_demand_30d == 0
    assert (out.loc[zero_30d, "daily_demand_7d"] == 0).all()


def test_trend_at_exact_thresholds_is_stable():
    # 严格不等号：11 == 10 * 1.10，9 == 10 * 0.90 → 都是 STABLE
    assert _demand_trend(11.0, 10.0) == "STABLE"
    assert _demand_trend(9.0, 10.0) == "STABLE"


# --- 11 / 12 / 13 / 14 数据质量：fail-fast，不静默修复 ---------------------

def test_negative_qty_raises():
    try:
        calculate_demand(orders(("O1", d(1), "SKU001", -5)), AS_OF)
    except ValueError as e:
        assert "negative qty" in str(e) and "O1" in str(e)
        return
    raise AssertionError("negative qty must raise ValueError")


def test_null_qty_raises():
    df = pd.DataFrame([("O1", d(1), "SKU001", None)], columns=["order_id", "order_date", "sku", "qty"])
    try:
        calculate_demand(df, AS_OF)
    except ValueError as e:
        assert "null qty" in str(e) and "O1" in str(e)
        return
    raise AssertionError("null qty must raise ValueError")


def test_null_sku_raises():
    df = pd.DataFrame([("O1", d(1), None, 10)], columns=["order_id", "order_date", "sku", "qty"])
    try:
        calculate_demand(df, AS_OF)
    except ValueError as e:
        assert "null sku" in str(e)
        return
    raise AssertionError("null sku must raise ValueError")


def test_null_order_date_raises():
    df = pd.DataFrame([("O1", None, "SKU001", 10)], columns=["order_id", "order_date", "sku", "qty"])
    try:
        calculate_demand(df, AS_OF)
    except ValueError as e:
        assert "null order_date" in str(e)
        return
    raise AssertionError("null order_date must raise ValueError")


def test_invalid_date_format_raises():
    try:
        calculate_demand(orders(("O1", "2026/09/20", "SKU001", 10)), AS_OF)
    except ValueError as e:
        assert "invalid order_date" in str(e)
        return
    raise AssertionError("bad date format must raise ValueError")


# --- 15 / 16 历史完整度 vs 订单活跃天数 ------------------------------------

def test_insufficient_history_only_when_provable():
    out = calculate_demand(
        orders(("O1", d(1), "SKU001", 30)), AS_OF, coverage_start=AS_OF - timedelta(days=19)
    ).iloc[0]
    assert out.history_days == 20
    assert out.data_quality_status == "INSUFFICIENT_HISTORY"


def test_full_history_with_few_order_days_is_not_insufficient():
    # 30 天覆盖，但只有 5 天有订单（d(0), d(5), d(10), d(15), d(20)）
    rows = [("O%d" % i, d(i * 5), "SKU001", 10) for i in range(5)]
    out = calculate_demand(orders(*rows), AS_OF, coverage_start=COVERAGE_START).iloc[0]
    assert out.history_days == 30
    assert out.order_active_days == 5
    assert out.data_quality_status == "OK"
    assert out.daily_demand_30d == 50 / 30  # 分母仍是 30，不是 5


def test_history_unknown_without_coverage_start():
    out = one(("O1", d(1), "SKU001", 30))
    assert out.history_days is None
    assert out.data_quality_status == "UNKNOWN_HISTORY"


def test_unknown_history_when_orders_start_inside_the_30d_window():
    # B: 30D 窗口起点是 d(29)=2026-08-22，而最早订单在 d(2)=2026-09-18。
    # “看不到 08-22/08-23 的订单” 不等于 “08-22/08-23 需求为 0”，也不等于历史不足，
    # 所以绝不能自动判定 INSUFFICIENT_HISTORY。
    row = calculate_demand(
        orders(("O1", d(2), "SKU001", 30), ("O2", d(1), "SKU001", 30)), AS_OF
    ).iloc[0]
    assert row.history_days is None
    assert row.data_quality_status == "UNKNOWN_HISTORY"


def test_active_days_is_not_history_days():
    # C: 窗口内 4 个活跃日，但没有起点信息 → history_days 仍然是 None
    rows = [("O%d" % i, d(i), "SKU001", 5) for i in range(4)]  # d(0)..d(3)
    row = calculate_demand(orders(*rows), AS_OF).iloc[0]
    assert row.order_active_days == 4
    assert row.history_days is None
    assert row.data_quality_status == "UNKNOWN_HISTORY"


def test_insufficient_history_is_never_inferred_from_orders():
    # 不论订单落在窗口的哪一段（甚至为空），只要拿不到起点，就不能标记 INSUFFICIENT_HISTORY
    frames = (
        orders(("O1", d(1), "SKU001", 7)),                              # 只有 1 天
        orders(("O1", d(0), "SKU001", 7), ("O2", d(29), "SKU001", 7)),  # 跳满整个窗口
        orders(("O1", d(40), "SKU001", 7)),                             # 全在窗口之外
        orders(),                                                       # 空文件
    )
    for frame in frames:
        out = calculate_demand(frame, AS_OF)
        assert "INSUFFICIENT_HISTORY" not in set(out.data_quality_status)


def test_calendar_denominator_without_coverage_start():
    # D: 30 天窗口里只有 1 天有订单，分母依然是 30 / 7，且历史仍未被证明
    row = calculate_demand(orders(("O1", d(1), "SKU001", 30)), AS_OF).iloc[0]
    assert row.order_active_days == 1
    assert row.daily_demand_30d == 1.0    # 30 / 30
    assert row.daily_demand_7d == 30 / 7  # 30 / 7
    assert row.data_quality_status == "UNKNOWN_HISTORY"


def test_coverage_start_after_as_of_raises():
    try:
        calculate_demand(orders(("O1", d(1), "SKU001", 30)), AS_OF, coverage_start=date(2026, 10, 1))
    except ValueError as e:
        assert "coverage_start" in str(e)
        return
    raise AssertionError("coverage_start after as_of_date must raise ValueError")


# --- 17 未知 SKU / 空输入 ---------------------------------------------------

def test_unknown_sku_is_still_computed():
    # demand engine 不依赖 products.csv：orders 里出现的 SKU 都算
    out = calculate_demand(orders(("O1", d(1), "SKU999", 30)), AS_OF)
    assert out.sku.tolist() == ["SKU999"]
    assert out.daily_demand_30d.iloc[0] == 1.0


def test_sku_without_orders_has_no_row():
    out = calculate_demand(orders(("O1", d(1), "SKU001", 30)), AS_OF)
    assert "SKU888" not in out.sku.tolist()  # 缺需求行由后续装配层补 0
    assert len(out) == 1


def test_empty_orders_gives_empty_result():
    empty = orders()
    out = calculate_demand(empty, AS_OF)
    assert list(out.columns) == list(COLUMNS)
    assert len(out) == 0


# --- 18 可复现性与 as_of_date 必填 -----------------------------------------

def test_result_is_reproducible():
    df = orders(("O1", d(1), "SKU001", 30), ("O2", d(3), "SKU002", 90))
    first = calculate_demand(df, AS_OF)
    second = calculate_demand(df, AS_OF)
    pd.testing.assert_frame_equal(first, second)


def test_as_of_date_is_required_and_must_be_a_date():
    df = orders(("O1", d(1), "SKU001", 30))
    try:
        calculate_demand(df)  # 不允许隐式使用今天
    except TypeError as e:
        assert "as_of_date" in str(e)
    else:
        raise AssertionError("as_of_date must be required")

    try:
        calculate_demand(df, "2026-09-20")  # 字符串也不行，必须显式给 date
    except TypeError as e:
        assert "as_of_date" in str(e)
        return
    raise AssertionError("as_of_date must be a datetime.date")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
