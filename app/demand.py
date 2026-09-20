"""Demand engine: order facts -> daily demand metrics.

Pure functions: no file IO, no system clock. `as_of_date` is always passed in,
so the same input always gives the same output.

一次计算回答三个问题：
- 每天卖多少（daily_demand_30d / daily_demand_7d）
- 需求在涨还是在跌（demand_trend）
- 这批数据够不够看（history_days / order_active_days / data_quality_status）

需求数值与历史完整度是**两件事，绝不互相代偿**：
- 需求数值只由订单窗口决定，分母固定 30 / 7，窗口内没订单的日子按 0 计入。
- 历史完整度描述“这批数据覆盖了多久”，只有能从数据证明时才行结论，
  否则为 UNKNOWN_HISTORY；绝不用“订单少”反推出“历史不足”。
"""
from datetime import date, timedelta

import pandas as pd

COLUMNS = (
    "sku",
    "daily_demand_30d",
    "daily_demand_7d",
    "demand_trend",
    "history_days",
    "order_active_days",
    "data_quality_status",
)

WINDOW_30D = 30
WINDOW_7D = 7
INCREASE_FACTOR = 1.10
DECREASE_FACTOR = 0.90


def calculate_demand(
    orders: pd.DataFrame,
    as_of_date: date,
    coverage_start: date | None = None,
) -> pd.DataFrame:
    """orders -> 一行一个 SKU（按 sku 升序），列见 COLUMNS。

    as_of_date: 分析基准日，**必填**，绝不使用系统时间，保证结果可复现。
    coverage_start: 数据导出/覆盖窗口的起点。只有提供它才能"证明"历史长度；
        不提供时 history_days 为空、data_quality_status = UNKNOWN_HISTORY（不猜）。

    窗口口径（闭区间，按自然日）：
        30D = [as_of_date - 29, as_of_date]      分母固定 30
        7D  = [as_of_date - 6,  as_of_date]      分母固定 7
    某天没有订单 = 当天需求 0，不减分母。
    晚于 as_of_date 的订单不进入任何窗口。

    ⚠️ 结果含义（当 data_quality_status == "UNKNOWN_HISTORY" 时）：
        固定分母 30 天里可能包含“根本没有历史数据”的日子，它们和“真实 0 需求”
        在数值上无法区分，也被当成 0 计入了分母。因此此时的 daily_demand_30d
        存在**历史完整性不确定性**，不能当成“完整 30 天需求”使用。
        本函数既不缩分母、也不猜起始日、也不改 status：
        机器可读的信号就是 data_quality_status。
    """
    if not isinstance(as_of_date, date):
        raise TypeError("as_of_date must be a datetime.date")

    data = _validated_orders(orders)
    history_days, quality = _history(coverage_start, as_of_date)

    end = pd.Timestamp(as_of_date)
    in_30d = data["order_date"].between(end - pd.Timedelta(days=WINDOW_30D - 1), end)
    in_7d = data["order_date"].between(end - pd.Timedelta(days=WINDOW_7D - 1), end)

    skus = sorted(data["sku"].unique())
    # 窗口内没有订单 → 需求 0。这是结构性补零，不是掩盖脏数据：
    # null / 非法 / 负数在上一步已经抛错，能走到这里的都是有效订单。
    total_30d = data[in_30d].groupby("sku")["qty"].sum().reindex(skus, fill_value=0)
    total_7d = data[in_7d].groupby("sku")["qty"].sum().reindex(skus, fill_value=0)
    active_days = data[in_30d].groupby("sku")["order_date"].nunique().reindex(skus, fill_value=0)

    # 分母固定为自然日数，且**不因历史不完整而缩小**：
    # 当 data_quality_status == "UNKNOWN_HISTORY" 时，下面这些 0 里可能混着
    # “无历史数据”的日子，故 daily_demand_30d 带历史完整性不确定性。
    # 处理方式：不改公式、不补猜、只由 data_quality_status 暴露风险。

    out = pd.DataFrame(
        {
            "sku": skus,
            "daily_demand_30d": total_30d.to_numpy() / WINDOW_30D,
            "daily_demand_7d": total_7d.to_numpy() / WINDOW_7D,
            "order_active_days": active_days.to_numpy().astype(int),
            "history_days": history_days,
            "data_quality_status": quality,
        }
    )
    out["demand_trend"] = [
        _demand_trend(d7, d30)
        for d7, d30 in zip(out["daily_demand_7d"], out["daily_demand_30d"])
    ]
    return out[list(COLUMNS)]


def _demand_trend(daily_demand_7d: float, daily_demand_30d: float) -> str:
    """严格不等号：恰好等于 1.10 / 0.90 都算 STABLE。

    注意 `dd30 == 0 且 dd7 > 0` 在 calculate_demand 里不可能出现（7D 窗口是 30D 的子集，
    有 7D 订单必有 30D 订单），但规则明确要求处理，故保留该分支。
    """
    if daily_demand_30d == 0:
        return "INCREASING" if daily_demand_7d > 0 else "NO_DEMAND"
    if daily_demand_7d > daily_demand_30d * INCREASE_FACTOR:
        return "INCREASING"
    if daily_demand_7d < daily_demand_30d * DECREASE_FACTOR:
        return "DECREASING"
    return "STABLE"


def _history(coverage_start: date | None, as_of_date: date) -> tuple[int | None, str]:
    """history_days 描述**数据覆盖范围**，与 order_active_days（订单活跃天数）无关。

    只有拿到 coverage_start 才能证明历史长度；拿不到就标记 UNKNOWN_HISTORY，
    绝不把"订单少"当成"历史不足"。
    """
    if coverage_start is None:
        return None, "UNKNOWN_HISTORY"
    if coverage_start > as_of_date:
        raise ValueError("coverage_start must not be after as_of_date")
    days = (as_of_date - coverage_start).days + 1
    return days, ("INSUFFICIENT_HISTORY" if days < WINDOW_30D else "OK")


def _validated_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """fail-fast：不做 abs()、不 fillna(0)、不猜业务含义。"""
    for column in ("sku", "order_date", "qty"):
        nulls = orders[column].isna()
        if nulls.any():
            raise ValueError(f"null {column} in {int(nulls.sum())} order(s): {_rows(orders, nulls)}")

    try:
        dates = pd.to_datetime(orders["order_date"], format="%Y-%m-%d")
    except (ValueError, TypeError) as e:
        raise ValueError(f"invalid order_date (expected YYYY-MM-DD): {e}") from e

    try:
        qty = pd.to_numeric(orders["qty"])
    except (ValueError, TypeError) as e:
        raise ValueError(f"invalid qty: {e}") from e

    negative = qty < 0
    if negative.any():
        raise ValueError(
            f"negative qty in {int(negative.sum())} order(s): {_rows(orders, negative)}"
            " - returns/refunds are not representable in this schema"
        )

    out = orders.assign(order_date=dates, qty=qty)
    return out[qty > 0]  # 0 数量订单既无需求，也不算活跃日


def _rows(orders: pd.DataFrame, mask) -> list:
    if "order_id" in orders.columns:
        return orders.loc[mask, "order_id"].tolist()
    return list(orders.index[mask])
