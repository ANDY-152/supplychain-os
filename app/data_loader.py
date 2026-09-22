"""Read CSV files from data/, with column checks."""
from datetime import date, datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INVENTORY_COLUMNS = {"sku", "current_stock", "safety_stock", "lead_time_days"}
PRODUCTS_COLUMNS = {"sku", "product_name", "category"}
ORDERS_COLUMNS = {"order_id", "order_date", "sku", "qty"}
IN_TRANSIT_COLUMNS = {"po_id", "sku", "qty", "expected_date", "status"}
# data_metadata.csv 只声明数据覆盖的起点；不声明其他元数据
DATA_METADATA_COLUMNS = {"key", "value"}
COVERAGE_START_KEY = "coverage_start"
# procurement_constraints.csv：按 SKU 的 Level 2 采购约束（moq / order_multiple）
PROCUREMENT_CONSTRAINTS_COLUMNS = {"sku", "moq", "order_multiple"}


def load_csv(name: str, required: set[str] = frozenset(), **read_csv_kwargs) -> pd.DataFrame:
    """load_csv("x.csv") -> DataFrame. Extra kwargs go to pd.read_csv.

    Pass encoding="utf-8-sig" (or "gbk") if a file was saved by Excel.
    Pass required={"sku", ...} to fail loudly on a bad file.
    Extra columns in the file are allowed and kept as-is.
    """
    df = pd.read_csv(DATA_DIR / name, **read_csv_kwargs)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df


def load_inventory(name: str = "inventory.csv", **read_csv_kwargs) -> pd.DataFrame:
    return load_csv(name, required=INVENTORY_COLUMNS, **read_csv_kwargs)


def load_products(name: str = "products.csv", **read_csv_kwargs) -> pd.DataFrame:
    return load_csv(name, required=PRODUCTS_COLUMNS, **read_csv_kwargs)


def load_orders(name: str = "orders.csv", **read_csv_kwargs) -> pd.DataFrame:
    return load_csv(name, required=ORDERS_COLUMNS, **read_csv_kwargs)


def load_in_transit(name: str = "in_transit.csv", **read_csv_kwargs) -> pd.DataFrame:
    return load_csv(name, required=IN_TRANSIT_COLUMNS, **read_csv_kwargs)


def load_data_metadata(name: str = "data_metadata.csv", **read_csv_kwargs) -> date:
    """data_metadata.csv -> 声明的数据覆盖起点 coverage_start（ISO YYYY-MM-DD）。

    只认 coverage_start 这一个 key。缺失 / 非法 / 重复一律明确失败，
    绝不从 orders.csv 的最早订单日期推断，也不做任何默认/静默填补。
    """
    df = load_csv(name, required=DATA_METADATA_COLUMNS, **read_csv_kwargs)
    matched = df[df["key"] == COVERAGE_START_KEY]
    if matched.empty:
        raise ValueError(f"data_metadata missing key: {COVERAGE_START_KEY}")
    if len(matched) > 1:
        raise ValueError(f"data_metadata duplicate key: {COVERAGE_START_KEY}")

    raw = matched["value"].iloc[0]
    try:
        return datetime.strptime(str(raw), "%Y-%m-%d").date()
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"invalid {COVERAGE_START_KEY} (expected YYYY-MM-DD): {raw!r}"
        ) from e


def load_procurement_constraints(
    name: str = "procurement_constraints.csv", **read_csv_kwargs
) -> pd.DataFrame:
    """procurement_constraints.csv -> 一行一个 SKU：`sku, moq, order_multiple`。

    Level 2 采购约束（当前仅 SKU 维度，无 supplier）：
    - 文件不存在 / 文件为空 / 无该 SKU 行 → “未配置”（返回空 frame），不报错；
      Level 2 在该 SKU 上保持 identity。
    - 文件存在但结构非法 → fail-loud（空 sku / 重复 / 未知 SKU / 负 MOQ /
      非整数或不小于 1 的倍数 / NaN / 非法字符串）。

    绝不 silently fillna，也不生成业务默认值：`moq == 0` 表示未启用 MOQ，
    `order_multiple == 1` 表示未启用倍数。
    """
    path = DATA_DIR / name
    if not path.exists() or path.stat().st_size == 0:
        return _empty_constraints()

    df = load_csv(name, required=PROCUREMENT_CONSTRAINTS_COLUMNS, **read_csv_kwargs)
    df = df[["sku", "moq", "order_multiple"]]
    if df.empty:
        return _empty_constraints()

    blank = df["sku"].isna() | (df["sku"].astype(str).str.strip() == "")
    if blank.any():
        raise ValueError(f"blank sku in procurement_constraints: {int(blank.sum())} row(s)")

    duplicated = df["sku"].duplicated()
    if duplicated.any():
        raise ValueError(
            f"duplicate constraint SKU: {sorted(df['sku'][duplicated].unique().tolist())}"
        )

    unknown = sorted(set(df["sku"]) - set(load_inventory()["sku"]))
    if unknown:
        raise ValueError(f"unknown constraint SKU: {unknown}")

    try:
        moq = pd.to_numeric(df["moq"])
    except (ValueError, TypeError) as e:
        raise ValueError(f"invalid moq: {e}") from e
    if moq.isna().any():
        raise ValueError("null moq in procurement_constraints")

    try:
        multiple = pd.to_numeric(df["order_multiple"])
    except (ValueError, TypeError) as e:
        raise ValueError(f"invalid order_multiple: {e}") from e
    if multiple.isna().any():
        raise ValueError("null order_multiple in procurement_constraints")

    if (moq < 0).any():
        raise ValueError(f"moq must be >= 0: {df['sku'][moq < 0].tolist()}")
    if (multiple < 1).any():
        raise ValueError(f"order_multiple must be >= 1: {df['sku'][multiple < 1].tolist()}")
    if (multiple % 1 != 0).any():
        raise ValueError(
            f"order_multiple must be an integer: {df['sku'][multiple % 1 != 0].tolist()}"
        )

    return df.assign(
        moq=moq.astype(float), order_multiple=multiple.astype(int)
    ).reset_index(drop=True)


def _empty_constraints() -> pd.DataFrame:
    return pd.DataFrame(columns=["sku", "moq", "order_multiple"])
