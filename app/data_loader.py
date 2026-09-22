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
