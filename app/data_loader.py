"""Read CSV files from data/, with column checks."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INVENTORY_COLUMNS = {"sku", "current_stock", "safety_stock", "lead_time_days"}
PRODUCTS_COLUMNS = {"sku", "product_name", "category"}
ORDERS_COLUMNS = {"order_id", "order_date", "sku", "qty"}
IN_TRANSIT_COLUMNS = {"po_id", "sku", "qty", "expected_date", "status"}


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
