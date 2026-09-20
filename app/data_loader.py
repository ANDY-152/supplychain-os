"""Read CSV files from data/, with column checks."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INVENTORY_COLUMNS = {"sku", "current_stock", "daily_demand", "safety_stock", "lead_time_days"}


def load_csv(name: str, required: set[str] = frozenset(), **read_csv_kwargs) -> pd.DataFrame:
    """load_csv("x.csv") -> DataFrame. Extra kwargs go to pd.read_csv.

    Pass encoding="utf-8-sig" (or "gbk") if a file was saved by Excel.
    Pass required={"sku", ...} to fail loudly on a bad file.
    """
    df = pd.read_csv(DATA_DIR / name, **read_csv_kwargs)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df


def load_inventory(name: str = "inventory.csv", **read_csv_kwargs) -> pd.DataFrame:
    return load_csv(name, required=INVENTORY_COLUMNS, **read_csv_kwargs)
