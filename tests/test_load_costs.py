"""load_costs contract tests. Run: pytest tests/test_load_costs.py

A3：`app/data_loader.py` 追加的 `load_costs`；既有 loader 未改。
本文件为新增测试。
"""
import pandas as pd

from app.data_loader import load_costs


def write(tmp_path, text, name="costs.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_costs_valid(tmp_path):
    path = write(tmp_path, "sku,unit_cost\nSKU001,12.5\nSKU002,3\n")
    df = load_costs(str(path))
    assert list(df.columns) == ["sku", "unit_cost"]
    assert df["unit_cost"].dtype == "float64"
    assert df["sku"].tolist() == ["SKU001", "SKU002"]
    assert df["unit_cost"].tolist() == [12.5, 3.0]


def test_load_costs_missing_file(tmp_path):
    try:
        load_costs(str(tmp_path / "nope.csv"))
    except ValueError as e:
        assert "missing or empty" in str(e)
        return
    raise AssertionError("missing file must raise")


def test_load_costs_empty_file(tmp_path):
    path = write(tmp_path, "")
    try:
        load_costs(str(path))
    except ValueError as e:
        assert "missing or empty" in str(e)
        return
    raise AssertionError("empty file must raise")


def test_load_costs_missing_column(tmp_path):
    path = write(tmp_path, "sku,price\nSKU001,1\n")
    try:
        load_costs(str(path))
    except ValueError as e:
        assert "Missing columns" in str(e)
        return
    raise AssertionError("missing column must raise")


def test_load_costs_blank_sku(tmp_path):
    path = write(tmp_path, "sku,unit_cost\n,1\n")
    try:
        load_costs(str(path))
    except ValueError as e:
        assert "blank sku" in str(e)
        return
    raise AssertionError("blank sku must raise")


def test_load_costs_duplicate_sku(tmp_path):
    path = write(tmp_path, "sku,unit_cost\nSKU001,1\nSKU001,2\n")
    try:
        load_costs(str(path))
    except ValueError as e:
        assert "duplicate" in str(e)
        return
    raise AssertionError("duplicate sku must raise")


def test_load_costs_non_numeric(tmp_path):
    path = write(tmp_path, "sku,unit_cost\nSKU001,abc\n")
    try:
        load_costs(str(path))
    except ValueError as e:
        assert "unit_cost" in str(e)
        return
    raise AssertionError("non-numeric unit_cost must raise")


def test_load_costs_nan(tmp_path):
    path = write(tmp_path, "sku,unit_cost\nSKU001,\n")
    try:
        load_costs(str(path))
    except ValueError as e:
        assert "unit_cost" in str(e)
        return
    raise AssertionError("NaN unit_cost must raise")


def test_load_costs_non_positive(tmp_path):
    for value in ("0", "-3"):
        path = write(tmp_path, f"sku,unit_cost\nSKU001,{value}\n", name=f"c{value}.csv")
        try:
            load_costs(str(path))
        except ValueError as e:
            assert "unit_cost" in str(e)
        else:
            raise AssertionError(f"non-positive unit_cost must raise: {value}")


def test_load_costs_infinite(tmp_path):
    path = write(tmp_path, "sku,unit_cost\nSKU001,inf\n")
    try:
        load_costs(str(path))
    except ValueError as e:
        assert "unit_cost" in str(e)
        return
    raise AssertionError("infinite unit_cost must raise")


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as d:
                try:
                    fn(Path(d))
                    print(f"ok   {name}")
                except Exception as e:  # noqa: BLE001
                    failures += 1
                    print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
