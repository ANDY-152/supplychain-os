"""确定性生成研究用模拟成本 data/costs.csv。

⚠️ 研究用模拟成本，非海尔真实采购成本。

规则（裁决清单 C-1..C-6）：
- 读取 data/inventory.csv 的**实际** SKU 集合（不硬编码数量），按该文件 SKU 顺序输出；
- 价目表为脚本内固定常量；缺少任一 SKU 的模拟价格即报错，不静默补价；
- 无随机数；同输入必产生同输出。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
INVENTORY_PATH = REPO / "data" / "inventory.csv"
OUTPUT_PATH = REPO / "data" / "costs.csv"

# 研究用模拟单价（元）；非海尔真实采购成本
SIMULATED_PRICES = {
    "SKU001": "12.50",
    "SKU002": "8.75",
    "SKU003": "3.20",
    "SKU004": "25.00",
    "SKU005": "6.40",
}


def generate(inventory_path: Path = INVENTORY_PATH, output_path: Path = OUTPUT_PATH) -> Path:
    inventory = pd.read_csv(inventory_path)
    if "sku" not in inventory.columns:
        raise ValueError(f"inventory missing column 'sku': {inventory_path}")

    skus = [str(value) for value in inventory["sku"].tolist()]
    missing = [sku for sku in skus if sku not in SIMULATED_PRICES]
    if missing:
        raise ValueError(f"missing simulated price for SKU(s): {missing}")

    rows = [f"{sku},{SIMULATED_PRICES[sku]}" for sku in skus]
    output_path.write_text("sku,unit_cost\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print(generate())

    # 自检：同输入重生成结果一致
    first = OUTPUT_PATH.read_text(encoding="utf-8")
    generate()
    assert OUTPUT_PATH.read_text(encoding="utf-8") == first, "generator is not deterministic"
