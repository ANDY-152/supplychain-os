[中文](./README.md) | [English](./README_EN.md)

# Supply Chain Decision Agent

一个面向**库存补货**与**采购建议**的确定性供应链决策引擎。

它把原始的订单、库存与在途事实，转化为每个 SKU 的一条可审计决策：需求有多少、库存位置在哪里、是否存在补货缺口，以及在 MOQ / 订货倍数约束之后应该下多少量。每条结果都携带其证据与数据质量状态，并可通过多日仿真逐日复现。

> 范围说明：这里的 “Agent” 指面向工具的决策组件 —— 一个可被调用的报告工具，加一层轻量 Web UI 封装。本仓库**不**实现自主规划、多 Agent 编排或基于 LLM 的决策。所有决策都来自明确、确定性的公式。

---

## 项目简介

该引擎是一个小而依赖极轻的 Python 包（仅 pandas；无数据库、无 Web 框架），围绕一个核心理念构建：**输入事实，输出可审计的决策**。

- 输入是纯 CSV 事实：订单、库存、在途采购订单、覆盖元数据，以及每个 SKU 的采购约束。
- 输出是每个 SKU 的决策帧，同时包含决策与其证据。
- 每次计算都锚定到一个显式的 `as_of_date`；核心从不读取系统时钟，因此相同输入永远产生相同输出。
- 脏数据或含义不清的数据会大声失败，而不是被静默“修复”。

---

## 核心架构

```
        CSV 事实 (data/)
  ┌───────────────────────────────────────────────────────────────┐
  │ inventory.csv   orders.csv   in_transit.csv                   │
  │ data_metadata.csv   procurement_constraints.csv               │
  └───────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    app/data_loader.py
                              │
                              ▼
                    app/demand.py            需求引擎
                    daily_demand_30d / daily_demand_7d
                    demand_trend / data_quality_status
                              │
                              ▼
                    app/inventory.py         库存决策 —— Level 1
                    reorder_point / inventory_position / shortage_qty
                    recommended_order_qty / status / 在途聚合
                              │
                              ▼
                    app/procurement.py       采购建议 —— Level 2
                    MOQ + 订货倍数 -> procurement_recommended_qty
                              │
                              ▼
                    app/demo.py              CLI 报告 (stdout)
                              │
                              └──► Web UI 集成（仅 shell 调用 + 展示，
                                   不承担任何业务计算）
```

**职责分离：** 每一层把上一层的输出当作普通输入，只追加自己的结论。任何层都不会反过来改写更早层的数字。

---

## 核心能力

- **需求引擎（Demand Engine）** —— 滚动 30 天 / 7 天日均需求、需求趋势与数据质量状态。
- **库存决策（Inventory Decision，Level 1）** —— 覆盖天数、再订货点、库存位置、带符号缺口、非负补货量，以及状态。
- **采购建议（Procurement Recommendation，Level 2）** —— 对 Level 1 缺口施加 MOQ 与订货倍数，给出最终建议采购量。
- **在途处理（In-transit handling）** —— 仅 `OPEN` 采购订单计入供给；逾期 `OPEN` 订单作为风险暴露，但不会被自动移除。
- **数据质量（Data Quality）** —— `OK` / `INSUFFICIENT_HISTORY` / `UNKNOWN_HISTORY`，外加结构化风险提示；数据质量永远不会静默改变核心数量。
- **多日仿真器（Multi-day Simulator）** —— 确定性的逐日状态推进，复用同一套需求 / 库存 / 采购逻辑。
- **审计日志（Audit Log）** —— 以 JSONL 追加记录真正产生过的决策，不做任何重算。

---

## 需求引擎（Demand Engine）

`app/demand.py` 提供一个纯函数：

```python
calculate_demand(orders, as_of_date, coverage_start=None) -> pd.DataFrame
```

输出列：`sku`、`daily_demand_30d`、`daily_demand_7d`、`demand_trend`、`history_days`、`order_active_days`、`data_quality_status`。

**30D / 7D 窗口**（闭区间、自然日、固定分母）：

```
daily_demand_30d = sum(qty in [as_of_date - 29, as_of_date]) / 30
daily_demand_7d  = sum(qty in [as_of_date - 6,  as_of_date]) / 7
```

- 某天没有订单按 `0` 需求计；分母始终是 30 / 7。
- 晚于 `as_of_date` 的订单不进入任何窗口。

**趋势**（严格不等号，因此恰好落在边界上是 `STABLE`）：

```
INCREASING   当 daily_demand_7d >  daily_demand_30d * 1.10
DECREASING   当 daily_demand_7d <  daily_demand_30d * 0.90
NO_DEMAND    当 daily_demand_30d == 0 且 daily_demand_7d == 0
其余情况     STABLE
```

**历史 / 质量**描述的是数据覆盖范围，而不是订单活跃程度。只有在覆盖范围可被证明时才下结论：

```
history_days = (as_of_date - coverage_start).days + 1   # 提供 coverage_start 时
未提供 coverage_start    -> UNKNOWN_HISTORY
history_days < 30        -> INSUFFICIENT_HISTORY
其余情况                  -> OK
```

引擎对空值、非数值、负数或格式错误的输入会大声失败，而不是强行转换。

---

## 库存决策 Level 1（Inventory Decision Level 1）

`app/inventory.py` 把库存事实、需求引擎输出与在途事实装配在一起。这里使用的日需求是需求引擎的 `daily_demand_30d`（`demand_basis = "daily_demand_30d"`）。

**核心公式：**

```
coverage_days         = current_stock / daily_demand
reorder_point         = daily_demand * lead_time_days + safety_stock
inventory_position    = current_stock + in_transit_stock        # 仅 OPEN 在途
shortage_qty          = reorder_point - inventory_position      # 带符号
recommended_order_qty = max(0, shortage_qty)                    # Level 1
```

**状态**只由现货决定（在途不影响状态）：

```
CRITICAL   current_stock < safety_stock
REORDER    current_stock < reorder_point
OVERSTOCK  coverage_days > 180
NORMAL     其余情况
```

**在途规则：**

- 只有 `status == "OPEN"` 计入 `in_transit_stock`。
- `ARRIVED` 与 `CANCELLED` 不计入。
- `OPEN` 且 `expected_date < as_of_date` 的订单仍然计入，同时上报到 `overdue_in_transit`（这是**子集**，仅为风险信息 —— 绝不会从库存位置中扣减）。
- `coverage_days` 只用现货计算；在途不进入该分子。

每条决策还携带一个明确的 `reason` 代码（例如 `STOCK_BELOW_REORDER_POINT`、`IN_TRANSIT_COVERS_SHORTAGE`、`SUPPLY_SUFFICIENT`，可带 `_OVERDUE` 后缀）。

---

## 采购建议 Level 2（Procurement Recommendation Level 2）

`app/procurement.py` 消费 Level 1 结果帧与每个 SKU 的约束（`moq`、`order_multiple`），并只追加三列：

```
procurement_recommended_qty   # 最终建议采购量
provisional                   # 置信标记（不改变数量）
procurement_reason            # 实际改变了数量的约束
```

**约束链**（只作用于 Level 1 的缺口）：

```
gap = recommended_order_qty

if gap <= 0:                         q = 0
else:
    q = gap
    if moq > 0 and moq > q:          q = moq            # MOQ 下限
    if order_multiple > 1:           q = ceil(q / order_multiple) * order_multiple

procurement_recommended_qty = q
provisional = data_quality_status != "OK"
```

`procurement_reason` 取值为 `NO_GAP`、`GAP_ONLY`、`MOQ_APPLIED`、`MULTIPLE_APPLIED`、`MOQ_AND_MULTIPLE_APPLIED` 之一，且只反映真正改变了最终数量的约束。

**Level 1 与 Level 2 不是同一个数字：**

| 层级 | 字段 | 含义 |
|---|---|---|
| Level 1 | `recommended_order_qty` | 由库存与需求导出的非负**补货缺口**。 |
| Level 2 | `procurement_recommended_qty` | 施加 MOQ 与订货倍数之后的**采购建议数量**。 |

Level 1 永远不会被 Level 2 改写；两者并排保留，使取整决策保持可审计。

---

## 数据质量（Data Quality）

需求引擎返回一个数据质量状态：

| 状态 | 含义 |
|---|---|
| `OK` | 可从 `coverage_start` 证明覆盖至少 30 天。 |
| `INSUFFICIENT_HISTORY` | 覆盖范围已知，但不足 30 天。 |
| `UNKNOWN_HISTORY` | 未提供覆盖起点；历史长度无法断言。 |

在不完整窗口上算出的数量仍会以固定 30 天分母产出 —— 风险通过状态暴露，而不是缩短分母或猜测起始日。

库存层还会额外发出结构化 `data_warnings`（每条含 `sku`、`code` 与可读 `message`）：

- `UNKNOWN_HISTORY`、`INSUFFICIENT_HISTORY` —— 证据缺口。
- `OVERDUE_INBOUND` —— 已逾期但仍为 `OPEN` 的在途订单。

**数据质量永远不会静默改变核心数量。** 它只影响提示通道以及 Level 2 的 `provisional` 标记。

---

## 多日仿真器（Multi-day Simulator）

`app/simulator.py` 将引擎在多个自然日上回放。它**直接复用现有的需求 / 库存 / 采购模块**，不重新实现任何公式。

每日执行顺序（固定）：

```
1. 到货：expected_date <= current_day 的 OPEN PO -> ARRIVED，
   数量计入 current_stock
2. 应用当天需求：current_stock -= demand_qty
3. calculate_demand()            (需求引擎)
4. analyze_inventory_frame()     (库存 Level 1)
5. apply_procurement()           (采购 Level 2)
6. order_placed_qty = procurement_recommended_qty
7. 创建新的 OPEN PO：expected_date = current_day + lead_time_days
8. 产出审计记录（DECISION + SKIP）
9. 进入下一天
```

关键性质：

- **到货先于下单**，因此某天创建的采购订单绝不会在同一天到货 —— 即使 `lead_time_days == 0` 也不会。
- **需求按字面扣减**：`stock_after = stock_before - demand_qty`；库存允许为负，未满足需求记为 `max(0, demand_qty - stock_before)`，绝不 clamp 到零。
- **`provisional=True` 不会阻止下单。** 数量照常生成。
- **`po_id` 是确定性的**（`PO-{run_id}-{seq:06d}`），并在一次运行内唯一。
- **相同输入产生相同运行。** 给定相同的 `run_id`、`start_day`、`days`、状态与 `demand_profile`，`run_simulation()` 返回完全一致的结果 —— 无随机性、无时钟依赖。

**需求分离（重要）：**

- `orders_log` 是**历史订单日志**。它是只读的，且*仅*供需求引擎使用。
- `demand_profile` 是 `DataFrame[sku, simulation_day, qty]`，*仅*用于当天库存消耗。
- 模拟需求**绝不会写回** `orders_log`。

**零需求 SKU：** 如果某个 SKU 没有需求证据、被库存层在当天跳过，仿真器不会删除该 SKU。它会记录一条 `SKIP` 审计条目（保留原 warning 的 field/message），不为它下单，并将该 SKU 带入下一天。

---

## 审计日志（Audit Log）

`app/audit_log.py` 提供两个小函数：

```python
build_audit_records(frame, action=None) -> list[dict]
append_audit_records(path, records) -> None
```

- `build_audit_records` **投影**决策帧为记录，并合并仿真器的动作字段（`order_placed_qty`、`po_id`、`expected_date`）。它不重算任何业务值。
- `append_audit_records` 以**追加**模式写 **JSONL**，UTF-8、`ensure_ascii=False`、一行一个 JSON object。它绝不覆盖，也绝不吞掉写入失败。

记录信封：`run_id`、`simulation_day`、`as_of_date`、`sku`、`record_type`（`DECISION` 或 `SKIP`）。DECISION 记录还会携带投影后的 Level 1 / Level 2 / 需求证据字段。

审计日志**只记录已经产生的决策** —— 它是投影/持久化层，而不是计算层。它不读取时钟，也不添加 timestamp。

---

## 项目结构

```
supplychain-os/
├── app/
│   ├── __init__.py
│   ├── data_loader.py    # CSV 加载与契约校验
│   ├── demand.py         # 需求引擎（30D / 7D、趋势、质量）
│   ├── inventory.py      # 库存决策 Level 1 + 在途聚合
│   ├── procurement.py    # 采购建议 Level 2（MOQ / 倍数）
│   ├── audit_log.py      # JSONL 审计投影（不重算）
│   ├── simulator.py      # 多日仿真器（复用以上各层）
│   └── demo.py           # CLI 入口：加载 -> 计算 -> 打印
├── data/
│   ├── inventory.csv
│   ├── orders.csv
│   ├── in_transit.csv
│   ├── products.csv
│   ├── data_metadata.csv            # 声明 coverage_start
│   └── procurement_constraints.csv  # 每 SKU 的 moq / order_multiple
├── tests/
│   ├── test_data_loader.py
│   ├── test_demand.py
│   ├── test_inventory.py
│   ├── test_procurement.py
│   ├── test_audit_log.py
│   └── test_simulator.py
├── requirements.txt
└── dev-requirements.txt
```

---

## 快速开始（Quick Start）

```bash
git clone <your-repo-url> supplychain-os
cd supplychain-os

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

运行时依赖：`pandas`。核心引擎没有数据库、没有 Web 框架、也没有 LLM 依赖。

---

## 运行 Demo（Run Demo）

用仓库内置样例数据运行引擎：

```bash
python -X utf8 -m app.demo
```

Demo 会加载 CSV、运行完整链路，并把决策报告打印到 stdout —— 包括基准日、需求口径、每个 SKU 的 Level 1 / Level 2 决策、需求证据、风险提示与汇总。默认使用当天日期；传入显式 `as_of_date`（测试即如此）可让输出完全可复现。

Demo 只是展示入口 —— 它自身不包含任何业务公式。

---

## 运行测试（Run Tests）

```bash
pip install -r dev-requirements.txt
python -m pytest -q
```

当前测试套件：**218 passed**。

测试覆盖需求窗口与趋势边界、Level 1 数学与状态规则、在途聚合、Level 2 约束链、输入 fail-fast 校验、审计日志投影与 JSONL 行为，以及仿真器的确定性 / 到货顺序 / 库存守恒。

---

## Web UI

引擎与 UI 无关：所有相关内容都由 `python -X utf8 -m app.demo` 以确定性纯文本打印。

一个 Web UI 集成（Pi Web UI 插件）可以封装这个入口：

- 服务端 shell 调用引擎并返回其 stdout；
- 客户端只对返回文本做解析、排序与搜索。

**Web UI 不承担任何核心业务计算。** 它不 import 引擎公式，也不重新推导任何数量。该插件是独立组件，不属于本仓库。

---

## 设计原则

- **每个公式只有一个来源。** 需求、库存、采购规则各自只存在于一个模块中；其余各层都调用它们。
- **事实与结论分层。** 需求证据、Level 1 缺口、Level 2 建议是彼此独立的字段，绝不互相覆盖。
- **构造上即确定性。** `as_of_date` 始终显式传入；核心不读取系统时钟，因此结果可复现。
- **大声失败。** 空值、负数量、畸形日期、重复键与未知状态都会抛错，而不是被静默转换。
- **质量是可见的，而不是纠正性的。** 不完整历史通过状态与提示暴露；它绝不静默改变数量。
- **仿真器复用引擎。** 它推进状态并编排现有模块，而不是重新实现它们。
- **审计日志只记录，不决策。** 它投影已经产生的决策，绝不重算。

---

## 当前状态（Current Status）

已实现并通过测试：

- 需求引擎（30D / 7D 需求、趋势、覆盖质量）。
- 库存决策 Level 1（再订货点、库存位置、缺口、补货量、状态）。
- 在途处理（仅 OPEN 计入供给，逾期作为风险）。
- 采购建议 Level 2（MOQ 与订货倍数）。
- 数据质量状态与结构化风险提示。
- 确定性的多日仿真器。
- 审计日志（追加式 JSONL 投影）。

测试套件：**218 passed**。

---

## 路线图（Roadmap）

以下均为 **planned / 未来**，**尚未实现**：

- `LICENSE` 文件（见下）—— *planned*。
- 打包元数据（如 `pyproject.toml`），支持 `pip install` —— *planned*。
- 面向仿真器的最小 CLI 封装 —— *planned*。
- 供应商维度属性与交期波动 —— *planned*。
- 更多需求策略 / 可配置窗口 —— *planned*。
- 面向更大数据集的情景工具 —— *planned*。

---

## 许可证（License）

本仓库目前**未包含** `LICENSE` 文件。在添加之前，代码按现状提供，不授予明确的再使用权利。添加一个 OSI 认可的开源许可证属于 planned 事项（见路线图）。

---

**Supply Chain Decision Agent** —— 一个面向库存补货与采购建议的确定性决策引擎。决策辅助，而非自动采购系统。
