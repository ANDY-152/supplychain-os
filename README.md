[中文](./README.md) | [English](./README_EN.md)

# Supply Chain Decision Agent

一个以**确定性规则**为核心的库存补货与采购决策工具：读取订单、库存、在途采购订单和采购约束，输出逐 SKU 的补货建议、预算分配结果及可审计记录。

> “Agent”在本项目中指可被工具调用的决策组件，不代表自主规划或多 Agent 编排。可选 LLM 仅解释已计算的结果，不参与任何数量或库存决策。

## 功能概览

| 模块 | 作用 |
|---|---|
| 需求引擎 | 计算 30 天 / 7 天日均需求、趋势和历史覆盖状态 |
| 库存决策（L1） | 计算再订货点、库存位置、补货缺口和库存状态 |
| 采购建议（L2） | 对补货缺口应用 MOQ 与订货倍数 |
| 预算分配（L3，可选） | 按共享预算分配 L2 建议量，支持 R0 / R1 / R2 |
| 多日仿真 | 按日处理到货、消耗、补货决策和新采购订单 |
| 审计日志 | 将已产生的决策以 JSONL 追加写入，不重新计算业务值 |
| LLM 解释（可选） | 将已计算结果转换为自然语言说明，默认关闭 |

## 决策流程

```text
CSV 输入
  → 输入校验
  → 需求分析
  → 库存决策 L1
  → 采购约束 L2
  → （可选）预算分配 L3
  → CLI / 仿真结果 / 审计记录
```

各层保留自己的结果字段，不互相覆盖：`recommended_order_qty` 是 L1 补货缺口，`procurement_recommended_qty` 是 L2 采购建议，`allocated_qty` 是启用 L3 后的最终分配量。

## L3 预算分配

| 规则 | 行为 |
|---|---|
| R0 | 不约束预算，透传 L2 建议量，作为基准 |
| R1 | 按比例缩放建议量，并按整包与采购约束向下调整 |
| R2 | 按确定性优先级排序，逐 SKU 分配可负担的完整包装 |

R1 / R2 使用整数刻度核算金额。分配需提供 SKU 成本和优先级；模拟器中的预算按周期重置，不跨周期结转。L3 默认关闭，未启用时保留原有 L1 / L2 行为。

## 主要规则与边界

- **需求窗口**：30 天与 7 天使用固定自然日窗口；没有订单的日期按零需求计入分母。历史覆盖不足通过数据质量状态显示，不擅自补齐数据。
- **库存位置**：只有 `OPEN` 在途订单计入供给。逾期 `OPEN` 订单仍计入，同时作为风险提示。
- **采购约束**：MOQ 和订货倍数仅作用于 L1 缺口；L1 原始结果保留，便于审计取整的影响。
- **仿真顺序**：先处理到货，再扣减当日需求，然后计算补货并创建新采购订单。库存允许为负，表示模型中的积压语义。
- **确定性**：核心计算使用显式日期和输入，不依赖系统时钟或随机数；相同状态和输入可复现相同结果。
- **数据处理**：不合规输入会报错，而不是静默转换。数据质量状态与警告用于暴露证据不足，不会自动改写数量。
- **审计**：日志只记录已生成的决策，不负责重新计算；启用 L3 时会附加分配相关字段。

## 快速开始

需要 Python 和 `pandas`。克隆仓库后执行：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -X utf8 -m app.demo
```

安装开发依赖并运行测试：

```bash
pip install -r dev-requirements.txt
python -m pytest -q
node --test tools/pi_llm_proxy.test.mjs  # 可选：仅在需要时运行本地代理测试
```

## 示例数据与输入

`data/` 中的 CSV 是演示输入。`data/costs.csv` 为示例 / 模拟单位成本，不是任何企业的真实采购成本。

| 文件 | 内容 |
|---|---|
| `inventory.csv` | SKU 现有库存与库存参数 |
| `orders.csv` | 历史订单，用于需求计算 |
| `in_transit.csv` | 在途采购订单及状态 |
| `products.csv` | SKU 产品属性 |
| `data_metadata.csv` | 数据覆盖起点 |
| `procurement_constraints.csv` | 每 SKU 的 MOQ 和订货倍数 |
| `costs.csv` | 示例 / 模拟单位成本，供 L3 使用 |

核心运行依赖只有 `pandas`；项目没有数据库或 Web 框架依赖。`tools/` 中的脚本是可选开发与实验工具，不是库存决策核心的运行时依赖。

## 可选 LLM 解释层

LLM 仅接收已完成计算的有限字段，生成解释文本；不计算、不更改需求、库存状态、补货量或预算分配。默认关闭，且核心功能无需配置 API 密钥。

启用时设置：

| 环境变量 | 用途 |
|---|---|
| `SCOS_LLM_EXPLANATION=1` | 显式启用解释层 |
| `OPENAI_API_KEY` | 从环境变量读取的 API Key；若使用本地代理，则为代理令牌 |
| `SCOS_LLM_MODEL` | 模型名称 |
| `SCOS_LLM_BASE_URL` | 可选的 OpenAI 兼容服务地址，默认 `https://api.openai.com/v1` |

详细实现见 `app/llm_explanation.py`。本地 Pi RPC 代理 `tools/pi_llm_proxy.mjs` 仅供开发使用，仅监听 `127.0.0.1`，需要本机 Pi 配置；不要将其暴露到公网。

## 项目结构

```text
app/       核心计算、输入校验、仿真、审计与可选解释层
data/      示例 CSV 输入
tests/    Python 回归与集成测试
tools/    可选本地工具、代理和 L3 实验脚本
```

关键模块：`demand.py`、`inventory.py`、`procurement.py`、`allocation.py`、`simulator.py`、`audit_log.py`。

## 当前限制

- 本项目是确定性决策支持工具，不是自动采购系统。
- R2 是按优先级的贪心整包分配规则，不是全局最优求解器。
- 不实现自主规划、多 Agent 编排或 LLM 驱动的库存决策。
- 仿真指标受需求假设与负库存 / 积压语义影响，不应直接解释为真实企业绩效。
- 尚未声明可复用的开源许可证；仓库目前没有 `LICENSE` 文件。添加许可证前，不应假设已授予他人特定的再分发或使用权。

## 后续方向

- 添加适当的开源许可证与 Python 项目打包元数据。
- 评估更大数据集、更多需求策略及供应商交期变化场景。

---

[English version](./README_EN.md)
