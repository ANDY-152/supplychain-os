[中文](./README.md) | [English](./README_EN.md)

# Supply Chain Decision Agent

一个面向**库存补货**与**采购建议**的确定性供应链决策引擎。

它把原始的订单、库存与在途事实，转化为每个 SKU 的一条可审计决策：需求有多少、库存位置在哪里、是否存在补货缺口，以及在 MOQ / 订货倍数约束之后应该下多少量。每条结果都携带其证据与数据质量状态，并可通过多日仿真逐日复现。

> 范围说明：这里的 “Agent” 指面向工具的决策组件 —— 一个可被调用的报告工具，加一层轻量 Web UI 封装。本仓库**不**实现自主规划、多 Agent 编排或基于 LLM 的决策。所有决策都来自明确、确定性的公式。
>
> 可选补充：仓库提供一个**默认关闭**的**大模型解释层**（`app/llm_explanation.py`），仅把已算好的决策结果转成自然语言说明，**不参与任何计算**。它不改变上述范围：决策仍全部来自确定性公式。

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

## 大模型解释层（可选，LLM Explanation Layer）

**作用与边界**：把**已经算好的**决策结果（逐 SKU 的库存位置、再订货点、缺口、建议采购量、状态、原因、数据质量等）交给大模型，生成一段自然语言说明。

- **不参与核心计算**：不参与需求计算、库存判断、再订货点计算、采购约束处理与补货数量计算；**不修改**任何数值或决策状态。
- **输入白名单**：只发送 `app/llm_explanation.py` 中 `EXPLANATION_FIELDS` 列出的字段，不发送无关内部字段。
- **调用位置**：仅在展示层 `app/demo.py` 中调用，独立于 `app/inventory.py` / `app/procurement.py`。
- **默认关闭**：只有显式设置开关才调用。

### 配置与启用

| 环境变量 | 必需 | 说明 |
|---|---|---|
| `SCOS_LLM_EXPLANATION` | 是（启用开关） | `1` / `true` / `yes` 时启用；其它值或不设置则关闭 |
| `OPENAI_API_KEY` | 启用时必需 | API Key，**只从环境变量读取**，不写入代码/配置/测试数据/日志 |
| `SCOS_LLM_MODEL` | 启用时必需 | 模型名 |
| `SCOS_LLM_BASE_URL` | 否 | OpenAI 兼容服务地址，默认 `https://api.openai.com/v1` |

接口为 OpenAI 兼容的 `POST {base_url}/chat/completions`。

```bash
# 启用（Windows cmd）
set SCOS_LLM_EXPLANATION=1
set OPENAI_API_KEY=your-key
set SCOS_LLM_MODEL=your-model
python -X utf8 -m app.demo
# PowerShell: $env:SCOS_LLM_EXPLANATION="1"; $env:OPENAI_API_KEY="..."; $env:SCOS_LLM_MODEL="..."

# 关闭
set SCOS_LLM_EXPLANATION=0
```

### 本地代理（可选：复用 Pi 已配置的凭据）

若要让解释层复用 **Pi 已配置的凭据** 访问固定模型 `deepseek-v4-pro`，使用 `tools/pi_llm_proxy.mjs`：它在 `127.0.0.1` 上提供一个 OpenAI 兼容入口，把请求转成 `pi --mode rpc`（stdio JSONL）调用。**代理自身不读取任何 provider 凭据**，模型鉴权由 Pi 完成。

> 定位：这是**可选的本机开发工具**，**不是**库存系统的运行时依赖 —— `app/` 的计算链路完全不使用它；解释层也可以直连任意 OpenAI 兼容服务而完全不启动代理。

```bash
# 启动（前台运行，Ctrl+C 关闭；不注册服务、不开机自启、不对外暴露）
node tools/pi_llm_proxy.mjs
# 启动横幅（stderr）给出实际端口与本次随机 Token（仅内存，只显示一次）
```

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `PI_LLM_PROXY_PORT` | `8791` | 监听端口；被占用时自动改用临时端口（横幅会说明） |
| `PI_CLI_PATH` | 自动解析 | Pi CLI 入口（pi-coding-agent/dist/bundle/cli.js）的绝对路径；仅在自动解析失败时设置 |
| `PI_LLM_PROXY_SHOW_TOKEN` | `1` | 设为 `0`/`false`/`no` 时**不在横幅中显示 Token**（脚本化、日志重定向场景） |
| `PI_LLM_PROXY_TOKEN` | 未设置 | 自行指定代理 Token（≥16 字符）以替代随机生成；`SHOW_TOKEN=0` 时客户端必须靠它认证 |

把 Python 侧指向代理（**`OPENAI_API_KEY` 填代理 Token，不是 DeepSeek API Key**）：

```bash
set OPENAI_API_KEY=<代理启动时显示的 Token>
set SCOS_LLM_BASE_URL=http://127.0.0.1:<实际端口>/v1
set SCOS_LLM_MODEL=deepseek-v4-pro
set SCOS_LLM_EXPLANATION=1
python -X utf8 -m app.demo
```

- **两种凭据必须区分**：这里的 `OPENAI_API_KEY` 是**本地代理 Token**，只存在于代理进程内存；DeepSeek 凭据由 Pi 自己保管，代理既不读取也不转发。
- **关闭**：在代理终端按 `Ctrl+C`；代理会终止在途 `pi` 子进程并关闭监听。客户端断开或超时同样会终止对应子进程。
- **一次代理请求不保证只产生一次上游模型调用**：代理只在收到 RPC 的 `agent_settled` 时才返回；若 Pi 内部发生自动重试，可能带来**额外的模型调用与费用**，代理无法封顶。
- **没有在途请求并发上限**：持有有效令牌的本地调用方可同时发起多个请求，代理会为每个请求各启动一个 `pi` 子进程，由此可能产生额外的上游模型调用与费用。
- **横幅中的 Token 是敏感凭据（SENSITIVE）**：它只显示一次、仅存在于内存，且**不是** DeepSeek API Key。请勿把 stderr 重定向到会被持久化或上传的文件。代理**不会**把 Token 写入应用日志、异常或普通诊断日志（日志只含 HTTP 方法、固定路由标签、状态码、耗时与固定原因标签）。
- **输入限制**：请求体 ≤ 256 KiB；`messages` 1–20 条且 `content` 必须为字符串；system 提示词 ≤ 8 KiB（超出返回 400，且不会启动子进程）。
- **环境前提**：需要本机已安装并配置好 Pi CLI，且 Pi 侧存在可用的 provider 凭据（模型名由代理固定传入，见下方限制；代理通过 `pi --mode rpc` 调用它）。仓库**不含** `package.json`，未引入任何 npm 依赖管理体系，**不要**为此新增依赖。
- **已知环境耦合（provider / model 是固定项）**：代理内置的 provider / model 是**固定常量**（`deepseek` / `deepseek-v4-pro`），**不能**通过环境变量或请求体的 `model` 字段切换 —— 该字段只做类型校验、不参与选择（Python 客户端仍需设置 `SCOS_LLM_MODEL`，但它经本代理时不改变实际使用的模型）。CLI 路径解析（`PI_CLI_PATH`，或 `%APPDATA%/npm`、`~/.pi/agent/npm` 下的 pi-coding-agent 包内路径）与**本机安装**绑定；换机器或上游改变包结构后需要重新调整。未做跨机器移植验证。
- **仅监听 `127.0.0.1`，不是生产服务**：不要暴露到网络、不要注册为常驻服务。**不含凭据不等于适合公开部署** —— `GET /health` 免鉴权、Token 为一次性内存值，都是面向本机单用户的设计。

#### 如何获取 Token / 关闭显示 / 关闭后如何配置客户端

- **默认（首次配置）**：启动代理后在 stderr 横幅中读取本次随机 Token，按下方示例写入 `OPENAI_API_KEY`。
- **关闭显示**：设置 `PI_LLM_PROXY_SHOW_TOKEN=0`，横幅只显示 `token : hidden`。
- **关闭后如何认证**：此时必须由你提供 Token —— 设置 `PI_LLM_PROXY_TOKEN=<你自备的长随机值>`（≥16 字符，否则代理拒绝启动），并在客户端把同一个值写入 `OPENAI_API_KEY`。**不要**把它写进仓库、日志或 issue；建议通过本机密钥管理器注入环境变量。
- 关闭显示并不会削弱代理本身的鉴权：`Authorization: Bearer` 校验逻辑不变，只是不再把凭据打印到终端。

#### HTTP 状态码（与实现一致）

| 状态码 | 触发条件 | 响应体 |
|---|---|---|
| `200` | 生成完成；`choices[0].finish_reason` ∈ `stop` / `length` / `unknown` | OpenAI 兼容完成对象 |
| `400` | 请求体不是合法 JSON 或不是对象；`messages` 为空或超过 20 条；`model`/`message.role`/`message.content` 类型不合法；system 提示词超过 8 KiB | `{"error":{"message",…,"type":"invalid_request_error"}}` |
| `401` | `Authorization: Bearer` 缺失或与本次 Token 不符 | `authentication_error` |
| `404` | 路径既不是 `/v1/chat/completions` 也不是 `GET /health` | `invalid_request_error` |
| `405` | 路径为 `/v1/chat/completions` 但方法不是 `POST` | `invalid_request_error` |
| `413` | 请求体超过 256 KiB | `invalid_request_error` |
| `499` | 客户端在响应前断开（代理已中止对应的 `pi` 子进程）；无响应体 | — |
| `502` | 子进程启动失败、RPC 命令失败、`stopReason` 为 `error`/`aborted`/`toolUse`、输出为空或非 JSON | `api_error` |
| `504` | 代理侧 RPC 超时（默认 120 秒） | `timeout_error` |
- **鉴权顺序**：`GET /health` 按既定规则免鉴权；**其他请求一律先鉴权**，通过后才做路由与方法检查。因此未携带合法 Token 时，未知路径与错误方法都返回 `401`；只有通过鉴权后才可能得到 `404` / `405` —— 未授权调用方无法用状态码探测存在哪些路由。

- `finish_reason` 说明：`stop` = 正常结束；`length` = 被输出长度上限截断（**文本仍会返回，但已标记截断**）；`unknown` = 上游未报告 `stopReason`（保持兼容，不虚构成功原因）。`error`/`aborted`/`toolUse` 不会以 200 返回。
- **客户端（`app/llm_explanation.py`）只接受 `finish_reason="stop"`**：`length`、`unknown`、字段缺失或任何其它取值（如 `content_filter`）都会抛出 `IncompleteExplanationError`（`RuntimeError` 的子类）。调用方**不会**收到被截断或未完成的文本；`app/demo.py` 捕获后打印 `[LLM_EXPLANATION_UNAVAILABLE] …`，确定性结果不受影响。
- 上述结束状态目前仅由**模拟测试**覆盖（代理侧用例 32–37、客户端侧 `finish_reason` 契约用例）；**未对真实上游逐一验证**这些状态。
- 注意：若本机设置了 `HTTP_PROXY` 或 `NODE_USE_ENV_PROXY`，Python 访问 `127.0.0.1` 需要 `NO_PROXY=127.0.0.1,localhost`（仅对该测试/运行进程设置即可）。

### 失败降级

请求超时、认证失败、网络错误、返回格式异常、请求构造或序列化失败（例如结果帧中出现无法 JSON 序列化的值）、结束状态不是 `stop`（`length` / `unknown` / 字段缺失 / 其它取值）或未配置时，解释层只打印 `[LLM_EXPLANATION_UNAVAILABLE] ...` 提示，**绝不中断**确定性决策输出，也**不伪造**成功的解释。默认 90 秒超时（代理侧 RPC 预算 120 秒，客户端先于代理放弃），不做自动重试；超时或客户端断开后会终止对应的 `pi` 子进程。

### 完成程度（如实）

- 已实现：白名单投影、OpenAI 兼容调用、超时与失败降级、展示层接入、可选本地代理（`tools/pi_llm_proxy.mjs`）、单元与集成测试（默认全部用 mock，无需真实 Key）。
- **真实端到端调用历史：两次独立调用，不是同一次测试**。① 阶段 2.7C：`app/llm_explanation.py` → 本地 `127.0.0.1` 代理 → 真实 `pi --mode rpc` → `deepseek-v4-pro`，返回非空中文解释（HTTP 200，约 30.12 秒）。② 阶段 3.0：同一链路复跑一次，用于验证严格的 `finish_reason` 契约，HTTP 200 / `finish=stop`，约 49 秒（代理日志 ms≈49164），Python 退出码 0。
- **尚未验证**：并发请求、长期运行稳定性、其他模型或服务商、以及默认 OpenAI 端点（需使用者自备可用服务）。
- **库存计算始终由确定性逻辑完成**：大模型只生成解释文本，不参与需求、库存、再订货点、采购约束与补货数量的任何计算。
- 解释文本**不写入**审计 JSONL（审计仅记录确定性决策投影，格式未改动）。

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
│   ├── demo.py           # CLI 入口：加载 -> 计算 -> 打印
│   └── llm_explanation.py # 可选：大模型解释层（默认关闭，只解释不改数）
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
│   ├── test_simulator.py
│   └── test_llm_explanation.py
├── requirements.txt
├── dev-requirements.txt
└── tools/                          # 可选本地开发工具（不是库存系统的运行时依赖）
    ├── pi_llm_proxy.mjs            # Pi RPC 代理（仅监听 127.0.0.1；需本机 Pi）
    └── pi_llm_proxy.test.mjs       # 代理测试：node --test tools/pi_llm_proxy.test.mjs
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

运行时依赖：`pandas`。核心引擎没有数据库、没有 Web 框架、也没有 LLM 依赖。可选的大模型解释层只使用 Python 标准库，**不新增任何依赖**。

---

## 运行 Demo（Run Demo）

用仓库内置样例数据运行引擎：

```bash
python -X utf8 -m app.demo
```

Demo 会加载 CSV、运行完整链路，并把决策报告打印到 stdout —— 包括基准日、需求口径、每个 SKU 的 Level 1 / Level 2 决策、需求证据、风险提示与汇总。默认使用当天日期；传入显式 `as_of_date`（测试即如此）可让输出完全可复现。

Demo 只是展示入口 —— 它自身不包含任何业务公式。

如需让大模型解释决策结果（**可选，默认关闭**），见下文“大模型解释层”。未设置 `SCOS_LLM_EXPLANATION` 时，解释层完全不调用。

---

## 运行测试（Run Tests）

```bash
pip install -r dev-requirements.txt
python -m pytest -q

# 可选本地代理的测试（只用 Node 内置模块，无需 npm install）
node --test tools/pi_llm_proxy.test.mjs
```

当前测试套件：**262 passed**（其中 `tests/test_llm_explanation.py` 44 条）。Node 侧代理测试 **37 passed / 1 skipped**：跳过者为用例 19 的真实 Pi 握手，仅在显式设置 `PI_RPC_SMOKE=1` 时运行，常规回归不设置。本机实测 Node v24.16.0；仓库未声明 Node 版本要求，也未引入 npm 依赖管理。

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
- 大模型决策解释层（可选，默认关闭；只解释、不改数；见上）。

测试套件：**262 passed**。

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
