# Supply Chain Decision Agent

A deterministic supply-chain decision engine for **inventory replenishment** and
**procurement recommendations**.

It turns raw order, inventory and in-transit facts into an auditable decision
per SKU: how much demand is there, where the stock position sits, whether a
replenishment gap exists, and what quantity should be ordered after MOQ / order
multiple constraints. Every result carries its evidence and its data-quality
status, and can be replayed day by day with a multi-day simulator.

> Scope note: "Agent" here refers to the tool-facing decision component — a
> callable report tool plus a thin Web UI wrapper. This repository does **not**
> implement autonomous planning, multi-agent orchestration, or LLM-based
> decision making. All decisions come from explicit, deterministic formulas.

---

## Overview / 项目简介

The engine is a small, dependency-light Python package (pandas only, no
database, no web framework) built around a single idea: **facts in, auditable
decisions out**.

- Inputs are plain CSV facts: orders, inventory, in-transit purchase orders,
  coverage metadata and per-SKU procurement constraints.
- Outputs are per-SKU decision frames with both the decision and its evidence.
- Every calculation is anchored to an explicit `as_of_date`; the core never
  reads the system clock, so the same inputs always produce the same outputs.
- Bad or ambiguous data fails loudly instead of being silently "fixed".

---

## Core Architecture

```
        CSV facts (data/)
  ┌───────────────────────────────────────────────────────────────┐
  │ inventory.csv   orders.csv   in_transit.csv                   │
  │ data_metadata.csv   procurement_constraints.csv               │
  └───────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    app/data_loader.py
                              │
                              ▼
                    app/demand.py            Demand Engine
                    daily_demand_30d / daily_demand_7d
                    demand_trend / data_quality_status
                              │
                              ▼
                    app/inventory.py         Inventory Decision — Level 1
                    reorder_point / inventory_position / shortage_qty
                    recommended_order_qty / status / in-transit aggregation
                              │
                              ▼
                    app/procurement.py       Procurement Recommendation — Level 2
                    MOQ + order multiple -> procurement_recommended_qty
                              │
                              ▼
                    app/demo.py              CLI report (stdout)
                              │
                              └──► Web UI integration (shell out + display only,
                                   no business computation)
```

**Separation of concerns:** each layer takes the previous layer's output as a
plain input and adds its own conclusions. No layer reaches back to rewrite an
earlier layer's numbers.

---

## Core Capabilities

- **Demand Engine** — trailing 30-day / 7-day average daily demand, demand
  trend, and data-quality status.
- **Inventory Decision (Level 1)** — coverage days, reorder point, inventory
  position, signed shortage, non-negative replenishment quantity, and status.
- **Procurement Recommendation (Level 2)** — applies MOQ and order multiple to
  the Level 1 gap and reports a final recommended order quantity.
- **In-transit handling** — only `OPEN` purchase orders count toward supply;
  overdue `OPEN` orders are surfaced as risk without being auto-removed.
- **Data Quality** — `OK` / `INSUFFICIENT_HISTORY` / `UNKNOWN_HISTORY` plus
  structured risk warnings; quality never silently changes core quantities.
- **Multi-day Simulator** — deterministic day-by-day state progression reusing
  the same demand / inventory / procurement logic.
- **Audit Log** — append-only JSONL records of decisions that were actually
  produced, with no recomputation.

---

## Demand Engine

`app/demand.py` exposes a pure function:

```python
calculate_demand(orders, as_of_date, coverage_start=None) -> pd.DataFrame
```

Output columns: `sku`, `daily_demand_30d`, `daily_demand_7d`, `demand_trend`,
`history_days`, `order_active_days`, `data_quality_status`.

**30D / 7D windows** (closed intervals, calendar days, fixed denominators):

```
daily_demand_30d = sum(qty in [as_of_date - 29, as_of_date]) / 30
daily_demand_7d  = sum(qty in [as_of_date - 6,  as_of_date]) / 7
```

- A day with no orders counts as `0` demand; the denominator stays 30 / 7.
- Orders dated after `as_of_date` are excluded from every window.

**Trend** (strict comparisons, so the exact boundaries are `STABLE`):

```
INCREASING   when daily_demand_7d >  daily_demand_30d * 1.10
DECREASING   when daily_demand_7d <  daily_demand_30d * 0.90
NO_DEMAND    when daily_demand_30d == 0 and daily_demand_7d == 0
otherwise STABLE
```

**History / quality** are about data coverage, not about order activity. They
are only asserted when coverage can be proven:

```
history_days = (as_of_date - coverage_start).days + 1   # when coverage_start is given
coverage_start is None            -> UNKNOWN_HISTORY
history_days < 30                 -> INSUFFICIENT_HISTORY
otherwise                         -> OK
```

The engine fails loudly on null, non-numeric, negative, or malformed inputs
rather than coercing them.

---

## Inventory Decision Level 1

`app/inventory.py` blends inventory facts with the Demand Engine output and
in-transit facts. The daily demand used here is the Demand Engine's
`daily_demand_30d` (`demand_basis = "daily_demand_30d"`).

**Core formulas:**

```
coverage_days         = current_stock / daily_demand
reorder_point         = daily_demand * lead_time_days + safety_stock
inventory_position    = current_stock + in_transit_stock        # OPEN in-transit only
shortage_qty          = reorder_point - inventory_position      # signed
recommended_order_qty = max(0, shortage_qty)                    # Level 1
```

**Status** is decided from on-hand stock only (in-transit does not change the
status):

```
CRITICAL   current_stock < safety_stock
REORDER    current_stock < reorder_point
OVERSTOCK  coverage_days > 180
NORMAL     otherwise
```

**In-transit rules:**

- Only `status == "OPEN"` counts toward `in_transit_stock`.
- `ARRIVED` and `CANCELLED` are excluded.
- An `OPEN` order whose `expected_date < as_of_date` still counts, and is also
  reported in `overdue_in_transit` (a **subset**, risk information only — it is
  never subtracted from the position).
- `coverage_days` uses on-hand stock only; in-transit is not in that numerator.

Each decision also carries an explicit `reason` code (e.g.
`STOCK_BELOW_REORDER_POINT`, `IN_TRANSIT_COVERS_SHORTAGE`,
`SUPPLY_SUFFICIENT`, with an optional `_OVERDUE` suffix).

---

## Procurement Recommendation Level 2

`app/procurement.py` consumes the Level 1 result frame and per-SKU constraints
(`moq`, `order_multiple`) and appends exactly three columns:

```
procurement_recommended_qty   # final recommended order quantity
provisional                   # confidence flag (does not change the quantity)
procurement_reason            # which constraint actually changed the quantity
```

**Constraint chain** (only ever applied to the Level 1 gap):

```
gap = recommended_order_qty

if gap <= 0:                         q = 0
else:
    q = gap
    if moq > 0 and moq > q:          q = moq            # MOQ floor
    if order_multiple > 1:           q = ceil(q / order_multiple) * order_multiple

procurement_recommended_qty = q
provisional = data_quality_status != "OK"
```

`procurement_reason` is one of `NO_GAP`, `GAP_ONLY`, `MOQ_APPLIED`,
`MULTIPLE_APPLIED`, `MOQ_AND_MULTIPLE_APPLIED`, and only reflects constraints
that actually changed the final quantity.

**Level 1 vs Level 2 — these are not the same number:**

| Layer | Field | Meaning |
|---|---|---|
| Level 1 | `recommended_order_qty` | Non-negative **replenishment gap** derived from stock and demand. |
| Level 2 | `procurement_recommended_qty` | **Procurement recommendation** after applying MOQ and order multiple. |

Level 1 is never rewritten by Level 2; both are kept side by side so the
rounding decision stays auditable.

---

## Data Quality

The Demand Engine returns a data-quality status:

| Status | Meaning |
|---|---|
| `OK` | Coverage of at least 30 days is provable from `coverage_start`. |
| `INSUFFICIENT_HISTORY` | Coverage is known but shorter than 30 days. |
| `UNKNOWN_HISTORY` | No coverage start was provided; history length cannot be asserted. |

A quantity computed over an incomplete window is still produced with the fixed
30-day denominator — the risk is exposed through the status, not by shrinking
the denominator or guessing a start date.

The inventory layer additionally emits structured `data_warnings` (each with a
`sku`, `code`, and human-readable `message`):

- `UNKNOWN_HISTORY`, `INSUFFICIENT_HISTORY` — evidence gaps.
- `OVERDUE_INBOUND` — an overdue but still `OPEN` in-transit order.

**Data quality never silently changes core quantities.** It affects the warning
channel and the Level 2 `provisional` flag only.

---

## Multi-day Simulator

`app/simulator.py` replays the engine across multiple days. It **reuses the
existing demand / inventory / procurement modules directly** and does not
re-implement any formula.

Daily order of operations (fixed):

```
1. Receive arrivals: OPEN POs with expected_date <= current_day -> ARRIVED,
   and their qty is added to current_stock
2. Apply the day's demand: current_stock -= demand_qty
3. calculate_demand()            (Demand Engine)
4. analyze_inventory_frame()     (Inventory Level 1)
5. apply_procurement()           (Procurement Level 2)
6. order_placed_qty = procurement_recommended_qty
7. Create new OPEN POs: expected_date = current_day + lead_time_days
8. Emit audit records (DECISION + SKIP)
9. Advance to the next day
```

Key properties:

- **Arrivals precede ordering**, so a purchase order created on a given day can
  never arrive on that same day — even when `lead_time_days == 0`.
- **Demand is applied literally**: `stock_after = stock_before - demand_qty`;
  stock may go negative, and unmet demand is recorded as
  `max(0, demand_qty - stock_before)`. It is never clamped to zero.
- **`provisional=True` does not block ordering.** The quantity is still placed.
- **`po_id` is deterministic** (`PO-{run_id}-{seq:06d}`) and unique within a run.
- **Same inputs produce the same run.** Given the same `run_id`, `start_day`,
  `days`, state and `demand_profile`, `run_simulation()` returns identical
  results — there is no randomness and no clock dependency.

**Demand separation (important):**

- `orders_log` is the **historical order log**. It is read-only and is used
  *only* by the Demand Engine.
- `demand_profile` is a `DataFrame[sku, simulation_day, qty]` used *only* for
  the day's stock consumption.
- Simulated demand is **never written back** into `orders_log`.

**Zero-demand SKUs:** if a SKU has no demand evidence and the inventory layer
skips it for a day, the simulator does not delete the SKU. It records a `SKIP`
audit entry (keeping the original warning field/message), places no order for
it, and carries the SKU into the next day.

---

## Audit Log

`app/audit_log.py` provides two small functions:

```python
build_audit_records(frame, action=None) -> list[dict]
append_audit_records(path, records) -> None
```

- `build_audit_records` **projects** the decision frame into records and merges
  the simulator's action fields (`order_placed_qty`, `po_id`, `expected_date`).
  It does not recompute any business value.
- `append_audit_records` writes **JSONL** in append mode, UTF-8,
  `ensure_ascii=False`, one JSON object per line. It never overwrites and never
  swallows write failures.

Record envelope: `run_id`, `simulation_day`, `as_of_date`, `sku`,
`record_type` (`DECISION` or `SKIP`). DECISION records additionally carry the
projected Level 1 / Level 2 / demand-evidence fields.

The audit log **only records decisions that were already produced** — it is a
projection/persistence layer, not a calculation layer. It does not read the
clock and adds no timestamp.

---

## Project Structure

```
supplychain-os/
├── app/
│   ├── __init__.py
│   ├── data_loader.py    # CSV loading and contract validation
│   ├── demand.py         # Demand Engine (30D / 7D, trend, quality)
│   ├── inventory.py      # Inventory Decision Level 1 + in-transit aggregation
│   ├── procurement.py    # Procurement Recommendation Level 2 (MOQ / multiple)
│   ├── audit_log.py      # JSONL audit projection (no recomputation)
│   ├── simulator.py      # Multi-day simulator (reuses the layers above)
│   └── demo.py           # CLI entry point: load -> compute -> print
├── data/
│   ├── inventory.csv
│   ├── orders.csv
│   ├── in_transit.csv
│   ├── products.csv
│   ├── data_metadata.csv            # declares coverage_start
│   └── procurement_constraints.csv  # per-SKU moq / order_multiple
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

## Quick Start

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

Runtime dependency: `pandas`. There is no database, no web framework, and no
LLM dependency in the core engine.

---

## Run Demo

Run the engine against the bundled sample data:

```bash
python -X utf8 -m app.demo
```

The demo loads the CSVs, runs the full pipeline, and prints a decision report to
stdout — including the as-of date, the demand basis, per-SKU Level 1 / Level 2
decisions, demand evidence, risk warnings, and a summary. By default it uses
today's date; passing an explicit `as_of_date` (as the tests do) makes the
output fully reproducible.

The demo is a display entry point only — it contains no business formulas of
its own.

---

## Run Tests

```bash
pip install -r dev-requirements.txt
python -m pytest -q
```

Current suite: **218 passed**.

The tests cover demand windows and trend boundaries, Level 1 math and status
rules, in-transit aggregation, Level 2 constraint chains, fail-fast input
validation, the audit-log projection and JSONL behavior, and simulator
determinism / arrival ordering / stock conservation.

---

## Web UI

The engine is UI-agnostic: everything relevant is printed as deterministic
plain text by `python -X utf8 -m app.demo`.

A Web UI integration (a Pi Web UI plugin) can wrap this entry point:

- the server side shells out to the engine and returns its stdout;
- the client side only parses, sorts and searches the returned text.

**The Web UI does not perform any core business calculation.** It does not
import the engine's formulas or re-derive any quantity. The plugin is a separate
component and is not part of this repository.

---

## Design Principles

- **One source of truth per formula.** Demand, inventory and procurement rules
  live in exactly one module each; every other layer calls them.
- **Facts and conclusions are layered.** Demand evidence, Level 1 gap, and
  Level 2 recommendation are separate fields that never overwrite each other.
- **Determinism by construction.** `as_of_date` is always explicit; the core
  does not read the system clock, so results are reproducible.
- **Fail loudly.** Nulls, negative quantities, malformed dates, duplicate keys,
  and unknown statuses raise errors instead of being silently coerced.
- **Quality is visible, not corrective.** Incomplete history is surfaced through
  status and warnings; it never silently changes a quantity.
- **The simulator reuses the engine.** It advances state and orchestrates the
  existing modules rather than reimplementing them.
- **The audit log records, it does not decide.** It projects already-produced
  decisions and never recalculates them.

---

## Current Status

Implemented and tested:

- Demand Engine (30D / 7D demand, trend, coverage quality).
- Inventory Decision Level 1 (ROP, position, shortage, replenishment quantity,
  status).
- In-transit handling (OPEN-only supply, overdue as risk).
- Procurement Recommendation Level 2 (MOQ and order multiple).
- Data-quality status and structured risk warnings.
- Multi-day deterministic Simulator.
- Audit Log (append-only JSONL projection).

Test suite: **218 passed**.

---

## Roadmap

The following are **planned / future** and are **not** implemented yet:

- A `LICENSE` file (see below) — *planned*.
- Packaging metadata (e.g. `pyproject.toml`) for `pip install` — *planned*.
- A minimal CLI wrapper around the simulator — *planned*.
- Supplier-level attributes and lead-time variability — *planned*.
- Additional demand policies / configurable windows — *planned*.
- Scenario tooling for larger datasets — *planned*.

---

## License

This repository does **not** currently include a `LICENSE` file. Until one is
added, the code is provided as-is without an explicit grant of reuse rights.
Adding an OSI-approved license is a planned item (see Roadmap).

---

**Supply Chain Decision Agent** — a deterministic decision engine for
inventory replenishment and procurement recommendations. Decision support, not
an automated purchasing system.
