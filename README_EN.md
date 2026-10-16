[中文](./README.md) | [English](./README_EN.md)

# Supply Chain Decision Agent

A deterministic decision-support tool for inventory replenishment and procurement. It reads order, stock, in-transit PO and procurement-constraint data, then produces per-SKU recommendations, optional budget allocations and auditable records.

> “Agent” means a tool-callable decision component here, not autonomous planning or multi-agent orchestration. The optional LLM layer only explains already-computed results; it does not make inventory or quantity decisions.

## Features

| Module | Purpose |
|---|---|
| Demand engine | 30-day / 7-day average demand, trend and history-coverage status |
| Inventory decision (L1) | Reorder point, inventory position, replenishment gap and status |
| Procurement recommendation (L2) | Apply MOQ and order multiples to the replenishment gap |
| Budget allocation (optional L3) | Allocate L2 recommendations under a shared budget using R0 / R1 / R2 |
| Multi-day simulator | Process arrivals, demand, replenishment decisions and new POs day by day |
| Audit log | Append decisions as JSONL without recalculating business values |
| LLM explanation (optional) | Turn completed results into natural-language explanations; off by default |

## Decision flow

```text
CSV inputs
  → validation
  → demand analysis
  → inventory decision L1
  → procurement constraints L2
  → (optional) budget allocation L3
  → CLI / simulation results / audit records
```

Each layer preserves its own output: `recommended_order_qty` is the L1 replenishment gap, `procurement_recommended_qty` is the L2 procurement recommendation, and `allocated_qty` is the final allocation when L3 is enabled.

## Budget allocation (L3)

| Rule | Behavior |
|---|---|
| R0 | Pass through the L2 recommendation without a budget constraint; baseline only |
| R1 | Scale recommendations proportionally, then round down to feasible packs |
| R2 | Allocate complete packs in deterministic priority order, checking the remaining budget per SKU |

R1 / R2 calculate amounts on an integer money scale. They require per-SKU costs and priorities. The simulator resets the budget by cycle without carrying unused funds into the next cycle. L3 is opt-in; when disabled, existing L1 / L2 behavior is retained.

## Core rules and boundaries

- **Demand windows:** 30-day and 7-day windows use calendar days and fixed denominators; dates without orders count as zero demand. Incomplete history is reported rather than silently filled in.
- **Inventory position:** Only `OPEN` in-transit POs count as supply. Overdue `OPEN` POs still count and are also surfaced as a risk.
- **Procurement constraints:** MOQ and order multiples apply to the L1 gap. The L1 result is preserved so the effect of rounding remains auditable.
- **Simulation order:** Process arrivals, apply the day's demand, compute replenishment, then create new POs. Stock may become negative, representing backlog semantics in the model.
- **Determinism:** Core calculations use explicit dates and inputs; they do not depend on the system clock or random values.
- **Input quality:** Invalid inputs raise errors instead of being silently coerced. Quality statuses and warnings expose evidence gaps without changing quantities.
- **Audit:** The audit layer records decisions already produced; it does not recompute them. L3 adds allocation fields when enabled.

## Quick start

Requires Python and `pandas`.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -X utf8 -m app.demo
```

Run the tests:

```bash
pip install -r dev-requirements.txt
python -m pytest -q
node --test tools/pi_llm_proxy.test.mjs  # Optional local proxy tests
```

## Sample data

CSV files in `data/` are demo inputs. `data/costs.csv` contains example / simulated unit costs, not real procurement costs from any company.

| File | Contents |
|---|---|
| `inventory.csv` | Current SKU inventory and parameters |
| `orders.csv` | Historical orders for demand calculation |
| `in_transit.csv` | In-transit POs and statuses |
| `products.csv` | SKU product attributes |
| `data_metadata.csv` | Data coverage start |
| `procurement_constraints.csv` | Per-SKU MOQ and order multiple |
| `costs.csv` | Example / simulated unit costs for L3 |

The core runtime dependency is `pandas`. The project has no database or web-framework dependency. Scripts in `tools/` are optional development and experiment tools, not runtime dependencies of the core decision engine.

## Optional LLM explanation

The LLM receives an allowlisted subset of already-computed fields and generates prose. It does not calculate or modify demand, stock status, order quantities or budget allocation. It is disabled by default; the core engine does not need API credentials.

When enabled, configure:

| Variable | Purpose |
|---|---|
| `SCOS_LLM_EXPLANATION=1` | Explicitly enable the explanation layer |
| `OPENAI_API_KEY` | API key from the environment; use the proxy token instead when using the local proxy |
| `SCOS_LLM_MODEL` | Model name |
| `SCOS_LLM_BASE_URL` | Optional OpenAI-compatible endpoint; defaults to `https://api.openai.com/v1` |

See `app/llm_explanation.py` for implementation. The local Pi RPC proxy, `tools/pi_llm_proxy.mjs`, is a developer utility that binds to `127.0.0.1` and requires a configured local Pi CLI. Do not expose it to the public internet.

## Repository layout

```text
app/       Core calculations, input validation, simulation, audit and optional explanation
data/     Sample CSV inputs
tests/    Python regression and integration tests
tools/    Optional local tools, proxy and L3 experiment scripts
```

Key modules: `demand.py`, `inventory.py`, `procurement.py`, `allocation.py`, `simulator.py` and `audit_log.py`.

## Current limitations

- This is a deterministic decision-support tool, not an automated purchasing system.
- R2 is a greedy whole-pack allocation rule; it is not a global optimizer.
- Autonomous planning, multi-agent orchestration and LLM-driven inventory decisions are out of scope.
- Simulation metrics depend on the demand assumptions and negative-stock / backlog semantics; they should not be interpreted as real-company performance.
- No reusable open-source license has been declared; the repository currently has no `LICENSE` file. Do not assume a specific redistribution or reuse grant before a license is added.

## Possible next steps

- Add an appropriate open-source license and Python packaging metadata.
- Evaluate larger datasets, alternative demand policies and variable supplier lead times.

---

[Chinese version](./README.md)
