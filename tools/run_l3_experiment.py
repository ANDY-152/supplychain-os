"""Level 3 实验 runner（最小实现，情景感知）。

用法：
  python -X utf8 tools/run_l3_experiment.py             # 正式矩阵（每情景 21 算例）
  python -X utf8 tools/run_l3_experiment.py --precheck  # 仅 S1-C 前置核验（少量算例，不写文件）

情景：BASELINE（原始库存，零补货基准）、S1-C（内存覆盖 SKU001/SKU003/SKU005 初始库存）。
正式矩阵：R0×1（标定）、R1×5 档、R2×5 档×3 优先级 = 21 算例/情景。
输出到 experiments/l3/<scenario>/（逐日 CSV、summary.json、manifest.json），不自动加入 Git。
不修改 L2/L3 核心算法；不修改数据文件。
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.data_loader import (  # noqa: E402
    load_costs,
    load_data_metadata,
    load_in_transit,
    load_inventory,
    load_orders,
    load_procurement_constraints,
    load_products,
)
from app.simulator import _to_cents, run_simulation  # noqa: E402
from tools.l3_experiment import (  # noqa: E402
    FIXED_B1_DAILY,
    PRIORITY_RULES,
    SCENARIO_ORDER,
    TIERS,
    aggregate_metrics,
    apply_scenario,
    build_demand_profile,
    calibrate_bfull_cents,
    convergence_report,
    file_sha256,
    git_snapshot,
    priority_map,
    tier_label,
    tier_limit_cents,
    validate_inputs,
)

START_DAY = date(2026, 9, 22)
DAYS = 30
CYCLE_DAYS = 30
OUT_DIR = REPO / "experiments" / "l3" / "v2_b1_fixed"
DATA_FILES = (
    "inventory.csv", "orders.csv", "in_transit.csv",
    "procurement_constraints.csv", "costs.csv", "data_metadata.csv", "products.csv",
)


def _slug(scenario):
    return scenario.replace("-", "").lower()


def _daily_rows(run_id, algorithm, tier, priority_rule, result):
    """每日 × 全部 SKU（来自 day.demand，含未进入决策帧/空帧日）。

    - 实际需求/库存来自 day.demand（全库存 SKU）；决策字段来自 day.decision_frame，
      无决策记录时写空值并置 has_decision=False（不得伪装为 0）。
    - budget_spent_today 为货币单位（兼容旧使用者）；budget_spent_today_cents 为整数分。
    """
    for day in result.daily:
        frame = {str(r.sku): r for r in day.decision_frame.itertuples()}
        demand_by_sku = {str(d["sku"]): d for d in day.demand}
        arrivals_by_sku = {}
        for a in day.arrivals:
            arrivals_by_sku[str(a["sku"])] = arrivals_by_sku.get(str(a["sku"]), 0.0) + float(a["qty"])
        orders_by_sku = {}
        for o in day.order_placed:
            orders_by_sku[str(o["sku"])] = orders_by_sku.get(str(o["sku"]), 0.0) + float(o["qty"])
        spent_cents = _to_cents("budget_spent_today", day.budget_spent_today)
        skus = list(demand_by_sku.keys())
        skus += [s for s in frame if s not in demand_by_sku]
        for sku in skus:
            d = demand_by_sku.get(sku, {})
            row = frame.get(sku)
            has = row is not None
            yield {
                "run_id": run_id,
                "algorithm": algorithm,
                "tier_label": tier_label(*tier) if tier else "",
                "priority_rule": priority_rule,
                "period_index": day.period_index,
                "simulation_day": day.simulation_day,
                "as_of_date": day.as_of_date.isoformat(),
                "sku": sku,
                "has_decision": has,
                "daily_demand_30d": getattr(row, "daily_demand_30d", None) if has else None,
                "current_stock": getattr(row, "current_stock", None) if has else d.get("stock_after"),
                "in_transit_stock": getattr(row, "in_transit_stock", None) if has else None,
                "procurement_recommended_qty": getattr(row, "procurement_recommended_qty", None) if has else None,
                "allocated_qty": getattr(row, "allocated_qty", None) if has else None,
                "allocation_reason": getattr(row, "allocation_reason", None) if has else None,
                "priority": getattr(row, "priority", None) if has else None,
                "allocation_rank": getattr(row, "allocation_rank", None) if has else None,
                "budget_available_before": day.budget_available_before,
                "budget_spent_today": day.budget_spent_today,
                "budget_spent_today_cents": spent_cents,
                "budget_remaining_after": day.budget_remaining_after,
                "demand_qty": d.get("demand_qty"),
                "unmet_demand": d.get("unmet_demand"),
                "stock_before": d.get("stock_before"),
                "stock_after": d.get("stock_after"),
                "arrival_qty": arrivals_by_sku.get(sku, 0.0),
                "order_placed_qty": orders_by_sku.get(sku, 0.0),
            }


def _order_qty_map(result):
    out = {}
    for day in result.daily:
        for o in day.order_placed:
            key = (day.simulation_day, str(o["sku"]))
            out[key] = out.get(key, 0.0) + float(o["qty"])
    return out


def _load_bundle():
    inventory = load_inventory()
    orders = load_orders()
    transit = load_in_transit()
    constraints = load_procurement_constraints()
    costs_df = load_costs()
    products = load_products()
    coverage_start = load_data_metadata()
    costs = {str(s): float(v) for s, v in zip(costs_df["sku"], costs_df["unit_cost"])}
    skus = [str(s) for s in inventory["sku"]]
    return dict(
        inventory=inventory, orders=orders, transit=transit, constraints=constraints,
        costs=costs, products=products, coverage_start=coverage_start, skus=skus,
    )


def _common_kwargs(bundle, demand_profile):
    return dict(
        start_day=START_DAY,
        days=DAYS,
        demand_profile=demand_profile,
        orders_log=bundle["orders"],
        constraints=bundle["constraints"],
        coverage_start=bundle["coverage_start"],
        cycle_days=CYCLE_DAYS,
        period_start=START_DAY,
        initial_budget_spent=0.0,
        # 主实验：补货决策统一使用固定 B1；诊断列（7d/trend）保持滚动
        daily_demand_override=FIXED_B1_DAILY,
    )


def run_scenario(scenario, bundle, write_outputs=True):
    """运行单情景的 21 算例矩阵；返回 (runs, convergence, bfull_cents, changes)。"""
    inv, changes = apply_scenario(bundle["inventory"], scenario)
    skus = bundle["skus"]
    costs = bundle["costs"]
    constraints = bundle["constraints"]
    demand_profile, b1 = build_demand_profile(START_DAY, DAYS, bundle["orders"], skus)
    priorities = {rule: priority_map(rule, skus, bundle["products"], bundle["inventory"]) for rule in PRIORITY_RULES}
    common = _common_kwargs(bundle, demand_profile)

    out_dir = OUT_DIR / _slug(scenario)
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
    runs = []

    def record(run_id, algorithm, tier, rule, result, limit_cents):
        metrics = aggregate_metrics(
            result, algorithm=algorithm, priority_rule=rule, tier=tier,
            budget_limit_cents=limit_cents, constraints=constraints,
        )
        metrics["run_id"] = run_id
        metrics["scenario"] = scenario
        if write_outputs:
            pd.DataFrame(_daily_rows(run_id, algorithm, tier, rule, result)).to_csv(
                out_dir / f"daily_{run_id}.csv", index=False, encoding="utf-8"
            )
        runs.append(metrics)

    # None sanity（不计入正式矩阵）；algorithm=None 不得带 L3 参数
    none_common = {
        k: v for k, v in common.items()
        if k not in ("cycle_days", "period_start", "initial_budget_spent")
    }
    none_result = run_simulation(
        run_id=f"{scenario}-NONE-sanity", algorithm=None, inventory_state=inv, **none_common
    )

    # R0 标定
    r0_result = run_simulation(
        run_id=f"{scenario}-R0", algorithm="R0", costs=costs,
        priority=priorities["P_cat"], budget_limit=None, inventory_state=inv, **common,
    )
    bfull_cents = calibrate_bfull_cents(r0_result, costs)
    record(f"{scenario}-R0", "R0", None, "P_cat", r0_result, None)

    convergence = []
    for tier in TIERS:
        limit_cents = tier_limit_cents(bfull_cents, *tier)
        limit = limit_cents / 100.0
        label = tier_label(*tier)

        r1 = run_simulation(
            run_id=f"{scenario}-R1-{label}", algorithm="R1", costs=costs,
            priority=priorities["P_cat"], budget_limit=limit, inventory_state=inv, **common,
        )
        record(f"{scenario}-R1-{label}", "R1", tier, "P_cat", r1, limit_cents)
        if tier == (1, 1):
            convergence.append(convergence_report(r1, r0_result, "R1"))

        for rule in PRIORITY_RULES:
            r2 = run_simulation(
                run_id=f"{scenario}-R2-{label}-{rule}", algorithm="R2", costs=costs,
                priority=priorities[rule], budget_limit=limit, inventory_state=inv, **common,
            )
            record(f"{scenario}-R2-{label}-{rule}", "R2", tier, rule, r2, limit_cents)
            if tier == (1, 1):
                convergence.append(convergence_report(r2, r0_result, "R2"))

    none_matches = _order_qty_map(none_result) == _order_qty_map(r0_result)

    summary = {
        "scenario": scenario,
        "scenario_changes": changes,
        "start_day": START_DAY.isoformat(),
        "coverage_start": bundle["coverage_start"].isoformat(),
        "days": DAYS,
        "cycle_days": CYCLE_DAYS,
        "period_start": START_DAY.isoformat(),
        "initial_budget_spent": 0.0,
        "b1_daily_qty": b1,
        "daily_demand_override": dict(FIXED_B1_DAILY),
        "bfull_cents": bfull_cents,
        "tiers": [{"num": n, "den": d, "limit_cents": tier_limit_cents(bfull_cents, n, d)} for n, d in TIERS],
        "priorities": priorities,
        "none_sanity_matches_r0": none_matches,
        "convergence_tier_1_0": convergence,
        "runs": runs,
    }
    manifest = {
        "tool": "tools/run_l3_experiment.py",
        "scenario": scenario,
        "scenario_changes": changes,
        "data_sha256": {name: file_sha256(REPO / "data" / name) for name in DATA_FILES},
        "b1": {
            "window_days": 30,
            "formula": "(window_total + 15) // 30",
            "start_day": START_DAY.isoformat(),
            "coverage_start": bundle["coverage_start"].isoformat(),
            "days": DAYS,
            "skus": skus,
            "daily_qty": b1,
            "consumption_source": "demand_profile (B1 fixed)",
            "replenishment_decision_source": "daily_demand_override (fixed B1)",
            "note": "daily_demand_30d overridden for decision; daily_demand_7d/demand_trend remain rolling diagnostics",
        },
        "budget": {
            "bfull_cents": bfull_cents,
            "tier_rule": "(num * B_full_cents) // den",
            "tiers": [{"num": n, "den": d} for n, d in TIERS],
            "cycle_days": CYCLE_DAYS,
            "period_start": START_DAY.isoformat(),
            "initial_budget_spent": 0.0,
        },
        "priority_rules": list(PRIORITY_RULES),
        "git": git_snapshot(REPO),
    }
    if write_outputs:
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return runs, convergence, bfull_cents, changes, none_matches


def _check_failures(runs, convergence, none_matches):
    failures = []
    if not none_matches:
        failures.append("None sanity 与 R0 下单量不一致")
    for run in runs:
        if run["budget_violation_count"]:
            failures.append(f"{run['run_id']}: 预算违反 {run['budget_violation_count']} 次")
        if run["pack_violation_count"]:
            failures.append(f"{run['run_id']}: 包装违规 {run['pack_violation_count']} 次")
    for conf in convergence:
        if not conf["passed"]:
            failures.append(f"{conf['algorithm']} tier=1.0 未收敛")
    return failures


def main():
    bundle = _load_bundle()
    problems = validate_inputs(
        bundle["skus"], bundle["costs"], bundle["constraints"], bundle["products"],
        bundle["orders"], bundle["transit"],
    )
    if problems:
        print("输入校验失败：")
        for p in problems:
            print(" -", p)
        return 2

    ok = True
    for scenario in SCENARIO_ORDER:
        runs, convergence, bfull, changes, none_matches = run_scenario(scenario, bundle)
        print(f"=== {scenario} changes={changes} B_full_cents={bfull} ===")
        for run in runs:
            print(
                f"  {run['run_id']:<28} service={run['service_rate']} "
                f"stockout={run['stockout_rate']} spent={run['total_spent_cents']} "
                f"util={run['budget_utilization']} pack_viol={run['pack_violation_count']} "
                f"budget_viol={run['budget_violation_count']} orders={run['order_count']}"
            )
        for conf in convergence:
            print(f"  convergence[{conf['algorithm']}] passed={conf['passed']}")
        failures = _check_failures(runs, convergence, none_matches)
        if failures:
            ok = False
            print("FAILURES:")
            for f in failures:
                print("  -", f)
    return 0 if ok else 1


def precheck():
    """S1-C 前置核验：少量算例，只打印、不写文件。"""
    bundle = _load_bundle()
    scenario = "S1-C"
    inv, changes = apply_scenario(bundle["inventory"], scenario)
    skus = bundle["skus"]
    costs = bundle["costs"]
    demand_profile, b1 = build_demand_profile(START_DAY, DAYS, bundle["orders"], skus)
    common = _common_kwargs(bundle, demand_profile)
    print("scenario_changes:", json.dumps(changes, ensure_ascii=False))
    print("priority maps:", {r: priority_map(r, skus, bundle["products"], bundle["inventory"]) for r in PRIORITY_RULES})

    r0 = run_simulation(
        run_id=f"{scenario}-precheck-R0", algorithm="R0", costs=costs,
        priority=priority_map("P_cat", skus, bundle["products"], bundle["inventory"]),
        budget_limit=None, inventory_state=inv, **common,
    )
    bfull = calibrate_bfull_cents(r0, costs)
    print("B_full_cents:", bfull, "tiers:",
          {tier_label(n, d): tier_limit_cents(bfull, n, d) for n, d in TIERS})

    def amap(res):
        m = {}
        for d in res.daily:
            for row in d.decision_frame.itertuples():
                m[(d.simulation_day, str(row.sku))] = float(row.allocated_qty)
        return m

    for tier in ((1, 2), (1, 1)):
        lim = tier_limit_cents(bfull, *tier) / 100.0
        r1 = run_simulation(
            run_id=f"{scenario}-precheck-R1-{tier_label(*tier)}", algorithm="R1",
            costs=costs, priority=priority_map("P_cat", skus, bundle["products"], bundle["inventory"]),
            budget_limit=lim, inventory_state=inv, **common,
        )
        r2 = run_simulation(
            run_id=f"{scenario}-precheck-R2-{tier_label(*tier)}", algorithm="R2",
            costs=costs, priority=priority_map("P_cat", skus, bundle["products"], bundle["inventory"]),
            budget_limit=lim, inventory_state=inv, **common,
        )
        a1, a2 = amap(r1), amap(r2)
        diffs = [k for k in set(a1) | set(a2) if a1.get(k, 0) != a2.get(k, 0)]
        c1 = convergence_report(r1, r0, "R1")
        c2 = convergence_report(r2, r0, "R2")
        print(
            f"tier {tier_label(*tier)}: sum_alloc R1={sum(a1.values()):.0f} R2={sum(a2.values()):.0f} "
            f"rule_diff_cells={len(diffs)} conv(R1,R2)=({c1['passed']},{c2['passed']})"
        )
    return 0


if __name__ == "__main__":
    if "--precheck" in sys.argv:
        raise SystemExit(precheck())
    raise SystemExit(main())
