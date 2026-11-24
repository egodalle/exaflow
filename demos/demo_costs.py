"""
Problem 6 - research ignores trading costs.

Break it: a 5-day reversal strategy, traded across every stock and fully
rebalanced daily, backtested *without* costs. It looks spectacular.

Fix it: the same strategy through the cost model (half-spread + square-root
market impact), then a cost-aware version that (a) trades only stocks with
over $10M daily volume and (b) moves halfway to its target each day instead of
all the way. Both are run across fund sizes to find each one's capacity.
"""
import json
from pathlib import Path

import polars as pl

from exaflow import backtest as bt
from exaflow.panel import simulate

OUT = Path("demo_results")
AUMS = [1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9]
LIQUID_ADV = 10e6
SMOOTHING = 0.5


def money(x: float) -> str:
    return f"${x / 1e9:.1f}B" if x >= 1e9 else f"${x / 1e6:.0f}M"


def main():
    OUT.mkdir(exist_ok=True)
    df, _ = simulate()                                    # same synthetic market as problem 2
    naive_t = bt.reversal_targets(df)
    aware_t = bt.reversal_targets(df, pl.col("adv_usd") > LIQUID_ADV)

    paper = bt.run(df, naive_t, aum=1e7, costs=False)
    naive_curve = bt.capacity(df, naive_t, AUMS)
    aware_curve = bt.capacity(df, aware_t, AUMS, smoothing=SMOOTHING)
    pick = lambda curve, a: next(r for r in curve if r["aum"] == a)
    n10, a10 = pick(naive_curve, 1e7), pick(aware_curve, 1e7)

    result = {
        "paper": {k: paper[k] for k in ("ann_return_pct", "sharpe", "daily_turnover_pct")},
        "naive_at_10m": n10, "aware_at_10m": a10,
        "naive_capacity_usd": bt.breakeven_aum(naive_curve),
        "aware_capacity_usd": bt.breakeven_aum(aware_curve),
        "naive_curve": naive_curve, "aware_curve": aware_curve,
        "settings": {"liquid_adv_usd": LIQUID_ADV, "smoothing": SMOOTHING,
                     "impact_Y": bt.IMPACT_Y},
    }
    (OUT / "costs.json").write_text(json.dumps(result, indent=2))

    print("\nPROBLEM 6: trading costs  (5-day reversal, $10M fund unless stated)\n")
    print(f"{'':34s}{'annual return':>15s}{'Sharpe':>9s}{'costs/yr':>11s}{'turnover/day':>14s}")
    print(f"{'on paper (no costs)':34s}{paper['ann_return_pct']:>14.1f}%{paper['sharpe']:>9.2f}{'0.0%':>11s}{paper['daily_turnover_pct']:>13.0f}%")
    print(f"{'same strategy, real costs':34s}{n10['ann_return_pct']:>14.1f}%{n10['sharpe']:>9.2f}{n10['ann_cost_pct']:>10.1f}%{n10['daily_turnover_pct']:>13.0f}%")
    print(f"{'cost-aware version':34s}{a10['ann_return_pct']:>14.1f}%{a10['sharpe']:>9.2f}{a10['ann_cost_pct']:>10.1f}%{a10['daily_turnover_pct']:>13.0f}%")
    print("\nnet annual return by fund size:")
    print(f"{'fund size':>12s}{'naive':>10s}{'cost-aware':>12s}")
    for n, a in zip(naive_curve, aware_curve):
        print(f"{money(n['aum']):>12s}{n['ann_return_pct']:>9.1f}%{a['ann_return_pct']:>11.1f}%")
    cap = result["aware_capacity_usd"]
    print(f"\ncapacity (net return reaches zero): naive "
          f"{'none, loses money at every size' if result['naive_capacity_usd'] is None else money(result['naive_capacity_usd'])}"
          f", cost-aware about {money(cap) if cap else 'n/a'}")


if __name__ == "__main__":
    main()
