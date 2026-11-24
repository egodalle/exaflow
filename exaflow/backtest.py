"""
Daily cross-sectional backtester with a trading-cost model.

Weights decided with information up to the close of day t earn the return of
day t+1. Costs are charged on every change in weight:

    cost_i = |dw_i| * ( spread_i / 2  +  Y * sigma_i * sqrt(|dw_i| * AUM / ADV_i) )
             \\________/   \\______________________________________________/
              cross the       square-root market impact: the more of a day's
              half-spread     volume you trade, the more the price moves against you

The square-root impact law is the standard empirical model (Y ~ 0.5-1). Because
impact grows with AUM, the same strategy has a *capacity*: the size at which
costs eat the edge.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl

IMPACT_Y = 0.7


def reversal_targets(df: pl.DataFrame, universe: pl.Expr | None = None,
                     window: int = 5) -> pl.DataFrame:
    """Dollar-neutral short-term reversal: buy the last `window` days' relative losers,
    short the winners."""
    d = (df.sort("symbol", "date")
           .with_columns(r=pl.col("ret").rolling_sum(window, min_samples=1).over("symbol")))
    d = d if universe is None else d.filter(universe)
    return (d.with_columns(s=-(pl.col("r") - pl.col("r").mean().over("date")))
             .with_columns(target=pl.col("s") / pl.col("s").abs().sum().over("date"))
             .select("date", "symbol", "target"))


def run(df: pl.DataFrame, targets: pl.DataFrame, aum: float = 10e6,
        costs: bool = True, smoothing: float = 1.0) -> dict:
    """Backtest `targets` (date, symbol, target weight).

    smoothing < 1 moves only part of the way to the target each day
    (w_t = w_{t-1} + smoothing * (target_t - w_{t-1})): less trading, slower signal.
    """
    full = (df.select("date", "symbol", "ret", "adv_usd", "spread_bps", "sigma")
              .join(targets, on=["date", "symbol"], how="left")
              .with_columns(pl.col("target").fill_null(0.0))
              .sort("symbol", "date"))
    if smoothing >= 1.0:
        full = full.with_columns(w=pl.col("target"))
    else:
        full = full.with_columns(
            w=pl.col("target").ewm_mean(alpha=smoothing, adjust=False).over("symbol"))
    full = full.with_columns(
        next_ret=pl.col("ret").shift(-1).over("symbol").fill_null(0.0),
        dw=(pl.col("w") - pl.col("w").shift(1).over("symbol").fill_null(0.0)).abs(),
    ).with_columns(
        pnl=pl.col("w") * pl.col("next_ret"),
        cost=pl.col("dw") * (pl.col("spread_bps") / 2e4 + IMPACT_Y * pl.col("sigma") *
                             (pl.col("dw") * aum / pl.col("adv_usd")).sqrt()),
    )
    daily = (full.group_by("date").agg(gross=pl.col("pnl").sum(), cost=pl.col("cost").sum(),
                                       turnover=pl.col("dw").sum())
                 .sort("date"))
    gross, cost = daily["gross"].to_numpy(), daily["cost"].to_numpy()
    net = gross - cost if costs else gross
    return {
        "aum": aum,
        "ann_return_pct": float(net.mean() * 252 * 100),
        "ann_vol_pct": float(net.std() * math.sqrt(252) * 100),
        "sharpe": float(net.mean() / net.std() * math.sqrt(252)) if net.std() > 0 else 0.0,
        "gross_sharpe": float(gross.mean() / gross.std() * math.sqrt(252)),
        "ann_cost_pct": float(cost.mean() * 252 * 100),
        "daily_turnover_pct": float(daily["turnover"].mean() * 100),
        "daily_net": net,
    }


def capacity(df, targets, aums, **kw) -> list[dict]:
    out = []
    for a in aums:
        r = run(df, targets, aum=a, **kw)
        r.pop("daily_net")
        out.append(r)
    return out


def breakeven_aum(curve: list[dict]) -> float | None:
    """Largest AUM in the curve where net annual return is still positive (log-interpolated)."""
    for lo, hi in zip(curve, curve[1:]):
        if lo["ann_return_pct"] > 0 >= hi["ann_return_pct"]:
            f = lo["ann_return_pct"] / (lo["ann_return_pct"] - hi["ann_return_pct"])
            return float(np.exp(np.log(lo["aum"]) + f * (np.log(hi["aum"]) - np.log(lo["aum"]))))
    return None
