"""
Problem 2 - research data isn't point-in-time.

Break it, two ways:
  1. Survivorship: build the stock universe from today's list of listed
     companies. Every company that failed during the sample silently vanishes
     from history, so backtests only ever hold the survivors.
  2. Silent revisions: the data vendor publishes corrections. Rerunning last
     month's research now gives a different answer, and nobody can say why.

Fix it:
  1. Point-in-time universe: on each day, use the stocks that were listed *that
     day*, including their final, delisting-day losses.
  2. Catalog snapshots: corrections are new file versions, old ones are kept,
     and a backtest pinned to a snapshot reads exactly the data it read then.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from exaflow import backtest as bt
from exaflow.catalog import Manifest
from exaflow.panel import read_panel, simulate, write_panel, write_version

ROOT = Path("data/demo_pit")
OUT = Path("demo_results")


def ann(daily: np.ndarray) -> float:
    return float(daily.mean() * 252 * 100)


def equal_weight(df: pl.DataFrame) -> np.ndarray:
    return df.group_by("date").agg(pl.col("ret").mean()).sort("date")["ret"].to_numpy()


def buy_losers(df: pl.DataFrame) -> np.ndarray:
    """Long-only contrarian: each day hold the 10% of stocks that fell most over the past month."""
    d = (df.sort("symbol", "date")
           .with_columns(m=pl.col("ret").rolling_sum(21, min_samples=21).over("symbol"),
                         nxt=pl.col("ret").shift(-1).over("symbol"))
           .drop_nulls(["m", "nxt"])
           .with_columns(rank=pl.col("m").rank("ordinal").over("date") /
                              pl.len().over("date")))
    return d.filter(pl.col("rank") <= 0.1).group_by("date").agg(pl.col("nxt").mean()) \
            .sort("date")["nxt"].to_numpy()


def main():
    shutil.rmtree(ROOT, ignore_errors=True)
    OUT.mkdir(exist_ok=True)
    df, master = simulate()
    s1 = write_panel(ROOT, df, master)

    # ---------- 1. survivorship ----------
    survivors = master.filter(pl.col("delisted").is_null())["symbol"]
    naive = df.filter(pl.col("symbol").is_in(survivors.implode()))
    surv = {
        "symbols_ever_listed": master.height,
        "symbols_listed_today": int(survivors.len()),
        "delisted_during_sample": int(master["delisted"].is_not_null().sum()),
        "equal_weight": {"survivors_only": ann(equal_weight(naive)),
                         "point_in_time": ann(equal_weight(df))},
        "buy_losers": {"survivors_only": ann(buy_losers(naive)),
                       "point_in_time": ann(buy_losers(df))},
    }

    # ---------- 2. silent revisions vs snapshots ----------
    research = lambda data: bt.run(data, bt.reversal_targets(data), aum=1e7, costs=False)
    before = research(read_panel(ROOT))                       # last month's run

    # the vendor restates 1% of 2023's prices (late corporate actions, fixed bad closes)
    rng = np.random.default_rng(7)
    y23 = df.filter(pl.col("date").dt.year() == 2023)
    fix = rng.random(y23.height) < 0.01
    y23 = y23.with_columns(ret=pl.when(pl.Series(fix))
                              .then((1 + pl.col("ret")) * (1 + pl.Series(rng.normal(0, 0.03, y23.height))) - 1)
                              .otherwise(pl.col("ret")))
    s2 = Manifest(ROOT / "prices").commit([write_version(ROOT, y23, 2023, version=1)])

    latest = research(read_panel(ROOT))                       # rerun today, unpinned
    pinned = research(read_panel(ROOT, as_of=s1))             # rerun pinned to snapshot 1
    rev = {
        "rows_restated": int(fix.sum()),
        "snapshots": len(Manifest(ROOT / "prices").snapshots()),
        "original_run": {"ann_return_pct": before["ann_return_pct"], "sharpe": before["sharpe"]},
        "rerun_latest": {"ann_return_pct": latest["ann_return_pct"], "sharpe": latest["sharpe"]},
        "rerun_pinned": {"ann_return_pct": pinned["ann_return_pct"], "sharpe": pinned["sharpe"]},
        "pinned_identical": bool(np.array_equal(before["daily_net"], pinned["daily_net"])),
        "latest_identical": bool(np.array_equal(before["daily_net"], latest["daily_net"])),
        "files_at_snapshot_1": sorted(e.path for e in Manifest(ROOT / "prices").entries(as_of=s1)),
        "files_now": sorted(e.path for e in Manifest(ROOT / "prices").entries()),
    }
    result = {"survivorship": surv, "revisions": rev}
    (OUT / "pit.json").write_text(json.dumps(result, indent=2))

    print(f"\nPROBLEM 2: point-in-time data  ({master.height:,} stocks ever listed, "
          f"{surv['delisted_during_sample']:,} delisted, {int(survivors.len()):,} still listed)\n")
    print(f"{'annual return':28s}{'survivors only':>16s}{'point-in-time':>16s}{'overstated by':>16s}")
    for k, label in [("equal_weight", "equal-weight all stocks"), ("buy_losers", "buy last month's losers")]:
        a, b = surv[k]["survivors_only"], surv[k]["point_in_time"]
        print(f"{label:28s}{a:>15.1f}%{b:>15.1f}%{a - b:>15.1f}%")
    print(f"\nvendor restated {rev['rows_restated']:,} rows of 2023 data after the research ran")
    print(f"{'':28s}{'annual return':>16s}{'Sharpe':>10s}{'same as original?':>20s}")
    print(f"{'original run':28s}{rev['original_run']['ann_return_pct']:>15.2f}%{rev['original_run']['sharpe']:>10.2f}")
    print(f"{'rerun on latest data':28s}{rev['rerun_latest']['ann_return_pct']:>15.2f}%{rev['rerun_latest']['sharpe']:>10.2f}{str(rev['latest_identical']):>20s}")
    print(f"{'rerun pinned to snapshot':28s}{rev['rerun_pinned']['ann_return_pct']:>15.2f}%{rev['rerun_pinned']['sharpe']:>10.2f}{str(rev['pinned_identical']):>20s}")
    print(f"\nfiles at snapshot 1: {rev['files_at_snapshot_1'][2]} ...   now: {rev['files_now'][2]} ...")


if __name__ == "__main__":
    main()
