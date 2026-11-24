"""
Research query layer.

Nobody scans an exabyte per question. The thing that makes an exabyte
store usable for research is *how little of it each query touches*.
This module measures that directly:

  1. point query  - one symbol, one day       (manifest prune -> 1 file)
  2. slice query  - all symbols, one week      (prune by date)
  3. full scan    - cross-sectional over everything

and reports bytes touched vs total for each.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb

from .catalog import Manifest, bucket_of


def timed(con, sql, params=None):
    t0 = time.perf_counter()
    df = con.execute(sql, params or []).fetch_df()
    return df, time.perf_counter() - t0


def files_sql(root: str, entries) -> str:
    paths = [str(Path(root) / e.path) for e in entries]
    return "[" + ",".join(f"'{p}'" for p in paths) + "]"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", default="data/ticks")
    ap.add_argument("--bars", default="data/bars")
    ap.add_argument("--buckets", type=int, default=16)
    ap.add_argument("--symbol", type=int, default=42)
    ap.add_argument("--bench-out", default=None)
    a = ap.parse_args(argv)

    ticks, bars = Manifest(a.ticks), Manifest(a.bars)
    tot = ticks.totals()
    dates = sorted({e.date for e in ticks.entries()})
    con = duckdb.connect()
    report = {"total_tick_bytes": tot["bytes"], "total_tick_rows": tot["rows"], "queries": []}

    # 1. point query on raw ticks: one symbol, one day
    b = bucket_of(a.symbol, a.buckets)
    hit = ticks.prune(dates={dates[-1]}, buckets={b})
    df, s = timed(con, f"""
        SELECT count(*) n, min(price_e4)/1e4 lo, max(price_e4)/1e4 hi,
               sum(price_e4*size)/sum(size)/1e4 vwap
        FROM read_parquet({files_sql(a.ticks, hit)}) WHERE symbol = ?""", [a.symbol])
    report["queries"].append({"name": f"point: symbol {a.symbol} on {dates[-1]} (raw ticks)",
                              "files": len(hit), "bytes": sum(e.bytes for e in hit),
                              "seconds": s, "result": df.to_dict("records")})

    # 2. slice: last 5 days, whole universe, top order-flow names per day
    hit = bars.prune(dates=set(dates[-5:]))
    df, s = timed(con, f"""
        SELECT trade_date, symbol, sum(volume) vol, avg(ofi) ofi FROM (
            SELECT symbol, volume, ofi, strftime(to_timestamp(bar_ts/1e9), '%Y-%m-%d') AS trade_date
            FROM read_parquet({files_sql(a.bars, hit)}, hive_partitioning=false))
        GROUP BY trade_date, symbol
        QUALIFY row_number() OVER (PARTITION BY trade_date ORDER BY avg(ofi) DESC) <= 3
        ORDER BY trade_date, ofi DESC""")
    report["queries"].append({"name": "slice: top-3 order-flow names per day, last 5 days (bars)",
                              "files": len(hit), "bytes": sum(e.bytes for e in hit),
                              "seconds": s, "result_rows": len(df)})

    # 3. full scan of raw ticks: cross-sectional daily stats
    allf = ticks.entries()
    df, s = timed(con, f"""
        SELECT count(*) n_trades, count(DISTINCT symbol) n_symbols, sum(size) shares
        FROM read_parquet({files_sql(a.ticks, allf)})""")
    report["queries"].append({"name": "full scan: every tick", "files": len(allf),
                              "bytes": tot["bytes"], "seconds": s,
                              "result": df.to_dict("records")})

    for q in report["queries"]:
        frac = q["bytes"] / tot["bytes"]
        q["fraction_of_dataset"] = frac
        print(f"  {q['name']:<62} files={q['files']:>4}  "
              f"touched={q['bytes']/1e6:9.1f} MB ({frac:8.4%})  {q['seconds']:.2f}s")

    if a.bench_out:
        Path(a.bench_out).write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
