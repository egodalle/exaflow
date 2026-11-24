"""
Feature pipeline: raw ticks -> 1-minute research bars.

For every (date, bucket) partition, compute per symbol per minute:
  open, high, low, close, volume, vwap, n_trades,
  order-flow imbalance (signed volume / volume), log return,
  and a rolling 30-minute realised volatility.

Design properties that matter at scale:
  * embarrassingly parallel over partitions (no shuffle needed, because
    the partition key (date, symbol-bucket) is also the compute key)
  * planning reads the manifest, not the filesystem
  * resumable: tasks already committed to the output manifest are skipped
  * atomic outputs: temp file + rename, committed by the driver only
  * chaos mode: --fail-rate injects random worker crashes to prove the
    retry path and that output is still exactly-once
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from .catalog import FileEntry, Manifest, sha256_file
from .executor import run_tasks


def _finish(bars):
    """Shared post-processing on the small bar frame (both engines)."""
    import polars as pl
    # Integer price units -> float with true IEEE division (NumPy). Polars' x/10_000
    # multiplies by the reciprocal, which can differ in the last bit; doing the
    # conversion in one place keeps every engine bit-identical.
    bars = bars.with_columns(**{
        c: pl.Series(c, bars[f"{c}_e4"].to_numpy() / 10_000) for c in ("open", "high", "low", "close")
    })
    return (
        bars.with_columns(
            vwap=pl.col("notional_e4") / pl.col("volume") / 10_000,
            ofi=pl.col("signed") / pl.col("volume"),
            ret=pl.col("close").log().diff().over("symbol"),
        )
        .with_columns(rv30=pl.col("ret").rolling_std(30).over("symbol") * (30 ** 0.5))
        .select("symbol", "bar_ts", "open", "high", "low", "close", "volume",
                "n_trades", "vwap", "ofi", "ret", "rv30")
    )


def _bars_fast(src):
    """Run-based NumPy kernel: input is sorted by (symbol, ts), so no hashing."""
    import polars as pl
    from .kernels import bars_from_file
    b = bars_from_file(src)
    return _finish(pl.DataFrame({
        "symbol": b["symbol"], "bar_ts": b["bar_ts"],
        "open_e4": b["open_e4"], "high_e4": b["high_e4"],
        "low_e4": b["low_e4"], "close_e4": b["close_e4"],
        "volume": b["volume"], "n_trades": b["n_trades"],
        "notional_e4": b["notional_e4"], "signed": b["signed"],
    }))


def _bars_polars(src):
    """Reference engine: generic hash group-by in Polars."""
    import polars as pl
    minute = 60_000_000_000
    bars = (
        pl.scan_parquet(src)
        .with_columns(
            bar_ts=(pl.col("ts") // minute) * minute,
            notional_e4=pl.col("price_e4") * pl.col("size").cast(pl.Int64),
            signed=pl.when(pl.col("side") > 0).then(pl.col("size").cast(pl.Int64))
                     .otherwise(-pl.col("size").cast(pl.Int64)),
        )
        .group_by("symbol", "bar_ts")
        .agg(
            open_e4=pl.col("price_e4").first(),
            high_e4=pl.col("price_e4").max(),
            low_e4=pl.col("price_e4").min(),
            close_e4=pl.col("price_e4").last(),
            volume=pl.col("size").cast(pl.Int64).sum(),
            notional_e4=pl.col("notional_e4").sum(),
            signed=pl.col("signed").sum(),
            n_trades=pl.len().cast(pl.UInt32),
        )
        .sort("symbol", "bar_ts")
        .collect()
    )
    return _finish(bars)


def bars_task(task: dict, attempt: int = 0) -> dict:
    if task.get("fail_rate", 0) and attempt == 0 and random.random() < task["fail_rate"]:
        raise RuntimeError("injected worker failure (chaos mode)")

    t0 = time.perf_counter()
    src = Path(task["src_root"]) / task["path"]
    engine = task.get("engine", "fast")
    bars = _bars_fast(src) if engine == "fast" else _bars_polars(src)

    rel = task["path"].replace("part-0", "bars-0")
    out = Path(task["dst_root"]) / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    bars.write_parquet(tmp, compression="zstd", compression_level=1)
    os.replace(tmp, out)

    return {
        "entry": FileEntry(path=rel, date=task["date"], bucket=task["bucket"],
                           rows=bars.height, bytes=out.stat().st_size,
                           ts_min=int(bars["bar_ts"].min()), ts_max=int(bars["bar_ts"].max()),
                           sha256=sha256_file(out)).__dict__,
        "in_rows": task["rows"], "in_bytes": task["bytes"],
        "seconds": time.perf_counter() - t0, "attempt": attempt,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="ticks -> 1-minute bars")
    ap.add_argument("--src", default="data/ticks")
    ap.add_argument("--dst", default="data/bars")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--backend", default="local", choices=["local", "ray"])
    ap.add_argument("--fail-rate", type=float, default=0.0)
    ap.add_argument("--fresh", action="store_true", help="ignore output manifest, recompute all")
    ap.add_argument("--bench-out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only first N partitions")
    ap.add_argument("--engine", default="fast", choices=["fast", "polars"],
                    help="fast = sorted-run NumPy kernel, polars = generic hash group-by")
    a = ap.parse_args(argv)

    src, dst = Manifest(a.src), Manifest(a.dst)
    if a.fresh and dst.path.exists():
        dst.path.unlink()
    done = dst.committed_paths()
    entries = sorted(src.entries(), key=lambda e: e.path)[: a.limit]
    tasks = [{"src_root": a.src, "dst_root": a.dst, "path": e.path, "date": e.date,
              "bucket": e.bucket, "rows": e.rows, "bytes": e.bytes, "fail_rate": a.fail_rate,
              "engine": a.engine}
             for e in entries if e.path.replace("part-0", "bars-0") not in done]
    # Longest-processing-time-first: start the biggest partitions first so a big
    # one isn't left running alone at the end while other cores sit idle.
    tasks.sort(key=lambda t: -t["bytes"])
    print(f"pipeline: {len(tasks)} partitions to process, {len(done)} already done, "
          f"backend={a.backend} workers={a.workers}")

    t0 = time.perf_counter()
    res = run_tasks(bars_task, tasks, backend=a.backend, workers=a.workers,
                    on_result=lambda r: dst.commit([FileEntry(**r["entry"])]))
    wall = time.perf_counter() - t0

    rows = sum(r["in_rows"] for r in res)
    byts = sum(r["in_bytes"] for r in res)
    retries = sum(1 for r in res if r["attempt"] > 0)
    stats = {
        "backend": a.backend, "engine": a.engine, "workers": a.workers, "partitions": len(res),
        "input_rows": rows, "input_bytes": byts, "wall_seconds": round(wall, 3),
        "rows_per_sec": rows / wall if wall else 0,
        "bytes_per_sec": byts / wall if wall else 0,
        "rows_per_sec_per_core": rows / wall / a.workers if wall else 0,
        "bytes_per_sec_per_core": byts / wall / a.workers if wall else 0,
        "retried_tasks": retries,
    }
    print(f"pipeline: {rows:,} ticks ({byts/1e9:.2f} GB compressed) in {wall:.1f}s -> "
          f"{stats['rows_per_sec']/1e6:.1f}M rows/s, "
          f"{stats['rows_per_sec_per_core']/1e6:.1f}M rows/s/core, retried={retries}, engine={a.engine}")
    if a.bench_out:
        Path(a.bench_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.bench_out).write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
