"""
Synthetic tick-data generator.

Produces realistic-shaped trade prints (timestamp ns, symbol, price, size,
side) for N symbols x D days. Liquidity follows a Zipf law, like real markets:
a few names trade 100x more than the tail, which is exactly the skew that
breaks naive partitioning.

Layout (Hive-style, same as you'd use on S3):
    <root>/date=YYYY-MM-DD/bucket=BBB/part-0.parquet

Each (date, bucket) is one deterministic task: same seed -> same bytes, so
regeneration is idempotent and any task can be retried safely.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import time
from pathlib import Path

import numpy as np

from .catalog import FileEntry, Manifest, bucket_of, sha256_file
from .executor import run_tasks

SESSION_START_H = 9      # 09:00
SESSION_SECONDS = 8 * 3600


def session_start_ns(date: str) -> int:
    d = dt.datetime.fromisoformat(date).replace(hour=SESSION_START_H, tzinfo=dt.timezone.utc)
    return int(d.timestamp()) * 1_000_000_000


def symbol_liquidity(n_symbols: int, ticks_per_symbol_day: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ranks = rng.permutation(n_symbols) + 1
    w = 1.0 / ranks ** 0.8
    w = w / w.mean()
    return np.maximum((w * ticks_per_symbol_day).astype(np.int64), 100)


def write_ticks(table, path) -> None:
    """Write a tick table sorted by (symbol, ts) in the house format."""
    import pyarrow.parquet as pq
    # Rows are sorted by (symbol, ts). Delta encoding stores the small step between
    # neighbouring timestamps/prices instead of full 8-byte values: ~40% smaller files
    # and faster decoding than plain zstd.
    pq.write_table(table, path, compression="zstd", row_group_size=1_000_000,
                   use_dictionary=["symbol", "side"],
                   column_encoding={"ts": "DELTA_BINARY_PACKED",
                                    "price_e4": "DELTA_BINARY_PACKED"},
                   sorting_columns=[pq.SortingColumn(1), pq.SortingColumn(0)])


def gen_task(task: dict, attempt: int = 0) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, date, bucket = Path(task["root"]), task["date"], task["bucket"]
    syms, counts = task["symbols"], task["counts"]
    # deterministic across processes/machines (Python's hash() is salted per process)
    seed = int.from_bytes(hashlib.blake2b(f"{date}|{bucket}|{task['seed']}".encode(),
                                          digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed)
    # Session times are pinned to UTC. Using the machine's local time zone made the
    # same command produce different bytes on different computers.
    day0 = session_start_ns(date)

    ts_parts, sym_parts, px_parts, sz_parts, side_parts = [], [], [], [], []
    for sym, n in zip(syms, counts):
        # Poisson arrivals: given n trades, their times are n sorted uniform draws
        # over the session (and never spill past the close).
        ts = day0 + np.sort(rng.integers(0, SESSION_SECONDS * 1_000_000_000, size=n))
        # geometric random walk, daily vol ~2%
        p0 = 10 + (sym * 7919 % 490)
        rets = rng.normal(0.0, 0.02 / np.sqrt(n), size=n)
        # fixed-point prices (1e-4 units), the way exchange feeds send them
        px = np.round(p0 * np.exp(np.cumsum(rets)) * 10_000).astype(np.int64)
        ts_parts.append(ts)
        sym_parts.append(np.full(n, sym, dtype=np.uint32))
        px_parts.append(px)
        sz_parts.append((rng.pareto(1.5, size=n) * 100 + 1).astype(np.int32))
        side_parts.append(rng.choice(np.array([-1, 1], dtype=np.int8), size=n))

    cols = {
        "ts": np.concatenate(ts_parts),
        "symbol": np.concatenate(sym_parts),
        "price_e4": np.concatenate(px_parts),
        "size": np.concatenate(sz_parts),
        "side": np.concatenate(side_parts),
    }
    fault_counts = {}
    if task.get("faults", 0) > 0:
        from .quality import inject_faults
        cols, fault_counts = inject_faults(cols, date, task["faults"], seed)
    table = pa.table(cols)

    rel = f"date={date}/bucket={bucket:03d}/part-0.parquet"
    out = root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    write_ticks(table, tmp)
    os.replace(tmp, out)  # atomic publish

    ts_col = table.column("ts")
    return {"entry": FileEntry(path=rel, date=date, bucket=bucket, rows=table.num_rows,
                               bytes=out.stat().st_size,
                               ts_min=int(ts_col[0].as_py()), ts_max=int(ts_col[-1].as_py()),
                               sha256=sha256_file(out)).__dict__,
            "faults": fault_counts}


def plan(root: str, n_symbols: int, n_days: int, n_buckets: int, ticks: int,
         start: str = "2026-01-05", seed: int = 1, faults: float = 0.0) -> list[dict]:
    counts = symbol_liquidity(n_symbols, ticks)
    by_bucket: dict[int, list[int]] = {}
    for s in range(n_symbols):
        by_bucket.setdefault(bucket_of(s, n_buckets), []).append(s)
    d0 = dt.date.fromisoformat(start)
    days, d = [], d0
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d += dt.timedelta(days=1)
    return [{"root": root, "date": day, "bucket": b, "symbols": syms,
             "counts": [int(counts[s]) for s in syms], "seed": seed, "faults": faults}
            for day in days for b, syms in sorted(by_bucket.items())]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate synthetic tick data")
    ap.add_argument("--root", default="data/ticks")
    ap.add_argument("--symbols", type=int, default=500)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--buckets", type=int, default=16)
    ap.add_argument("--ticks", type=int, default=50_000, help="avg trades per symbol per day")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--backend", default="local", choices=["local", "ray"])
    ap.add_argument("--faults", type=float, default=0.0,
                    help="simulate a dirty vendor feed: add this fraction of bad rows")
    a = ap.parse_args(argv)

    man = Manifest(a.root)
    done = man.committed_paths()
    tasks = [t for t in plan(a.root, a.symbols, a.days, a.buckets, a.ticks, faults=a.faults)
             if f"date={t['date']}/bucket={t['bucket']:03d}/part-0.parquet" not in done]
    print(f"generate: {len(tasks)} tasks to run ({len(done)} already committed)")
    t0 = time.perf_counter()
    injected: dict[str, int] = {}

    def _done(r):
        man.commit([FileEntry(**r["entry"])])
        for k, v in r.get("faults", {}).items():
            injected[k] = injected.get(k, 0) + v

    run_tasks(gen_task, tasks, backend=a.backend, workers=a.workers, on_result=_done)
    if injected:
        import json
        (Path(a.root) / "_faults.json").write_text(json.dumps(injected, indent=2))
        print(f"generate: injected faults (ground truth) {injected}")
    el = time.perf_counter() - t0
    tot = man.totals()
    print(f"generate: {tot['rows']:,} rows, {tot['bytes']/1e9:.2f} GB in {tot['files']} files "
          f"({el:.1f}s this run)")


if __name__ == "__main__":
    main()
