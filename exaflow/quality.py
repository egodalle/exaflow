"""
Data quality gate.

Problem: real feeds contain bad prints (fat-finger prices), duplicated messages,
zero or negative sizes, trades stamped outside the session, and rows that
arrive out of order. Any one of them silently corrupts bars and signals.

Fix: every raw partition passes through check_and_clean() before it gets a
catalog entry. Bad rows are removed and written to a quarantine file with the
reason; out-of-order rows are re-sorted; the catalog entry records the
quality stats. A partition with too many bad rows is rejected outright,
so nothing downstream ever reads it.

inject_faults() is the other half: it turns a clean partition into a
realistic dirty one *with ground truth*, so the gate's detection can be
measured, not assumed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .catalog import FileEntry, Manifest, sha256_file
from .executor import run_tasks

SESSION_NS = 8 * 3600 * 1_000_000_000
PRICE_JUMP = 0.05          # |log(price / local median)| above this = bad print
MEDIAN_WINDOW = 21         # trades in the centred median window
FAULT_TYPES = ("duplicate", "bad_size", "off_session", "bad_price")


# --------------------------------------------------------------------------
# Break it: realistic faults with ground truth
# --------------------------------------------------------------------------
def inject_faults(cols: dict, date: str, rate: float, seed: int):
    """Add `rate` x n bad rows (and some out-of-order swaps) to sorted tick columns."""
    from .generate import session_start_ns

    rng = np.random.default_rng(seed ^ 0x5EED)
    ts, sym, px = cols["ts"], cols["symbol"], cols["price_e4"]
    size, side = cols["size"], cols["side"]
    n = len(ts)
    k = max(int(n * rate), len(FAULT_TYPES))
    day0 = session_start_ns(date)

    src = rng.integers(0, n - 1, size=k)                      # row each fault copies
    kind = np.arange(k) % len(FAULT_TYPES)                    # even mix of fault types
    e_ts, e_sym, e_px = ts[src].copy(), sym[src].copy(), px[src].copy()
    e_size, e_side = size[src].copy(), side[src].copy()

    m = kind == 1                                             # bad_size: 0 or negative
    e_size[m] = np.where(rng.random(m.sum()) < 0.5, 0, -e_size[m])
    m = kind == 2                                             # off_session: up to 1h outside
    off = rng.integers(1, 3600, m.sum()) * 1_000_000_000
    e_ts[m] = np.where(rng.random(m.sum()) < 0.5, day0 - off, day0 + SESSION_NS + off)
    m = kind == 3                                             # bad_price: fat finger
    mult = rng.choice(np.array([10.0, 0.1, 1.5, 0.6]), m.sum())
    e_px[m] = np.round(e_px[m] * mult).astype(np.int64)
    same_sym = sym[src[m] + 1] == sym[src[m]]                 # sit between two real trades
    e_ts[m] = np.where(same_sym, (ts[src[m]] + ts[src[m] + 1]) // 2, ts[src[m]])

    # interleave: each extra row lands right after the row it copied
    key = np.concatenate([np.arange(n, dtype=np.float64), src + 0.5])
    order = np.argsort(key, kind="stable")
    out = {
        "ts": np.concatenate([ts, e_ts])[order],
        "symbol": np.concatenate([sym, e_sym])[order],
        "price_e4": np.concatenate([px, e_px])[order],
        "size": np.concatenate([size, e_size])[order],
        "side": np.concatenate([side, e_side])[order],
    }
    # out-of-order delivery: swap some neighbouring rows (no data is wrong, order is)
    # positions are distinct and two apart, so no row takes part in two swaps
    # (overlapping vectorised swaps would duplicate one row and drop another)
    swaps = rng.choice(np.arange(0, len(order) - 1, 2), size=max(k // 4, 1), replace=False)
    for c in out.values():
        a, b = c[swaps].copy(), c[swaps + 1].copy()
        c[swaps], c[swaps + 1] = b, a
    counts = {t: int((kind == i).sum()) for i, t in enumerate(FAULT_TYPES)}
    counts["swapped_pairs"] = int(len(swaps))
    return out, counts


# --------------------------------------------------------------------------
# Fix it: detect, quarantine, repair order
# --------------------------------------------------------------------------
def check_and_clean(table, date: str):
    """Return (clean_table, quarantine_table, stats)."""
    import polars as pl
    from .generate import session_start_ns

    day0 = session_start_ns(date)
    df = pl.from_arrow(table).with_row_index("row")
    n_in = df.height

    # rows that arrived out of (symbol, ts) order
    out_of_order = int(df.select(
        ((pl.col("symbol") < pl.col("symbol").shift(1)) |
         ((pl.col("symbol") == pl.col("symbol").shift(1)) & (pl.col("ts") < pl.col("ts").shift(1))))
        .fill_null(False).sum()).item())

    df = df.sort("symbol", "ts", "row")
    reason = (
        pl.when(pl.col("size") <= 0).then(pl.lit("bad_size"))
        .when((pl.col("ts") < day0) | (pl.col("ts") >= day0 + SESSION_NS)).then(pl.lit("off_session"))
        # any repeat of an earlier identical message, not just of the row right
        # before it (another bad row with the same timestamp can sit in between)
        .when(~pl.struct("symbol", "ts", "price_e4", "size", "side").is_first_distinct())
        .then(pl.lit("duplicate"))
        .otherwise(None)
    )
    df = df.with_columns(reason=reason)

    # bad prints: compare each price to the median of nearby trades in the same symbol,
    # computed on rows that passed the cheap checks (so junk can't skew the median)
    ok = df.filter(pl.col("reason").is_null()).with_columns(
        med=pl.col("price_e4").cast(pl.Float64)
             .rolling_median(MEDIAN_WINDOW, center=True, min_samples=1).over("symbol"))
    bad_px = ok.filter((pl.col("price_e4") / pl.col("med")).log().abs() > PRICE_JUMP)["row"]
    df = df.with_columns(
        reason=pl.when(pl.col("row").is_in(bad_px.implode())).then(pl.lit("bad_price"))
                 .otherwise(pl.col("reason")))

    clean = df.filter(pl.col("reason").is_null()).drop("row", "reason")
    quarantine = df.filter(pl.col("reason").is_not_null()).drop("row")
    stats = {
        "rows_in": n_in, "rows_clean": clean.height, "out_of_order": out_of_order,
        "quarantined": {r: int(c) for r, c in
                        quarantine.group_by("reason").len().iter_rows()},
    }
    stats["bad_fraction"] = (n_in - clean.height) / n_in if n_in else 0.0
    return clean.to_arrow(), quarantine.to_arrow(), stats


def gate_task(task: dict, attempt: int = 0) -> dict:
    import pyarrow.parquet as pq
    from .generate import write_ticks

    src = Path(task["src_root"]) / task["path"]
    clean, quarantine, stats = check_and_clean(pq.read_table(src), task["date"])
    if stats["bad_fraction"] > task["max_bad"]:
        return {"rejected": True, "path": task["path"], "stats": stats}

    out = Path(task["dst_root"]) / task["path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    write_ticks(clean, tmp)
    os.replace(tmp, out)
    if quarantine.num_rows:
        q = Path(task["quarantine_root"]) / task["path"].replace("part-0", "quarantine-0")
        q.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(quarantine, q, compression="zstd")
    ts = clean.column("ts")
    return {"rejected": False, "stats": stats,
            "entry": FileEntry(path=task["path"], date=task["date"], bucket=task["bucket"],
                               rows=clean.num_rows, bytes=out.stat().st_size,
                               ts_min=int(np.min(ts)), ts_max=int(np.max(ts)),
                               sha256=sha256_file(out), quality=stats).__dict__}


def main(argv=None):
    ap = argparse.ArgumentParser(description="quality gate: raw ticks -> clean ticks")
    ap.add_argument("--src", default="data/raw")
    ap.add_argument("--dst", default="data/ticks")
    ap.add_argument("--quarantine", default="data/quarantine")
    ap.add_argument("--max-bad", type=float, default=0.01,
                    help="reject a partition if more than this fraction of rows is bad")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)

    src, dst = Manifest(a.src), Manifest(a.dst)
    done = dst.committed_paths()
    tasks = [{"src_root": a.src, "dst_root": a.dst, "quarantine_root": a.quarantine,
              "path": e.path, "date": e.date, "bucket": e.bucket, "max_bad": a.max_bad}
             for e in src.entries() if e.path not in done]
    t0 = time.perf_counter()
    totals, rejected = {"rows_in": 0, "rows_clean": 0, "out_of_order": 0}, []
    reasons: dict[str, int] = {}

    def _done(r):
        st = r["stats"]
        if r["rejected"]:
            rejected.append(r["path"])
            return
        dst.commit([FileEntry(**r["entry"])])
        for k in totals:
            totals[k] += st[k]
        for k, v in st["quarantined"].items():
            reasons[k] = reasons.get(k, 0) + v

    run_tasks(gate_task, tasks, workers=a.workers, on_result=_done)
    report = {**totals, "quarantined": reasons, "rejected_partitions": rejected,
              "seconds": round(time.perf_counter() - t0, 2)}
    Path(a.dst, "_quality_report.json").write_text(json.dumps(report, indent=2))
    print(f"quality: {totals['rows_in']:,} rows in, {totals['rows_clean']:,} clean, "
          f"quarantined {reasons}, re-sorted {totals['out_of_order']:,} out-of-order rows, "
          f"rejected {len(rejected)} partitions")
    return report


if __name__ == "__main__":
    main()
