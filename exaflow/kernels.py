"""
Fast bar kernel.

Ticks inside a partition are stored sorted by (symbol, ts), so every
(symbol, minute) bar is one contiguous run of rows. That means aggregation
needs no hash table: find where the key changes, then reduce each run with
NumPy's reduceat (one sequential pass, cache-friendly, no allocation per group).

All sums are integers (price in 1e-4 units, sizes, signed sizes), so results
are exact and independent of evaluation order.
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

MINUTE = 60_000_000_000


def bars_from_file(path) -> dict[str, np.ndarray]:
    t = pq.read_table(path, columns=["ts", "symbol", "price_e4", "size", "side"],
                      use_threads=False)
    return bars_from_arrays(*(t.column(c).to_numpy() for c in
                              ("ts", "symbol", "price_e4", "size", "side")))


def bars_from_arrays(ts, sym, px, size, side) -> dict[str, np.ndarray]:
    n = len(ts)
    minute = ts // MINUTE
    # boundary wherever the minute or the symbol changes (measured faster than
    # building a combined int64 key, which costs three extra passes)
    change = np.empty(n, dtype=bool)
    change[0] = True
    np.not_equal(minute[1:], minute[:-1], out=change[1:])
    change[1:] |= sym[1:] != sym[:-1]
    starts = np.flatnonzero(change)
    last = np.empty_like(starts)
    last[:-1] = starts[1:] - 1
    last[-1] = n - 1

    signed = size * side                      # side is +1/-1 (int8)
    return {
        "symbol": sym[starts].astype(np.uint32),
        "bar_ts": minute[starts] * MINUTE,
        "open_e4": px[starts],
        "high_e4": np.maximum.reduceat(px, starts),
        "low_e4": np.minimum.reduceat(px, starts),
        "close_e4": px[last],
        "volume": np.add.reduceat(size, starts, dtype=np.int64),
        "notional_e4": np.add.reduceat(px * size, starts),          # int64 * int32 -> int64, exact
        "signed": np.add.reduceat(signed, starts, dtype=np.int64),
        "n_trades": (last - starts + 1).astype(np.uint32),
    }
