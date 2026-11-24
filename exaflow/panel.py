"""
Daily equity panel with a realistic life cycle: listings, delistings, liquidity.

Used by the point-in-time and trading-cost demos. Everything planted here is a
*known* effect, so each tool can be checked against the answer:

  * ~30% of stocks delist during the sample. Most are failures: they slide for
    months and lose about a third of their value on the delisting day, which is
    roughly what the academic record shows for performance delistings.
  * a weak short-term reversal: yesterday's idiosyncratic move partly reverses.
    It is real, it is small, and it lives mostly in illiquid names, which is
    exactly the kind of signal that looks great before trading costs.
  * liquidity (average daily dollar volume) varies ~10,000x across stocks, and
    spreads and volatility are wider for small names.

Output:
  <root>/prices/year=YYYY/part-N.parquet   daily rows, one file per year (catalogued)
  <root>/security_master.parquet           symbol, listed, delisted (null = still listed)
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import numpy as np

from .catalog import FileEntry, Manifest, sha256_file

REVERSAL = -0.08         # idio_t = REVERSAL * mean(idio over the last WINDOW days) + noise
WINDOW = 5
SECTOR_VOL = 0.008


def business_days(start: str, n: int) -> list[str]:
    d, out = dt.date.fromisoformat(start), []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def simulate(n_symbols: int = 1500, years: int = 5, seed: int = 11,
             reversal: float = REVERSAL, window: int = WINDOW):
    import polars as pl

    rng = np.random.default_rng(seed)
    days = business_days("2021-01-04", 252 * years)
    T, N = len(days), n_symbols

    # life cycle
    listed = np.where(rng.random(N) < 0.7, 0, rng.integers(1, T - 60, N))
    delists = rng.random(N) < 0.30
    delisted = np.where(delists, np.minimum(listed + rng.integers(120, T, N), T - 1), T)
    delisted = np.where(delisted <= listed + 60, T, delisted)          # keep lives >= 60 days

    # liquidity and risk
    adv = np.exp(rng.normal(np.log(8e6), 1.9, N)).clip(5e4, 5e9)        # $ per day
    sigma = (0.008 + 0.06 / np.sqrt(adv / 1e6 + 4)).clip(0.008, 0.04)    # daily idio vol
    spread_bps = (2 + 30 / np.sqrt(adv / 1e6)).clip(1, 150)              # quoted spread
    beta = rng.normal(1.0, 0.3, N).clip(0.2, 2.0)

    mkt = rng.normal(0.0004, 0.011, T)
    # sector shocks: risk a long-short book cannot diversify away (keeps Sharpe ratios honest)
    sector = rng.integers(0, 20, N)
    sec_ret = rng.normal(0, SECTOR_VOL, (T, 20))
    eps = rng.normal(0, 1, (T, N)) * sigma
    idio = np.empty((T, N))
    idio[0] = eps[0]
    for t in range(1, T):
        idio[t] = reversal * idio[max(0, t - window):t].mean(axis=0) + eps[t]

    # failures slide before they delist, then drop on the last day
    tt = np.arange(T)[:, None]
    slide = (tt >= delisted - 120) & (tt < delisted) & delists
    idio += np.where(slide, -0.004, 0.0)
    last = (tt == delisted - 1) & delists
    idio = np.where(last, np.log(rng.uniform(0.45, 0.85, N)), idio)

    # subtract the lognormal convexity term so a stock's expected simple return is
    # its market exposure, not a free 'volatility bonus' for equal-weight portfolios
    ret = np.expm1(beta * mkt[:, None] + sec_ret[:, sector] + idio
                   - 0.5 * (sigma ** 2 + (beta * 0.011) ** 2 + SECTOR_VOL ** 2))
    alive = (tt >= listed) & (tt < delisted)

    t_idx, s_idx = np.nonzero(alive)
    df = pl.DataFrame({
        "date": np.array(days)[t_idx],
        "symbol": s_idx.astype(np.int32),
        "ret": ret[t_idx, s_idx],
        "adv_usd": adv[s_idx],
        "spread_bps": spread_bps[s_idx],
        "sigma": sigma[s_idx],
    }).with_columns(pl.col("date").str.to_date())
    master = pl.DataFrame({
        "symbol": np.arange(N, dtype=np.int32),
        "listed": [dt.date.fromisoformat(days[i]) for i in listed],
        "delisted": [dt.date.fromisoformat(days[i]) if i < T else None for i in delisted],
    })
    return df, master


def write_panel(root: str, df, master) -> int:
    """One catalogued file per year; returns the snapshot time."""
    import polars as pl
    root = Path(root)
    man = Manifest(root / "prices")
    entries = []
    for (year,), part in df.group_by(pl.col("date").dt.year(), maintain_order=True):
        entries.append(write_version(root, part, int(year), version=0))
    master.write_parquet(root / "security_master.parquet")
    return man.commit(entries)


def write_version(root, part, year: int, version: int) -> FileEntry:
    rel = f"year={year}/part-{version}.parquet"
    out = Path(root) / "prices" / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    part.sort("date", "symbol").write_parquet(tmp, compression="zstd")
    os.replace(tmp, out)
    d = part["date"]
    epoch = lambda x: int(dt.datetime.combine(x, dt.time(), dt.timezone.utc).timestamp() * 1e9)
    return FileEntry(path=rel, date=str(year), bucket=0, rows=part.height,
                     bytes=out.stat().st_size, ts_min=epoch(d.min()), ts_max=epoch(d.max()),
                     sha256=sha256_file(out))


def read_panel(root: str, as_of: int | None = None):
    """The price history as the catalog knew it at `as_of` (default: now)."""
    import polars as pl
    entries = sorted(Manifest(Path(root) / "prices").entries(as_of=as_of), key=lambda e: e.date)
    return pl.concat([pl.read_parquet(Path(root) / "prices" / e.path) for e in entries])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/panel")
    ap.add_argument("--symbols", type=int, default=1500)
    ap.add_argument("--years", type=int, default=5)
    a = ap.parse_args(argv)
    df, master = simulate(a.symbols, a.years)
    write_panel(a.root, df, master)
    print(f"panel: {df.height:,} stock-days, {master.height:,} symbols, "
          f"{master['delisted'].is_not_null().sum():,} delisted")


if __name__ == "__main__":
    main()
