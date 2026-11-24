"""Compare two feature-store outputs row for row.

Usage: python3 compare_outputs.py [clean_dir] [other_dir]
Exits non-zero if they differ, so it can gate CI.
"""
import sys

import polars as pl

a_dir = sys.argv[1] if len(sys.argv) > 1 else "data/bars"
b_dir = sys.argv[2] if len(sys.argv) > 2 else "data/bars_chaos"
a = pl.read_parquet(f"{a_dir}/**/*.parquet").sort("symbol", "bar_ts")
b = pl.read_parquet(f"{b_dir}/**/*.parquet").sort("symbol", "bar_ts")
same = a.equals(b)
print(f"clean rows: {a.height:,}   chaos rows: {b.height:,}   identical: {same}")
sys.exit(0 if same else 1)
