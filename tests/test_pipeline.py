"""End-to-end correctness: chaos run == clean run, and bars reconcile to raw ticks."""
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import polars as pl


def run(*args):
    subprocess.run([sys.executable, "-m", *args], check=True, capture_output=True)


def test_end_to_end(tmp_path: Path):
    ticks, clean, chaos = tmp_path / "ticks", tmp_path / "clean", tmp_path / "chaos"
    run("exaflow.generate", "--root", str(ticks), "--symbols", "40", "--days", "2",
        "--buckets", "4", "--ticks", "3000")
    run("exaflow.pipeline", "--src", str(ticks), "--dst", str(clean))
    run("exaflow.pipeline", "--src", str(ticks), "--dst", str(chaos), "--fail-rate", "0.5")

    # 0. both engines (fast NumPy kernel, Polars group-by) agree exactly
    ref = tmp_path / "ref"
    run("exaflow.pipeline", "--src", str(ticks), "--dst", str(ref), "--engine", "polars")
    r = pl.read_parquet(str(ref / "**/*.parquet")).sort("symbol", "bar_ts")
    f = pl.read_parquet(str(clean / "**/*.parquet")).sort("symbol", "bar_ts")
    assert r.equals(f)

    # 1. exactly-once under failures: identical file set, identical content
    def entries(root):
        return {json.loads(l)["path"]: json.loads(l)["rows"]
                for l in (root / "_manifest.jsonl").read_text().splitlines()}
    assert entries(clean) == entries(chaos)
    a = pl.read_parquet(str(clean / "**/*.parquet")).sort("symbol", "bar_ts")
    b = pl.read_parquet(str(chaos / "**/*.parquet")).sort("symbol", "bar_ts")
    assert a.equals(b)

    # 2. bars reconcile to raw ticks: volume and VWAP per symbol-day
    con = duckdb.connect()
    raw = con.execute(f"""
        SELECT symbol, sum(size) vol, sum(price_e4::HUGEINT*size)/sum(size)/1e4 vwap
        FROM read_parquet('{ticks}/**/*.parquet', hive_partitioning=false)
        GROUP BY symbol ORDER BY symbol""").fetch_df()
    agg = con.execute(f"""
        SELECT symbol, sum(volume) vol, sum(vwap*volume)/sum(volume) vwap
        FROM read_parquet('{clean}/**/*.parquet', hive_partitioning=false)
        GROUP BY symbol ORDER BY symbol""").fetch_df()
    assert raw["vol"].to_list() == agg["vol"].to_list()
    assert (raw["vwap"] - agg["vwap"]).abs().max() < 1e-9

    # 3. regeneration is deterministic (idempotent tasks)
    regen = tmp_path / "ticks2"
    run("exaflow.generate", "--root", str(regen), "--symbols", "40", "--days", "2",
        "--buckets", "4", "--ticks", "3000")
    sha = lambda r: {json.loads(l)["path"]: json.loads(l)["sha256"]
                     for l in (r / "_manifest.jsonl").read_text().splitlines()}
    assert sha(ticks) == sha(regen)
