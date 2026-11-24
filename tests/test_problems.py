"""Tests for the three problem fixes: quality gate, point-in-time catalog, cost model."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from exaflow import backtest as bt
from exaflow.catalog import FileEntry, Manifest
from exaflow.panel import read_panel, simulate, write_panel, write_version


def run(*args):
    subprocess.run([sys.executable, "-m", *args], check=True, capture_output=True)


def bars(d):
    return pl.read_parquet(f"{d}/**/*.parquet", hive_partitioning=False).sort("symbol", "bar_ts")


def test_quality_gate_catches_every_fault_and_restores_truth(tmp_path: Path):
    args = ["--symbols", "40", "--days", "2", "--buckets", "4", "--ticks", "3000"]
    run("exaflow.generate", "--root", str(tmp_path / "truth"), *args)
    run("exaflow.generate", "--root", str(tmp_path / "raw"), *args, "--faults", "0.005")
    run("exaflow.quality", "--src", str(tmp_path / "raw"), "--dst", str(tmp_path / "clean"),
        "--quarantine", str(tmp_path / "q"))
    import json
    injected = json.loads((tmp_path / "raw/_faults.json").read_text())
    caught = json.loads((tmp_path / "clean/_quality_report.json").read_text())["quarantined"]
    for t in ("duplicate", "bad_size", "off_session", "bad_price"):
        assert caught.get(t, 0) == injected[t], t
    for d in ("truth", "clean", "raw"):
        run("exaflow.pipeline", "--src", str(tmp_path / d), "--dst", str(tmp_path / f"b_{d}"))
    assert bars(tmp_path / "b_clean").equals(bars(tmp_path / "b_truth"))
    assert not bars(tmp_path / "b_raw").equals(bars(tmp_path / "b_truth"))


def test_quality_gate_rejects_a_mostly_bad_partition(tmp_path: Path):
    args = ["--symbols", "20", "--days", "1", "--buckets", "2", "--ticks", "2000"]
    run("exaflow.generate", "--root", str(tmp_path / "raw"), *args, "--faults", "0.05")
    run("exaflow.quality", "--src", str(tmp_path / "raw"), "--dst", str(tmp_path / "clean"),
        "--quarantine", str(tmp_path / "q"), "--max-bad", "0.01")
    assert Manifest(tmp_path / "clean").entries() == []


def test_catalog_snapshots_time_travel(tmp_path: Path):
    m = Manifest(tmp_path)
    e = lambda p: FileEntry(path=p, date="2026-01-05", bucket=0, rows=1, bytes=1,
                            ts_min=0, ts_max=0, sha256="x")
    s1 = m.commit([e("v0.parquet")])
    s2 = m.commit([e("v1.parquet")])                      # a correction supersedes v0
    assert s2 > s1
    assert [x.path for x in m.entries()] == ["v1.parquet"]
    assert [x.path for x in m.entries(as_of=s1)] == ["v0.parquet"]
    assert m.snapshots() == [s1, s2]


def test_pinned_backtest_is_reproducible_after_a_correction(tmp_path: Path):
    df, master = simulate(n_symbols=200, years=1)
    s1 = write_panel(tmp_path, df, master)
    research = lambda d: bt.run(d, bt.reversal_targets(d), costs=False)["daily_net"]
    before = research(read_panel(tmp_path))
    fixed = df.with_columns(ret=pl.col("ret") * 1.01)
    Manifest(tmp_path / "prices").commit([write_version(tmp_path, fixed, 2021, version=1)])
    assert not np.array_equal(research(read_panel(tmp_path)), before)
    assert np.array_equal(research(read_panel(tmp_path, as_of=s1)), before)


def test_cost_model_behaves():
    df, _ = simulate(n_symbols=300, years=1)
    t = bt.reversal_targets(df)
    gross = bt.run(df, t, costs=False)
    small, big = bt.run(df, t, aum=1e6), bt.run(df, t, aum=1e9)
    assert small["ann_return_pct"] < gross["ann_return_pct"]          # costs never help
    assert big["ann_cost_pct"] > small["ann_cost_pct"]                # impact grows with size
    slow = bt.run(df, t, aum=1e6, smoothing=0.5)
    assert slow["daily_turnover_pct"] < small["daily_turnover_pct"]   # smoothing trades less
