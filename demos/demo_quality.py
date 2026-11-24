"""
Problem 4 - bad data gets into signals.

Break it: generate a clean market (ground truth), then the same market as a
dirty vendor feed (0.2% bad rows + out-of-order delivery). Build bars from the
dirty feed directly and measure the damage against the truth.

Fix it: run the dirty feed through the quality gate, build bars again, and
check they match the truth exactly. Detection is scored against the injected
ground truth.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path("data/demo_quality")
OUT = Path("demo_results")
ARGS = ["--symbols", "200", "--days", "5", "--buckets", "8", "--ticks", "20000"]


def sh(*args):
    subprocess.run([sys.executable, "-m", *args], check=True, stdout=subprocess.DEVNULL)


def bars(d):
    return pl.read_parquet(f"{d}/**/*.parquet", hive_partitioning=False)


def damage(truth: pl.DataFrame, other: pl.DataFrame) -> dict:
    j = truth.join(other, on=["symbol", "bar_ts"], how="full", suffix="_x", coalesce=True)
    both = j.filter(pl.col("close").is_not_null() & pl.col("close_x").is_not_null())
    wrong = both.filter(
        (pl.col("high") != pl.col("high_x")) | (pl.col("low") != pl.col("low_x")) |
        (pl.col("open") != pl.col("open_x")) | (pl.col("close") != pl.col("close_x")) |
        (pl.col("volume") != pl.col("volume_x")))
    big = lambda df, col: int(df.filter(pl.col(col).abs() > 0.05).height)
    return {
        "bars_truth": truth.height,
        "bars_wrong": wrong.height,
        "bars_phantom": int(j.filter(pl.col("close").is_null()).height),
        "bars_missing": int(j.filter(pl.col("close_x").is_null()).height),
        "worst_high_error_pct": round(float(((both["high_x"] / both["high"]) - 1).abs().max() * 100), 1),
        "fake_5pct_moves": big(other, "ret"),
        "true_5pct_moves": big(truth, "ret"),
        "max_rv30_x_truth": round(float(other["rv30"].max() / truth["rv30"].max()), 1),
    }


def main():
    shutil.rmtree(ROOT, ignore_errors=True)
    OUT.mkdir(exist_ok=True)
    sh("exaflow.generate", "--root", f"{ROOT}/truth", *ARGS)
    sh("exaflow.generate", "--root", f"{ROOT}/raw", *ARGS, "--faults", "0.002")
    injected = json.loads((ROOT / "raw/_faults.json").read_text())

    # break it: no gate
    sh("exaflow.pipeline", "--src", f"{ROOT}/truth", "--dst", f"{ROOT}/bars_truth")
    sh("exaflow.pipeline", "--src", f"{ROOT}/raw", "--dst", f"{ROOT}/bars_nogate")
    # fix it: gate first
    sh("exaflow.quality", "--src", f"{ROOT}/raw", "--dst", f"{ROOT}/clean",
       "--quarantine", f"{ROOT}/quarantine")
    sh("exaflow.pipeline", "--src", f"{ROOT}/clean", "--dst", f"{ROOT}/bars_gated")
    report = json.loads((ROOT / "clean/_quality_report.json").read_text())

    truth = bars(f"{ROOT}/bars_truth")
    before = damage(truth, bars(f"{ROOT}/bars_nogate"))
    after = damage(truth, bars(f"{ROOT}/bars_gated"))
    caught = report["quarantined"]
    detection = {t: {"injected": injected[t], "caught": caught.get(t, 0)}
                 for t in ("duplicate", "bad_size", "off_session", "bad_price")}
    result = {"rows": report["rows_in"], "injected": injected, "detection": detection,
              "out_of_order_resorted": report["out_of_order"],
              "without_gate": before, "with_gate": after,
              "gated_identical_to_truth": bars(f"{ROOT}/bars_gated").sort("symbol", "bar_ts")
                                          .equals(truth.sort("symbol", "bar_ts"))}
    (OUT / "quality.json").write_text(json.dumps(result, indent=2))

    print(f"\nPROBLEM 4: bad data  ({report['rows_in']:,} raw rows, "
          f"{sum(v for k, v in injected.items() if k != 'swapped_pairs'):,} bad + "
          f"{injected['swapped_pairs']} out-of-order pairs injected)\n")
    print(f"{'':28s}{'no gate':>12s}{'with gate':>12s}")
    for k, label in [("bars_wrong", "bars with wrong values"), ("bars_phantom", "phantom bars"),
                     ("bars_missing", "missing bars"), ("worst_high_error_pct", "worst high error %"),
                     ("fake_5pct_moves", "1-min moves > 5%"), ("max_rv30_x_truth", "max volatility vs truth")]:
        print(f"{label:28s}{before[k]:>12}{after[k]:>12}")
    print(f"{'(true 1-min moves > 5%)':28s}{before['true_5pct_moves']:>12}")
    print("\ndetection (caught / injected): " +
          ", ".join(f"{t} {d['caught']}/{d['injected']}" for t, d in detection.items()))
    print(f"gated bars identical to truth: {result['gated_identical_to_truth']}")


if __name__ == "__main__":
    main()
