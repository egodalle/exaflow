"""
Exabyte scale model.

Takes the *measured* per-core throughput from the benchmark runs and turns
it into the numbers you need to discuss an exabyte deployment:
compute (cores / nodes), storage layout (files, partitions, metadata size),
and what a research query actually costs given partition pruning.

Every input is either measured or stated as an explicit assumption,
so each number can be challenged and recomputed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EB = 1e18


def fmt(n: float) -> str:
    for unit, div in ((" quadrillion", 1e15), (" trillion", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n/div:,.1f}{unit}"
    return f"{n:,.0f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default="bench")
    ap.add_argument("--dataset-bytes", type=float, default=1 * EB)
    ap.add_argument("--cores-per-node", type=int, default=64)
    ap.add_argument("--target-file-mb", type=float, default=512,
                    help="target compressed size per data file at scale")
    ap.add_argument("--node-net-gbps", type=float, default=100,
                    help="NIC per node; caps per-node read throughput")
    ap.add_argument("--efficiency", type=float, default=0.7,
                    help="assumed cluster efficiency vs perfect linear (stragglers, scheduling)")
    a = ap.parse_args(argv)

    bd = Path(a.bench_dir)
    runs = sorted((json.loads(p.read_text()) for p in bd.glob("pipeline_w*.json")),
                  key=lambda r: r["workers"])
    if not runs:
        raise SystemExit("no pipeline_w*.json benchmark files found")
    best = max(runs, key=lambda r: r["bytes_per_sec_per_core"])
    query = json.loads((bd / "query.json").read_text()) if (bd / "query.json").exists() else None

    per_core = best["bytes_per_sec_per_core"]
    bytes_per_row = best["input_bytes"] / best["input_rows"]
    D = a.dataset_bytes
    out = []
    p = out.append

    p("# Exabyte scale model\n")
    p("## 1. Measured (this machine)\n")
    p("| workers | ticks | compressed input | wall | rows/s | rows/s/core | MB/s/core | speedup | efficiency |")
    p("|---|---|---|---|---|---|---|---|---|")
    base = runs[0]
    for r in runs:
        sp = r["rows_per_sec"] / base["rows_per_sec"]
        eff = sp / (r["workers"] / base["workers"])
        p(f"| {r['workers']} | {fmt(r['input_rows'])} | {r['input_bytes']/1e9:.2f} GB | "
          f"{r['wall_seconds']:.1f}s | {fmt(r['rows_per_sec'])} | {fmt(r['rows_per_sec_per_core'])} | "
          f"{r['bytes_per_sec_per_core']/1e6:.1f} | {sp:.2f}x | {eff:.0%} |")
    p(f"\nCompressed bytes per tick: **{bytes_per_row:.1f} B** (zstd Parquet, columnar).")

    p("\n## 2. Full pass over the dataset (worst case: touch every byte)\n")
    p(f"Dataset: **{D/EB:g} EB** compressed  ->  ~{fmt(D/bytes_per_row)} ticks at this density.\n")
    p(f"Assumptions: {a.cores_per_node} cores/node, {a.node_net_gbps:g} Gb/s NIC/node, "
      f"{a.efficiency:.0%} cluster efficiency.\n")
    node_cpu = per_core * a.cores_per_node
    node_net = a.node_net_gbps * 1e9 / 8
    node_eff = min(node_cpu, node_net) * a.efficiency
    bound = "CPU" if node_cpu < node_net else "network"
    p(f"Per-node effective throughput: **{node_eff/1e9:.2f} GB/s** ({bound}-bound).\n")
    p("| finish a full pass in | nodes needed | cores |")
    p("|---|---|---|")
    for label, secs in (("1 hour", 3600), ("24 hours", 86400), ("7 days", 7 * 86400),
                        ("30 days", 30 * 86400)):
        nodes = math.ceil(D / (node_eff * secs))
        p(f"| {label} | {nodes:,} | {nodes*a.cores_per_node:,} |")

    rows_core = best["rows_per_sec_per_core"]
    p(f"\nIn trades rather than bytes: one node processes ~{rows_core*a.cores_per_node*a.efficiency/1e6:,.0f}M "
      f"trades/s, so **100 quadrillion trades in 7 days needs "
      f"{math.ceil(1e17/(rows_core*a.cores_per_node*a.efficiency*7*86400)):,} nodes**. "
      "(Byte-based node counts rise when compression improves, because 1 EB then holds more trades.)")

    p("\n## 3. Storage layout at 1 EB\n")
    nfiles = D / (a.target_file_mb * 1e6)
    manifest_bytes = nfiles * 300  # ~300 B per manifest entry (path, stats, checksum)
    p(f"- Target file size {a.target_file_mb:g} MB -> **{fmt(nfiles)} data files**.")
    p(f"- Manifest at ~300 B/entry -> **{manifest_bytes/1e9:,.0f} GB of metadata**. "
      "That is too big for one flat file, so it becomes a tree: "
      "snapshot -> manifest lists -> manifests (the Iceberg design), "
      "partitioned by date so planning a query reads only the relevant branch.")
    p(f"- LISTing {fmt(nfiles)} objects at 1,000 keys/call = {fmt(nfiles/1000)} API calls. "
      "That is why planning must go through the catalog and never through LIST.")
    p("- Small-file problem: tick partitions for illiquid names are tiny. Compaction jobs "
      "merge them up to the target size; the manifest swap makes compaction atomic.")

    if query:
        p("\n## 4. What a research query actually touches (measured here)\n")
        p("| query | files | fraction of dataset | at 1 EB that is |")
        p("|---|---|---|---|")
        for q in query["queries"]:
            f = q["fraction_of_dataset"]
            if q["name"].startswith("point"):
                at_eb = f"{q['bytes']/1e6:,.1f} MB (fixed: one symbol-day doesn't grow)"
            else:
                at_eb = f"{f * D / 1e15:,.1f} PB if the same fraction"
            p(f"| {q['name']} | {q['files']} | {f:.4%} | {at_eb} |")
        p("\nThe point query's fraction shrinks as the dataset grows (one symbol-day is "
          "a fixed size, the dataset is not), so at 1 EB it is far smaller than the "
          "local measurement. Partition pruning is what turns an exabyte store into "
          "seconds-to-minutes research queries.")

    text = "\n".join(out)
    (bd / "SCALE_MODEL.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
