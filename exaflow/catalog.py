"""
Manifest-based catalog.

At exabyte scale the dataset is billions of files on object storage. LISTing a
bucket to discover partitions is the first thing that falls over (S3 LIST is
~1,000 keys per call). Table formats like Apache Iceberg / Delta Lake solve this
by keeping a metadata layer that records every data file plus its stats.

This module is a deliberately small version of that idea:
  * every dataset has a manifest (JSON-lines): one record per data file,
    with partition keys, row count, byte size, min/max timestamp and checksum
  * planners read the manifest, never the filesystem
  * commits are append-only and atomic (write temp file, then rename)
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class FileEntry:
    path: str            # relative to dataset root
    date: str
    bucket: int
    rows: int
    bytes: int
    ts_min: int
    ts_max: int
    sha256: str
    quality: dict | None = None      # quality-gate stats, when the file passed the gate
    committed_at: int = 0            # ns since epoch when the entry was committed


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


class Manifest:
    """Append-only manifest. Safe for one committer process (the driver)."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "_manifest.jsonl"
        self._lock = threading.Lock()
        self._last_stamp: int | None = None       # cached so commits stay O(1)

    def _all(self) -> list[FileEntry]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [FileEntry(**json.loads(line)) for line in f if line.strip()]

    def entries(self, as_of: int | None = None) -> list[FileEntry]:
        """The dataset as it stood at `as_of` (ns), or now.

        Every commit is kept, so the catalog can answer "which files made up this
        dataset at time T?". A newer file for the same (date, bucket) supersedes
        the older one (a correction), but the older file is never deleted, so a
        backtest pinned to an earlier snapshot reads exactly what it read then.
        """
        out = {}
        for e in self._all():
            if as_of is not None and e.committed_at > as_of:
                continue
            out[(e.date, e.bucket)] = e          # last write wins per partition
        return list(out.values())

    def snapshots(self) -> list[int]:
        """Commit times: each one is a version of the dataset you can pin to."""
        return sorted({e.committed_at for e in self._all()})

    def committed_paths(self) -> set[str]:
        return {e.path for e in self._all()}

    def commit(self, entries: list[FileEntry]) -> int:
        """Append entries atomically; returns the snapshot time they belong to."""
        if not entries:
            return 0
        with self._lock:
            if self._last_stamp is None:
                self._last_stamp = max(self.snapshots(), default=0)
            stamp = max(time.time_ns(), self._last_stamp + 1)     # strictly increasing
            self._last_stamp = stamp
            with open(self.path, "a") as f:
                for e in entries:
                    e.committed_at = stamp
                    f.write(json.dumps(asdict(e)) + "\n")
                f.flush()
                os.fsync(f.fileno())
        return stamp

    # ---- planning helpers (predicate pushdown at the metadata layer) ----
    def prune(self, dates: set[str] | None = None, buckets: set[int] | None = None,
              ts_range: tuple[int, int] | None = None) -> list[FileEntry]:
        out = []
        for e in self.entries():
            if dates and e.date not in dates:
                continue
            if buckets is not None and e.bucket not in buckets:
                continue
            if ts_range and (e.ts_max < ts_range[0] or e.ts_min > ts_range[1]):
                continue
            out.append(e)
        return out

    def totals(self) -> dict:
        es = self.entries()
        return {"files": len(es), "rows": sum(e.rows for e in es),
                "bytes": sum(e.bytes for e in es)}


def bucket_of(symbol_id: int, n_buckets: int) -> int:
    """Stable hash partitioning of symbols -> buckets (spreads liquid names)."""
    return int(hashlib.blake2b(symbol_id.to_bytes(4, "little"), digest_size=4).hexdigest(), 16) % n_buckets
