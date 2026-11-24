"""
Pluggable executor: the same pure task function runs on

  * local  - a process pool on one machine (one process per core)
  * ray    - a Ray cluster (laptop -> thousands of nodes, same code)

Tasks are pure functions of (input path, params) -> output file + stats.
That property is what makes the system scale: tasks can be retried, run
twice, run anywhere, and the driver only ever commits results to the
manifest, so a crashed worker never leaves a half-written dataset.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable


def _single_thread_env():
    # Each worker owns exactly one core; measuring per-core throughput is
    # what lets us extrapolate linearly to N cores.
    for k in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "ARROW_NUM_THREADS"):
        os.environ[k] = "1"


def _with_retries(fn, task, max_retries):
    attempt = 0
    while True:
        try:
            return fn(task, attempt)
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"task {task} failed after {attempt} attempts: {exc}") from exc


def run_tasks(fn: Callable[[Any, int], dict], tasks: Iterable[Any], *,
              backend: str = "local", workers: int | None = None,
              max_retries: int = 3, on_result: Callable[[dict], None] | None = None,
              progress_every: int = 25) -> list[dict]:
    tasks = list(tasks)
    workers = workers or os.cpu_count() or 1
    results: list[dict] = []
    t0 = time.perf_counter()

    def _report(r):
        results.append(r)
        if on_result:
            on_result(r)
        n = len(results)
        if n % progress_every == 0 or n == len(tasks):
            el = time.perf_counter() - t0
            print(f"  [{backend}] {n}/{len(tasks)} tasks  {el:6.1f}s", flush=True)

    if backend == "local":
        with ProcessPoolExecutor(max_workers=workers, initializer=_single_thread_env) as ex:
            futs = [ex.submit(_with_retries, fn, t, max_retries) for t in tasks]
            for f in as_completed(futs):
                _report(f.result())

    elif backend == "ray":
        import ray
        if not ray.is_initialized():
            # Locally this starts a mini-cluster; with RAY_ADDRESS set it joins a real one.
            ray.init(num_cpus=workers if not os.environ.get("RAY_ADDRESS") else None,
                     include_dashboard=False, log_to_driver=False,
                     runtime_env={"env_vars": {"POLARS_MAX_THREADS": "1"}})

        # Application errors retry inside the task (attempt counter visible to fn);
        # Ray's max_retries covers lost workers/nodes.
        remote = ray.remote(num_cpus=1, max_retries=max_retries)(
            lambda t: _with_retries(fn, t, max_retries))
        # Bounded in-flight window = backpressure. Never submit a billion tasks at once.
        window = workers * 4
        pending, it = [], iter(tasks)
        for t in it:
            pending.append(remote.remote(t))
            if len(pending) >= window:
                break
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            _report(ray.get(done[0]))
            nxt = next(it, None)
            if nxt is not None:
                pending.append(remote.remote(nxt))
    else:
        raise ValueError(f"unknown backend {backend}")

    return results
