"""Heartbeat progress reporting for ProcessPoolExecutor batches.

`as_completed` and `pool.map` only print when a task finishes, so a single
slow task (adaptive quadrature needing many subdivisions, a root-find that
takes longer than usual) produces total silence for its whole duration --
indistinguishable from a hang. `run_pool_with_heartbeat` prints on every
completion AND on a wall-clock timer even when nothing has completed, so
silence is capped at `heartbeat_seconds` regardless of how long individual
tasks take.
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Callable, Sequence


def run_pool_with_heartbeat(
    pool: ProcessPoolExecutor,
    tasks: Sequence[object],
    worker_fn: Callable[[object], object],
    heartbeat_seconds: float = 12.0,
    label: str = "task",
    quiet: bool = False,
) -> list[object]:
    """Run `worker_fn` over `tasks` on `pool`; return results in task order.

    Prints `[label] k/n done ...` on every completion (with elapsed/eta), and
    `[label] still running ...` every `heartbeat_seconds` of wall-clock time
    if nothing has completed in that window. Pass `quiet=True` to suppress
    all printing while keeping the same blocking/collection behaviour.
    """
    total = len(tasks)
    if total == 0:
        return []

    def emit(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    start = time.perf_counter()
    future_to_index = {pool.submit(worker_fn, task): index for index, task in enumerate(tasks)}
    results: list[object] = [None] * total
    pending = set(future_to_index)
    completed = 0
    emit(f"[{label}] queued {total} tasks on up to {pool._max_workers} workers.")
    while pending:
        done, pending = wait(pending, timeout=heartbeat_seconds, return_when=FIRST_COMPLETED)
        elapsed = time.perf_counter() - start
        if done:
            for future in done:
                results[future_to_index[future]] = future.result()
            completed += len(done)
            eta = elapsed * (total - completed) / completed if completed else float("nan")
            emit(
                f"[{label}] {completed}/{total} done  elapsed={elapsed:.1f}s  "
                f"eta={eta:.1f}s  in_flight={len(pending)}"
            )
        else:
            emit(
                f"[{label}] still running: {completed}/{total} done, {len(pending)} in "
                f"flight, elapsed={elapsed:.1f}s (no completions in the last "
                f"{heartbeat_seconds:.0f}s -- normal for slow methods/high resolution, not a hang)"
            )
    return results
