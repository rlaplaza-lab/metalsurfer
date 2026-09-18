"""Joblib-style worker resolution for threaded placement / site stages."""

from __future__ import annotations

import os


def resolve_materialize_workers(
    n_jobs: int,
    *,
    n_tasks: int | None = None,
    cpu_count: int | None = None,
) -> int:
    """Resolve joblib-style ``n_jobs`` to a concrete thread-pool size.

    ``1`` is serial, ``>1`` is that many workers, ``-1`` uses all CPUs, and
    values ``< -1`` use ``max(1, cpu_count + 1 + n_jobs)`` (so ``-2`` is all
    but one CPU). When ``n_tasks`` is set, the result is capped at ``n_tasks``.

    Parameters
    ----------
    n_jobs
        Number of parallel workers (joblib convention).
    n_tasks
        Optional cap on workers based on task count.
    cpu_count
        Optional CPU count override.
    """
    if n_jobs == 0:
        raise ValueError("n_jobs must be != 0")
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    cpus = max(int(cpus), 1)
    if n_jobs < 0:
        workers = cpus + 1 + int(n_jobs) if n_jobs < -1 else cpus
        workers = max(1, workers)
    else:
        workers = max(1, int(n_jobs))
    if n_tasks is not None:
        workers = min(workers, max(1, int(n_tasks)))
    return workers
