"""Lightweight runtime timing hooks for local perf instrumentation."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import os
import time
from typing import Iterator

_ENABLED = os.environ.get("ECHOZERO_PERF", "0").strip().lower() in {"1", "true", "yes", "on"}

# name -> [count, total_ms, max_ms]
_STATS: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])


def is_enabled() -> bool:
    return _ENABLED


@contextmanager
def timed(name: str) -> Iterator[None]:
    if not _ENABLED:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        bucket = _STATS[name]
        bucket[0] += 1.0
        bucket[1] += elapsed_ms
        if elapsed_ms > bucket[2]:
            bucket[2] = elapsed_ms


def snapshot() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, (count, total_ms, max_ms) in _STATS.items():
        avg = (total_ms / count) if count else 0.0
        out[name] = {
            "count": float(count),
            "total_ms": float(total_ms),
            "avg_ms": float(avg),
            "max_ms": float(max_ms),
        }
    return out


def reset() -> None:
    _STATS.clear()
