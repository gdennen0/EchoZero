"""
Test helpers: Golden file comparison, performance assertions, and test factories.
Exists because tests need shared utilities that aren't fixtures (no pytest dependency).
Used by test files directly via import — not auto-injected like conftest.py fixtures.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"


def assert_matches_golden(
    test_name: str,
    actual: Any,
    *,
    tolerance: float | None = None,
    update: bool = False,
) -> None:
    """Compare actual output against a golden file snapshot.

    On first run (or with update=True), creates the golden file.
    On subsequent runs, loads the golden file and compares.

    Args:
        test_name: Name for the golden file (becomes {test_name}_expected.json).
        actual: The actual output to compare. Must be JSON-serializable.
        tolerance: If set, floating-point values are compared within this tolerance.
        update: If True, overwrite the golden file with actual output.
    """
    golden_path = GOLDEN_DIR / f"{test_name}_expected.json"

    if update or not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(actual, indent=2, default=str))
        return

    expected = json.loads(golden_path.read_text())

    if tolerance is not None:
        _assert_close(expected, actual, tolerance, path="root")
    else:
        assert actual == expected, (
            f"Golden file mismatch for '{test_name}'.\n"
            f"Expected: {json.dumps(expected, indent=2)}\n"
            f"Actual: {json.dumps(actual, indent=2)}\n"
            f"To update, run with update=True or delete {golden_path}"
        )


def _assert_close(expected: Any, actual: Any, tolerance: float, path: str) -> None:
    """Recursively compare structures with floating-point tolerance."""
    if isinstance(expected, float) and isinstance(actual, (float, int)):
        assert abs(expected - actual) <= tolerance, (
            f"Float mismatch at {path}: expected {expected}, got {actual} "
            f"(difference {abs(expected - actual)}, tolerance {tolerance})"
        )
    elif isinstance(expected, list) and isinstance(actual, list):
        assert len(expected) == len(actual), (
            f"List length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
        )
        for i, (e, a) in enumerate(zip(expected, actual)):
            _assert_close(e, a, tolerance, f"{path}[{i}]")
    elif isinstance(expected, dict) and isinstance(actual, dict):
        assert set(expected.keys()) == set(actual.keys()), (
            f"Key mismatch at {path}: expected {set(expected.keys())}, got {set(actual.keys())}"
        )
        for key in expected:
            _assert_close(expected[key], actual[key], tolerance, f"{path}.{key}")
    else:
        assert expected == actual, f"Value mismatch at {path}: expected {expected}, got {actual}"


@contextmanager
def assert_faster_than(seconds: float) -> Generator[None, None, None]:
    """Assert that a block of code completes within a time limit.

    Usage:
        with assert_faster_than(10.0):
            process_audio(long_file)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    assert elapsed <= seconds, (
        f"Performance assertion failed: took {elapsed:.2f}s, limit was {seconds:.2f}s"
    )


def make_event(
    time: float = 0.0,
    duration: float = 0.0,
    label: str | None = None,
    **classifications: Any,
) -> dict[str, Any]:
    """Factory for creating test event dicts with sensible defaults.

    Usage:
        make_event(time=0.5, label="kick")
        make_event(time=1.0, duration=0.1, kick={"confidence": 0.9})
    """
    cls: dict[str, Any] = {}
    if label is not None:
        cls[label] = {"label": label, "confidence": 1.0}
    cls.update(classifications)

    return {
        "time": time,
        "duration": duration,
        "classifications": cls,
        "metadata": {},
    }


def make_events_at(
    *times: float,
    label: str | None = None,
) -> list[dict[str, Any]]:
    """Factory for creating a list of events at specific times.

    Usage:
        make_events_at(0.1, 0.2, 0.5, 1.0, label="onset")
    """
    return [make_event(time=t, label=label) for t in times]
