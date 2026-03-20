"""
Tests for test helpers themselves — making sure our test infrastructure is correct.
Exists because broken test helpers produce false confidence in passing tests.
Run these first: if helpers are broken, all other test results are suspect.
"""

import pytest

from tests.helpers import (
    assert_faster_than,
    assert_matches_golden,
    make_event,
    make_events_at,
)


def test_make_event_defaults() -> None:
    """Event factory produces correct structure with defaults."""
    event = make_event()
    assert event["time"] == 0.0
    assert event["duration"] == 0.0
    assert event["classifications"] == {}
    assert event["metadata"] == {}


def test_make_event_with_label() -> None:
    """Event factory adds classification when label is provided."""
    event = make_event(time=0.5, label="kick")
    assert event["time"] == 0.5
    assert "kick" in event["classifications"]
    assert event["classifications"]["kick"]["confidence"] == 1.0


def test_make_events_at() -> None:
    """Batch event factory creates events at specified times."""
    events = make_events_at(0.1, 0.2, 0.5, label="onset")
    assert len(events) == 3
    assert events[0]["time"] == 0.1
    assert events[1]["time"] == 0.2
    assert events[2]["time"] == 0.5
    assert all("onset" in e["classifications"] for e in events)


def test_assert_faster_than_passes() -> None:
    """Performance assertion passes for fast code."""
    with assert_faster_than(1.0):
        x = sum(range(100))
        assert x == 4950


def test_assert_faster_than_fails() -> None:
    """Performance assertion fails for slow code."""
    import time

    with pytest.raises(AssertionError, match="Performance assertion failed"):
        with assert_faster_than(0.001):
            time.sleep(0.01)


def test_golden_file_create_and_compare(tmp_path: object) -> None:
    """Golden file helper creates file on first run, compares on second."""
    import tests.helpers as h

    original_dir = h.GOLDEN_DIR
    # Temporarily redirect golden dir to tmp
    h.GOLDEN_DIR = tmp_path  # type: ignore[assignment]

    try:
        data = {"events": [{"time": 0.1}, {"time": 0.2}]}
        # First call creates the file
        assert_matches_golden("test_create", data)
        # Second call compares successfully
        assert_matches_golden("test_create", data)
    finally:
        h.GOLDEN_DIR = original_dir


def test_golden_file_tolerance(tmp_path: object) -> None:
    """Golden file comparison respects floating-point tolerance."""
    import tests.helpers as h

    original_dir = h.GOLDEN_DIR
    h.GOLDEN_DIR = tmp_path  # type: ignore[assignment]

    try:
        original = {"time": 0.10000}
        assert_matches_golden("test_tol", original)
        # Slightly different value — within tolerance
        close_enough = {"time": 0.10001}
        assert_matches_golden("test_tol", close_enough, tolerance=0.001)
    finally:
        h.GOLDEN_DIR = original_dir
