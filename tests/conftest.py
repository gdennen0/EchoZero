"""
Shared test fixtures for EchoZero test suite.
Provides reusable fixtures for database, audio, events, and temporary directories.
All fixtures follow pytest conventions — import-free via conftest auto-discovery.
"""

from __future__ import annotations

import sqlite3
import tempfile
import shutil
from pathlib import Path
from uuid import uuid4
from typing import Generator

import numpy as np
import pytest


# Keep all test temp dirs inside the repository and avoid OS temp locations
_TMP_ROOT = Path(__file__).resolve().parent / ".local-pytest-tmp"


def pytest_configure(config) -> None:
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    # Force pytest's internal tmp directories into a writable repo-local path.
    config.option.basetemp = str(_TMP_ROOT / "run")


@pytest.fixture
def tmp_path() -> Path:
    """Provide a writable temp directory under the repo-local cache."""
    root = _TMP_ROOT / "path-fixtures"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def in_memory_db() -> Generator[sqlite3.Connection, None, None]:
    """Provide a fresh in-memory SQLite connection for each test."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    yield conn
    conn.close()


@pytest.fixture
def sample_audio_samples() -> np.ndarray:
    """Generate a 1-second click track at 44100Hz for testing.

    Produces 10 clicks (impulses) evenly spaced at 100ms intervals.
    Each click is a single-sample impulse at full amplitude.
    """
    sample_rate = 44100
    duration_seconds = 1.0
    num_samples = int(sample_rate * duration_seconds)
    audio = np.zeros(num_samples, dtype=np.float32)

    click_count = 10
    click_interval = num_samples // click_count
    for i in range(click_count):
        audio[i * click_interval] = 1.0

    return audio


@pytest.fixture
def sample_rate() -> int:
    """Standard sample rate used across tests."""
    return 44100


@pytest.fixture
def sample_events() -> list[dict[str, object]]:
    """Generate 10 deterministic onset events for testing.

    Events are spaced at 100ms intervals, matching the click track fixture.
    Each event is a dict with time, duration, and classification fields.
    """
    events = []
    for i in range(10):
        events.append(
            {
                "time": i * 0.1,
                "duration": 0.0,
                "classifications": {},
                "metadata": {},
            }
        )
    return events


@pytest.fixture
def tmp_project_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for project file tests, cleaned up after."""
    with tempfile.TemporaryDirectory(prefix="echozero_test_") as tmpdir:
        yield Path(tmpdir)
