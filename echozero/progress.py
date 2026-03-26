"""
Runtime reporting: Side-channel for progress updates and execution lifecycle reports.
Exists because UI needs sub-block granularity and execution status separate from domain events.
Used by ExecutionEngine to report per-block progress and lifecycle; consumed by UI and Coordinator.

Reports are observations of a running process — NOT domain state changes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressReport:
    """A single progress update from a block execution phase."""

    block_id: str
    phase: str
    percent: float
    message: str
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        # Clamp percent to [0.0, 1.0] — frozen dataclass requires object.__setattr__
        clamped = max(0.0, min(1.0, self.percent))
        if clamped != self.percent:
            object.__setattr__(self, "percent", clamped)


@dataclass(frozen=True)
class ExecutionStartedReport:
    """Observation: a block began executing within a pipeline run."""

    block_id: str
    execution_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ExecutionCompletedReport:
    """Observation: a block finished executing — success or failure."""

    block_id: str
    execution_id: str
    success: bool
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


RuntimeReport = Union[ProgressReport, ExecutionStartedReport, ExecutionCompletedReport]


# ---------------------------------------------------------------------------
# RuntimeBus
# ---------------------------------------------------------------------------


class RuntimeBus:
    """Fan-out pub/sub for runtime reports — completely separate from EventBus (DocumentBus)."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[RuntimeReport], None]] = []

    def subscribe(self, callback: Callable[[RuntimeReport], None]) -> None:
        """Register a callback to receive runtime reports."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[RuntimeReport], None]) -> None:
        """Remove a previously registered callback."""
        self._subscribers.remove(callback)

    def publish(self, report: RuntimeReport) -> None:
        """Deliver a runtime report to all subscribers immediately."""
        for subscriber in self._subscribers:
            try:
                subscriber(report)
            except Exception as exc:
                logger.warning(
                    "RuntimeBus: subscriber %r raised %r", subscriber, exc
                )

    def clear(self) -> None:
        """Remove all subscribers — primarily for test teardown."""
        self._subscribers.clear()

