"""
Runtime reporting tests: Verify pub/sub delivery, unsubscribe, clear, percent clamping, and lifecycle reports.
Exists because the runtime side-channel must be reliable — dropped or corrupt reports break the UI.
Tests assert on output values per STYLE.md testing rules; no smoke-only checks.
"""

from __future__ import annotations

import time

from echozero.progress import (
    ExecutionCompletedReport,
    ExecutionStartedReport,
    ProgressReport,
    RuntimeBus,
    RuntimeReport,
)


# ---------------------------------------------------------------------------
# ProgressReport value object
# ---------------------------------------------------------------------------


class TestProgressReport:
    """Verify frozen dataclass construction and percent clamping."""

    def test_creates_with_all_fields(self) -> None:
        report = ProgressReport(
            block_id="b1",
            phase="loading",
            percent=0.5,
            message="Halfway there",
            timestamp=1000.0,
        )
        assert report.block_id == "b1"
        assert report.phase == "loading"
        assert report.percent == 0.5
        assert report.message == "Halfway there"
        assert report.timestamp == 1000.0

    def test_timestamp_defaults_to_current_time(self) -> None:
        before = time.time()
        report = ProgressReport(block_id="b1", phase="x", percent=0.0, message="")
        after = time.time()
        assert before <= report.timestamp <= after

    def test_percent_clamped_to_zero_when_negative(self) -> None:
        report = ProgressReport(block_id="b1", phase="x", percent=-0.5, message="")
        assert report.percent == 0.0

    def test_percent_clamped_to_one_when_over(self) -> None:
        report = ProgressReport(block_id="b1", phase="x", percent=1.5, message="")
        assert report.percent == 1.0

    def test_percent_zero_not_clamped(self) -> None:
        report = ProgressReport(block_id="b1", phase="x", percent=0.0, message="")
        assert report.percent == 0.0

    def test_percent_one_not_clamped(self) -> None:
        report = ProgressReport(block_id="b1", phase="x", percent=1.0, message="")
        assert report.percent == 1.0

    def test_is_frozen(self) -> None:
        report = ProgressReport(block_id="b1", phase="x", percent=0.5, message="")
        try:
            report.percent = 0.9  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# ExecutionStartedReport
# ---------------------------------------------------------------------------


class TestExecutionStartedReport:
    """Verify frozen dataclass construction for execution started reports."""

    def test_creates_with_all_fields(self) -> None:
        report = ExecutionStartedReport(
            block_id="b1", execution_id="run-1", timestamp=1000.0
        )
        assert report.block_id == "b1"
        assert report.execution_id == "run-1"
        assert report.timestamp == 1000.0

    def test_timestamp_defaults_to_current_time(self) -> None:
        before = time.time()
        report = ExecutionStartedReport(block_id="b1", execution_id="run-1")
        after = time.time()
        assert before <= report.timestamp <= after

    def test_is_frozen(self) -> None:
        report = ExecutionStartedReport(block_id="b1", execution_id="run-1")
        try:
            report.block_id = "b2"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# ExecutionCompletedReport
# ---------------------------------------------------------------------------


class TestExecutionCompletedReport:
    """Verify frozen dataclass construction for execution completed reports."""

    def test_creates_success_report(self) -> None:
        report = ExecutionCompletedReport(
            block_id="b1", execution_id="run-1", success=True, timestamp=1000.0
        )
        assert report.block_id == "b1"
        assert report.execution_id == "run-1"
        assert report.success is True
        assert report.error is None
        assert report.timestamp == 1000.0

    def test_creates_failure_report_with_error(self) -> None:
        report = ExecutionCompletedReport(
            block_id="b1",
            execution_id="run-1",
            success=False,
            error="Something broke",
        )
        assert report.success is False
        assert report.error == "Something broke"

    def test_timestamp_defaults_to_current_time(self) -> None:
        before = time.time()
        report = ExecutionCompletedReport(
            block_id="b1", execution_id="run-1", success=True
        )
        after = time.time()
        assert before <= report.timestamp <= after

    def test_is_frozen(self) -> None:
        report = ExecutionCompletedReport(
            block_id="b1", execution_id="run-1", success=True
        )
        try:
            report.success = False  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# RuntimeBus pub/sub
# ---------------------------------------------------------------------------


class TestRuntimeBus:
    """Verify fan-out delivery, unsubscribe, and clear for all report types."""

    def test_publish_delivers_to_subscriber(self) -> None:
        bus = RuntimeBus()
        received: list[RuntimeReport] = []
        bus.subscribe(received.append)

        report = ProgressReport(block_id="b1", phase="run", percent=0.5, message="half")
        bus.publish(report)

        assert len(received) == 1
        assert received[0] is report

    def test_multiple_subscribers_all_receive(self) -> None:
        bus = RuntimeBus()
        received_a: list[RuntimeReport] = []
        received_b: list[RuntimeReport] = []
        bus.subscribe(received_a.append)
        bus.subscribe(received_b.append)

        report = ProgressReport(block_id="b1", phase="run", percent=0.3, message="msg")
        bus.publish(report)

        assert len(received_a) == 1
        assert len(received_b) == 1
        assert received_a[0] is report
        assert received_b[0] is report

    def test_unsubscribe_stops_delivery(self) -> None:
        bus = RuntimeBus()
        received: list[RuntimeReport] = []
        bus.subscribe(received.append)

        report_1 = ProgressReport(block_id="b1", phase="a", percent=0.1, message="")
        bus.publish(report_1)
        assert len(received) == 1

        bus.unsubscribe(received.append)

        report_2 = ProgressReport(block_id="b1", phase="b", percent=0.2, message="")
        bus.publish(report_2)
        assert len(received) == 1  # No new delivery

    def test_clear_removes_all_subscribers(self) -> None:
        bus = RuntimeBus()
        received_a: list[RuntimeReport] = []
        received_b: list[RuntimeReport] = []
        bus.subscribe(received_a.append)
        bus.subscribe(received_b.append)

        bus.clear()

        report = ProgressReport(block_id="b1", phase="x", percent=0.5, message="")
        bus.publish(report)

        assert len(received_a) == 0
        assert len(received_b) == 0

    def test_publish_with_no_subscribers_does_nothing(self) -> None:
        bus = RuntimeBus()
        report = ProgressReport(block_id="b1", phase="x", percent=0.5, message="")
        bus.publish(report)  # Should not raise

    def test_multiple_publishes_accumulate(self) -> None:
        bus = RuntimeBus()
        received: list[RuntimeReport] = []
        bus.subscribe(received.append)

        for i in range(5):
            bus.publish(
                ProgressReport(
                    block_id="b1", phase="step", percent=i / 4.0, message=f"step {i}"
                )
            )

        assert len(received) == 5
        assert isinstance(received[0], ProgressReport)
        assert received[0].percent == 0.0
        assert isinstance(received[4], ProgressReport)
        assert received[4].percent == 1.0

    def test_broken_subscriber_does_not_stop_others(self) -> None:
        bus = RuntimeBus()
        received: list[RuntimeReport] = []

        def bad_subscriber(report: RuntimeReport) -> None:
            raise RuntimeError("I broke")

        bus.subscribe(bad_subscriber)
        bus.subscribe(received.append)

        report = ProgressReport(block_id="b1", phase="x", percent=0.5, message="")
        bus.publish(report)

        assert len(received) == 1
        assert received[0] is report

    def test_delivers_execution_started_report(self) -> None:
        bus = RuntimeBus()
        received: list[RuntimeReport] = []
        bus.subscribe(received.append)

        report = ExecutionStartedReport(block_id="b1", execution_id="run-1")
        bus.publish(report)

        assert len(received) == 1
        assert isinstance(received[0], ExecutionStartedReport)
        assert received[0].block_id == "b1"

    def test_delivers_execution_completed_report(self) -> None:
        bus = RuntimeBus()
        received: list[RuntimeReport] = []
        bus.subscribe(received.append)

        report = ExecutionCompletedReport(
            block_id="b1", execution_id="run-1", success=True
        )
        bus.publish(report)

        assert len(received) == 1
        assert isinstance(received[0], ExecutionCompletedReport)
        assert received[0].success is True

    def test_mixed_report_types_all_delivered(self) -> None:
        bus = RuntimeBus()
        received: list[RuntimeReport] = []
        bus.subscribe(received.append)

        bus.publish(ExecutionStartedReport(block_id="b1", execution_id="run-1"))
        bus.publish(ProgressReport(block_id="b1", phase="run", percent=0.5, message="half"))
        bus.publish(ExecutionCompletedReport(block_id="b1", execution_id="run-1", success=True))

        assert len(received) == 3
        assert isinstance(received[0], ExecutionStartedReport)
        assert isinstance(received[1], ProgressReport)
        assert isinstance(received[2], ExecutionCompletedReport)


# ---------------------------------------------------------------------------
# Backward compatibility alias

