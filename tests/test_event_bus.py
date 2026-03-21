"""
EventBus tests: Verify subscription, dispatch, ordering, and fault-tolerance.
Exists because the bus is the backbone of decoupled communication (FP1).
Tests assert on output values per STYLE.md testing rules; no smoke-only checks.
"""

from __future__ import annotations

import time

import pytest

from echozero.domain.enums import BlockState
from echozero.domain.events import (
    BlockAddedEvent,
    BlockRemovedEvent,
    BlockStateChangedEvent,
    DomainEvent,
    ExecutionCompletedEvent,
    create_event_id,
)
from echozero.event_bus import EventBus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_block_added(
    correlation_id: str = "cmd-1",
    block_id: str = "b1",
    block_type: str = "LoadAudio",
) -> BlockAddedEvent:
    """Create a BlockAddedEvent with sensible defaults."""
    return BlockAddedEvent(
        event_id=create_event_id(),
        timestamp=time.time(),
        correlation_id=correlation_id,
        block_id=block_id,
        block_type=block_type,
    )


def _make_block_removed(
    correlation_id: str = "cmd-1",
    block_id: str = "b1",
) -> BlockRemovedEvent:
    """Create a BlockRemovedEvent with sensible defaults."""
    return BlockRemovedEvent(
        event_id=create_event_id(),
        timestamp=time.time(),
        correlation_id=correlation_id,
        block_id=block_id,
    )


@pytest.fixture
def bus() -> EventBus:
    """Provide a fresh EventBus for each test."""
    return EventBus()


# ---------------------------------------------------------------------------
# Subscription and delivery
# ---------------------------------------------------------------------------


class TestSubscribeAndReceive:
    """Verify basic subscribe-publish-receive cycle."""

    def test_handler_receives_published_event(self, bus: EventBus) -> None:
        received: list[BlockAddedEvent] = []
        bus.subscribe(BlockAddedEvent, received.append)

        event = _make_block_added()
        bus.publish(event)

        assert len(received) == 1
        assert received[0] is event

    def test_parent_type_receives_child_events(self, bus: EventBus) -> None:
        received: list[DomainEvent] = []
        bus.subscribe(DomainEvent, received.append)

        added = _make_block_added()
        removed = _make_block_removed()
        bus.publish(added)
        bus.publish(removed)

        assert len(received) == 2
        assert isinstance(received[0], BlockAddedEvent)
        assert isinstance(received[1], BlockRemovedEvent)

    def test_multiple_handlers_all_called(self, bus: EventBus) -> None:
        results_a: list[BlockAddedEvent] = []
        results_b: list[BlockAddedEvent] = []
        bus.subscribe(BlockAddedEvent, results_a.append)
        bus.subscribe(BlockAddedEvent, results_b.append)

        event = _make_block_added()
        bus.publish(event)

        assert len(results_a) == 1
        assert len(results_b) == 1
        assert results_a[0] is results_b[0] is event

    def test_handler_order_matches_subscription_order(self, bus: EventBus) -> None:
        call_order: list[str] = []
        bus.subscribe(BlockAddedEvent, lambda _: call_order.append("first"))
        bus.subscribe(BlockAddedEvent, lambda _: call_order.append("second"))
        bus.subscribe(BlockAddedEvent, lambda _: call_order.append("third"))

        bus.publish(_make_block_added())

        assert call_order == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# Unsubscribe
# ---------------------------------------------------------------------------


class TestUnsubscribe:
    """Verify handler removal stops delivery."""

    def test_unsubscribed_handler_not_called(self, bus: EventBus) -> None:
        received: list[DomainEvent] = []
        bus.subscribe(BlockAddedEvent, received.append)
        bus.unsubscribe(BlockAddedEvent, received.append)

        bus.publish(_make_block_added())

        assert len(received) == 0


# ---------------------------------------------------------------------------
# Re-entrant publish
# ---------------------------------------------------------------------------


class TestReentrantPublish:
    """Verify breadth-first ordering when handlers publish new events."""

    def test_reentrant_event_delivered_after_current_batch(self, bus: EventBus) -> None:
        delivery_order: list[str] = []

        def on_added(event: BlockAddedEvent) -> None:
            delivery_order.append("added")
            # Re-entrant: publish a removal during handling of add
            bus.publish(_make_block_removed(correlation_id=event.correlation_id))

        def on_removed(event: BlockRemovedEvent) -> None:
            delivery_order.append("removed")

        bus.subscribe(BlockAddedEvent, on_added)
        bus.subscribe(BlockRemovedEvent, on_removed)

        bus.publish(_make_block_added())

        assert delivery_order == ["added", "removed"]


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------


class TestExceptionHandling:
    """Verify one broken handler doesn't stop others."""

    def test_exception_does_not_stop_other_handlers(self, bus: EventBus) -> None:
        received: list[BlockAddedEvent] = []

        def bad_handler(event: BlockAddedEvent) -> None:
            raise RuntimeError("I broke")

        bus.subscribe(BlockAddedEvent, bad_handler)
        bus.subscribe(BlockAddedEvent, received.append)

        bus.publish(_make_block_added())

        assert len(received) == 1


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    """Verify clear removes all subscriptions."""

    def test_clear_removes_all_subscriptions(self, bus: EventBus) -> None:
        received: list[DomainEvent] = []
        bus.subscribe(BlockAddedEvent, received.append)
        bus.subscribe(BlockRemovedEvent, received.append)
        bus.clear()

        bus.publish(_make_block_added())
        bus.publish(_make_block_removed())

        assert len(received) == 0


# ---------------------------------------------------------------------------
# Event identity and metadata
# ---------------------------------------------------------------------------


class TestEventMetadata:
    """Verify auto-generated fields on domain events."""

    def test_event_has_auto_generated_id(self) -> None:
        event_a = _make_block_added()
        event_b = _make_block_added()

        assert len(event_a.event_id) == 32  # UUID4 hex
        assert event_a.event_id != event_b.event_id

    def test_event_has_timestamp(self) -> None:
        before = time.time()
        event = _make_block_added()
        after = time.time()

        assert before <= event.timestamp <= after

    def test_correlation_id_groups_related_events(self) -> None:
        correlation = "cmd-42"
        added = _make_block_added(correlation_id=correlation)
        state_changed = BlockStateChangedEvent(
            event_id=create_event_id(),
            timestamp=time.time(),
            correlation_id=correlation,
            block_id="b1",
            old_state=BlockState.FRESH,
            new_state=BlockState.STALE,
        )

        assert added.correlation_id == state_changed.correlation_id == correlation


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Verify safe behavior for boundary conditions."""

    def test_publish_with_no_subscribers_does_not_error(self, bus: EventBus) -> None:
        event = _make_block_added()
        bus.publish(event)
        # No assertion needed beyond not raising — but verify bus is still functional
        received: list[DomainEvent] = []
        bus.subscribe(BlockAddedEvent, received.append)
        bus.publish(_make_block_added())

        assert len(received) == 1
