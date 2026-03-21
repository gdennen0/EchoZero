"""
EventBus: Decoupled notification system for domain events.
Exists because blocks must stay isolated (FP1) — no block knows about other blocks.
Consumers subscribe by event type; events publish breadth-first after UoW commits.
"""

from __future__ import annotations

import collections
from collections.abc import Callable
from typing import Any

from echozero.domain.events import DomainEvent


class EventBus:
    """Routes domain events to subscribed handlers using breadth-first delivery."""

    def __init__(self) -> None:
        self._handlers: dict[type[DomainEvent], list[Callable[..., Any]]] = {}
        self._queue: collections.deque[DomainEvent] = collections.deque()
        self._publishing: bool = False

    def subscribe(self, event_type: type[DomainEvent], handler: Callable[..., Any]) -> None:
        """Register a handler to receive events of the given type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: type[DomainEvent], handler: Callable[..., Any]) -> None:
        """Remove a handler from the given event type's subscription list."""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)

    def publish(self, event: DomainEvent) -> None:
        """Dispatch an event to all matching handlers using breadth-first ordering."""
        self._queue.append(event)
        if self._publishing:
            # Re-entrant call — event is queued, will be processed after current batch
            return
        self._publishing = True
        try:
            while self._queue:
                current = self._queue.popleft()
                self._dispatch(current)
        finally:
            self._publishing = False

    def clear(self) -> None:
        """Remove all subscriptions — primarily for test teardown."""
        self._handlers.clear()

    def _dispatch(self, event: DomainEvent) -> None:
        """Invoke all handlers whose subscribed type matches the event's MRO."""
        for event_type, handlers in self._handlers.items():
            if isinstance(event, event_type):
                for handler in handlers:
                    try:
                        handler(event)
                    except Exception as exc:
                        # Never let one broken handler kill the bus
                        print(
                            f"EventBus: handler {handler!r} raised {exc!r} for {type(event).__name__}"
                        )
