from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from echozero.domain.events import (
    FoundryArtifactFinalizedEvent,
    FoundryArtifactValidatedEvent,
    FoundryRunCreatedEvent,
    FoundryRunStartedEvent,
)
from echozero.event_bus import EventBus


@dataclass(slots=True)
class FoundryActivityItem:
    kind: str
    message: str
    correlation_id: str
    timestamp: float


class FoundryActivityViewModel:
    """UI-facing Foundry activity feed subscribed to Project EventBus."""

    def __init__(self, event_bus: EventBus, max_items: int = 200):
        self._event_bus = event_bus
        self._max_items = max_items
        self._items: list[FoundryActivityItem] = []
        self._latest_run_status: dict[str, str] = {}
        self._listener: Callable[[FoundryActivityItem], None] | None = None

        event_bus.subscribe(FoundryRunCreatedEvent, self._on_run_created)
        event_bus.subscribe(FoundryRunStartedEvent, self._on_run_started)
        event_bus.subscribe(FoundryArtifactFinalizedEvent, self._on_artifact_finalized)
        event_bus.subscribe(FoundryArtifactValidatedEvent, self._on_artifact_validated)

    @property
    def items(self) -> list[FoundryActivityItem]:
        return list(self._items)

    @property
    def latest_run_status(self) -> dict[str, str]:
        return dict(self._latest_run_status)

    def set_listener(self, listener: Callable[[FoundryActivityItem], None] | None) -> None:
        self._listener = listener

    def dispose(self) -> None:
        self._event_bus.unsubscribe(FoundryRunCreatedEvent, self._on_run_created)
        self._event_bus.unsubscribe(FoundryRunStartedEvent, self._on_run_started)
        self._event_bus.unsubscribe(FoundryArtifactFinalizedEvent, self._on_artifact_finalized)
        self._event_bus.unsubscribe(FoundryArtifactValidatedEvent, self._on_artifact_validated)

    def _push(self, item: FoundryActivityItem) -> None:
        self._items.append(item)
        if len(self._items) > self._max_items:
            self._items = self._items[-self._max_items :]
        if self._listener is not None:
            self._listener(item)

    def _on_run_created(self, event: FoundryRunCreatedEvent) -> None:
        self._latest_run_status[event.run_id] = event.status
        self._push(
            FoundryActivityItem(
                kind="run_created",
                message=f"Run created: {event.run_id} ({event.status})",
                correlation_id=event.correlation_id,
                timestamp=event.timestamp,
            )
        )

    def _on_run_started(self, event: FoundryRunStartedEvent) -> None:
        self._latest_run_status[event.run_id] = event.status
        self._push(
            FoundryActivityItem(
                kind="run_started",
                message=f"Run started: {event.run_id}",
                correlation_id=event.correlation_id,
                timestamp=event.timestamp,
            )
        )

    def _on_artifact_finalized(self, event: FoundryArtifactFinalizedEvent) -> None:
        self._push(
            FoundryActivityItem(
                kind="artifact_finalized",
                message=f"Artifact finalized: {event.artifact_id}",
                correlation_id=event.correlation_id,
                timestamp=event.timestamp,
            )
        )

    def _on_artifact_validated(self, event: FoundryArtifactValidatedEvent) -> None:
        outcome = "ok" if event.ok else "failed"
        self._push(
            FoundryActivityItem(
                kind="artifact_validated",
                message=(
                    f"Artifact validation {outcome}: {event.artifact_id} "
                    f"(errors={event.error_count}, warnings={event.warning_count})"
                ),
                correlation_id=event.correlation_id,
                timestamp=event.timestamp,
            )
        )
