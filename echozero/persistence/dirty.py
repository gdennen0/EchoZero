"""
DirtyTracker: Change-detection for autosave and "unsaved changes" indicators.
Exists because the UI needs to know when to show a save indicator, and the autosave
timer needs to know when to flush. Subscribes to EventBus mutation events (FP7 compliant:
domain objects don't know about persistence) and also supports manual marking for
persistence-layer mutations that bypass the event bus (reordering, visibility changes).
"""

from __future__ import annotations

from datetime import datetime, timezone

from echozero.domain.events import (
    BlockAddedEvent,
    BlockRemovedEvent,
    ConnectionAddedEvent,
    ConnectionRemovedEvent,
    DomainEvent,
    SettingsChangedEvent,
)
from echozero.event_bus import EventBus

# Event types that indicate a structural mutation worth tracking for persistence.
# BlockStateChangedEvent is intentionally excluded — it's transient execution state,
# not a structural change that warrants an "unsaved changes" indicator.
_MUTATION_EVENTS: tuple[type[DomainEvent], ...] = (
    BlockAddedEvent,
    BlockRemovedEvent,
    ConnectionAddedEvent,
    ConnectionRemovedEvent,
    SettingsChangedEvent,
)


class DirtyTracker:
    """Tracks whether the project has unsaved changes since the last save/clear."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._dirty: bool = False
        self._dirty_entity_ids: set[str] = set()
        self._last_saved_at: datetime | None = None
        self._event_bus = event_bus
        if event_bus is not None:
            self._subscribe(event_bus)

    def _subscribe(self, event_bus: EventBus) -> None:
        """Wire up handlers for all mutation event types."""
        for event_type in _MUTATION_EVENTS:
            event_bus.subscribe(event_type, self._on_mutation)

    def _unsubscribe(self) -> None:
        """Remove handlers from the event bus."""
        if self._event_bus is not None:
            for event_type in _MUTATION_EVENTS:
                try:
                    self._event_bus.unsubscribe(event_type, self._on_mutation)
                except Exception:
                    pass  # Already removed or EventBus changed exception type

    def _on_mutation(self, event: DomainEvent) -> None:
        """Handler called by the EventBus for any mutation event."""
        self._dirty = True
        # Extract entity ID from known event shapes
        entity_id = getattr(event, "block_id", None)
        if entity_id is not None:
            self._dirty_entity_ids.add(entity_id)

    def is_dirty(self) -> bool:
        """Whether there are unsaved changes."""
        return self._dirty

    def mark_dirty(self, entity_id: str | None = None) -> None:
        """Manually mark the project as dirty.

        Use for persistence-layer mutations that don't go through the event bus
        (e.g., reordering songs, updating layer visibility).
        """
        self._dirty = True
        if entity_id is not None:
            self._dirty_entity_ids.add(entity_id)

    def clear(self) -> None:
        """Reset dirty state — called after a successful save."""
        self._dirty = False
        self._dirty_entity_ids.clear()
        self._last_saved_at = datetime.now(timezone.utc)

    @property
    def last_saved_at(self) -> datetime | None:
        """The timestamp of the last successful save/clear, or None if never saved."""
        return self._last_saved_at

    @property
    def dirty_ids(self) -> set[str]:
        """The set of entity IDs that have been dirtied since the last clear."""
        return set(self._dirty_entity_ids)
