"""
Take System: Manages versioned snapshots of pipeline outputs for user curation.

Replaces the deleted OverrideStore. Takes are DAW-style take lanes — each pipeline
run creates a new take, users compare and cherry-pick the best results into main.

Invariants:
1. Every TakeLayer has exactly one Take where is_main=True.
2. A Take's data field is never mutated after creation. Edit = new Take.
3. Take IDs are globally unique and never reused.
4. Sync only reads/writes the Take where is_main=True.
5. Undo always restores a previous valid TakeLayer state.

See: sass/seminars/echozero-take-system/synthesis.md
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Literal, Union

from echozero.domain.types import AudioData, EventData, Event


# ---------------------------------------------------------------------------
# Take source / provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TakeSource:
    """What pipeline run produced this take. Purely informational."""

    block_id: str
    block_type: str
    settings_snapshot: dict[str, Any]
    run_id: str


# ---------------------------------------------------------------------------
# Take
# ---------------------------------------------------------------------------

TakeOrigin = Literal["pipeline", "user", "merge", "sync"]


@dataclass(frozen=True)
class Take:
    """
    A named, immutable snapshot of EventData or AudioData.

    One take per layer is flagged is_main=True.
    Sync only reads/writes the main take.
    """

    id: str
    label: str
    data: Union[EventData, AudioData]
    origin: TakeOrigin
    source: TakeSource | None
    created_at: datetime
    is_main: bool = False
    notes: str = ""

    @staticmethod
    def create(
        data: Union[EventData, AudioData],
        label: str,
        origin: TakeOrigin = "pipeline",
        source: TakeSource | None = None,
        is_main: bool = False,
        notes: str = "",
    ) -> Take:
        """Factory: create a new Take with a fresh UUID and timestamp."""
        return Take(
            id=str(uuid.uuid4()),
            label=label,
            data=data,
            origin=origin,
            source=source,
            created_at=datetime.now(timezone.utc),
            is_main=is_main,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# TakeLayer
# ---------------------------------------------------------------------------


class TakeLayerError(Exception):
    """Raised when a TakeLayer invariant is violated."""


@dataclass
class TakeLayer:
    """
    Mutable container of frozen Takes. Lives in the Editor.
    Exactly one take has is_main=True at all times.
    """

    layer_id: str
    takes: list[Take]

    def __post_init__(self) -> None:
        self._validate_main_invariant()

    # -- queries --

    def main_take(self) -> Take:
        """Return the main take. Raises if invariant is violated."""
        for t in self.takes:
            if t.is_main:
                return t
        raise TakeLayerError(
            f"TakeLayer '{self.layer_id}' has no main take"
        )

    def get_take(self, take_id: str) -> Take:
        """Return a take by ID. Raises if not found."""
        for t in self.takes:
            if t.id == take_id:
                return t
        raise TakeLayerError(
            f"Take '{take_id}' not found in layer '{self.layer_id}'"
        )

    @property
    def take_count(self) -> int:
        return len(self.takes)

    # -- mutations --

    def add_take(self, take: Take) -> None:
        """Append a take. Must not be main (new takes never auto-promote)."""
        if take.is_main:
            raise TakeLayerError(
                "New takes must not be main. Use promote_to_main() instead."
            )
        self.takes.append(take)

    def promote_to_main(self, take_id: str) -> None:
        """Promote a take to main. Demotes current main."""
        found = False
        new_takes: list[Take] = []
        for t in self.takes:
            if t.id == take_id:
                new_takes.append(replace(t, is_main=True))
                found = True
            elif t.is_main:
                new_takes.append(replace(t, is_main=False))
            else:
                new_takes.append(t)
        if not found:
            raise TakeLayerError(
                f"Take '{take_id}' not found in layer '{self.layer_id}'"
            )
        self.takes = new_takes

    def replace_take(self, take_id: str, new_take: Take) -> None:
        """Replace a take in-place (same position). Used for edits."""
        for i, t in enumerate(self.takes):
            if t.id == take_id:
                self.takes[i] = new_take
                self._validate_main_invariant()
                return
        raise TakeLayerError(
            f"Take '{take_id}' not found in layer '{self.layer_id}'"
        )

    def remove_take(self, take_id: str) -> Take:
        """Remove a take by ID. Cannot remove the main take."""
        for i, t in enumerate(self.takes):
            if t.id == take_id:
                if t.is_main:
                    raise TakeLayerError("Cannot remove the main take")
                return self.takes.pop(i)
        raise TakeLayerError(
            f"Take '{take_id}' not found in layer '{self.layer_id}'"
        )

    def reorder_takes(self, take_ids: list[str]) -> None:
        """Reorder takes by ID list. All IDs must be present."""
        take_map = {t.id: t for t in self.takes}
        if set(take_ids) != set(take_map.keys()):
            raise TakeLayerError("take_ids must contain all take IDs exactly once")
        self.takes = [take_map[tid] for tid in take_ids]

    # -- snapshot (for undo) --

    def snapshot(self) -> TakeLayerSnapshot:
        """Capture current state for undo."""
        return TakeLayerSnapshot(
            layer_id=self.layer_id,
            takes=tuple(self.takes),
        )

    def restore(self, snap: TakeLayerSnapshot) -> None:
        """Restore from a snapshot."""
        self.takes = list(snap.takes)
        self._validate_main_invariant()

    # -- internal --

    def _validate_main_invariant(self) -> None:
        main_count = sum(1 for t in self.takes if t.is_main)
        if len(self.takes) > 0 and main_count != 1:
            raise TakeLayerError(
                f"TakeLayer '{self.layer_id}' has {main_count} main takes "
                f"(expected exactly 1)"
            )


# ---------------------------------------------------------------------------
# Snapshot (for undo)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TakeLayerSnapshot:
    """Immutable snapshot of a TakeLayer's state. Used for undo."""

    layer_id: str
    takes: tuple[Take, ...]


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

MergeStrategy = Literal["additive", "subtract", "intersect", "replace_range"]


def merge_events(
    target_events: tuple[Event, ...],
    source_events: tuple[Event, ...],
    strategy: MergeStrategy = "additive",
    time_epsilon: float = 0.05,
    time_range: tuple[float, float] | None = None,
) -> tuple[Event, ...]:
    """
    Merge source events into target events using the given strategy.

    Strategies:
        additive: Union of all events (keep_both on overlap).
        subtract: Remove target events that match source events (within epsilon).
        intersect: Keep only target events that have a match in source.
        replace_range: In the given time_range, replace target events with source events.

    Args:
        target_events: The events being merged INTO (usually main take).
        source_events: The events being merged FROM.
        strategy: The merge strategy.
        time_epsilon: Time proximity for matching (seconds).
        time_range: Required for replace_range strategy.

    Returns:
        New tuple of merged events.
    """
    if strategy == "additive":
        return target_events + source_events

    elif strategy == "subtract":
        source_times = [e.time for e in source_events]
        return tuple(
            e for e in target_events
            if not _has_time_match(e.time, source_times, time_epsilon)
        )

    elif strategy == "intersect":
        source_times = [e.time for e in source_events]
        return tuple(
            e for e in target_events
            if _has_time_match(e.time, source_times, time_epsilon)
        )

    elif strategy == "replace_range":
        if time_range is None:
            raise ValueError("replace_range strategy requires time_range")
        start, end = time_range
        # Remove target events in range, add source events in range
        kept = tuple(e for e in target_events if e.time < start or e.time > end)
        inserted = tuple(e for e in source_events if start <= e.time <= end)
        # Merge and sort by time
        combined = kept + inserted
        return tuple(sorted(combined, key=lambda e: e.time))

    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")


def _has_time_match(
    time: float, candidates: list[float], epsilon: float
) -> bool:
    """Check if any candidate is within epsilon of the given time."""
    return any(abs(time - c) <= epsilon for c in candidates)


# ---------------------------------------------------------------------------
# Merge takes (high-level)
# ---------------------------------------------------------------------------


def merge_take_into(
    layer: TakeLayer,
    source_take_id: str,
    target_take_id: str,
    strategy: MergeStrategy = "additive",
    event_indices: set[int] | None = None,
    time_range: tuple[float, float] | None = None,
    time_epsilon: float = 0.05,
) -> Take:
    """
    Merge events from source take into target take. Returns the new target take.

    Args:
        layer: The TakeLayer containing both takes.
        source_take_id: Take to merge FROM.
        target_take_id: Take to merge INTO.
        strategy: Merge strategy.
        event_indices: If set, cherry-pick only these event indices from source.
        time_range: Required for replace_range strategy.
        time_epsilon: Time proximity for matching.

    Returns:
        New Take replacing the target, with merged events.
    """
    source = layer.get_take(source_take_id)
    target = layer.get_take(target_take_id)

    if not isinstance(source.data, EventData) or not isinstance(target.data, EventData):
        raise TakeLayerError("Merge is only supported for EventData takes")

    # Get source events (all or cherry-picked)
    if source.data.layers and target.data.layers:
        # For now: merge first layer of each. Multi-layer merge TBD.
        source_events = source.data.layers[0].events
        target_events = target.data.layers[0].events

        if event_indices is not None:
            source_events = tuple(
                e for i, e in enumerate(source_events) if i in event_indices
            )

        merged_events = merge_events(
            target_events,
            source_events,
            strategy=strategy,
            time_epsilon=time_epsilon,
            time_range=time_range,
        )

        # Build new EventData with merged events
        new_layer = replace(target.data.layers[0], events=merged_events)
        remaining_layers = target.data.layers[1:]
        new_data = EventData(layers=(new_layer,) + remaining_layers)
    else:
        new_data = target.data

    merged_take = Take(
        id=target.id,
        label=target.label,
        data=new_data,
        origin="merge",
        source=target.source,
        created_at=target.created_at,
        is_main=target.is_main,
        notes=f"Merged from '{source.label}' ({strategy})",
    )

    return merged_take
