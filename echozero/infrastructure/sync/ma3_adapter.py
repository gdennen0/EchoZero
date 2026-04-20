"""MA3 transfer adapter seam and snapshot coercion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class MA3TrackSnapshot:
    """Normalized snapshot of one MA3 track exposed to the sync layer."""

    coord: str
    name: str
    note: str | None = None
    event_count: int | None = None


@dataclass(frozen=True, slots=True)
class MA3EventSnapshot:
    """Normalized snapshot of one MA3 event exposed to the sync layer."""

    event_id: str
    label: str
    start: float | None = None
    end: float | None = None
    cmd: str | None = None


class MA3Adapter(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_status(self) -> dict[str, Any]: ...

    def list_tracks(self) -> Sequence[MA3TrackSnapshot]: ...

    def list_track_events(self, track_coord: str) -> Sequence[MA3EventSnapshot]: ...

    def apply_push_transfer(
        self,
        *,
        target_track_coord: str,
        selected_events: Sequence[Any],
        transfer_mode: str = "merge",
    ) -> None: ...


def coerce_track_snapshot(raw_track: Any) -> MA3TrackSnapshot:
    """Coerce bridge payloads into the canonical MA3 track snapshot shape."""

    if isinstance(raw_track, MA3TrackSnapshot):
        return raw_track

    coord = _value(raw_track, "coord")
    name = _value(raw_track, "name")
    note = _value(raw_track, "note")
    event_count = _value(raw_track, "event_count")
    return MA3TrackSnapshot(
        coord=str(coord or ""),
        name=str(name or ""),
        note=None if note is None else str(note),
        event_count=_optional_int(event_count),
    )


def coerce_event_snapshot(raw_event: Any) -> MA3EventSnapshot:
    """Coerce bridge payloads into the canonical MA3 event snapshot shape."""

    if isinstance(raw_event, MA3EventSnapshot):
        return raw_event

    event_id = _value(raw_event, "event_id")
    if event_id in {None, ""}:
        event_id = _value(raw_event, "id")
    label = _value(raw_event, "label")
    if label in {None, ""}:
        label = _value(raw_event, "name")
    return MA3EventSnapshot(
        event_id=str(event_id or ""),
        label=str(label or "Event"),
        start=_optional_float(_value(raw_event, "start")),
        end=_optional_float(_value(raw_event, "end")),
        cmd=None if _value(raw_event, "cmd") is None else str(_value(raw_event, "cmd")),
    )


def track_snapshot_payload(raw_track: Any) -> dict[str, object]:
    """Serialize one MA3 track snapshot into a plain adapter payload."""

    snapshot = coerce_track_snapshot(raw_track)
    return {
        "coord": snapshot.coord,
        "name": snapshot.name,
        "note": snapshot.note,
        "event_count": snapshot.event_count,
    }


def event_snapshot_payload(raw_event: Any) -> dict[str, object]:
    """Serialize one MA3 event snapshot into a plain adapter payload."""

    snapshot = coerce_event_snapshot(raw_event)
    return {
        "event_id": snapshot.event_id,
        "label": snapshot.label,
        "start": snapshot.start,
        "end": snapshot.end,
    }


def _value(raw: Any, key: str) -> Any:
    if isinstance(raw, dict):
        return raw.get(key)
    return getattr(raw, key, None)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
