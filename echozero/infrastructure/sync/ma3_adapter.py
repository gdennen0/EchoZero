"""MA3 transfer adapter seam and snapshot coercion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class MA3TrackSnapshot:
    """Normalized snapshot of one MA3 track exposed to the sync layer."""

    coord: str
    name: str
    number: int | None = None
    timecode_name: str | None = None
    note: str | None = None
    event_count: int | None = None
    sequence_no: int | None = None


@dataclass(frozen=True, slots=True)
class MA3TimecodeSnapshot:
    """Normalized snapshot of one MA3 timecode pool exposed to the sync layer."""

    number: int
    name: str | None = None


@dataclass(frozen=True, slots=True)
class MA3TrackGroupSnapshot:
    """Normalized snapshot of one MA3 track group exposed to the sync layer."""

    number: int
    name: str
    track_count: int | None = None


@dataclass(frozen=True, slots=True)
class MA3SequenceSnapshot:
    """Normalized snapshot of one MA3 sequence exposed to the sync layer."""

    number: int
    name: str
    cue_count: int | None = None


@dataclass(frozen=True, slots=True)
class MA3SequenceRangeSnapshot:
    """Normalized snapshot of one resolved MA3 current-song sequence range."""

    song_label: str | None
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class MA3EventSnapshot:
    """Normalized snapshot of one MA3 event exposed to the sync layer."""

    event_id: str
    label: str
    start: float | None = None
    end: float | None = None
    cmd: str | None = None
    cue_number: int | None = None


class MA3Adapter(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_status(self) -> dict[str, Any]: ...

    def list_tracks(self, *, timecode_no: int | None = None) -> Sequence[MA3TrackSnapshot]: ...

    def list_timecodes(self) -> Sequence[MA3TimecodeSnapshot]: ...

    def list_track_groups(self, *, timecode_no: int) -> Sequence[MA3TrackGroupSnapshot]: ...

    def list_track_events(self, track_coord: str) -> Sequence[MA3EventSnapshot]: ...

    def list_sequences(
        self,
        *,
        start_no: int | None = None,
        end_no: int | None = None,
    ) -> Sequence[MA3SequenceSnapshot]: ...

    def get_current_song_sequence_range(self) -> MA3SequenceRangeSnapshot | None: ...

    def assign_track_sequence(
        self,
        *,
        target_track_coord: str,
        sequence_no: int,
    ) -> None: ...

    def create_sequence_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> MA3SequenceSnapshot: ...

    def create_sequence_in_current_song_range(
        self,
        *,
        preferred_name: str | None = None,
    ) -> MA3SequenceSnapshot: ...

    def create_timecode_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> MA3TimecodeSnapshot: ...

    def create_track_group_next_available(
        self,
        *,
        timecode_no: int,
        preferred_name: str | None = None,
    ) -> MA3TrackGroupSnapshot: ...

    def create_track(
        self,
        *,
        timecode_no: int,
        track_group_no: int,
        preferred_name: str | None = None,
    ) -> MA3TrackSnapshot: ...

    def prepare_track_for_events(self, *, target_track_coord: str) -> None: ...

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
    timecode_name = _value(raw_track, "timecode_name")
    if timecode_name in {None, ""}:
        timecode_name = _value(raw_track, "timecodeName")
    number = _value(raw_track, "number")
    if number in {None, ""}:
        number = _value(raw_track, "no")
    note = _value(raw_track, "note")
    event_count = _value(raw_track, "event_count")
    sequence_no = _value(raw_track, "sequence_no")
    if sequence_no in {None, ""}:
        sequence_no = _value(raw_track, "seq")
    return MA3TrackSnapshot(
        coord=str(coord or ""),
        name=str(name or ""),
        number=_optional_positive_int(number),
        timecode_name=None if timecode_name in {None, ""} else str(timecode_name),
        note=None if note is None else str(note),
        event_count=_optional_int(event_count),
        sequence_no=_optional_positive_int(sequence_no),
    )


def coerce_timecode_snapshot(raw_timecode: Any) -> MA3TimecodeSnapshot:
    """Coerce bridge payloads into the canonical MA3 timecode snapshot shape."""

    if isinstance(raw_timecode, MA3TimecodeSnapshot):
        return raw_timecode

    number = _value(raw_timecode, "number")
    if number in {None, ""}:
        number = _value(raw_timecode, "no")
    name = _value(raw_timecode, "name")
    return MA3TimecodeSnapshot(
        number=int(_optional_int(number) or 0),
        name=None if name in {None, ""} else str(name),
    )


def coerce_trackgroup_snapshot(raw_trackgroup: Any) -> MA3TrackGroupSnapshot:
    """Coerce bridge payloads into the canonical MA3 track-group snapshot shape."""

    if isinstance(raw_trackgroup, MA3TrackGroupSnapshot):
        return raw_trackgroup

    number = _value(raw_trackgroup, "number")
    if number in {None, ""}:
        number = _value(raw_trackgroup, "no")
    name = _value(raw_trackgroup, "name")
    track_count = _value(raw_trackgroup, "track_count")
    if track_count in {None, ""}:
        track_count = _value(raw_trackgroup, "trackCount")
    return MA3TrackGroupSnapshot(
        number=int(_optional_int(number) or 0),
        name=str(name or ""),
        track_count=_optional_int(track_count),
    )


def coerce_sequence_snapshot(raw_sequence: Any) -> MA3SequenceSnapshot:
    """Coerce bridge payloads into the canonical MA3 sequence snapshot shape."""

    if isinstance(raw_sequence, MA3SequenceSnapshot):
        return raw_sequence

    number = _value(raw_sequence, "number")
    if number in {None, ""}:
        number = _value(raw_sequence, "no")
    cue_count = _value(raw_sequence, "cue_count")
    if cue_count in {None, ""}:
        cue_count = _value(raw_sequence, "cueCount")
    return MA3SequenceSnapshot(
        number=int(_optional_int(number) or 0),
        name=str(_value(raw_sequence, "name") or ""),
        cue_count=_optional_positive_int(cue_count),
    )


def coerce_sequence_range_snapshot(raw_range: Any) -> MA3SequenceRangeSnapshot | None:
    """Coerce bridge payloads into the canonical MA3 sequence range shape."""

    if raw_range is None:
        return None
    if isinstance(raw_range, MA3SequenceRangeSnapshot):
        return raw_range

    start = _optional_positive_int(_value(raw_range, "start"))
    end = _optional_positive_int(_value(raw_range, "end"))
    if start is None or end is None:
        return None
    song_label = _value(raw_range, "song_label")
    if song_label in {None, ""}:
        song_label = _value(raw_range, "songLabel")
    return MA3SequenceRangeSnapshot(
        song_label=None if song_label in {None, ""} else str(song_label),
        start=start,
        end=end,
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
    cue_number = _value(raw_event, "cue_number")
    if cue_number in {None, ""}:
        cue_number = _value(raw_event, "cue_no")
    if cue_number in {None, ""}:
        cue_number = _value(raw_event, "cueno")
    if cue_number in {None, ""}:
        cue_number = _value(raw_event, "cueNo")
    start = _optional_float(_value(raw_event, "start"))
    if start is None:
        start = _optional_float(_value(raw_event, "time"))
    end = _optional_float(_value(raw_event, "end"))
    duration = _optional_float(_value(raw_event, "duration"))
    if duration is None:
        duration = _optional_float(_value(raw_event, "dur"))
    if duration is not None and duration <= 0.0:
        duration = None
    if end is None and start is not None and duration is not None and duration > 0.0:
        end = start + duration
    if start is not None and end is not None and end <= start:
        end = None
    return MA3EventSnapshot(
        event_id=str(event_id or ""),
        label=str(label or "Event"),
        start=start,
        end=end,
        cmd=None if _value(raw_event, "cmd") is None else str(_value(raw_event, "cmd")),
        cue_number=_optional_positive_int(cue_number),
    )


def track_snapshot_payload(raw_track: Any) -> dict[str, object]:
    """Serialize one MA3 track snapshot into a plain adapter payload."""

    snapshot = coerce_track_snapshot(raw_track)
    return {
        "coord": snapshot.coord,
        "name": snapshot.name,
        "number": snapshot.number,
        "timecode_name": snapshot.timecode_name,
        "note": snapshot.note,
        "event_count": snapshot.event_count,
        "sequence_no": snapshot.sequence_no,
    }


def timecode_snapshot_payload(raw_timecode: Any) -> dict[str, object]:
    """Serialize one MA3 timecode snapshot into a plain adapter payload."""

    snapshot = coerce_timecode_snapshot(raw_timecode)
    return {
        "number": snapshot.number,
        "name": snapshot.name,
    }


def trackgroup_snapshot_payload(raw_trackgroup: Any) -> dict[str, object]:
    """Serialize one MA3 track-group snapshot into a plain adapter payload."""

    snapshot = coerce_trackgroup_snapshot(raw_trackgroup)
    return {
        "number": snapshot.number,
        "name": snapshot.name,
        "track_count": snapshot.track_count,
    }


def sequence_snapshot_payload(raw_sequence: Any) -> dict[str, object]:
    """Serialize one MA3 sequence snapshot into a plain adapter payload."""

    snapshot = coerce_sequence_snapshot(raw_sequence)
    return {
        "number": snapshot.number,
        "name": snapshot.name,
        "cue_count": snapshot.cue_count,
    }


def sequence_range_snapshot_payload(raw_range: Any) -> dict[str, object] | None:
    """Serialize one MA3 sequence range snapshot into a plain adapter payload."""

    snapshot = coerce_sequence_range_snapshot(raw_range)
    if snapshot is None:
        return None
    return {
        "song_label": snapshot.song_label,
        "start": snapshot.start,
        "end": snapshot.end,
    }


def event_snapshot_payload(raw_event: Any) -> dict[str, object]:
    """Serialize one MA3 event snapshot into a plain adapter payload."""

    snapshot = coerce_event_snapshot(raw_event)
    return {
        "event_id": snapshot.event_id,
        "label": snapshot.label,
        "start": snapshot.start,
        "end": snapshot.end,
        "cue_number": snapshot.cue_number,
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


def _optional_positive_int(value: Any) -> int | None:
    parsed = _optional_int(value)
    if parsed is None or parsed < 1:
        return None
    return parsed
