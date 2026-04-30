"""MA3 transfer adapter seam and snapshot coercion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from echozero.application.shared.cue_numbers import (
    CueNumber,
    cue_number_from_ref_text as shared_cue_number_from_ref_text,
    parse_positive_cue_number,
)

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
    cue_number: CueNumber | None = None
    cue_ref: str | None = None
    color: str | None = None
    notes: str | None = None
    payload_ref: str | None = None


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
        start_offset_seconds: float = 0.0,
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
    cue_ref = _value(raw_event, "cue_ref")
    if cue_ref in {None, ""}:
        cue_ref = _value(raw_event, "cueRef")
    if cue_ref in {None, ""}:
        cue_ref = _value(raw_event, "cue_label")
    if cue_ref in {None, ""}:
        cue_ref = _value(raw_event, "cueLabel")
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
    resolved_cue_number = parse_positive_cue_number(cue_number)
    label_text = str(label or "Event").strip() or "Event"
    resolved_cue_ref = None if cue_ref in {None, ""} else str(cue_ref).strip() or None
    if resolved_cue_ref is None:
        inferred_cue_ref, inferred_label = _split_transport_cue_prefix(
            label_text,
            cue_number=resolved_cue_number,
        )
        resolved_cue_ref = inferred_cue_ref
        label_text = inferred_label
    else:
        label_text = _strip_transport_cue_prefix(label_text, resolved_cue_ref)
    if resolved_cue_ref is None and resolved_cue_number is not None:
        resolved_cue_ref = str(resolved_cue_number)
    return MA3EventSnapshot(
        event_id=str(event_id or ""),
        label=label_text or resolved_cue_ref or "Event",
        start=start,
        end=end,
        cmd=None if _value(raw_event, "cmd") is None else str(_value(raw_event, "cmd")),
        cue_number=resolved_cue_number,
        cue_ref=resolved_cue_ref,
        color=None if _value(raw_event, "color") in {None, ""} else str(_value(raw_event, "color")),
        notes=(
            None
            if _value(raw_event, "notes") in {None, ""}
            and _value(raw_event, "note") in {None, ""}
            else str(_value(raw_event, "notes") or _value(raw_event, "note"))
        ),
        payload_ref=(
            None
            if _value(raw_event, "payload_ref") in {None, ""}
            and _value(raw_event, "payloadRef") in {None, ""}
            else str(_value(raw_event, "payload_ref") or _value(raw_event, "payloadRef"))
        ),
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
        "cue_ref": snapshot.cue_ref,
        "color": snapshot.color,
        "notes": snapshot.notes,
        "payload_ref": snapshot.payload_ref,
    }


def transport_event_label(raw_event: Any) -> str:
    """Render one cue-aware event label for MA transport round-trips."""

    snapshot = coerce_event_snapshot(raw_event)
    cue_ref = transport_event_cue_ref(raw_event)
    label = str(snapshot.label or "").strip()
    if cue_ref is None:
        return label or "Event"
    if not label:
        return cue_ref
    if label.casefold() == cue_ref.casefold():
        return cue_ref
    if label.casefold().startswith(f"{cue_ref.casefold()} "):
        return label
    return f"{cue_ref} {label}"


def transport_event_cue_ref(raw_event: Any) -> str | None:
    """Resolve only the explicit cue-ref payload we should write back to MA."""

    cue_ref = _value(raw_event, "cue_ref")
    if cue_ref in {None, ""}:
        cue_ref = _value(raw_event, "cueRef")
    if cue_ref in {None, ""}:
        cue_ref = _value(raw_event, "cue_label")
    if cue_ref in {None, ""}:
        cue_ref = _value(raw_event, "cueLabel")
    if cue_ref in {None, ""}:
        return None
    normalized = str(cue_ref).strip()
    return normalized or None


def _value(raw: Any, key: str) -> Any:
    if isinstance(raw, dict):
        return raw.get(key)
    return getattr(raw, key, None)


def _cue_number_from_ref_text(cue_ref: str | None) -> CueNumber | None:
    return shared_cue_number_from_ref_text(cue_ref)


def _split_transport_cue_prefix(
    label: str,
    *,
    cue_number: CueNumber | None,
) -> tuple[str | None, str]:
    text = str(label or "").strip()
    if not text:
        return None, "Event"
    first_token, _separator, _remainder = text.partition(" ")
    candidate = first_token.rstrip(":")
    if cue_number is None:
        return None, text
    if _cue_number_from_ref_text(candidate) != cue_number:
        return None, text
    return candidate, _strip_transport_cue_prefix(text, candidate)


def _strip_transport_cue_prefix(label: str, cue_ref: str) -> str:
    text = str(label or "").strip()
    normalized_cue_ref = str(cue_ref or "").strip()
    if not text or not normalized_cue_ref:
        return text or normalized_cue_ref or "Event"
    if text.casefold() == normalized_cue_ref.casefold():
        return normalized_cue_ref
    if text.casefold().startswith(f"{normalized_cue_ref.casefold()}:"):
        remainder = text[len(normalized_cue_ref) + 1 :].strip()
        return remainder or normalized_cue_ref
    if text.casefold().startswith(f"{normalized_cue_ref.casefold()} "):
        remainder = text[len(normalized_cue_ref) :].strip()
        return remainder or normalized_cue_ref
    return text


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
