"""Compatibility exports for external transport normalization.
Exists while older UI imports move to the application transport package.
Connects timeline callers to the canonical `echozero.application.transport.external` module.
"""

from echozero.application.shared.ids import SongVersionId, TimelineId
from echozero.application.timeline.models import Timeline
from echozero.application.transport.external import (
    build_external_transport_intents,
    normalize_external_transport_command,
    normalize_external_transport_intents as _normalize_external_transport_intents,
    resolve_adjacent_section_start,
    resolve_external_transport_action,
    resolve_external_transport_delta_seconds,
    resolve_external_transport_is_playing,
    resolve_external_transport_seek_seconds,
)


def normalize_external_transport_intents(
    payload: dict[str, object] | None,
    *,
    is_playing: bool,
    timeline: Timeline | None = None,
    playhead_seconds: float = 0.0,
):
    """Return external transport intents with backward-compatible defaults."""

    resolved_timeline = timeline or Timeline(
        id=TimelineId("external_transport_compat"),
        song_version_id=SongVersionId("external_transport_compat"),
    )
    return _normalize_external_transport_intents(
        payload,
        timeline=resolved_timeline,
        is_playing=is_playing,
        playhead_seconds=playhead_seconds,
    )


__all__ = [
    "build_external_transport_intents",
    "normalize_external_transport_command",
    "normalize_external_transport_intents",
    "resolve_adjacent_section_start",
    "resolve_external_transport_action",
    "resolve_external_transport_delta_seconds",
    "resolve_external_transport_is_playing",
    "resolve_external_transport_seek_seconds",
]
