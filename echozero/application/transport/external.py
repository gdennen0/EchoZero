"""External transport command normalization and intent planning.
Exists so MA3, OSC, and other controllers share one app-level transport policy.
Connects raw bridge payloads to timeline intents without letting controllers own timeline truth.
"""

from __future__ import annotations

from collections.abc import Iterable

from echozero.application.timeline.intents import Pause, Play, Seek, Stop, TimelineIntent
from echozero.application.timeline.models import SectionCue, Timeline
from echozero.application.transport.models import (
    ExternalTransportAction,
    ExternalTransportCommand,
)

_ACTION_ALIASES = {
    "play": ExternalTransportAction.PLAY,
    "playing": ExternalTransportAction.PLAY,
    "pause": ExternalTransportAction.PAUSE,
    "paused": ExternalTransportAction.PAUSE,
    "stop": ExternalTransportAction.STOP,
    "stopped": ExternalTransportAction.STOP,
    "toggle": ExternalTransportAction.TOGGLE,
    "play_pause": ExternalTransportAction.TOGGLE,
    "toggle_play_pause": ExternalTransportAction.TOGGLE,
    "seek": ExternalTransportAction.SEEK,
    "scrub": ExternalTransportAction.SEEK,
    "scrubbed": ExternalTransportAction.SEEK,
    "scrubbing": ExternalTransportAction.SEEK,
    "move": ExternalTransportAction.MOVE,
    "nudge": ExternalTransportAction.MOVE,
    "jump_previous_section": ExternalTransportAction.JUMP_PREVIOUS_SECTION,
    "previous_section": ExternalTransportAction.JUMP_PREVIOUS_SECTION,
    "prev_section": ExternalTransportAction.JUMP_PREVIOUS_SECTION,
    "jump_next_section": ExternalTransportAction.JUMP_NEXT_SECTION,
    "next_section": ExternalTransportAction.JUMP_NEXT_SECTION,
}
_TRUE_STATES = {"play", "playing", "run", "running", "go"}
_FALSE_STATES = {"pause", "paused", "stop", "stopped"}
_SECTION_EDGE_EPSILON_SECONDS = 0.025
_PREVIOUS_SECTION_RESTART_GRACE_SECONDS = 1.0


def normalize_external_transport_command(
    payload: dict[str, object] | None,
) -> ExternalTransportCommand | None:
    """Normalize one raw external transport payload into a canonical command."""

    if not isinstance(payload, dict):
        return None

    fields = payload.get("fields")
    field_values = fields if isinstance(fields, dict) else {}
    action = _resolve_action(payload, field_values)
    if action is None:
        is_playing = resolve_external_transport_is_playing(payload)
        if is_playing is True:
            action = ExternalTransportAction.PLAY
        elif is_playing is False:
            action = ExternalTransportAction.PAUSE
    if action is None:
        return None

    return ExternalTransportCommand(
        action=action,
        position_seconds=resolve_external_transport_seek_seconds(payload),
        delta_seconds=resolve_external_transport_delta_seconds(payload),
        source=_optional_text(_first_present(payload, field_values, ("source",))),
        request_id=_optional_text(_first_present(payload, field_values, ("request_id", "id"))),
        metadata=dict(payload),
    )


def build_external_transport_intents(
    command: ExternalTransportCommand,
    *,
    timeline: Timeline,
    is_playing: bool,
    playhead_seconds: float,
) -> tuple[TimelineIntent, ...]:
    """Build canonical timeline intents for one normalized external transport command."""

    playing = bool(is_playing)
    playhead = _clamp_to_timeline(playhead_seconds, timeline=timeline)

    if command.action is ExternalTransportAction.PLAY:
        intents: list[TimelineIntent] = []
        if command.position_seconds is not None:
            intents.append(
                Seek(
                    position=_clamp_to_timeline(
                        command.position_seconds,
                        timeline=timeline,
                    ),
                ),
            )
        if not playing:
            intents.append(Play())
        return tuple(intents)

    if command.action is ExternalTransportAction.PAUSE:
        return (Pause(),) if playing else ()

    if command.action is ExternalTransportAction.STOP:
        return (Stop(),)

    if command.action is ExternalTransportAction.TOGGLE:
        return (Pause(),) if playing else (Play(),)

    if command.action is ExternalTransportAction.SEEK:
        if command.position_seconds is None:
            return ()
        return (Seek(position=_clamp_to_timeline(command.position_seconds, timeline=timeline)),)

    if command.action is ExternalTransportAction.MOVE:
        if command.delta_seconds is None:
            return ()
        return (
            Seek(
                position=_clamp_to_timeline(
                    playhead + float(command.delta_seconds),
                    timeline=timeline,
                ),
            ),
        )

    if command.action is ExternalTransportAction.JUMP_PREVIOUS_SECTION:
        target = resolve_adjacent_section_start(
            timeline.section_cues,
            direction=-1,
            playhead_seconds=playhead,
        )
        return (Seek(position=target),) if target is not None else ()

    if command.action is ExternalTransportAction.JUMP_NEXT_SECTION:
        target = resolve_adjacent_section_start(
            timeline.section_cues,
            direction=1,
            playhead_seconds=playhead,
        )
        return (Seek(position=target),) if target is not None else ()

    return ()


def normalize_external_transport_intents(
    payload: dict[str, object] | None,
    *,
    timeline: Timeline,
    is_playing: bool,
    playhead_seconds: float,
) -> tuple[TimelineIntent, ...]:
    """Normalize a raw payload and return the app-level timeline intent sequence."""

    command = normalize_external_transport_command(payload)
    if command is None:
        return ()
    return build_external_transport_intents(
        command,
        timeline=timeline,
        is_playing=is_playing,
        playhead_seconds=playhead_seconds,
    )


def resolve_adjacent_section_start(
    section_cues: Iterable[SectionCue],
    *,
    direction: int,
    playhead_seconds: float,
) -> float | None:
    """Resolve previous or next section start from EchoZero section cues."""

    cues = sorted({max(0.0, float(cue.start)) for cue in section_cues})
    if not cues:
        return None
    playhead = max(0.0, float(playhead_seconds))
    if direction < 0:
        edge_seconds = max(
            _SECTION_EDGE_EPSILON_SECONDS,
            _PREVIOUS_SECTION_RESTART_GRACE_SECONDS,
        )
        previous = [start for start in cues if start < playhead - edge_seconds]
        return previous[-1] if previous else cues[0]
    next_cues = [start for start in cues if start > playhead + _SECTION_EDGE_EPSILON_SECONDS]
    return next_cues[0] if next_cues else cues[-1]


def resolve_external_transport_seek_seconds(payload: dict[str, object] | None) -> float | None:
    """Resolve the explicit external playhead or seek target, if present."""

    if not isinstance(payload, dict):
        return None
    fields = payload.get("fields")
    field_values = fields if isinstance(fields, dict) else {}
    value = _first_float(
        payload,
        field_values,
        ("playhead_seconds", "to_seconds", "position_seconds", "position", "playhead", "seconds"),
    )
    return None if value is None else max(0.0, value)


def resolve_external_transport_delta_seconds(payload: dict[str, object] | None) -> float | None:
    """Resolve a signed external move delta, if present."""

    if not isinstance(payload, dict):
        return None
    fields = payload.get("fields")
    field_values = fields if isinstance(fields, dict) else {}
    return _first_float(payload, field_values, ("delta_seconds", "delta", "by_seconds", "step"))


def resolve_external_transport_action(payload: dict[str, object] | None) -> str | None:
    """Resolve an explicit external transport action as a canonical string."""

    if not isinstance(payload, dict):
        return None
    fields = payload.get("fields")
    field_values = fields if isinstance(fields, dict) else {}
    action = _resolve_action(payload, field_values)
    return None if action is None else str(action.value)


def resolve_external_transport_is_playing(payload: dict[str, object] | None) -> bool | None:
    """Resolve a presentation state update from explicit booleans or state words."""

    if not isinstance(payload, dict):
        return None
    fields = payload.get("fields")
    field_values = fields if isinstance(fields, dict) else {}
    for value in _candidate_values(payload, ("is_playing", "playing")):
        parsed = _coerce_bool(value)
        if parsed is not None:
            return parsed
    for value in _candidate_values(field_values, ("is_playing", "playing")):
        parsed = _coerce_bool(value)
        if parsed is not None:
            return parsed
    for value in _candidate_values(payload, ("state",)):
        parsed = _state_to_is_playing(value)
        if parsed is not None:
            return parsed
    for value in _candidate_values(field_values, ("state",)):
        parsed = _state_to_is_playing(value)
        if parsed is not None:
            return parsed
    return None


def _resolve_action(
    payload: dict[str, object],
    fields: dict[str, object],
) -> ExternalTransportAction | None:
    for value in _candidate_values(payload, ("change", "action", "state")):
        normalized = _normalize_action(value)
        if normalized is not None:
            return normalized
    for value in _candidate_values(fields, ("action", "change", "state")):
        normalized = _normalize_action(value)
        if normalized is not None:
            return normalized
    return None


def _normalize_action(value: object) -> ExternalTransportAction | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _ACTION_ALIASES.get(text)


def _clamp_to_timeline(position_seconds: float, *, timeline: Timeline) -> float:
    position = max(0.0, float(position_seconds))
    timeline_end = max(0.0, float(getattr(timeline, "end", 0.0) or 0.0))
    if timeline_end <= 0.0:
        return position
    return min(position, timeline_end)


def _first_float(
    payload: dict[str, object],
    fields: dict[str, object],
    keys: tuple[str, ...],
) -> float | None:
    for value in _candidate_values(payload, keys):
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    for value in _candidate_values(fields, keys):
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _first_present(
    payload: dict[str, object],
    fields: dict[str, object],
    keys: tuple[str, ...],
) -> object | None:
    for value in _candidate_values(payload, keys):
        if value is not None:
            return value
    for value in _candidate_values(fields, keys):
        if value is not None:
            return value
    return None


def _candidate_values(payload: dict[str, object], keys: Iterable[str]) -> Iterable[object]:
    for key in keys:
        if key in payload:
            yield payload.get(key)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def _state_to_is_playing(value: object) -> bool | None:
    text = str(value or "").strip().lower()
    if text in _TRUE_STATES:
        return True
    if text in _FALSE_STATES:
        return False
    return None


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
