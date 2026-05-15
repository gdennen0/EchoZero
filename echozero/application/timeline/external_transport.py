"""Normalize external transport updates into canonical timeline intents.
Exists to keep live-sync transport policy aligned with local EZ controls.
Connects MA3 OSC payloads to the same Play/Pause/Stop/Seek intent path used by UI controls.
"""

from __future__ import annotations

from collections.abc import Iterable

from echozero.application.timeline.intents import Pause, Play, Seek, Stop, TimelineIntent

_TRANSPORT_ACTIONS = {
    "play",
    "pause",
    "stop",
    "seek",
    "scrubbed",
    "toggle",
    "play_pause",
    "toggle_play_pause",
    "jump_previous_section",
    "jump_next_section",
}
_TOGGLE_ACTIONS = {"toggle", "play_pause", "toggle_play_pause"}
_SEEK_ACTIONS = {"seek", "scrub", "scrubbed"}
_TRUE_STATES = {"play", "playing", "run", "running", "go"}
_FALSE_STATES = {"pause", "paused", "stop", "stopped"}


def normalize_external_transport_intents(
    payload: dict[str, object] | None,
    *,
    is_playing: bool,
) -> tuple[TimelineIntent, ...]:
    """Return the canonical local timeline intent sequence for one external transport update."""

    if not isinstance(payload, dict):
        return ()

    playing = bool(is_playing)
    action = resolve_external_transport_action(payload)
    seek_seconds = resolve_external_transport_seek_seconds(payload)

    if action in {"jump_previous_section", "jump_next_section"}:
        return ()

    if action in _SEEK_ACTIONS:
        return (Seek(position=seek_seconds),) if seek_seconds is not None else ()

    if action == "stop":
        return (Stop(),)

    if action in _TOGGLE_ACTIONS or (action == "pause" and _payload_marks_toggle(payload)):
        return (Pause(),) if playing else (Play(),)

    if action == "play":
        intents: list[TimelineIntent] = []
        if seek_seconds is not None:
            intents.append(Seek(position=seek_seconds))
        if not playing:
            intents.append(Play())
        return tuple(intents)

    if action == "pause":
        return (Pause(),) if playing else ()

    state_is_playing = resolve_external_transport_is_playing(payload)
    if state_is_playing is True:
        return (Play(),) if not playing else ()
    if state_is_playing is False:
        return (Pause(),) if playing else ()

    if seek_seconds is not None:
        return (Seek(position=seek_seconds),)
    return ()


def resolve_external_transport_seek_seconds(payload: dict[str, object] | None) -> float | None:
    """Resolve the explicit external playhead/seek target, if present."""

    if not isinstance(payload, dict):
        return None
    for key in ("playhead_seconds", "to_seconds", "position", "playhead", "seconds"):
        value = _optional_float(payload.get(key))
        if value is not None:
            return max(0.0, value)
    fields = payload.get("fields")
    if isinstance(fields, dict):
        for key in ("to_seconds", "position", "playhead", "seconds"):
            value = _optional_float(fields.get(key))
            if value is not None:
                return max(0.0, value)
    return None


def resolve_external_transport_action(payload: dict[str, object] | None) -> str | None:
    """Resolve an explicit external transport action without inferring toggles."""

    if not isinstance(payload, dict):
        return None
    for value in _candidate_values(payload, ("change", "action", "state")):
        normalized = _normalize_action(value)
        if normalized is not None:
            return normalized
    fields = payload.get("fields")
    if isinstance(fields, dict):
        for value in _candidate_values(fields, ("action", "change", "state")):
            normalized = _normalize_action(value)
            if normalized is not None:
                return normalized
    return None


def resolve_external_transport_is_playing(payload: dict[str, object] | None) -> bool | None:
    """Resolve a presentation state update from explicit booleans or non-action state words."""

    if not isinstance(payload, dict):
        return None
    for value in _candidate_values(payload, ("is_playing", "playing")):
        parsed = _coerce_bool(value)
        if parsed is not None:
            return parsed
    for value in _candidate_values(payload, ("state",)):
        parsed = _state_to_is_playing(value)
        if parsed is not None:
            return parsed
    fields = payload.get("fields")
    if isinstance(fields, dict):
        for value in _candidate_values(fields, ("is_playing", "playing")):
            parsed = _coerce_bool(value)
            if parsed is not None:
                return parsed
        for value in _candidate_values(fields, ("state",)):
            parsed = _state_to_is_playing(value)
            if parsed is not None:
                return parsed
    return None


def _candidate_values(payload: dict[str, object], keys: Iterable[str]) -> Iterable[object]:
    for key in keys:
        if key in payload:
            yield payload.get(key)


def _normalize_action(value: object) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in _TRANSPORT_ACTIONS:
        return text
    if text in {"scrub", "scrubbing"}:
        return "scrubbed"
    if text in {"previous_section", "prev_section"}:
        return "jump_previous_section"
    if text == "next_section":
        return "jump_next_section"
    return None


def _payload_marks_toggle(payload: dict[str, object]) -> bool:
    for value in _candidate_values(payload, ("toggle", "is_toggle")):
        parsed = _coerce_bool(value)
        if parsed is True:
            return True
    fields = payload.get("fields")
    if isinstance(fields, dict):
        for value in _candidate_values(fields, ("toggle", "is_toggle")):
            parsed = _coerce_bool(value)
            if parsed is True:
                return True
    return False


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
    "normalize_external_transport_intents",
    "resolve_external_transport_action",
    "resolve_external_transport_is_playing",
    "resolve_external_transport_seek_seconds",
]
