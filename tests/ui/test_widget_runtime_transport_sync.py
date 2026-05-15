from __future__ import annotations

import pytest

from echozero.application.timeline.external_transport import normalize_external_transport_intents
from echozero.application.timeline.intents import Pause, Play, Seek, Stop
from echozero.ui.qt.timeline.widget_runtime_mixin import (
    _resolve_transport_action,
    _resolve_transport_seek_seconds,
)


def test_resolve_transport_seek_seconds_prefers_normalized_playhead_value() -> None:
    payload = {"playhead_seconds": 14.25, "fields": {"to_seconds": 12.0}}

    resolved = _resolve_transport_seek_seconds(payload)

    assert resolved == pytest.approx(14.25)


def test_resolve_transport_seek_seconds_falls_back_to_nested_fields() -> None:
    payload = {"change": "scrubbed", "fields": {"to_seconds": "9.5"}}

    resolved = _resolve_transport_seek_seconds(payload)

    assert resolved == pytest.approx(9.5)


def test_resolve_transport_seek_seconds_returns_none_without_position_values() -> None:
    assert _resolve_transport_seek_seconds({"change": "state", "is_playing": True}) is None


def test_resolve_transport_action_prefers_change_and_supports_play_pause_stop() -> None:
    assert _resolve_transport_action({"change": "play", "fields": {}}) == "play"
    assert _resolve_transport_action({"change": "pause", "fields": {}}) == "pause"
    assert _resolve_transport_action({"change": "stop", "fields": {}}) == "stop"


def test_resolve_transport_action_falls_back_to_nested_fields_state() -> None:
    payload = {"change": "state", "fields": {"state": "play"}}
    assert _resolve_transport_action(payload) == "play"


def test_resolve_transport_action_returns_seek_action_for_scrubbed_updates() -> None:
    payload = {"change": "scrubbed", "fields": {"to_seconds": 9.5}}
    assert _resolve_transport_action(payload) == "scrubbed"


def test_external_transport_pause_is_idempotent_not_toggle() -> None:
    playing_intents = normalize_external_transport_intents(
        {"change": "pause", "action": "pause", "is_playing": False},
        is_playing=True,
    )
    paused_intents = normalize_external_transport_intents(
        {"change": "pause", "action": "pause", "is_playing": False},
        is_playing=False,
    )

    assert [type(intent) for intent in playing_intents] == [Pause]
    assert paused_intents == ()


def test_external_transport_toggle_uses_current_ez_playing_state() -> None:
    paused_intents = normalize_external_transport_intents(
        {"change": "toggle", "action": "toggle"},
        is_playing=False,
    )
    playing_intents = normalize_external_transport_intents(
        {"change": "toggle", "action": "toggle"},
        is_playing=True,
    )

    assert [type(intent) for intent in paused_intents] == [Play]
    assert [type(intent) for intent in playing_intents] == [Pause]


def test_external_transport_play_pause_combo_uses_current_ez_playing_state() -> None:
    paused_intents = normalize_external_transport_intents(
        {"change": "play_pause", "action": "play_pause", "toggle": True},
        is_playing=False,
    )
    playing_intents = normalize_external_transport_intents(
        {"change": "play_pause", "action": "play_pause", "toggle": True},
        is_playing=True,
    )

    assert [type(intent) for intent in paused_intents] == [Play]
    assert [type(intent) for intent in playing_intents] == [Pause]


def test_external_transport_play_with_position_seeks_before_playing() -> None:
    intents = normalize_external_transport_intents(
        {"change": "play", "action": "play", "to_seconds": 4.25},
        is_playing=False,
    )

    assert [type(intent) for intent in intents] == [Seek, Play]
    assert intents[0].position == pytest.approx(4.25)


def test_external_transport_stop_does_not_emit_extra_seek_from_status_playhead() -> None:
    intents = normalize_external_transport_intents(
        {"change": "stop", "action": "stop", "to_seconds": 7.0},
        is_playing=True,
    )

    assert [type(intent) for intent in intents] == [Stop]
