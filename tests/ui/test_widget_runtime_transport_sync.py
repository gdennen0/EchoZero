from __future__ import annotations

import pytest

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


def test_resolve_transport_action_returns_none_when_transport_action_missing() -> None:
    assert _resolve_transport_action({"change": "scrubbed", "fields": {"to_seconds": 9.5}}) is None
