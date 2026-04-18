"""Stable timeline fixture loader for Stage Zero UI development."""

from __future__ import annotations

import json
from pathlib import Path

from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    LayerStatusPresentation,
    TakeActionPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import FollowMode, LayerKind
from echozero.application.shared.ids import EventId, LayerId, TakeId, TimelineId
from echozero.ui.qt.timeline.style import TIMELINE_STYLE, fixture_color, fixture_take_action_label

_DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "realistic_timeline_fixture.json"


def fixture_path() -> Path:
    return _DEFAULT_FIXTURE_PATH


def load_realistic_timeline_fixture(path: str | Path | None = None) -> TimelinePresentation:
    source = Path(path) if path is not None else _DEFAULT_FIXTURE_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))

    timeline_data = payload["timeline"]
    layers_data = payload.get("layers", [])

    layers = [_parse_layer(layer_data) for layer_data in layers_data]

    return TimelinePresentation(
        timeline_id=TimelineId(timeline_data["id"]),
        title=timeline_data["title"],
        layers=layers,
        playhead=float(timeline_data.get("playhead", 0.0)),
        is_playing=bool(timeline_data.get("is_playing", False)),
        follow_mode=FollowMode(timeline_data.get("follow_mode", FollowMode.CENTER.value)),
        selected_layer_id=_layer_id_or_none(timeline_data.get("selected_layer_id")),
        selected_take_id=_take_id_or_none(timeline_data.get("selected_take_id")),
        selected_event_ids=[EventId(event_id) for event_id in timeline_data.get("selected_event_ids", [])],
        pixels_per_second=float(timeline_data.get("pixels_per_second", 100.0)),
        scroll_x=float(timeline_data.get("scroll_x", 0.0)),
        scroll_y=float(timeline_data.get("scroll_y", 0.0)),
        current_time_label=timeline_data.get("current_time_label", "00:00.00"),
        end_time_label=timeline_data.get("end_time_label", "00:00.00"),
    )


def _parse_layer(data: dict) -> LayerPresentation:
    status_data = data.get("status", {})
    resolved_color = _resolved_color(data)
    return LayerPresentation(
        layer_id=LayerId(data["id"]),
        title=data["title"],
        subtitle=data.get("subtitle", ""),
        kind=LayerKind(data.get("kind", LayerKind.EVENT.value)),
        is_selected=bool(data.get("is_selected", False)),
        is_expanded=bool(data.get("is_expanded", False)),
        events=[_parse_event(event, default_color=resolved_color) for event in data.get("events", [])],
        takes=[_parse_take(take, default_color=resolved_color) for take in data.get("takes", [])],
        visible=bool(data.get("visible", True)),
        locked=bool(data.get("locked", False)),
        gain_db=float(data.get("gain_db", 0.0)),
        pan=float(data.get("pan", 0.0)),
        color=resolved_color,
        badges=list(data.get("badges", [])),
        waveform_key=data.get("waveform_key"),
        source_audio_path=data.get("source_audio_path"),
        status=LayerStatusPresentation(
            stale=bool(status_data.get("stale", False)),
            manually_modified=bool(status_data.get("manually_modified", False)),
            source_label=status_data.get("source_label", ""),
            sync_label=status_data.get("sync_label", TIMELINE_STYLE.fixture.default_sync_label),
        ),
    )


def _parse_take(data: dict, *, default_color: str | None) -> TakeLanePresentation:
    return TakeLanePresentation(
        take_id=TakeId(data["id"]),
        name=data["name"],
        is_main=bool(data.get("is_main", False)),
        kind=LayerKind(data.get("kind", LayerKind.EVENT.value)),
        events=[_parse_event(event, default_color=default_color) for event in data.get("events", [])],
        source_ref=data.get("source_ref"),
        waveform_key=data.get("waveform_key"),
        source_audio_path=data.get("source_audio_path"),
        actions=[
            TakeActionPresentation(
                action_id=action["action_id"],
                label=action.get("label", fixture_take_action_label(action["action_id"])),
            )
            for action in data.get("actions", [])
        ],
    )


def _parse_event(data: dict, *, default_color: str | None) -> EventPresentation:
    return EventPresentation(
        event_id=EventId(data["id"]),
        start=float(data["start"]),
        end=float(data["end"]),
        label=data["label"],
        color=data.get("color", default_color),
        muted=bool(data.get("muted", False)),
        is_selected=bool(data.get("is_selected", False)),
        badges=list(data.get("badges", [])),
    )


def _resolved_color(data: dict) -> str | None:
    token = data.get("color_token")
    if token:
        return fixture_color(token)
    return data.get("color")


def _layer_id_or_none(value: str | None) -> LayerId | None:
    return LayerId(value) if value else None


def _take_id_or_none(value: str | None) -> TakeId | None:
    return TakeId(value) if value else None
