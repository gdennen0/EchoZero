"""Audio engine v2 planner parity harness coverage."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from echozero.application.audio_engine_v2.graph import HardwareOutputRoute, TrackRoute
from echozero.application.audio_engine_v2.parity import (
    GraphEditKind,
    ParityPlanningError,
    build_shadow_graph_from_playback_projection,
    build_shadow_graph_from_track_plan,
    classify_graph_edit,
)
from echozero.application.shared.enums import LayerKind, PlaybackMode


def _audio_loader(_path: str | Path) -> tuple[np.ndarray, int]:
    return np.array([0.25, -0.25], dtype=np.float32), 44100


def _audio_layer(
    layer_id: str,
    *,
    title: str | None = None,
    source_audio_path: str | None = None,
    output_bus: str | None = None,
    muted: bool = False,
    soloed: bool = False,
    gain_db: float = 0.0,
    takes: list[object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        layer_id=layer_id,
        title=title or layer_id,
        kind=LayerKind.AUDIO,
        main_take_id=None,
        source_audio_path=source_audio_path or f"{layer_id}.wav",
        playback_enabled=False,
        playback_mode=PlaybackMode.NONE,
        playback_source_ref=None,
        events=[],
        takes=takes or [],
        output_bus=output_bus,
        muted=muted,
        soloed=soloed,
        gain_db=gain_db,
    )


def _event_layer(
    layer_id: str,
    *,
    output_bus: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        layer_id=layer_id,
        title="Kick",
        kind=LayerKind.EVENT,
        main_take_id=None,
        source_audio_path="kick.wav",
        playback_enabled=True,
        playback_mode=PlaybackMode.EVENT_SLICE,
        playback_source_ref="kick.wav",
        events=[SimpleNamespace(start=0.0, muted=False, badges=())],
        takes=[],
        output_bus=output_bus,
        muted=False,
        soloed=False,
        gain_db=0.0,
    )


def _presentation(
    layers: list[object],
    *,
    selected_layer_id: str,
    selected_take_id: str | None = None,
    playback_output_channels: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        layers=layers,
        selected_layer_id=selected_layer_id,
        selected_take_id=selected_take_id,
        playback_output_channels=playback_output_channels,
    )


def test_shadow_parity_uses_current_app_projection_for_selected_song() -> None:
    runtime_projection = _presentation(
        [
            _audio_layer(
                "layer_song",
                title="Song",
                output_bus="master,outputs_3_3",
                gain_db=-3.0,
            )
        ],
        selected_layer_id="layer_song",
    )

    report = build_shadow_graph_from_playback_projection(
        runtime_projection,
        audio_loader=_audio_loader,
        graph_id="selected-song",
        master_output_bus="outputs_1_2",
    )

    assert len(report.graph.tracks) == 1
    assert report.graph.tracks[0].route == TrackRoute.to_master_and_hardware(
        (HardwareOutputRoute(3, 3),)
    )
    assert report.summary.tracks[0].v1_output_bus == "master,outputs_3_3"
    assert report.summary.tracks[0].v2_targets == ("master", "outputs_3_3")
    assert report.summary.tracks[0].gain_db == -3.0
    assert report.diagnostics == ()


def test_shadow_parity_ignores_selected_version_without_audition_intent() -> None:
    take = SimpleNamespace(
        take_id="version_b",
        name="Version B",
        source_audio_path="song-version-b.wav",
        playback_source_ref=None,
        events=[],
    )
    presentation = _presentation(
        [_audio_layer("layer_song", title="Song", takes=[take])],
        selected_layer_id="layer_song",
        selected_take_id="version_b",
    )

    report = build_shadow_graph_from_playback_projection(
        presentation,
        audio_loader=_audio_loader,
        graph_id="selected-version",
    )

    assert report.summary.structure_signature == (("layer_song", "audio:layer_song.wav"),)
    assert report.summary.tracks[0].name == "Song"


def test_shadow_parity_preserves_event_slice_route_summary() -> None:
    presentation = _presentation(
        [_event_layer("layer_event", output_bus="outputs_4_4")],
        selected_layer_id="layer_event",
    )

    report = build_shadow_graph_from_playback_projection(
        presentation,
        audio_loader=_audio_loader,
        graph_id="event-slice",
    )

    assert report.summary.tracks[0].source_key.startswith("event:kick.wav:")
    assert report.summary.tracks[0].mode == PlaybackMode.EVENT_SLICE.value
    assert report.summary.tracks[0].v2_targets == ("outputs_4_4",)


def test_shadow_parity_covers_mute_solo_gain_and_route_semantics() -> None:
    presentation = _presentation(
        [
            _audio_layer("muted", output_bus="none", muted=True, gain_db=-6.0),
            _audio_layer("solo", output_bus=None, soloed=True, gain_db=2.5),
            _audio_layer("direct", output_bus="outputs_3_3"),
            _audio_layer("mirrored", output_bus="master,outputs_4_4"),
        ],
        selected_layer_id="solo",
    )

    report = build_shadow_graph_from_playback_projection(
        presentation,
        audio_loader=_audio_loader,
        graph_id="mix-routes",
        master_output_bus="outputs_1_2",
    )
    summaries = {track.track_id: track for track in report.summary.tracks}

    assert summaries["muted"].v2_targets == ()
    assert summaries["muted"].muted is True
    assert summaries["muted"].gain_db == -6.0
    assert summaries["solo"].v2_targets == ("master",)
    assert summaries["solo"].soloed is True
    assert summaries["solo"].muted is False
    assert summaries["direct"].v2_targets == ("outputs_3_3",)
    assert summaries["direct"].muted is True
    assert summaries["mirrored"].v2_targets == ("master", "outputs_4_4")


def test_shadow_parity_summaries_are_deterministic_and_classify_edits() -> None:
    base = _presentation(
        [_audio_layer("song", output_bus=None, gain_db=0.0)],
        selected_layer_id="song",
    )
    mixed = _presentation(
        [_audio_layer("song", output_bus=None, gain_db=-1.0)],
        selected_layer_id="song",
    )
    routed = _presentation(
        [_audio_layer("song", output_bus="outputs_3_3", gain_db=0.0)],
        selected_layer_id="song",
    )
    structured = _presentation(
        [_audio_layer("song", source_audio_path="alternate.wav")],
        selected_layer_id="song",
    )

    first = build_shadow_graph_from_playback_projection(
        base,
        audio_loader=_audio_loader,
        graph_id="deterministic",
    )
    second = build_shadow_graph_from_playback_projection(
        base,
        audio_loader=_audio_loader,
        graph_id="deterministic",
    )
    mix_report = build_shadow_graph_from_playback_projection(
        mixed,
        audio_loader=_audio_loader,
        graph_id="deterministic",
    )
    route_report = build_shadow_graph_from_playback_projection(
        routed,
        audio_loader=_audio_loader,
        graph_id="deterministic",
    )
    structure_report = build_shadow_graph_from_playback_projection(
        structured,
        audio_loader=_audio_loader,
        graph_id="deterministic",
    )

    assert second.summary == first.summary
    assert classify_graph_edit(first.graph, second.graph).kind is GraphEditKind.UNCHANGED
    assert classify_graph_edit(first.graph, mix_report.graph).kind is GraphEditKind.MIX
    assert classify_graph_edit(first.graph, route_report.graph).kind is GraphEditKind.ROUTE
    assert classify_graph_edit(first.graph, structure_report.graph).kind is GraphEditKind.STRUCTURE


def test_shadow_parity_rejects_unknown_route_tokens_observably() -> None:
    plan = SimpleNamespace(
        tracks=(
            SimpleNamespace(
                track_id="song",
                name="Song",
                source_key="audio:song.wav",
                gain_db=0.0,
                muted=False,
                soloed=False,
                output_bus="not_a_real_bus",
                sample_rate=44100,
            ),
        )
    )

    with pytest.raises(ParityPlanningError, match="unsupported output route token"):
        build_shadow_graph_from_track_plan(plan, graph_id="bad-route")
