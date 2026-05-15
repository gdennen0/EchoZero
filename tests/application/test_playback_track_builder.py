"""Playback-track planner coverage for output-bus normalization."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from echozero.application.playback.tracks import PlaybackTrackBuilder
from echozero.application.shared.ids import ObjectContentId, ObjectRevisionId, TimelineObjectId
from echozero.application.shared.enums import PlaybackMode
from echozero.application.timeline.object_content import SourceRef


def _presentation(*, output_bus: str | None, playback_output_channels: int) -> object:
    layer = SimpleNamespace(
        layer_id="layer_song",
        title="Song",
        source_audio_path="song.wav",
        output_bus=output_bus,
        muted=False,
        soloed=False,
    )
    return SimpleNamespace(
        layers=[layer],
        selected_layer_id="layer_song",
        selected_take_id=None,
        playback_output_channels=playback_output_channels,
    )


def test_playback_track_builder_preserves_wide_output_bus_when_device_supports_it() -> None:
    builder = PlaybackTrackBuilder(
        lambda _path: (np.array([0.25, -0.25], dtype=np.float32), 44100)
    )
    presentation = _presentation(output_bus="outputs_1_4", playback_output_channels=4)

    plan = builder.build_track_plan(presentation)

    assert len(plan.tracks) == 1
    assert plan.tracks[0].output_bus == "outputs_1_4"


def test_playback_track_builder_prunes_output_bus_that_exceeds_device_channels() -> None:
    builder = PlaybackTrackBuilder(
        lambda _path: (np.array([0.25, -0.25], dtype=np.float32), 44100)
    )
    presentation = _presentation(
        output_bus="outputs_1_1,outputs_7_8",
        playback_output_channels=4,
    )

    plan = builder.build_track_plan(presentation)

    assert len(plan.tracks) == 1
    assert plan.tracks[0].output_bus == "outputs_1_1"


def test_playback_track_builder_disables_route_when_all_outputs_exceed_device_channels() -> None:
    builder = PlaybackTrackBuilder(
        lambda _path: (np.array([0.25, -0.25], dtype=np.float32), 44100)
    )
    presentation = _presentation(output_bus="outputs_7_8", playback_output_channels=4)

    plan = builder.build_track_plan(presentation)

    assert len(plan.tracks) == 1
    assert plan.tracks[0].output_bus == "none"


def test_playback_track_builder_ignores_event_layer_source_audio_when_event_playback_disabled() -> (
    None
):
    builder = PlaybackTrackBuilder(
        lambda _path: (np.array([0.25, -0.25], dtype=np.float32), 44100)
    )
    presentation = SimpleNamespace(
        layers=[
            SimpleNamespace(
                layer_id="layer_event",
                title="Kick",
                kind="event",
                source_audio_path="drums.wav",
                playback_enabled=False,
                playback_mode=PlaybackMode.NONE,
                playback_source_ref="drums.wav",
                events=[],
                output_bus=None,
                muted=False,
                soloed=False,
                takes=[],
            )
        ],
        selected_layer_id="layer_event",
        selected_take_id=None,
        playback_output_channels=2,
    )

    plan = builder.build_track_plan(presentation)

    assert len(plan.tracks) == 0


def test_playback_track_builder_uses_event_slice_mode_for_event_layers() -> None:
    builder = PlaybackTrackBuilder(
        lambda _path: (np.array([0.25, -0.25], dtype=np.float32), 44100)
    )
    presentation = SimpleNamespace(
        layers=[
            SimpleNamespace(
                layer_id="layer_event",
                title="Kick",
                kind="event",
                source_audio_path="drums.wav",
                playback_enabled=True,
                playback_mode=PlaybackMode.EVENT_SLICE,
                playback_source_ref="drums.wav",
                events=[SimpleNamespace(start=0.0, muted=False, badges=())],
                output_bus=None,
                muted=False,
                soloed=False,
                takes=[],
            )
        ],
        selected_layer_id="layer_event",
        selected_take_id=None,
        playback_output_channels=2,
    )

    plan = builder.build_track_plan(presentation)

    assert len(plan.tracks) == 1
    assert plan.tracks[0].source_key.startswith("event:")


def test_playback_track_builder_uses_object_source_ref_without_legacy_audio_path() -> None:
    observed_paths: list[str] = []

    def _load_audio(path: str):
        observed_paths.append(path)
        return np.array([0.25, -0.25], dtype=np.float32), 44100

    builder = PlaybackTrackBuilder(_load_audio)
    presentation = SimpleNamespace(
        layers=[
            SimpleNamespace(
                layer_id="layer_event",
                title="Kick",
                kind="event",
                source_audio_path=None,
                source_content_ref=SourceRef(
                    object_id=TimelineObjectId("object_song_version"),
                    content_id=ObjectContentId("content_song_audio_version"),
                    revision_id=ObjectRevisionId("revision_song_audio_hash"),
                    role="imported_song_audio",
                    locator="song.wav",
                ),
                playback_enabled=True,
                playback_mode=PlaybackMode.EVENT_SLICE,
                playback_source_ref=None,
                events=[SimpleNamespace(start=0.0, muted=False, badges=())],
                output_bus=None,
                muted=False,
                soloed=False,
                takes=[],
            )
        ],
        selected_layer_id="layer_event",
        selected_take_id=None,
        playback_output_channels=2,
    )

    plan = builder.build_track_plan(presentation)

    assert len(plan.tracks) == 1
    assert plan.tracks[0].source_key.startswith("event:")
    assert observed_paths == ["song.wav"]


def test_playback_track_builder_event_slice_overlap_scales_to_prevent_hard_clip() -> None:
    builder = PlaybackTrackBuilder(lambda _path: (np.ones(1024, dtype=np.float32), 44100))
    presentation = SimpleNamespace(
        layers=[
            SimpleNamespace(
                layer_id="layer_event",
                title="Kick",
                kind="event",
                source_audio_path="drums.wav",
                playback_enabled=True,
                playback_mode=PlaybackMode.EVENT_SLICE,
                playback_source_ref="drums.wav",
                events=[
                    SimpleNamespace(start=0.0, muted=False, badges=()),
                    SimpleNamespace(start=0.0, muted=False, badges=()),
                ],
                output_bus=None,
                muted=False,
                soloed=False,
                takes=[],
            )
        ],
        selected_layer_id="layer_event",
        selected_take_id=None,
        playback_output_channels=2,
    )

    plan = builder.build_track_plan(presentation)

    assert len(plan.tracks) == 1
    rendered = plan.tracks[0].buffer
    assert rendered is not None
    assert float(np.max(np.abs(rendered))) <= 1.0
    assert float(np.max(np.abs(rendered))) >= 0.95


def test_playback_track_builder_event_slice_applies_boundary_fades_for_long_clips() -> None:
    builder = PlaybackTrackBuilder(lambda _path: (np.ones(2048, dtype=np.float32), 44100))
    presentation = SimpleNamespace(
        layers=[
            SimpleNamespace(
                layer_id="layer_event",
                title="Kick",
                kind="event",
                source_audio_path="drums.wav",
                playback_enabled=True,
                playback_mode=PlaybackMode.EVENT_SLICE,
                playback_source_ref="drums.wav",
                events=[SimpleNamespace(start=0.0, muted=False, badges=())],
                output_bus=None,
                muted=False,
                soloed=False,
                takes=[],
            )
        ],
        selected_layer_id="layer_event",
        selected_take_id=None,
        playback_output_channels=2,
    )

    plan = builder.build_track_plan(presentation)

    rendered = plan.tracks[0].buffer
    assert rendered is not None
    assert float(rendered[0]) < 0.5
    assert float(rendered[-1]) < 0.5
    assert float(rendered[len(rendered) // 2]) > 0.9


def test_playback_track_builder_event_slice_applies_boundary_fades_for_tiny_clips() -> None:
    source = np.ones(12, dtype=np.float32)
    builder = PlaybackTrackBuilder(lambda _path: (source, 44100))
    presentation = SimpleNamespace(
        layers=[
            SimpleNamespace(
                layer_id="layer_event",
                title="Tiny Click",
                kind="event",
                source_audio_path="tiny.wav",
                playback_enabled=True,
                playback_mode=PlaybackMode.EVENT_SLICE,
                playback_source_ref="tiny.wav",
                events=[SimpleNamespace(start=0.0, muted=False, badges=())],
                output_bus=None,
                muted=False,
                soloed=False,
                takes=[],
            )
        ],
        selected_layer_id="layer_event",
        selected_take_id=None,
        playback_output_channels=2,
    )

    plan = builder.build_track_plan(presentation)

    rendered = plan.tracks[0].buffer
    assert rendered is not None
    assert float(rendered[0]) == 0.0
    assert float(rendered[-1]) == 0.0
    assert float(np.max(rendered)) > 0.2
    np.testing.assert_array_equal(source, np.ones(12, dtype=np.float32))
