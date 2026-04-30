from __future__ import annotations

from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    LayerStatusPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, SongVersionId, TakeId, TimelineId
from echozero.application.timeline.models import EventRef, Timeline
from echozero.ui.qt.app_shell_timeline_state import (
    clear_selected_events,
    resolve_event_clip_preview,
    restore_timeline_targets,
    surface_new_take_rows,
)


def test_restore_timeline_targets_preserves_layer_and_clears_event_selection():
    timeline = Timeline(
        id=TimelineId("timeline_runtime"),
        song_version_id=SongVersionId("song_version_runtime"),
    )
    timeline.selection.selected_event_ids = [EventId("stale_evt")]
    timeline.selection.selected_event_refs = [
        EventRef(
            layer_id=LayerId("layer_a"),
            take_id=TakeId("take_alt"),
            event_id=EventId("stale_evt"),
        )
    ]

    prior = _presentation(
        selected_layer_id=LayerId("layer_a"),
        selected_take_id=TakeId("take_alt"),
    )
    current = _presentation(
        selected_layer_id=LayerId("layer_b"),
        selected_take_id=TakeId("take_b_alt"),
    )

    restore_timeline_targets(
        timeline=timeline,
        prior_presentation=prior,
        current_presentation=current,
    )

    assert timeline.selection.selected_layer_id == LayerId("layer_a")
    assert timeline.selection.selected_layer_ids == [LayerId("layer_a")]
    assert timeline.selection.selected_take_id == TakeId("take_alt")
    assert timeline.selection.selected_event_ids == []
    assert timeline.selection.selected_event_refs == []


def test_surface_new_take_rows_prefers_selected_source_layer_and_clears_event_selection():
    timeline = Timeline(
        id=TimelineId("timeline_runtime"),
        song_version_id=SongVersionId("song_version_runtime"),
    )
    timeline.selection.selected_event_ids = [EventId("stale_evt")]
    timeline.selection.selected_event_refs = [
        EventRef(
            layer_id=LayerId("layer_a"),
            take_id=TakeId("take_alt"),
            event_id=EventId("stale_evt"),
        )
    ]

    prior = _presentation(selected_layer_id=LayerId("layer_a"))
    current = _presentation(
        selected_layer_id=LayerId("layer_a"),
        extra_layers=[
            LayerPresentation(
                layer_id=LayerId("layer_rendered"),
                title="Rendered",
                kind=LayerKind.EVENT,
                takes=[
                    TakeLanePresentation(
                        take_id=TakeId("take_new"),
                        name="Take 2",
                    )
                ],
                status=LayerStatusPresentation(source_layer_id="layer_a"),
            )
        ],
    )

    surface_new_take_rows(
        timeline=timeline,
        prior_presentation=prior,
        current_presentation=current,
    )

    assert timeline.selection.selected_layer_id == LayerId("layer_rendered")
    assert timeline.selection.selected_layer_ids == [LayerId("layer_rendered")]
    assert timeline.selection.selected_take_id == TakeId("take_new")
    assert timeline.selection.selected_event_ids == []
    assert timeline.selection.selected_event_refs == []


def test_resolve_event_clip_preview_falls_back_to_source_layer_audio():
    presentation = TimelinePresentation(
        timeline_id=TimelineId("timeline_preview"),
        title="Preview",
        layers=[
            LayerPresentation(
                layer_id=LayerId("source_audio"),
                title="Source",
                kind=LayerKind.AUDIO,
                source_audio_path="/tmp/source.wav",
                gain_db=0.0,
                status=LayerStatusPresentation(),
            ),
            LayerPresentation(
                layer_id=LayerId("layer_events"),
                title="Events",
                kind=LayerKind.EVENT,
                events=[
                    EventPresentation(
                        event_id=EventId("evt_1"),
                        start=1.25,
                        end=1.75,
                        label="Kick",
                    )
                ],
                status=LayerStatusPresentation(source_layer_id="source_audio"),
                gain_db=-3.0,
            ),
        ],
    )

    clip = resolve_event_clip_preview(
        presentation,
        layer_id=LayerId("layer_events"),
        take_id=None,
        event_id=EventId("evt_1"),
    )

    assert clip.source_ref == "/tmp/source.wav"
    assert clip.start_seconds == 1.25
    assert clip.end_seconds == 1.75
    assert clip.gain_db == -3.0


def test_clear_selected_events_resets_ids_and_refs():
    timeline = Timeline(
        id=TimelineId("timeline_runtime"),
        song_version_id=SongVersionId("song_version_runtime"),
    )
    timeline.selection.selected_event_ids = [EventId("evt_1")]
    timeline.selection.selected_event_refs = [
        EventRef(
            layer_id=LayerId("layer_a"),
            take_id=TakeId("take_alt"),
            event_id=EventId("evt_1"),
        )
    ]

    clear_selected_events(timeline)

    assert timeline.selection.selected_event_ids == []
    assert timeline.selection.selected_event_refs == []


def _presentation(
    *,
    selected_layer_id: LayerId | None = None,
    selected_take_id: TakeId | None = None,
    extra_layers: list[LayerPresentation] | None = None,
) -> TimelinePresentation:
    layers = [
        LayerPresentation(
            layer_id=LayerId("layer_a"),
            title="Layer A",
            main_take_id=TakeId("take_main"),
            kind=LayerKind.EVENT,
            takes=[
                TakeLanePresentation(
                    take_id=TakeId("take_alt"),
                    name="Take Alt",
                )
            ],
            status=LayerStatusPresentation(),
        ),
        LayerPresentation(
            layer_id=LayerId("layer_b"),
            title="Layer B",
            main_take_id=TakeId("take_b_main"),
            kind=LayerKind.EVENT,
            takes=[
                TakeLanePresentation(
                    take_id=TakeId("take_b_alt"),
                    name="Take B Alt",
                )
            ],
            status=LayerStatusPresentation(),
        ),
    ]
    if extra_layers:
        layers.extend(extra_layers)
    return TimelinePresentation(
        timeline_id=TimelineId("timeline_runtime"),
        title="Runtime",
        layers=layers,
        selected_layer_id=selected_layer_id,
        selected_layer_ids=[selected_layer_id] if selected_layer_id is not None else [],
        selected_take_id=selected_take_id,
    )
