"""Project timeline baseline builders for the Qt app shell.
Exists to derive canonical timeline state from ProjectStorage records.
Connects runtime startup/refresh flows to presentation overlays and waveform registration.
"""

from __future__ import annotations

from dataclasses import replace

from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import (
    LayerId,
    RegionId,
    SongId,
    SongVersionId,
    TakeId,
    TimelineId,
)
from echozero.application.timeline.models import (
    Layer,
    LayerPresentationHints,
    Take,
    Timeline,
    TimelineRegion,
)
from echozero.persistence.session import ProjectStorage
from echozero.ui.qt.app_shell_project_timeline_overlay import (
    apply_timeline_presentation_overlay,
    available_song_options,
    available_song_version_options,
    empty_overlay,
    format_time,
    layer_badges,
)
from echozero.ui.qt.app_shell_project_timeline_selection import (
    resolve_project_timeline_selection,
)
from echozero.ui.qt.app_shell_project_timeline_storage import (
    audio_presentation_fields,
    build_storage_layer,
    ensure_registered_waveform,
    event_label,
    events_from_take,
    resolve_project_audio_path,
    resolve_projected_event_id,
    resolve_storage_layer_kind,
    source_ref,
    take_kind,
)
from echozero.ui.qt.app_shell_project_timeline_types import (
    AudioPresentationFields,
    TimelinePresentationOverlay,
)
from echozero.ui.qt.timeline.style import TIMELINE_STYLE


def build_project_native_baseline_timeline(
    project_storage: ProjectStorage,
    *,
    active_song_id: SongId | None = None,
    active_song_version_id: SongVersionId | None = None,
) -> tuple[Timeline, TimelinePresentationOverlay, SongId | None, SongVersionId | None]:
    """Build the canonical runtime timeline and overlay from the active project state."""

    project = project_storage.project
    selection = resolve_project_timeline_selection(
        project_storage,
        active_song_id=active_song_id,
        active_song_version_id=active_song_version_id,
    )
    active_song = selection.active_song
    version = selection.active_version

    if active_song is None:
        return (
            build_empty_project_timeline(project_storage),
            empty_overlay(
                project.name,
                available_songs=selection.available_songs,
            ),
            None,
            None,
        )

    if version is None:
        return (
            build_empty_project_timeline(project_storage),
            empty_overlay(
                project.name,
                active_song_id=active_song.id,
                active_song_title=active_song.title,
                available_songs=selection.available_songs,
                available_song_versions=selection.available_song_versions,
            ),
            SongId(active_song.id),
            None,
        )

    timeline_id = TimelineId(f"timeline_{project.id}")
    source_audio_path = resolve_project_audio_path(project_storage, version.audio_file)
    waveform_key = ensure_registered_waveform(f"song-{version.id}", source_audio_path)
    source_layer_id = LayerId("source_audio")
    source_take_id = TakeId(f"take_source_{version.id}")
    layers: list[Layer] = [
        Layer(
            id=source_layer_id,
            timeline_id=timeline_id,
            name=active_song.title,
            kind=LayerKind.AUDIO,
            order_index=0,
            takes=[
                Take(
                    id=source_take_id,
                    layer_id=source_layer_id,
                    name="Main",
                    source_ref="Imported track",
                )
            ],
            playback=replace(
                Layer(
                    id=source_layer_id,
                    timeline_id=timeline_id,
                    name="",
                    kind=LayerKind.AUDIO,
                    order_index=0,
                ).playback,
                armed_source_ref=str(source_audio_path),
            ),
            presentation_hints=LayerPresentationHints(
                color=TIMELINE_STYLE.fixture.layer_color_tokens.get("song"),
            ),
        )
    ]
    layer_audio: dict[LayerId, AudioPresentationFields] = {
        source_layer_id: AudioPresentationFields(
            waveform_key=waveform_key,
            source_audio_path=str(source_audio_path),
            playback_source_ref=str(source_audio_path),
        )
    }
    take_audio: dict[TakeId, AudioPresentationFields] = {
        source_take_id: AudioPresentationFields(
            waveform_key=waveform_key,
            source_audio_path=str(source_audio_path),
            playback_source_ref=str(source_audio_path),
        )
    }
    for layer_record in project_storage.layers.list_by_version(version.id):
        layer, layer_fields, take_fields = build_storage_layer(
            project_storage, timeline_id, layer_record
        )
        if layer is not None:
            layers.append(layer)
            layer_audio[layer.id] = layer_fields
            take_audio.update(take_fields)
    regions = [
        TimelineRegion(
            id=RegionId(region.id),
            start=float(region.start_seconds),
            end=float(region.end_seconds),
            label=region.label,
            color=region.color,
            order_index=int(region.order_index),
            kind=region.kind,
        )
        for region in project_storage.timeline_regions.list_by_version(version.id)
    ]

    timeline = Timeline(
        id=timeline_id,
        song_version_id=SongVersionId(version.id),
        end=version.duration_seconds,
        layers=layers,
        regions=regions,
    )
    timeline.selection.selected_layer_id = source_layer_id
    timeline.selection.selected_layer_ids = [source_layer_id]
    timeline.playback_target.layer_id = source_layer_id

    return (
        timeline,
        TimelinePresentationOverlay(
            project_title=project.name,
            end_time_label=format_time(version.duration_seconds),
            bpm=project.settings.bpm,
            active_song_id=active_song.id,
            active_song_title=active_song.title,
            active_song_version_id=version.id,
            active_song_version_label=version.label,
            active_song_version_ma3_timecode_pool_no=version.ma3_timecode_pool_no,
            available_songs=selection.available_songs,
            available_song_versions=selection.available_song_versions,
            layer_audio=layer_audio,
            take_audio=take_audio,
        ),
        SongId(active_song.id),
        SongVersionId(version.id),
    )


def build_empty_project_timeline(project_storage: ProjectStorage) -> Timeline:
    """Build an empty timeline for projects that do not yet have an active song version."""

    project = project_storage.project
    timeline_id = TimelineId(f"timeline_{project.id}")
    return Timeline(
        id=timeline_id,
        song_version_id=SongVersionId("song_version_empty"),
        layers=[],
    )


__all__ = [
    "AudioPresentationFields",
    "TimelinePresentationOverlay",
    "apply_timeline_presentation_overlay",
    "audio_presentation_fields",
    "available_song_options",
    "available_song_version_options",
    "build_empty_project_timeline",
    "build_project_native_baseline_timeline",
    "build_storage_layer",
    "empty_overlay",
    "ensure_registered_waveform",
    "event_label",
    "events_from_take",
    "format_time",
    "layer_badges",
    "resolve_project_audio_path",
    "resolve_projected_event_id",
    "resolve_storage_layer_kind",
    "source_ref",
    "take_kind",
]
