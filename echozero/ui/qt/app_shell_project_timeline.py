"""Project timeline baseline builders for the Qt app shell.
Exists to derive canonical timeline state from ProjectStorage records.
Connects runtime startup/refresh flows to presentation overlays and waveform registration.
"""

from __future__ import annotations

from dataclasses import replace

from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import (
    LayerId,
    ObjectContentId,
    ObjectRevisionId,
    SongId,
    SongVersionId,
    TakeId,
    TimelineId,
    TimelineObjectId,
)
from echozero.application.timeline.models import (
    Layer,
    LayerPresentationHints,
    Take,
    Timeline,
    derive_section_cues_from_layers,
)
from echozero.application.timeline.object_content_persistence import (
    imported_song_content_id,
    imported_song_object_id,
    imported_song_revision_id,
    require_source_ref,
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
    VideoPresentationFields,
)
from echozero.persistence.video import resolve_project_video_path
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
    source_ref = require_source_ref(
        project_storage,
        object_id=imported_song_object_id(version.id),
        content_id=imported_song_content_id(version.id),
        revision_id=imported_song_revision_id(version.audio_hash),
        role="imported_song_audio",
        locator=str(source_audio_path),
    )
    source_object_id = source_ref.object_id
    source_content_id = source_ref.content_id
    source_revision_id = source_ref.revision_id
    source_layer_id = LayerId(f"layer_song_{version.id}")
    source_take_id = TakeId(f"take_source_{version.id}")
    layers: list[Layer] = [
        Layer(
            id=source_layer_id,
            timeline_id=timeline_id,
            name=active_song.title,
            kind=LayerKind.AUDIO,
            order_index=0,
            object_id=source_object_id,
            main_content_id=source_content_id,
            main_revision_id=source_revision_id,
            takes=[
                Take(
                    id=source_take_id,
                    layer_id=source_layer_id,
                    name="Main",
                    source_ref="Imported track",
                    object_id=source_object_id,
                    content_id=source_content_id,
                    revision_id=source_revision_id,
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
    layer_video: dict[LayerId, VideoPresentationFields] = {}
    video_attachment = project_storage.song_video_attachments.get_by_song(active_song.id)
    if video_attachment is not None:
        video_layer_id = LayerId(f"layer_video_{active_song.id}")
        video_layer = Layer(
            id=video_layer_id,
            timeline_id=timeline_id,
            name="Video Reference",
            kind=LayerKind.REFERENCE,
            order_index=1,
            presentation_hints=LayerPresentationHints(
                color=TIMELINE_STYLE.fixture.layer_color_tokens.get("reference", "#9a948c"),
            ),
        )
        layers.append(video_layer)
        placement = project_storage.song_video_placements.get(version.id)
        video_start_seconds = 0.0 if placement is None else float(placement.video_start_seconds)
        video_path = resolve_project_video_path(
            project_storage.working_dir,
            video_attachment.video_file,
        )
        if video_attachment.extracted_audio_file:
            video_audio_path = resolve_project_audio_path(
                project_storage,
                video_attachment.extracted_audio_file,
            )
            video_waveform_key = ensure_registered_waveform(
                f"video-audio-{video_attachment.id}-{video_attachment.extracted_audio_hash or video_attachment.video_hash}",
                video_audio_path,
            )
            layer_audio[video_layer_id] = AudioPresentationFields(
                waveform_key=video_waveform_key,
                source_audio_path=str(video_audio_path),
                playback_source_ref=None,
            )
        layer_video[video_layer_id] = VideoPresentationFields(
            video_path=str(video_path),
            video_start_seconds=video_start_seconds,
            video_duration_seconds=float(video_attachment.duration_seconds),
        )
    for layer_record in project_storage.layers.list_by_version(version.id):
        layer, layer_fields, take_fields = build_storage_layer(
            project_storage, timeline_id, layer_record
        )
        if layer is not None:
            if video_attachment is not None:
                layer.order_index += 2
            layers.append(layer)
            layer_audio[layer.id] = layer_fields
            take_audio.update(take_fields)
    _expand_synthetic_source_layer_groups(layers)
    timeline = Timeline(
        id=timeline_id,
        song_version_id=SongVersionId(version.id),
        end=version.duration_seconds,
        layers=layers,
        section_cues=derive_section_cues_from_layers(layers),
    )
    timeline.selection.selected_layer_id = source_layer_id
    timeline.selection.selected_layer_ids = [source_layer_id]

    return (
        timeline,
        TimelinePresentationOverlay(
            project_title=project.name,
            end_time_label=format_time(version.duration_seconds),
            bpm=version.bpm if version.bpm is not None else project.settings.bpm,
            bpm_confidence=version.bpm_confidence,
            beat_anchor_seconds=version.beat_anchor_seconds,
            active_song_id=active_song.id,
            active_song_title=active_song.title,
            active_song_version_id=version.id,
            active_song_version_label=version.label,
            active_song_version_ma3_timecode_pool_no=version.ma3_timecode_pool_no,
            available_songs=selection.available_songs,
            available_song_versions=selection.available_song_versions,
            layer_audio=layer_audio,
            take_audio=take_audio,
            layer_video=layer_video,
        ),
        SongId(active_song.id),
        SongVersionId(version.id),
    )


def _expand_synthetic_source_layer_groups(layers: list[Layer]) -> None:
    """Default imported song source rows open when generated child layers exist."""

    parent_layer_ids = {
        layer.parent_layer_id for layer in layers if layer.parent_layer_id is not None
    }
    for layer in layers:
        if layer.id in parent_layer_ids and str(layer.id).startswith("layer_song_"):
            layer.presentation_hints.expanded = True


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
