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
from echozero.application.timeline.video_placement import VideoPlacement
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
    storage_layer_records = project_storage.layers.list_by_version(version.id)
    video_layer_records = [
        layer_record
        for layer_record in storage_layer_records
        if _layer_record_is_video_reference(layer_record)
    ]
    video_attachment = None
    if not video_layer_records:
        video_attachment = project_storage.song_video_attachments.get_by_song(active_song.id)
    if video_attachment is not None:
        video_layer_id = LayerId(f"layer_video_{active_song.id}")
        layers.append(
            Layer(
                id=video_layer_id,
                timeline_id=timeline_id,
                name="Video Reference",
                kind=LayerKind.REFERENCE,
                order_index=1,
                presentation_hints=LayerPresentationHints(
                    color=TIMELINE_STYLE.fixture.layer_color_tokens.get(
                        "reference",
                        "#9a948c",
                    ),
                ),
            )
        )
        placement = project_storage.song_video_placements.get(version.id)
        video_start_seconds = 0.0 if placement is None else float(placement.video_start_seconds)
        video_trim_start_seconds = (
            0.0 if placement is None else float(placement.video_trim_start_seconds)
        )
        video_loop_enabled = False if placement is None else bool(placement.video_loop_enabled)
        video_placement = VideoPlacement(
            start_seconds=video_start_seconds,
            trim_start_seconds=video_trim_start_seconds,
            visible_duration_seconds=(
                0.0
                if placement is None or placement.video_visible_duration_seconds is None
                else float(placement.video_visible_duration_seconds)
            ),
            source_duration_seconds=float(video_attachment.duration_seconds),
            loop_enabled=video_loop_enabled,
        ).normalized()
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
                "video-audio-"
                f"{video_attachment.id}-"
                f"{video_attachment.extracted_audio_hash or video_attachment.video_hash}",
                video_audio_path,
            )
            layer_audio[video_layer_id] = AudioPresentationFields(
                waveform_key=video_waveform_key,
                source_audio_path=str(video_audio_path),
                playback_source_ref=None,
            )
        layer_video[video_layer_id] = VideoPresentationFields(
            video_path=str(video_path),
            video_start_seconds=video_placement.start_seconds,
            video_trim_start_seconds=video_placement.trim_start_seconds,
            video_duration_seconds=float(video_attachment.duration_seconds),
            video_visible_duration_seconds=video_placement.visible_duration_seconds,
            video_loop_enabled=video_placement.loop_enabled,
        )
    for layer_record in storage_layer_records:
        layer, layer_fields, take_fields = build_storage_layer(
            project_storage, timeline_id, layer_record
        )
        if layer is not None:
            if video_attachment is not None:
                layer.order_index += 2
            layers.append(layer)
            layer_audio[layer.id] = layer_fields
            take_audio.update(take_fields)
            video_fields, video_audio_fields = _video_fields_for_storage_layer(
                project_storage,
                layer_id=layer.id,
                object_id=layer.object_id,
            )
            if video_fields is not None:
                layer_video[layer.id] = video_fields
            if video_audio_fields is not None:
                layer_audio[layer.id] = video_audio_fields
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


def _layer_record_is_video_reference(layer_record) -> bool:
    state_flags = getattr(layer_record, "state_flags", {}) or {}
    return (
        str(state_flags.get("reference_kind") or "").strip().lower() == "video"
        or bool(state_flags.get("package_video_layer"))
    )


def _video_fields_for_storage_layer(
    project_storage: ProjectStorage,
    *,
    layer_id: LayerId,
    object_id: TimelineObjectId | None,
) -> tuple[VideoPresentationFields | None, AudioPresentationFields | None]:
    if object_id is None:
        return None, None
    object_record = project_storage.timeline_objects.get(str(object_id))
    if object_record is None:
        return None, None
    content = project_storage.object_contents.get(object_record.main_content_id)
    if content is None or content.content_kind != "video_clip":
        return None, None
    payload = content.payload if isinstance(content.payload, dict) else {}
    video_file = str(payload.get("video_file") or "").strip()
    if not video_file:
        return None, None
    duration_seconds = float(payload.get("duration_seconds") or 0.0)
    placement = VideoPlacement(
        start_seconds=float(payload.get("video_start_seconds") or 0.0),
        trim_start_seconds=float(payload.get("video_trim_start_seconds") or 0.0),
        visible_duration_seconds=float(payload.get("video_visible_duration_seconds") or 0.0),
        source_duration_seconds=duration_seconds,
        loop_enabled=bool(payload.get("video_loop_enabled", False)),
    ).normalized()
    video_path = resolve_project_video_path(project_storage.working_dir, video_file)
    audio_fields: AudioPresentationFields | None = None
    extracted_audio_file = str(payload.get("extracted_audio_file") or "").strip()
    if extracted_audio_file:
        video_audio_path = resolve_project_audio_path(project_storage, extracted_audio_file)
        waveform_key = ensure_registered_waveform(
            "video-layer-audio-"
            f"{layer_id}-{payload.get('extracted_audio_hash') or content.revision_id}",
            video_audio_path,
        )
        audio_fields = AudioPresentationFields(
            waveform_key=waveform_key,
            source_audio_path=str(video_audio_path),
            playback_source_ref=None,
        )
    return (
        VideoPresentationFields(
            video_path=str(video_path),
            video_start_seconds=placement.start_seconds,
            video_trim_start_seconds=placement.trim_start_seconds,
            video_duration_seconds=duration_seconds,
            video_visible_duration_seconds=placement.visible_duration_seconds,
            video_loop_enabled=placement.loop_enabled,
        ),
        audio_fields,
    )


__all__ = [
    "AudioPresentationFields",
    "TimelinePresentationOverlay",
    "VideoPresentationFields",
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
