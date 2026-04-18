"""Project timeline baseline builders for the Qt app shell.
Exists to derive canonical timeline state from ProjectStorage records.
Connects runtime startup/refresh flows to presentation overlays and waveform registration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, SongId, SongVersionId, TakeId, TimelineId
from echozero.application.timeline.models import Event, Layer, LayerPresentationHints, LayerProvenance, LayerStatus, Take, Timeline
from echozero.domain.types import AudioData, EventData, Event as DomainEvent
from echozero.persistence.session import ProjectStorage
from echozero.ui.qt.timeline.style import TIMELINE_STYLE
from echozero.ui.qt.timeline.waveform_cache import get_cached_waveform, register_waveform_from_audio_file


@dataclass(slots=True)
class AudioPresentationFields:
    """Audio-specific presentation fields layered onto timeline rows and takes."""

    waveform_key: str | None = None
    source_audio_path: str | None = None
    playback_source_ref: str | None = None


@dataclass(slots=True)
class TimelinePresentationOverlay:
    """Presentation overlay fields applied after baseline timeline assembly."""

    project_title: str
    end_time_label: str
    layer_audio: dict[LayerId, AudioPresentationFields]
    take_audio: dict[TakeId, AudioPresentationFields]


def build_project_native_baseline_timeline(
    project_storage: ProjectStorage,
    *,
    active_song_id: SongId | None = None,
    active_song_version_id: SongVersionId | None = None,
) -> tuple[Timeline, TimelinePresentationOverlay, SongId | None, SongVersionId | None]:
    """Build the canonical runtime timeline and overlay from the active project state."""

    project = project_storage.project
    songs = project_storage.songs.list_by_project(project.id)
    requested_song_id = str(active_song_id) if active_song_id is not None else None
    requested_version_id = str(active_song_version_id) if active_song_version_id is not None else None

    active_song = None
    active_version = None
    if requested_version_id is not None:
        active_version = project_storage.song_versions.get(requested_version_id)
        if active_version is not None:
            active_song = project_storage.songs.get(active_version.song_id)
    if active_song is None and requested_song_id is not None:
        active_song = next((song for song in songs if song.id == requested_song_id), None)
    if active_song is None:
        active_song = next((song for song in songs if song.active_version_id), None)
    if active_song is not None and active_version is None and active_song.active_version_id is not None:
        active_version = project_storage.song_versions.get(active_song.active_version_id)

    if active_song is None:
        return build_empty_project_timeline(project_storage), empty_overlay(project.name), None, None

    version = active_version
    if version is None and active_song.active_version_id is not None:
        version = project_storage.song_versions.get(active_song.active_version_id)
    if version is None:
        return (
            build_empty_project_timeline(project_storage),
            empty_overlay(project.name),
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
                Layer(id=source_layer_id, timeline_id=timeline_id, name="", kind=LayerKind.AUDIO, order_index=0).playback,
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
        layer, layer_fields, take_fields = build_storage_layer(project_storage, timeline_id, layer_record)
        if layer is not None:
            layers.append(layer)
            layer_audio[layer.id] = layer_fields
            take_audio.update(take_fields)

    timeline = Timeline(
        id=timeline_id,
        song_version_id=SongVersionId(version.id),
        end=version.duration_seconds,
        layers=layers,
    )
    timeline.selection.selected_layer_id = source_layer_id
    timeline.selection.selected_layer_ids = [source_layer_id]
    timeline.playback_target.layer_id = source_layer_id

    return (
        timeline,
        TimelinePresentationOverlay(
            project_title=project.name,
            end_time_label=format_time(version.duration_seconds),
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


def resolve_project_audio_path(project_storage: ProjectStorage, audio_file: str) -> Path:
    """Resolve a stored project audio reference to an absolute local path."""

    raw_path = Path(audio_file)
    if raw_path.is_absolute():
        return raw_path
    return (project_storage.working_dir / raw_path).resolve()


def ensure_registered_waveform(key: str, audio_path: Path) -> str | None:
    """Register and return the waveform cache key when the audio file exists."""

    if not audio_path.exists():
        return None
    if get_cached_waveform(key) is None:
        register_waveform_from_audio_file(key, audio_path)
    return key


def audio_presentation_fields(project_storage: ProjectStorage, take) -> AudioPresentationFields:
    """Build audio presentation fields for one take when it carries audio data."""

    if not isinstance(take.data, AudioData):
        return AudioPresentationFields()
    audio_path = resolve_project_audio_path(project_storage, take.data.file_path)
    waveform_key = ensure_registered_waveform(f"take-{take.id}", audio_path)
    return AudioPresentationFields(
        waveform_key=waveform_key,
        source_audio_path=str(audio_path),
        playback_source_ref=str(audio_path),
    )


def build_storage_layer(
    project_storage: ProjectStorage,
    timeline_id: TimelineId,
    layer_record,
) -> tuple[Layer | None, AudioPresentationFields, dict[TakeId, AudioPresentationFields]]:
    """Convert one stored layer record plus takes into runtime timeline structures."""

    takes = project_storage.takes.list_by_layer(layer_record.id)
    if not takes:
        return None, AudioPresentationFields(), {}
    main_take = next((take for take in takes if take.is_main), takes[0])
    main_kind = take_kind(main_take)
    main_audio = audio_presentation_fields(project_storage, main_take)
    take_audio: dict[TakeId, AudioPresentationFields] = {}
    timeline_takes: list[Take] = []
    for take in takes:
        take_id = TakeId(str(take.id))
        timeline_takes.append(
            Take(
                id=take_id,
                layer_id=LayerId(str(layer_record.id)),
                name=take.label,
                events=events_from_take(take),
                source_ref=source_ref(take.source),
            )
        )
        take_audio[take_id] = audio_presentation_fields(project_storage, take)

    layer_id = LayerId(str(layer_record.id))
    source_pipeline = layer_record.source_pipeline or {}
    provenance = layer_record.provenance or {}
    layer = Layer(
        id=layer_id,
        timeline_id=timeline_id,
        name=layer_record.name.title(),
        kind=main_kind,
        order_index=int(layer_record.order) + 1,
        takes=timeline_takes,
        playback=replace(
            Layer(id=layer_id, timeline_id=timeline_id, name="", kind=main_kind, order_index=0).playback,
            armed_source_ref=main_audio.playback_source_ref,
        ),
        status=LayerStatus(
            stale=bool(layer_record.state_flags.get("stale", False)),
            manually_modified=bool(layer_record.state_flags.get("manually_modified", False)),
            stale_reason=layer_record.state_flags.get("stale_reason"),
        ),
        provenance=LayerProvenance(
            source_layer_id=LayerId(str(provenance["source_layer_id"])) if provenance.get("source_layer_id") else None,
            source_song_version_id=SongVersionId(str(provenance["source_song_version_id"])) if provenance.get("source_song_version_id") else None,
            source_run_id=provenance.get("source_run_id"),
            pipeline_id=source_pipeline.get("pipeline_id") or provenance.get("pipeline_id"),
            output_name=source_pipeline.get("output_name") or provenance.get("output_name"),
        ),
        presentation_hints=LayerPresentationHints(
            visible=bool(layer_record.visible),
            locked=bool(layer_record.locked),
            expanded=len(timeline_takes) > 1,
            color=layer_record.color,
        ),
    )
    return layer, main_audio, take_audio


def take_kind(take) -> LayerKind:
    """Resolve the runtime layer kind from one stored take payload."""

    if isinstance(take.data, EventData):
        return LayerKind.EVENT
    return LayerKind.AUDIO


def events_from_take(take) -> list[Event]:
    """Project event-data takes into ordered timeline events."""

    if not isinstance(take.data, EventData):
        return []
    events: list[DomainEvent] = []
    for layer in take.data.layers:
        events.extend(layer.events)
    events.sort(key=lambda event: (event.time, event.duration, str(event.id)))
    return [
        Event(
            id=EventId(str(event.id)),
            take_id=TakeId(str(take.id)),
            start=float(event.time),
            end=float(event.time + max(event.duration, 0.08)),
            label=event_label(event),
        )
        for event in events
    ]


def event_label(event: DomainEvent) -> str:
    """Choose the most useful display label for one domain event."""

    if isinstance(event.classifications, dict) and event.classifications:
        first_key = next(iter(event.classifications.keys()))
        value = event.classifications.get(first_key)
        if isinstance(value, str) and value.strip():
            return value.strip().title()
        if isinstance(first_key, str) and first_key.strip():
            return first_key.strip().replace("_", " ").title()
    return "Onset"


def source_ref(source) -> str | None:
    """Format the provenance source reference for one generated take."""

    if source is None:
        return None
    run_id = getattr(source, "run_id", "")
    block_type = getattr(source, "block_type", "")
    if run_id and block_type:
        return f"{block_type}:{str(run_id)[:8]}"
    if run_id:
        return str(run_id)
    return None


def empty_overlay(project_title: str) -> TimelinePresentationOverlay:
    """Build the empty-state presentation overlay for a project shell."""

    return TimelinePresentationOverlay(
        project_title=project_title,
        end_time_label="00:00.00",
        layer_audio={},
        take_audio={},
    )


def apply_timeline_presentation_overlay(
    presentation: TimelinePresentation,
    *,
    overlay: TimelinePresentationOverlay,
) -> TimelinePresentation:
    """Apply audio and waveform overlay fields onto a presentation snapshot."""

    layers: list[LayerPresentation] = []
    for layer in presentation.layers:
        layer_fields = overlay.layer_audio.get(layer.layer_id, AudioPresentationFields())
        takes = [
            replace(
                take,
                waveform_key=overlay.take_audio.get(take.take_id, AudioPresentationFields()).waveform_key,
                source_audio_path=overlay.take_audio.get(take.take_id, AudioPresentationFields()).source_audio_path,
                playback_source_ref=overlay.take_audio.get(take.take_id, AudioPresentationFields()).playback_source_ref,
            )
            for take in layer.takes
        ]
        layers.append(
            replace(
                layer,
                badges=layer_badges(layer.title, layer.kind),
                waveform_key=layer_fields.waveform_key,
                source_audio_path=layer_fields.source_audio_path,
                playback_source_ref=layer_fields.playback_source_ref,
                takes=takes,
            )
        )

    return replace(
        presentation,
        title=overlay.project_title,
        current_time_label=format_time(presentation.playhead),
        end_time_label=overlay.end_time_label,
        layers=layers,
    )


def layer_badges(name: str, kind: LayerKind) -> list[str]:
    """Compute default layer badges for runtime presentation."""

    badges = ["main", kind.value]
    if kind is LayerKind.AUDIO and name.strip().lower() != "imported song":
        badges.append("stem")
    if "drum" in name.strip().lower():
        badges.append("drums")
    return badges


def format_time(seconds: float) -> str:
    """Format seconds for the timeline shell time labels."""

    mins = int(seconds // 60)
    secs = seconds - (mins * 60)
    return f"{mins:02d}:{secs:05.2f}"
