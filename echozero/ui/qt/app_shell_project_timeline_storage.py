"""Project timeline storage and waveform helpers for the Qt app shell.
Exists to keep project-record projection and audio registration out of the public root.
Connects persisted layers and takes to runtime timeline rows plus waveform overlays.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from echozero.application.mixer.models import LayerMixerState
from echozero.application.shared.cue_numbers import CueNumber, cue_number_text, parse_positive_cue_number
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import (
    EventId,
    LayerId,
    SectionCueId,
    SongVersionId,
    TakeId,
    TimelineId,
)
from echozero.application.timeline.models import (
    CueMetadata,
    Event,
    Layer,
    LayerPresentationHints,
    LayerProvenance,
    LayerStatus,
    LayerSyncState,
    SectionCue,
    Take,
)
from echozero.domain.types import AudioData
from echozero.domain.types import Event as DomainEvent
from echozero.domain.types import EventData
from echozero.persistence.session import ProjectStorage
from echozero.ui.qt.app_shell_layer_storage import (
    STATE_FLAG_MA3_TRACK_COORD,
    STATE_FLAG_OUTPUT_BUS,
)
from echozero.ui.qt.app_shell_project_timeline_types import AudioPresentationFields
from echozero.ui.qt.timeline.waveform_cache import (
    get_cached_waveform,
    register_waveform_from_audio_file,
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

    audio_file_path = _take_source_audio_file_path(take)
    if audio_file_path is None:
        return AudioPresentationFields()
    audio_path = resolve_project_audio_path(project_storage, audio_file_path)
    waveform_key = ensure_registered_waveform(f"take-{take.id}", audio_path)
    source_audio_path = str(audio_path)
    return AudioPresentationFields(
        waveform_key=waveform_key,
        source_audio_path=source_audio_path,
        playback_source_ref=source_audio_path,
    )


def _take_source_audio_file_path(take) -> str | None:
    if isinstance(take.data, AudioData):
        return str(take.data.file_path)
    source = getattr(take, "source", None)
    if source is None:
        return None
    settings_snapshot = getattr(source, "settings_snapshot", {}) or {}
    source_audio_path = settings_snapshot.get("source_audio_path")
    if source_audio_path is None:
        return None
    candidate = str(source_audio_path).strip()
    return candidate or None


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
    main_kind = resolve_storage_layer_kind(layer_record, main_take)
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
    state_flags = layer_record.state_flags or {}
    layer = Layer(
        id=layer_id,
        timeline_id=timeline_id,
        name=layer_record.name.title(),
        kind=main_kind,
        order_index=int(layer_record.order) + 1,
        takes=timeline_takes,
        playback=replace(
            Layer(
                id=layer_id,
                timeline_id=timeline_id,
                name="",
                kind=main_kind,
                order_index=0,
            ).playback,
            armed_source_ref=main_audio.playback_source_ref,
        ),
        status=LayerStatus(
            stale=bool(state_flags.get("stale", False)),
            manually_modified=bool(state_flags.get("manually_modified", False)),
            stale_reason=state_flags.get("stale_reason"),
        ),
        mixer=LayerMixerState(
            output_bus=_state_flag_output_bus(state_flags),
        ),
        sync=LayerSyncState(ma3_track_coord=_state_flag_ma3_track_coord(state_flags)),
        provenance=LayerProvenance(
            source_layer_id=(
                LayerId(str(provenance["source_layer_id"]))
                if provenance.get("source_layer_id")
                else None
            ),
            source_song_version_id=(
                SongVersionId(str(provenance["source_song_version_id"]))
                if provenance.get("source_song_version_id")
                else None
            ),
            source_run_id=provenance.get("source_run_id"),
            pipeline_id=source_pipeline.get("pipeline_id") or provenance.get("pipeline_id"),
            output_name=source_pipeline.get("output_name") or provenance.get("output_name"),
        ),
        presentation_hints=LayerPresentationHints(
            visible=bool(layer_record.visible),
            locked=bool(layer_record.locked),
            expanded=layer_take_lanes_expanded(layer_record),
            color=layer_record.color,
        ),
    )
    return layer, main_audio, take_audio


def layer_take_lanes_expanded(layer_record) -> bool:
    """Resolve persisted take-lane expansion state, defaulting to collapsed."""

    state_flags = getattr(layer_record, "state_flags", {}) or {}
    if "take_lanes_expanded" in state_flags:
        return bool(state_flags.get("take_lanes_expanded"))
    if "expanded" in state_flags:
        return bool(state_flags.get("expanded"))
    if "take_selector_expanded" in state_flags:
        return bool(state_flags.get("take_selector_expanded"))
    if "collapsed" in state_flags:
        return not bool(state_flags.get("collapsed"))
    return False


def _state_flag_ma3_track_coord(state_flags: dict[str, object]) -> str | None:
    raw_coord = state_flags.get(STATE_FLAG_MA3_TRACK_COORD)
    if raw_coord is None:
        return None
    track_coord = str(raw_coord).strip()
    return track_coord or None


def _state_flag_output_bus(state_flags: dict[str, object]) -> str | None:
    raw_value = state_flags.get(STATE_FLAG_OUTPUT_BUS)
    if raw_value is None:
        return None
    output_bus = str(raw_value).strip()
    return output_bus or None


def take_kind(take) -> LayerKind:
    """Resolve the runtime layer kind from one stored take payload."""

    if isinstance(take.data, EventData):
        return LayerKind.EVENT
    return LayerKind.AUDIO


def resolve_storage_layer_kind(layer_record, main_take) -> LayerKind:
    """Resolve the runtime layer kind, preserving stored manual intent when present."""

    state_flags = getattr(layer_record, "state_flags", {}) or {}
    manual_kind = state_flags.get("manual_kind")
    if isinstance(manual_kind, str) and manual_kind:
        try:
            return LayerKind(manual_kind)
        except ValueError:
            pass
    return take_kind(main_take)


def events_from_take(take) -> list[Event]:
    """Project event-data takes into ordered timeline events."""

    if not isinstance(take.data, EventData):
        return []
    projected_events: list[tuple[str, int, DomainEvent]] = []
    for layer in take.data.layers:
        layer_id = str(layer.id)
        for event_index, event in enumerate(layer.events):
            projected_events.append((layer_id, event_index, event))
    projected_events.sort(
        key=lambda item: (item[2].time, item[2].duration, str(item[2].id), item[0], item[1])
    )
    return [
        _event_from_domain_event(
            take=take,
            domain_layer_id=layer_id,
            domain_event=event,
            event_index=event_index,
        )
        for layer_id, event_index, event in projected_events
    ]


def section_cues_from_take(take) -> list[SectionCue]:
    """Project event-data takes into ordered canonical section cues."""

    if not isinstance(take.data, EventData):
        return []
    projected_events: list[tuple[str, int, DomainEvent]] = []
    for layer in take.data.layers:
        layer_id = str(layer.id)
        for event_index, event in enumerate(layer.events):
            projected_events.append((layer_id, event_index, event))
    projected_events.sort(
        key=lambda item: (item[2].time, item[2].duration, str(item[2].id), item[0], item[1])
    )
    cues: list[SectionCue] = []
    for layer_id, event_index, event in projected_events:
        metadata = event_cue_metadata(event)
        cue_ref = metadata.cue_ref
        if cue_ref is None:
            continue
        cue_id = SectionCueId(
            resolve_projected_event_id(
                take_id=str(take.id),
                domain_layer_id=layer_id,
                domain_event=event,
                event_index=event_index,
            )
        )
        cues.append(
            SectionCue(
                id=cue_id,
                start=float(event.time),
                cue_ref=cue_ref,
                name=metadata.label or cue_ref,
                color=metadata.color,
                notes=metadata.notes,
                payload_ref=metadata.payload_ref,
            )
        )
    return cues


def _event_from_domain_event(
    *,
    take,
    domain_layer_id: str,
    domain_event: DomainEvent,
    event_index: int,
) -> Event:
    metadata = event_cue_metadata(domain_event)
    return Event(
        id=EventId(
            resolve_projected_event_id(
                take_id=str(take.id),
                domain_layer_id=domain_layer_id,
                domain_event=domain_event,
                event_index=event_index,
            )
        ),
        take_id=TakeId(str(take.id)),
        start=float(domain_event.time),
        end=float(domain_event.time + max(domain_event.duration, 0.08)),
        origin=domain_event.origin,
        classifications=dict(domain_event.classifications or {}),
        metadata=dict(domain_event.metadata or {}),
        cue_number=event_cue_number(domain_event),
        source_event_id=domain_event.source_event_id or str(domain_event.id),
        parent_event_id=domain_event.parent_event_id,
        payload_ref=metadata.payload_ref,
        label=metadata.label or "Onset",
        cue_ref=metadata.cue_ref,
        color=metadata.color,
        notes=metadata.notes,
        muted=event_muted(domain_event),
    )


def resolve_projected_event_id(
    *,
    take_id: str,
    domain_layer_id: str,
    domain_event: DomainEvent,
    event_index: int,
) -> str:
    """Resolve the runtime event id, falling back for legacy stored events."""

    if domain_event.source_event_id is not None or domain_event.parent_event_id is not None:
        return str(domain_event.id)

    return (
        f"take:{take_id}"
        f"|layer:{domain_layer_id}"
        f"|event:{str(domain_event.id)}"
        f"|index:{event_index}"
    )


def event_label(event: DomainEvent) -> str:
    """Choose the most useful display label for one domain event."""

    if isinstance(event.classifications, dict) and event.classifications:
        for preferred_key in ("label", "class"):
            preferred_value = event.classifications.get(preferred_key)
            if isinstance(preferred_value, str) and preferred_value.strip():
                return preferred_value.strip().replace("_", " ").title()
        first_key = next(iter(event.classifications.keys()))
        value = event.classifications.get(first_key)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("_", " ").title()
        if isinstance(first_key, str) and first_key.strip():
            return first_key.strip().replace("_", " ").title()
    return "Onset"


def event_cue_metadata(event: DomainEvent) -> CueMetadata:
    """Resolve one shared cue metadata envelope from a stored domain event."""

    return CueMetadata(
        cue_ref=event_cue_ref(event),
        label=event_label(event),
        color=event_color(event),
        notes=event_notes(event),
        payload_ref=event_payload_ref(event),
    )


def event_cue_number(event: DomainEvent) -> CueNumber:
    """Resolve the canonical cue number from stored event metadata."""

    cue_number = parse_positive_cue_number(event.metadata.get("cue_number"))
    return cue_number if cue_number is not None else 1


def event_payload_ref(event: DomainEvent) -> str | None:
    """Resolve a stable payload reference when one is available in metadata."""

    raw_value = event.metadata.get("payload_ref")
    if raw_value is None:
        return None
    payload_ref = str(raw_value).strip()
    return payload_ref or None


def event_cue_ref(event: DomainEvent) -> str | None:
    """Resolve a preserved cue-ref string when available."""

    raw_value = event.metadata.get("cue_ref")
    if raw_value in {None, ""}:
        raw_value = event.metadata.get("cueRef")
    if raw_value not in {None, ""}:
        cue_ref = str(raw_value).strip()
        return cue_ref or None
    raw_cue_number = event.metadata.get("cue_number")
    return cue_number_text(parse_positive_cue_number(raw_cue_number))


def event_color(event: DomainEvent) -> str | None:
    """Resolve persisted event color metadata, if present."""

    raw_value = event.metadata.get("color")
    if raw_value is None:
        return None
    color = str(raw_value).strip()
    return color or None


def event_notes(event: DomainEvent) -> str | None:
    """Resolve persisted cue/event notes metadata, if present."""

    raw_value = event.metadata.get("notes")
    if raw_value in {None, ""}:
        raw_value = event.metadata.get("note")
    if raw_value is None:
        return None
    notes = str(raw_value).strip()
    return notes or None


def event_muted(event: DomainEvent) -> bool:
    """Resolve persisted event mute metadata."""

    return bool(event.metadata.get("muted", False))


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
