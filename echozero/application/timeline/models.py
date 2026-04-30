"""Core timeline application models."""

from dataclasses import dataclass, field
from typing import Any

from echozero.application.shared.cue_numbers import (
    CueNumber,
    coerce_positive_cue_number,
    cue_number_from_ref_text,
)
from echozero.application.shared.ids import (
    EventId,
    LayerId,
    RegionId,
    SectionCueId,
    SongVersionId,
    TakeId,
    TimelineId,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ranges import TimeRange
from echozero.application.mixer.models import LayerMixerState
from echozero.application.playback.models import LayerPlaybackState
from echozero.application.sync.models import LiveSyncState, coerce_live_sync_state


@dataclass(slots=True)
class LayerSyncState:
    mode: str = "none"
    connected: bool = False
    offset_ms: float = 0.0
    target_ref: str | None = None
    show_manager_block_id: str | None = None
    ma3_track_coord: str | None = None
    derived_from_source: bool = False
    live_sync_state: LiveSyncState = LiveSyncState.OFF
    live_sync_pause_reason: str | None = None
    live_sync_divergent: bool = False

    def __post_init__(self) -> None:
        self.live_sync_state = coerce_live_sync_state(self.live_sync_state)
        if self.live_sync_pause_reason is not None:
            reason = self.live_sync_pause_reason.strip()
            self.live_sync_pause_reason = reason or None


@dataclass(slots=True)
class LayerPresentationHints:
    visible: bool = True
    locked: bool = False
    expanded: bool = False
    height: float = 40.0
    color: str | None = None
    group_id: str | None = None
    group_name: str | None = None
    group_index: int | None = None
    show_take_selector: bool = True
    show_take_lane: bool = False


@dataclass(slots=True)
class LayerStatus:
    stale: bool = False
    manually_modified: bool = False
    stale_reason: str | None = None


@dataclass(slots=True)
class LayerArtifactProvenance:
    schema: str = "echozero.model-artifact.v1"
    role: str = ""
    kind: str = ""
    locator: str = ""
    content_type: str | None = None


@dataclass(slots=True)
class LayerAnalysisBuildProvenance:
    schema: str = "echozero.analysis-build.v1"
    build_id: str | None = None
    execution_id: str | None = None
    pipeline_id: str | None = None
    pipeline_config_id: str | None = None
    block_id: str | None = None
    block_type: str | None = None
    output_name: str | None = None
    data_type: str | None = None
    generated_at: str | None = None


@dataclass(slots=True)
class LayerProvenance:
    source_layer_id: LayerId | None = None
    source_song_version_id: SongVersionId | None = None
    source_run_id: str | None = None
    pipeline_id: str | None = None
    output_name: str | None = None
    analysis_build: LayerAnalysisBuildProvenance | None = None
    artifacts: tuple[LayerArtifactProvenance, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        build = self.analysis_build
        if isinstance(build, dict):
            build = LayerAnalysisBuildProvenance(**build)

        if build is None and any(
            value is not None
            for value in (self.source_run_id, self.pipeline_id, self.output_name)
        ):
            build = LayerAnalysisBuildProvenance(
                build_id=self.source_run_id,
                execution_id=self.source_run_id,
                pipeline_id=self.pipeline_id,
                output_name=self.output_name,
            )

        if build is not None:
            if self.source_run_id is None:
                self.source_run_id = build.execution_id or build.build_id
            if self.pipeline_id is None:
                self.pipeline_id = build.pipeline_id
            if self.output_name is None:
                self.output_name = build.output_name
        self.analysis_build = build

        normalized_artifacts: list[LayerArtifactProvenance] = []
        for artifact in self.artifacts:
            if isinstance(artifact, LayerArtifactProvenance):
                normalized_artifacts.append(artifact)
                continue
            if isinstance(artifact, dict):
                normalized_artifacts.append(LayerArtifactProvenance(**artifact))
                continue
            raise TypeError(f"Unsupported layer artifact provenance: {type(artifact)!r}")
        self.artifacts = tuple(normalized_artifacts)


@dataclass(frozen=True, slots=True)
class EventRef:
    layer_id: LayerId
    take_id: TakeId
    event_id: EventId


@dataclass(frozen=True, slots=True)
class CueMetadata:
    cue_ref: str | None = None
    label: str = ""
    color: str | None = None
    notes: str | None = None
    payload_ref: str | None = None

    def __post_init__(self) -> None:
        cue_ref = None if self.cue_ref is None else str(self.cue_ref).strip()
        object.__setattr__(self, "cue_ref", cue_ref or None)
        label = str(self.label or "").strip()
        object.__setattr__(self, "label", label)
        color = None if self.color is None else str(self.color).strip()
        object.__setattr__(self, "color", color or None)
        notes = None if self.notes is None else str(self.notes).strip()
        object.__setattr__(self, "notes", notes or None)
        payload_ref = None if self.payload_ref is None else str(self.payload_ref).strip()
        object.__setattr__(self, "payload_ref", payload_ref or None)


@dataclass(slots=True)
class Event:
    id: EventId
    take_id: TakeId
    start: float
    end: float
    origin: str = "user"
    classifications: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    cue_number: CueNumber = 1
    source_event_id: str | None = None
    parent_event_id: str | None = None
    payload_ref: str | None = None
    label: str = "Event"
    cue_ref: str | None = None
    color: str | None = None
    notes: str | None = None
    muted: bool = False

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"Event.start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(
                f"Event.end must be >= start, got start={self.start}, end={self.end}"
            )
        origin = str(self.origin or "").strip()
        self.origin = origin or "user"
        self.classifications = dict(self.classifications or {})
        self.metadata = dict(self.metadata or {})
        try:
            cue_number = coerce_positive_cue_number(self.cue_number)
        except ValueError as exc:
            raise ValueError(
                f"Event.cue_number must be positive numeric data, got {self.cue_number!r}"
            ) from exc
        self.cue_number = cue_number
        self.label = str(self.label or "").strip() or "Event"
        if self.cue_ref is not None:
            cue_ref = str(self.cue_ref).strip()
            self.cue_ref = cue_ref or None
        if self.color is not None:
            color = str(self.color).strip()
            self.color = color or None
        if self.notes is not None:
            notes = str(self.notes).strip()
            self.notes = notes or None
        if self.payload_ref is not None:
            payload_ref = str(self.payload_ref).strip()
            self.payload_ref = payload_ref or None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def review_metadata(self) -> dict[str, Any]:
        raw_value = self.metadata.get("review")
        return dict(raw_value) if isinstance(raw_value, dict) else {}

    @property
    def detection_metadata(self) -> dict[str, Any]:
        raw_value = self.metadata.get("detection")
        return dict(raw_value) if isinstance(raw_value, dict) else {}

    @property
    def promotion_state(self) -> str:
        review_value = str(self.review_metadata.get("promotion_state", "")).strip().lower()
        if review_value in {"promoted", "demoted"}:
            return review_value
        detection_value = str(self.detection_metadata.get("promotion_state", "")).strip().lower()
        if detection_value in {"promoted", "demoted"}:
            return detection_value
        detection_threshold = _coerce_optional_bool(self.detection_metadata.get("threshold_passed"))
        if detection_threshold is not None:
            return "promoted" if detection_threshold else "demoted"
        return "promoted"

    @property
    def review_state(self) -> str:
        review_value = str(self.review_metadata.get("review_state", "")).strip().lower()
        if review_value in {"unreviewed", "corrected", "signed_off"}:
            return review_value
        return "unreviewed"

    @property
    def origin_kind(self) -> str:
        normalized = self.origin.strip().lower()
        if normalized in {"manual_added", "manual", "user", "ma3_pull"}:
            return "manual_added"
        return "model_detected"

    @property
    def is_promoted(self) -> bool:
        return self.promotion_state == "promoted"

    @property
    def cue_metadata(self) -> CueMetadata:
        return CueMetadata(
            cue_ref=self.cue_ref,
            label=self.label,
            color=self.color,
            notes=self.notes,
            payload_ref=self.payload_ref,
        )


@dataclass(slots=True)
class SectionCue:
    id: SectionCueId
    start: float
    cue_ref: str
    name: str
    color: str | None = None
    notes: str | None = None
    payload_ref: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"SectionCue.start must be >= 0, got {self.start}")
        cue_ref = str(self.cue_ref or "").strip()
        if not cue_ref:
            raise ValueError("SectionCue.cue_ref must be a non-empty string")
        self.cue_ref = cue_ref
        name = str(self.name or "").strip()
        self.name = name or cue_ref
        if self.color is not None:
            color = str(self.color).strip()
            self.color = color or None
        if self.notes is not None:
            notes = str(self.notes).strip()
            self.notes = notes or None
        if self.payload_ref is not None:
            payload_ref = str(self.payload_ref).strip()
            self.payload_ref = payload_ref or None

    @property
    def cue_metadata(self) -> CueMetadata:
        return CueMetadata(
            cue_ref=self.cue_ref,
            label=self.name,
            color=self.color,
            notes=self.notes,
            payload_ref=self.payload_ref,
        )


def _coerce_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None


@dataclass(slots=True)
class SectionRegion:
    cue_id: SectionCueId
    start: float
    end: float
    cue_ref: str
    name: str
    color: str | None = None
    notes: str | None = None
    payload_ref: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"SectionRegion.start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(
                f"SectionRegion.end must be >= start, got start={self.start}, end={self.end}"
            )
        cue_ref = str(self.cue_ref or "").strip()
        if not cue_ref:
            raise ValueError("SectionRegion.cue_ref must be a non-empty string")
        self.cue_ref = cue_ref
        name = str(self.name or "").strip()
        self.name = name or cue_ref
        if self.color is not None:
            color = str(self.color).strip()
            self.color = color or None
        if self.notes is not None:
            notes = str(self.notes).strip()
            self.notes = notes or None
        if self.payload_ref is not None:
            payload_ref = str(self.payload_ref).strip()
            self.payload_ref = payload_ref or None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class TimelineRegion:
    id: RegionId
    start: float
    end: float
    label: str = "Region"
    color: str | None = None
    order_index: int = 0
    kind: str = "custom"

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"TimelineRegion.start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(
                f"TimelineRegion.end must be >= start, got start={self.start}, end={self.end}"
            )
        self.label = (self.label or "").strip() or "Region"
        self.kind = (self.kind or "").strip().lower() or "custom"

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class Take:
    id: TakeId
    layer_id: LayerId
    name: str
    version_label: str | None = None
    events: list[Event] = field(default_factory=list)
    source_ref: str | None = None
    available: bool = True
    is_comped: bool = False


@dataclass(slots=True)
class Layer:
    id: LayerId
    timeline_id: TimelineId
    name: str
    kind: LayerKind
    order_index: int
    takes: list[Take] = field(default_factory=list)
    mixer: LayerMixerState = field(default_factory=LayerMixerState)
    playback: LayerPlaybackState = field(default_factory=LayerPlaybackState)
    sync: LayerSyncState = field(default_factory=LayerSyncState)
    status: LayerStatus = field(default_factory=LayerStatus)
    provenance: LayerProvenance = field(default_factory=LayerProvenance)
    presentation_hints: LayerPresentationHints = field(default_factory=LayerPresentationHints)


@dataclass(slots=True)
class TimelineSelection:
    selected_layer_id: LayerId | None = None
    selected_layer_ids: list[LayerId] = field(default_factory=list)
    selected_take_id: TakeId | None = None
    selected_event_refs: list[EventRef] = field(default_factory=list)
    selected_event_ids: list[EventId] = field(default_factory=list)
    selected_region_id: RegionId | None = None


@dataclass(slots=True)
class TimelineViewport:
    pixels_per_second: float = 100.0
    scroll_x: float = 0.0
    scroll_y: float = 0.0


@dataclass(slots=True)
class Timeline:
    id: TimelineId
    song_version_id: SongVersionId
    start: float = 0.0
    end: float = 0.0
    layers: list[Layer] = field(default_factory=list)
    section_cues: list[SectionCue] = field(default_factory=list)
    regions: list[TimelineRegion] = field(default_factory=list)
    loop_region: TimeRange | None = None
    selection: TimelineSelection = field(default_factory=TimelineSelection)
    viewport: TimelineViewport = field(default_factory=TimelineViewport)


def cue_number_from_ref(cue_ref: str | None, *, fallback: CueNumber = 1) -> CueNumber:
    """Resolve a best-effort positive cue number from one preserved cue ref."""

    parsed = cue_number_from_ref_text(cue_ref)
    if parsed is not None:
        return parsed
    try:
        return coerce_positive_cue_number(fallback)
    except ValueError:
        return 1


def derive_section_cues_from_events(events: list[Event]) -> list[SectionCue]:
    """Project canonical section cues from section-layer events."""

    projected = [
        SectionCue(
            id=SectionCueId(str(event.id)),
            start=float(event.start),
            cue_ref=str(event.cue_ref).strip(),
            name=event.label,
            color=event.color,
            notes=event.notes,
            payload_ref=event.payload_ref,
        )
        for event in events
        if event.cue_ref is not None and str(event.cue_ref).strip()
    ]
    return sorted(
        projected,
        key=lambda cue: (float(cue.start), str(cue.id)),
    )


def derive_section_cues_from_layers(
    layers: list[Layer],
    *,
    preferred_layer_id: LayerId | None = None,
) -> list[SectionCue]:
    """Resolve canonical section cues from one preferred section layer when available."""

    section_layers = sorted(
        (layer for layer in layers if layer.kind is LayerKind.SECTION),
        key=lambda layer: (int(layer.order_index), str(layer.id)),
    )
    if not section_layers:
        return []

    source_layer = section_layers[0]
    if preferred_layer_id is not None:
        preferred = next(
            (
                layer
                for layer in section_layers
                if layer.id == preferred_layer_id
            ),
            None,
        )
        if preferred is not None:
            source_layer = preferred

    main_take = source_layer.takes[0] if source_layer.takes else None
    if main_take is None:
        return []
    return derive_section_cues_from_events(main_take.events)


def derive_section_regions(
    section_cues: list[SectionCue],
    *,
    timeline_end: float,
) -> list[SectionRegion]:
    if not section_cues:
        return []
    ordered = sorted(
        enumerate(section_cues),
        key=lambda item: (float(item[1].start), int(item[0]), str(item[1].id)),
    )
    resolved_end = max(float(timeline_end), float(ordered[-1][1].start))
    regions: list[SectionRegion] = []
    for index, (_original_index, cue) in enumerate(ordered):
        if index + 1 < len(ordered):
            end = max(float(cue.start), float(ordered[index + 1][1].start))
        else:
            end = resolved_end
        regions.append(
            SectionRegion(
                cue_id=cue.id,
                start=float(cue.start),
                end=float(end),
                cue_ref=cue.cue_ref,
                name=cue.name,
                color=cue.color,
                notes=cue.notes,
                payload_ref=cue.payload_ref,
            )
        )
    return regions
