"""Timeline object/content references for the EZ app model.
Exists because timeline rows are presentation while objects and content are app truth.
Connects pipeline provenance, playback, preview, and sync without magic layer ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from echozero.application.shared.ids import (
    ObjectCandidateId,
    ObjectContentId,
    ObjectRevisionId,
    SongVersionId,
    TimelineObjectId,
)


class ObjectContentKind(str, Enum):
    """Supported main-content payload categories for a timeline object."""

    AUDIO_CLIP = "audio_clip"
    GENERATED_AUDIO = "generated_audio"
    EVENT_SET = "event_set"
    SECTION_CUE_SET = "section_cue_set"
    AUTOMATION = "automation"


LEGACY_SOURCE_AUDIO_LAYER_ID = "source_audio"


def is_imported_song_object_id(object_id: object | None) -> bool:
    """Return whether an object id names imported song audio."""

    return str(object_id or "").strip().startswith("object_song_")


def is_imported_song_layer(layer: object) -> bool:
    """Resolve the temporary source-audio alias without making it persisted truth."""

    object_id = getattr(layer, "object_id", None)
    layer_id = getattr(layer, "id", getattr(layer, "layer_id", None))
    return is_imported_song_object_id(object_id) or str(layer_id) == LEGACY_SOURCE_AUDIO_LAYER_ID


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A typed reference to the content revision that produced another content item."""

    object_id: TimelineObjectId
    content_id: ObjectContentId
    revision_id: ObjectRevisionId
    role: str = "source"
    locator: str | None = None

    def __post_init__(self) -> None:
        if not str(self.object_id).strip():
            raise ValueError("SourceRef.object_id must be non-empty")
        if not str(self.content_id).strip():
            raise ValueError("SourceRef.content_id must be non-empty")
        if not str(self.revision_id).strip():
            raise ValueError("SourceRef.revision_id must be non-empty")
        role = str(self.role or "").strip()
        object.__setattr__(self, "role", role or "source")
        if self.locator is not None:
            locator = str(self.locator).strip()
            object.__setattr__(self, "locator", locator or None)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this source reference into a JSON-compatible payload."""

        return {
            "object_id": str(self.object_id),
            "content_id": str(self.content_id),
            "revision_id": str(self.revision_id),
            "role": self.role,
            "locator": self.locator,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceRef":
        """Build a source reference from persisted JSON data."""

        return cls(
            object_id=TimelineObjectId(str(payload.get("object_id") or "")),
            content_id=ObjectContentId(str(payload.get("content_id") or "")),
            revision_id=ObjectRevisionId(str(payload.get("revision_id") or "")),
            role=str(payload.get("role") or "source"),
            locator=(
                None if payload.get("locator") in (None, "") else str(payload.get("locator"))
            ),
        )


@dataclass(frozen=True, slots=True)
class AnalysisBuildRef:
    """Pipeline execution identity for generated object content."""

    pipeline_id: str
    pipeline_config_id: str | None
    block_id: str | None
    block_type: str | None
    output_name: str
    execution_id: str
    build_id: str | None = None
    generated_at: str | None = None

    def __post_init__(self) -> None:
        if not str(self.pipeline_id).strip():
            raise ValueError("AnalysisBuildRef.pipeline_id must be non-empty")
        if not str(self.output_name).strip():
            raise ValueError("AnalysisBuildRef.output_name must be non-empty")
        if not str(self.execution_id).strip():
            raise ValueError("AnalysisBuildRef.execution_id must be non-empty")


@dataclass(slots=True)
class ObjectMainContent:
    """Live content for playback, export, sync, and freshness comparisons."""

    id: ObjectContentId
    object_id: TimelineObjectId
    revision_id: ObjectRevisionId
    kind: ObjectContentKind
    payload: dict[str, Any] = field(default_factory=dict)
    source_ref: SourceRef | None = None
    analysis_build: AnalysisBuildRef | None = None


@dataclass(slots=True)
class ObjectCandidate:
    """A rerun or comparison result that is not live truth until promoted."""

    id: ObjectCandidateId
    object_id: TimelineObjectId
    content: ObjectMainContent
    label: str = ""


@dataclass(slots=True)
class TimelineObject:
    """A logical song object whose main content is independent of timeline layout."""

    id: TimelineObjectId
    song_version_id: SongVersionId
    name: str
    main_content: ObjectMainContent
    candidates: list[ObjectCandidate] = field(default_factory=list)
