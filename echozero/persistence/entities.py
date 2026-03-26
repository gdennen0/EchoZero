"""
Persistence entities: Frozen dataclasses for EchoZero's project storage model.
Exists because the domain layer's types are engine-facing (runtime pipeline data),
while persistence needs additional UI/project state (color, order, visibility).
These DTOs map 1:1 to SQLite rows; repositories translate between them and domain types.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

LayerType = Literal["analysis", "structure", "manual"]


# ---------------------------------------------------------------------------
# Project settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectSettings:
    """Global project configuration — sample rate, tempo, timecode."""

    sample_rate: int = 44100
    bpm: float | None = None
    bpm_confidence: float | None = None
    timecode_fps: float | None = None


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Project:
    """Top-level container for a live-production project file."""

    id: str
    name: str
    settings: ProjectSettings
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Song
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Song:
    """A song in the setlist — owns one or more SongVersions."""

    id: str
    project_id: str
    title: str
    artist: str
    order: int
    active_version_id: str | None = None


# ---------------------------------------------------------------------------
# Song version
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SongVersion:
    """A specific mix or arrangement of a song, tied to one audio file."""

    id: str
    song_id: str
    label: str
    audio_file: str
    duration_seconds: float
    original_sample_rate: int
    audio_hash: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Layer record (persistence DTO — NOT the domain Layer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerRecord:
    """
    Persistence-layer entity for a timeline layer. NOT the domain Layer.
    Carries UI state (color, order, visible, locked) that the pipeline engine ignores.
    """

    id: str
    song_version_id: str
    name: str
    layer_type: LayerType
    color: str | None
    order: int
    visible: bool
    locked: bool
    parent_layer_id: str | None
    source_pipeline: dict[str, Any] | None
    created_at: datetime


# ---------------------------------------------------------------------------
# Song pipeline config (ActionSet replacement)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SongPipelineConfig:
    """Per-song pipeline configuration. Replaces EZ1 ActionSets."""

    id: str
    song_version_id: str
    pipeline_id: str
    bindings: dict[str, Any]  # param_name -> value
    created_at: datetime
