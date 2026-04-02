"""Persistence record shapes for the new EchoZero application architecture.

These are storage-facing records, not UI-facing models.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProjectRecord:
    id: str
    name: str
    songs: list[str] = field(default_factory=list)
    active_song_id: str | None = None
    session_id: str | None = None


@dataclass(slots=True)
class SongRecord:
    id: str
    project_id: str
    title: str
    versions: list[str] = field(default_factory=list)
    active_version_id: str | None = None


@dataclass(slots=True)
class SongVersionRecord:
    id: str
    song_id: str
    name: str
    timeline_id: str
    layer_order: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SessionRecord:
    id: str
    project_id: str
    active_song_id: str | None = None
    active_song_version_id: str | None = None
    active_timeline_id: str | None = None
    ui_prefs_ref: str | None = None
    transport: dict[str, Any] = field(default_factory=dict)
    mixer: dict[str, Any] = field(default_factory=dict)
    playback: dict[str, Any] = field(default_factory=dict)
    sync: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventRecord:
    id: str
    take_id: str
    start: float
    end: float
    payload_ref: str | None = None
    label: str = "Event"
    color: str | None = None
    muted: bool = False


@dataclass(slots=True)
class TakeRecord:
    id: str
    layer_id: str
    name: str
    version_label: str | None = None
    events: list[EventRecord] = field(default_factory=list)
    source_ref: str | None = None
    available: bool = True
    is_comped: bool = False


@dataclass(slots=True)
class LayerRecord:
    id: str
    timeline_id: str
    name: str
    kind: str
    order_index: int
    takes: list[TakeRecord] = field(default_factory=list)
    active_take_id: str | None = None
    mixer: dict[str, Any] = field(default_factory=dict)
    playback: dict[str, Any] = field(default_factory=dict)
    sync: dict[str, Any] = field(default_factory=dict)
    presentation_hints: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TimelineRecord:
    id: str
    song_version_id: str
    start: float = 0.0
    end: float = 0.0
    layers: list[LayerRecord] = field(default_factory=list)
    loop_region: dict[str, Any] | None = None
    selection: dict[str, Any] = field(default_factory=dict)
    viewport: dict[str, Any] = field(default_factory=dict)
