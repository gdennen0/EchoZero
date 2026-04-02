"""Song and song-version application models."""

from dataclasses import dataclass, field

from echozero.application.shared.ids import SongId, SongVersionId, ProjectId, TimelineId, LayerId


@dataclass(slots=True)
class SongVersion:
    id: SongVersionId
    song_id: SongId
    name: str
    timeline_id: TimelineId
    layer_order: list[LayerId] = field(default_factory=list)


@dataclass(slots=True)
class Song:
    id: SongId
    project_id: ProjectId
    title: str
    versions: list[SongVersionId] = field(default_factory=list)
    active_version_id: SongVersionId | None = None
