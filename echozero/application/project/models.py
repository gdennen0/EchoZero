"""Core project-level application models."""

from dataclasses import dataclass, field

from echozero.application.shared.ids import ProjectId, SessionId, SongId


@dataclass(slots=True)
class Project:
    id: ProjectId
    name: str
    songs: list[SongId] = field(default_factory=list)
    active_song_id: SongId | None = None
    session_id: SessionId | None = None
