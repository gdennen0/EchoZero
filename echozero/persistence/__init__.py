"""
Persistence package: SQLite-backed storage for EchoZero project files.
Exists because projects need durable, versioned storage that survives app restarts.
All SQL is encapsulated here — no other package imports sqlite3.

Modules:
    base          — BaseRepository[T] generic abstract base
    entities      — Project, Song, SongVersion, LayerRecord, SongPipelineConfig, ProjectSettings
    schema        — DDL, version tracking, migration infrastructure
    repositories/ — CRUD for each entity type
    dirty         — DirtyTracker for change detection and autosave
    session       — ProjectSession lifecycle manager
    archive       — .ez pack/unpack with atomic write
    audio         — Audio import, content-addressing, hash verification
"""

from echozero.persistence.archive import is_valid_ez, pack_ez, unpack_ez
from echozero.persistence.audio import (
    compute_audio_hash,
    import_audio,
    resolve_audio_path,
    verify_audio,
)
from echozero.persistence.base import BaseRepository
from echozero.persistence.dirty import DirtyTracker
from echozero.persistence.entities import (
    LayerRecord,
    LayerType,
    Project,
    ProjectSettings,
    Song,
    SongPipelineConfig,
    SongVersion,
)
from echozero.persistence.repositories import (
    LayerRepository,
    PipelineConfigRepository,
    ProjectRepository,
    SongRepository,
    SongVersionRepository,
    TakeRepository,
)
from echozero.persistence.schema import SCHEMA_VERSION, init_db
from echozero.persistence.session import ProjectSession

__all__ = [
    # Base
    "BaseRepository",
    # Entities
    "ProjectSettings",
    "Project",
    "Song",
    "SongVersion",
    "LayerRecord",
    "LayerType",
    "SongPipelineConfig",
    # Schema
    "SCHEMA_VERSION",
    "init_db",
    # Repositories
    "ProjectRepository",
    "SongRepository",
    "SongVersionRepository",
    "LayerRepository",
    "TakeRepository",
    "PipelineConfigRepository",
    # Session + tracking
    "DirtyTracker",
    "ProjectSession",
    # Archive
    "pack_ez",
    "unpack_ez",
    "is_valid_ez",
    # Audio
    "import_audio",
    "compute_audio_hash",
    "verify_audio",
    "resolve_audio_path",
]
