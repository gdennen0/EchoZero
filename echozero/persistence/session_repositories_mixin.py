"""Repository accessors for project storage sessions.
Exists to keep ProjectStorage focused on lifecycle while repositories stay discoverable.
Connects open SQLite sessions to persistence repository objects.
"""

from __future__ import annotations

from echozero.persistence.repositories import (
    LayerRepository,
    ObjectCandidateRepository,
    ObjectContentRepository,
    PipelineConfigRepository,
    ProjectRepository,
    SongDefaultPipelineConfigRepository,
    SongRepository,
    SongVersionRepository,
    TakeRepository,
    TimelineObjectRepository,
)


class ProjectStorageRepositoriesMixin:
    """Repository accessors shared by project storage sessions."""

    @property
    def projects(self) -> ProjectRepository:
        """Access the project repository."""
        self._check_closed()
        return ProjectRepository(self.db)

    @property
    def songs(self) -> SongRepository:
        """Access the song repository."""
        self._check_closed()
        return SongRepository(self.db)

    @property
    def song_versions(self) -> SongVersionRepository:
        """Access the song version repository."""
        self._check_closed()
        return SongVersionRepository(self.db)

    @property
    def layers(self) -> LayerRepository:
        """Access the layer repository."""
        self._check_closed()
        return LayerRepository(self.db)

    @property
    def takes(self) -> TakeRepository:
        """Access the take repository."""
        self._check_closed()
        return TakeRepository(self.db)

    @property
    def timeline_objects(self) -> TimelineObjectRepository:
        """Access timeline object records."""
        self._check_closed()
        return TimelineObjectRepository(self.db)

    @property
    def object_contents(self) -> ObjectContentRepository:
        """Access object content records."""
        self._check_closed()
        return ObjectContentRepository(self.db)

    @property
    def object_candidates(self) -> ObjectCandidateRepository:
        """Access object candidate records."""
        self._check_closed()
        return ObjectCandidateRepository(self.db)

    @property
    def pipeline_configs(self) -> PipelineConfigRepository:
        """Access the pipeline config repository."""
        self._check_closed()
        return PipelineConfigRepository(self.db)

    @property
    def song_default_pipeline_configs(self) -> SongDefaultPipelineConfigRepository:
        """Access the song default pipeline config repository."""
        self._check_closed()
        return SongDefaultPipelineConfigRepository(self.db)
