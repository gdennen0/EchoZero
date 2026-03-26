"""
Repository modules: CRUD operations for each persistence entity type.
Exists because each entity needs isolated, testable data access logic.
All repositories take a sqlite3.Connection and return domain/entity types.
"""

from echozero.persistence.repositories.layer import LayerRepository
from echozero.persistence.repositories.pipeline_config import PipelineConfigRepository
from echozero.persistence.repositories.project import ProjectRepository
from echozero.persistence.repositories.song import SongRepository, SongVersionRepository
from echozero.persistence.repositories.take import TakeRepository

__all__ = [
    "ProjectRepository",
    "SongRepository",
    "SongVersionRepository",
    "LayerRepository",
    "TakeRepository",
    "PipelineConfigRepository",
]
