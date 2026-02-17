"""
Projects feature module.

Usage:
    from src.features.projects.domain import Project, ActionSet, ActionItem
    from src.features.projects.application import ProjectService
    from src.features.projects.infrastructure import SQLiteProjectRepository
"""
# Only export domain by default - application and infrastructure via submodules
from src.features.projects.domain import (
    Project,
    ActionSet,
    ActionItem,
    ProjectRepository,
    ActionSetRepository,
    ActionItemRepository,
)

__all__ = [
    'Project',
    'ActionSet',
    'ActionItem',
    'ProjectRepository',
    'ActionSetRepository',
    'ActionItemRepository',
]
