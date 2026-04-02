"""In-memory project repository implementation for the new application architecture."""

from echozero.application.project.models import Project
from echozero.application.project.repository import ProjectRepository
from echozero.application.shared.ids import ProjectId


class InMemoryProjectRepository(ProjectRepository):
    def __init__(self) -> None:
        self._projects: dict[ProjectId, Project] = {}

    def get(self, project_id: ProjectId) -> Project:
        return self._projects[project_id]

    def save(self, project: Project) -> None:
        self._projects[project.id] = project
