"""Project persistence boundary for the new EchoZero application layer."""

from abc import ABC, abstractmethod

from echozero.application.project.models import Project
from echozero.application.shared.ids import ProjectId


class ProjectRepository(ABC):
    @abstractmethod
    def get(self, project_id: ProjectId) -> Project:
        raise NotImplementedError

    @abstractmethod
    def save(self, project: Project) -> None:
        raise NotImplementedError
