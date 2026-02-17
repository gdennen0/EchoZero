"""
Projects API - Feature-specific facade for project operations.

Provides a focused API for project management, delegating to the
main ApplicationFacade for actual implementation.

This enables gradual migration from the monolithic facade to
feature-specific APIs while maintaining backwards compatibility.
"""
from typing import TYPE_CHECKING, Optional, List

from src.application.api.result_types import CommandResult

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class ProjectsAPI:
    """
    Projects feature API.
    
    Provides project management operations:
    - Create/load/delete projects
    - Export/import projects
    - Recent projects list
    
    Usage:
        projects = ProjectsAPI(facade)
        result = projects.create_project("My Project", "/path/to/save")
    """
    
    def __init__(self, facade: "ApplicationFacade"):
        """Initialize with reference to main facade."""
        self._facade = facade
    
    def create_project(self, name: str, save_directory: Optional[str] = None) -> CommandResult:
        """Create a new project."""
        return self._facade.create_project(name, save_directory)
    
    def load_project(self, project_id: str) -> CommandResult:
        """Load an existing project."""
        return self._facade.load_project(project_id)
    
    def delete_project(self, project_id: str) -> CommandResult:
        """Delete a project."""
        return self._facade.delete_project(project_id)
    
    def get_current_project(self) -> CommandResult:
        """Get the currently active project."""
        return self._facade.get_current_project()
    
    def export_project(self, output_path: str) -> CommandResult:
        """Export project to a file."""
        return self._facade.export_project(output_path)
    
    def import_project(self, import_path: str) -> CommandResult:
        """Import project from a file."""
        return self._facade.import_project(import_path)
    
    def list_recent_projects(self, limit: int = 10) -> CommandResult:
        """List recently accessed projects."""
        return self._facade.list_recent_projects(limit)
