"""
Feature-specific API facades.

Provides organized access to application functionality by feature.
Each facade provides a focused API for a specific domain area.

Usage:
    from src.application.api.features import ProjectsAPI, BlocksAPI
    
    # Or access via the main facade
    facade.projects.create_project("My Project")
    facade.blocks.add_block("LoadAudio")

Available APIs:
- ProjectsAPI: Project management operations
- BlocksAPI: Block CRUD and processing
- ConnectionsAPI: Connection management
- ExecutionAPI: Block execution
- SetlistsAPI: Setlist management
- MA3API: grandMA3 integration
"""
from src.application.api.features.projects_api import ProjectsAPI
from src.application.api.features.blocks_api import BlocksAPI
from src.application.api.features.connections_api import ConnectionsAPI
from src.application.api.features.setlists_api import SetlistsAPI

__all__ = [
    'ProjectsAPI',
    'BlocksAPI',
    'ConnectionsAPI',
    'SetlistsAPI',
]
