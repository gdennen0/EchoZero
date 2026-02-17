"""
Application layer for blocks feature.

Contains:
- BlockService - block CRUD operations
- BlockStatusService - block status management
- EditorAPI - unified API for Editor block operations
"""
from src.features.blocks.application.block_service import BlockService
from src.features.blocks.application.block_status_service import BlockStatusService
from src.features.blocks.application.editor_api import (
    EditorAPI,
    EditorAPIError,
    LayerInfo,
    EventInfo,
    create_editor_api,
)

__all__ = [
    'BlockService',
    'BlockStatusService',
    'EditorAPI',
    'EditorAPIError',
    'LayerInfo',
    'EventInfo',
    'create_editor_api',
]
