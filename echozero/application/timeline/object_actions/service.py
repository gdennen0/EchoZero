"""Public service exports for timeline object-action execution.
Exists to expose one explicit execution owner instead of a settings-first alias chain.
Connects timeline callers to the canonical object-action service surface.
"""

from echozero.application.timeline.object_action_settings_service import (
    ObjectActionExecutionService,
    ObjectActionSettingsService,
)

ObjectActionService = ObjectActionExecutionService

__all__ = [
    "ObjectActionExecutionService",
    "ObjectActionService",
    "ObjectActionSettingsService",
]
