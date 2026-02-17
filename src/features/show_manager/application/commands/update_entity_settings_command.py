"""
Update Entity Settings Command

Updates per-entity sync settings.
Updates entity settings (sync direction, conflict resolution, excluded).
"""
from typing import TYPE_CHECKING, Optional, Dict, Any

from src.application.commands.base_command import EchoZeroCommand

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class UpdateEntitySettingsCommand(EchoZeroCommand):
    """
    Update per-entity sync settings (undoable).
    
    Redo: Updates entity settings
    Undo: Restores original settings
    
    Handles:
    - Updating entity settings (sync_direction, conflict_resolution, excluded)
    - Persisting to ShowManager settings
    - Updating entity in controller
    
    Args:
        facade: ApplicationFacade instance
        entity_id: Entity ID (Editor layer ID or MA3 track coord)
        entity_type: "editor" or "ma3"
        settings: Dict with settings to update
    """
    
    COMMAND_TYPE = "layer_sync.update_entity_settings"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        show_manager_block_id: str,
        entity_id: str,
        entity_type: str,
        settings: Dict[str, Any]
    ):
        super().__init__(facade, f"Update Entity Settings: {entity_id}")
        
        self._show_manager_block_id = show_manager_block_id
        self._entity_id = entity_id
        self._entity_type = entity_type
        self._settings = settings
        
        # State for undo
        self._original_settings: Optional[Dict[str, Any]] = None
    
    def redo(self):
        """Update entity settings."""
        from src.application.settings.show_manager_settings import ShowManagerSettingsManager
        from src.utils.message import Log
        
        settings_manager = ShowManagerSettingsManager(self._facade, self._show_manager_block_id)
        entity = settings_manager.get_synced_layer(self._entity_type, self._entity_id)
        if not entity:
            Log.warning(
                f"UpdateEntitySettingsCommand: Entity not found "
                f"{self._entity_type}:{self._entity_id}"
            )
            return
        
        current_settings = entity.get("settings", {}) if isinstance(entity, dict) else {}
        if self._original_settings is None:
            self._original_settings = current_settings.copy()
        
        updated_settings = {**current_settings, **self._settings}
        updated = settings_manager.update_synced_layer(
            self._entity_type,
            self._entity_id,
            {"settings": updated_settings}
        )
        if updated:
            settings_manager.force_save()
        else:
            Log.warning(
                f"UpdateEntitySettingsCommand: Failed to update settings for "
                f"{self._entity_type}:{self._entity_id}"
            )
    
    def undo(self):
        """Restore original settings."""
        if self._original_settings is None:
            return
        
        from src.application.settings.show_manager_settings import ShowManagerSettingsManager
        
        settings_manager = ShowManagerSettingsManager(self._facade, self._show_manager_block_id)
        updated = settings_manager.update_synced_layer(
            self._entity_type,
            self._entity_id,
            {"settings": self._original_settings.copy()}
        )
        if updated:
            settings_manager.force_save()
