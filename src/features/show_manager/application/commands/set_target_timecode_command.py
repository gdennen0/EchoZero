"""
Set Target Timecode Command

Sets the target timecode number that ShowManager uses to fetch MA3 data.
"""

from typing import TYPE_CHECKING

from src.application.commands.base_command import EchoZeroCommand
from src.utils.message import Log

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class SetTargetTimecodeCommand(EchoZeroCommand):
    """
    Set the target timecode number for ShowManager block (undoable).
    
    Redo: Updates target_timecode setting
    Undo: Restores previous target_timecode value
    
    This setting controls which timecode pool in MA3 the ShowManager
    fetches structure (tracks) and events from.
    
    Args:
        facade: ApplicationFacade instance
        show_manager_block_id: ShowManager block ID
        timecode_no: Target timecode number (e.g., 101, 102, etc.)
    """
    
    COMMAND_TYPE = "layer_sync.set_target_timecode"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        show_manager_block_id: str,
        timecode_no: int
    ):
        super().__init__(facade, f"Set Target Timecode: {timecode_no}")
        
        self._show_manager_block_id = show_manager_block_id
        self._new_timecode = timecode_no
        
        # State for undo
        self._old_timecode: int | None = None
    
    def redo(self):
        """Set the target timecode."""
        from src.application.settings.show_manager_settings import ShowManagerSettingsManager
        
        try:
            settings_manager = ShowManagerSettingsManager(self._facade, self._show_manager_block_id)
            
            # Capture old value for undo (first time only)
            if self._old_timecode is None:
                self._old_timecode = settings_manager.target_timecode
            
            # Validate timecode number
            if self._new_timecode < 1:
                self._log_error(f"Invalid timecode number: {self._new_timecode}. Must be >= 1")
                raise ValueError(f"Timecode number must be >= 1, got {self._new_timecode}")
            
            # Update setting
            settings_manager.target_timecode = self._new_timecode
            
            Log.info(f"Set target timecode to {self._new_timecode} for ShowManager block {self._show_manager_block_id}")
            Log.debug(f"ShowManager[{self._show_manager_block_id}]: Updated target_timecode from {self._old_timecode} to {self._new_timecode}")
        except Exception as e:
            self._log_error(f"Failed to set target timecode: {e}")
            raise
    
    def undo(self):
        """Restore previous target timecode."""
        if self._old_timecode is None:
            return
        
        from src.application.settings.show_manager_settings import ShowManagerSettingsManager
        
        try:
            settings_manager = ShowManagerSettingsManager(self._facade, self._show_manager_block_id)
            settings_manager.target_timecode = self._old_timecode
            
            Log.info(f"Restored target timecode to {self._old_timecode} for ShowManager block {self._show_manager_block_id}")
        except Exception as e:
            self._log_error(f"Failed to restore target timecode: {e}")
            # Don't raise - undo failures should be logged but not crash
