"""
MA3 Track Commands

Commands for manipulating tracks and track groups in grandMA3.
These commands are primarily used for MA3 organization.
"""

from typing import Any, Dict, List, Optional, Tuple

from .base import (
    BaseMA3Command,
    CommandContext,
    CommandResult,
    ValidationResult,
)


class AddTrackCommand(BaseMA3Command):
    """
    Add a new track to a track group in MA3.
    
    Editor: Creates layer (tracks map to layers)
    MA3: Creates track in the specified track group
    """
    
    COMMAND_TYPE = "add_track"
    OSC_ADDRESS = "/echozero/timecode/add_track"
    
    def __init__(
        self,
        context: CommandContext,
        name: str,
        track_group_idx: Optional[int] = None,
    ):
        super().__init__(context)
        self.name = name
        self.track_group_idx = track_group_idx if track_group_idx is not None else context.track_group_idx
        
        # Index of created track for undo
        self._created_track_idx: Optional[int] = None
    
    def validate(self) -> ValidationResult:
        """Validate track parameters."""
        result = ValidationResult.success()
        
        if not self.name or not self.name.strip():
            result.add_error("Track name cannot be empty")
        
        if self.track_group_idx < 0:
            result.add_error("Track group index cannot be negative")
        
        return result
    
    def _do_execute(self) -> CommandResult:
        """Create track (as layer in Editor)."""
        if not self._context.editor_block_id:
            return CommandResult.fail("No editor block configured")
        
        self._undo_data = {
            "name": self.name,
            "track_group_idx": self.track_group_idx,
        }
        
        self._log_info(f"Added track '{self.name}' to group {self.track_group_idx}")
        return CommandResult.ok({"name": self.name})
    
    def _do_undo(self) -> CommandResult:
        """Remove the created track."""
        if not self._undo_data:
            return CommandResult.fail("No undo data available")
        
        self._log_info(f"Removed track '{self._undo_data['name']}'")
        return CommandResult.ok()
    
    def to_osc(self) -> Tuple[str, List[Any]]:
        """Convert to OSC message format."""
        return (
            self.OSC_ADDRESS,
            [
                self._context.timecode_no,
                self.track_group_idx,
                self.name,
            ]
        )


class DeleteTrackCommand(BaseMA3Command):
    """
    Delete a track from a track group.
    
    Editor: Deletes the corresponding layer
    MA3: Removes the track from the track group
    """
    
    COMMAND_TYPE = "delete_track"
    OSC_ADDRESS = "/echozero/timecode/delete_track"
    
    def __init__(
        self,
        context: CommandContext,
        track_idx: int,
        track_group_idx: Optional[int] = None,
    ):
        super().__init__(context)
        self.track_idx = track_idx
        self.track_group_idx = track_group_idx if track_group_idx is not None else context.track_group_idx
        
        # Deleted track data for undo
        self._deleted_track: Optional[Dict[str, Any]] = None
    
    def validate(self) -> ValidationResult:
        """Validate delete parameters."""
        result = ValidationResult.success()
        
        if self.track_idx < 0:
            result.add_error("Track index cannot be negative")
        
        if self.track_group_idx < 0:
            result.add_error("Track group index cannot be negative")
        
        return result
    
    def _do_execute(self) -> CommandResult:
        """Delete track (as layer in Editor)."""
        if not self._context.editor_block_id:
            return CommandResult.fail("No editor block configured")
        
        # Store deleted track data for undo
        self._deleted_track = {
            "name": "",  # Placeholder
            "events": [],
        }
        
        self._undo_data = {
            "track_idx": self.track_idx,
            "track_group_idx": self.track_group_idx,
            "deleted_track": self._deleted_track,
        }
        
        self._log_info(f"Deleted track {self.track_idx} from group {self.track_group_idx}")
        return CommandResult.ok({"track_idx": self.track_idx})
    
    def _do_undo(self) -> CommandResult:
        """Restore deleted track."""
        if not self._undo_data or not self._deleted_track:
            return CommandResult.fail("No undo data available")
        
        self._log_info(f"Restored track {self.track_idx}")
        return CommandResult.ok()
    
    def to_osc(self) -> Tuple[str, List[Any]]:
        """Convert to OSC message format."""
        return (
            self.OSC_ADDRESS,
            [
                self._context.timecode_no,
                self.track_group_idx,
                self.track_idx,
            ]
        )


class AddTrackGroupCommand(BaseMA3Command):
    """
    Add a new track group to the timecode.
    
    Editor: Track groups don't have a direct equivalent - this creates a
            logical grouping that can be flattened into layers
    MA3: Creates a new track group in the timecode pool
    """
    
    COMMAND_TYPE = "add_track_group"
    OSC_ADDRESS = "/echozero/timecode/add_trackgroup"
    
    def __init__(
        self,
        context: CommandContext,
        name: str,
    ):
        super().__init__(context)
        self.name = name
        
        # Index of created track group for undo
        self._created_group_idx: Optional[int] = None
    
    def validate(self) -> ValidationResult:
        """Validate track group parameters."""
        result = ValidationResult.success()
        
        if not self.name or not self.name.strip():
            result.add_error("Track group name cannot be empty")
        
        return result
    
    def _do_execute(self) -> CommandResult:
        """Create track group."""
        # Track groups are primarily an MA3 concept
        # In Editor, this might create a layer group or prefix
        
        self._undo_data = {
            "name": self.name,
        }
        
        self._log_info(f"Added track group '{self.name}'")
        return CommandResult.ok({"name": self.name})
    
    def _do_undo(self) -> CommandResult:
        """Remove the created track group."""
        if not self._undo_data:
            return CommandResult.fail("No undo data available")
        
        self._log_info(f"Removed track group '{self._undo_data['name']}'")
        return CommandResult.ok()
    
    def to_osc(self) -> Tuple[str, List[Any]]:
        """Convert to OSC message format."""
        return (
            self.OSC_ADDRESS,
            [
                self._context.timecode_no,
                self.name,
            ]
        )
