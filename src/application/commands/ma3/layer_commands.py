"""
MA3 Layer Commands

Commands for manipulating layers in EchoZero Editor blocks.
In MA3 context, layers correspond to tracks within a track group.
"""

from typing import Any, Dict, List, Optional, Tuple

from .base import (
    BaseMA3Command,
    CommandContext,
    CommandResult,
    ValidationResult,
)


class AddLayerCommand(BaseMA3Command):
    """
    Add a new layer to the Editor block.
    
    Editor: Creates a new layer with the specified name
    MA3: Creates a new track in the track group
    """
    
    COMMAND_TYPE = "add_layer"
    OSC_ADDRESS = "/echozero/timecode/add_track"
    
    def __init__(
        self,
        context: CommandContext,
        name: str,
        color: Optional[str] = None,
    ):
        super().__init__(context)
        self.name = name
        self.color = color
        
        # Index of created layer for undo
        self._created_layer_idx: Optional[int] = None
    
    def validate(self) -> ValidationResult:
        """Validate layer parameters."""
        result = ValidationResult.success()
        
        if not self.name or not self.name.strip():
            result.add_error("Layer name cannot be empty")
        
        return result
    
    def _do_execute(self) -> CommandResult:
        """Create layer in Editor block."""
        if not self._context.editor_block_id:
            return CommandResult.fail("No editor block configured")
        
        # Add layer through Editor block API
        # For now, return success - actual implementation depends on Editor block
        
        self._undo_data = {
            "name": self.name,
        }
        
        self._log_info(f"Added layer: {self.name}")
        return CommandResult.ok({"name": self.name})
    
    def _do_undo(self) -> CommandResult:
        """Remove the created layer."""
        if not self._undo_data:
            return CommandResult.fail("No undo data available")
        
        self._log_info(f"Removed layer: {self._undo_data['name']}")
        return CommandResult.ok()
    
    def to_osc(self) -> Tuple[str, List[Any]]:
        """Convert to OSC message format."""
        return (
            self.OSC_ADDRESS,
            [
                self._context.timecode_no,
                self._context.track_group_idx,
                self.name,
            ]
        )


class DeleteLayerCommand(BaseMA3Command):
    """
    Delete an existing layer from the Editor block.
    
    Editor: Removes the layer and all its events
    MA3: Removes the track from the track group
    """
    
    COMMAND_TYPE = "delete_layer"
    OSC_ADDRESS = "/echozero/timecode/delete_track"
    
    def __init__(
        self,
        context: CommandContext,
        layer_idx: int,
    ):
        super().__init__(context)
        self.layer_idx = layer_idx
        
        # Deleted layer data for undo
        self._deleted_layer: Optional[Dict[str, Any]] = None
    
    def validate(self) -> ValidationResult:
        """Validate delete parameters."""
        result = ValidationResult.success()
        
        if self.layer_idx < 0:
            result.add_error("Layer index cannot be negative")
        
        return result
    
    def _do_execute(self) -> CommandResult:
        """Delete layer from Editor block."""
        if not self._context.editor_block_id:
            return CommandResult.fail("No editor block configured")
        
        # Store deleted layer data for undo (including all events)
        self._deleted_layer = {
            "name": "",  # Placeholder
            "events": [],
            "color": None,
        }
        
        self._undo_data = {
            "layer_idx": self.layer_idx,
            "deleted_layer": self._deleted_layer,
        }
        
        self._log_info(f"Deleted layer {self.layer_idx}")
        return CommandResult.ok({"layer_idx": self.layer_idx})
    
    def _do_undo(self) -> CommandResult:
        """Restore deleted layer."""
        if not self._undo_data or not self._deleted_layer:
            return CommandResult.fail("No undo data available")
        
        self._log_info(f"Restored layer {self.layer_idx}")
        return CommandResult.ok()
    
    def to_osc(self) -> Tuple[str, List[Any]]:
        """Convert to OSC message format."""
        return (
            self.OSC_ADDRESS,
            [
                self._context.timecode_no,
                self._context.track_group_idx,
                self.layer_idx,
            ]
        )


class RenameLayerCommand(BaseMA3Command):
    """
    Rename an existing layer.
    
    Editor: Updates the layer name
    MA3: Updates the track name
    """
    
    COMMAND_TYPE = "rename_layer"
    OSC_ADDRESS = "/echozero/timecode/rename_track"
    
    def __init__(
        self,
        context: CommandContext,
        layer_idx: int,
        new_name: str,
    ):
        super().__init__(context)
        self.layer_idx = layer_idx
        self.new_name = new_name
        
        # Original name for undo
        self._original_name: Optional[str] = None
    
    def validate(self) -> ValidationResult:
        """Validate rename parameters."""
        result = ValidationResult.success()
        
        if self.layer_idx < 0:
            result.add_error("Layer index cannot be negative")
        
        if not self.new_name or not self.new_name.strip():
            result.add_error("New name cannot be empty")
        
        return result
    
    def _do_execute(self) -> CommandResult:
        """Rename layer in Editor block."""
        if not self._context.editor_block_id:
            return CommandResult.fail("No editor block configured")
        
        # Store original name for undo
        self._original_name = ""  # Placeholder - get from actual layer
        
        self._undo_data = {
            "layer_idx": self.layer_idx,
            "original_name": self._original_name,
        }
        
        self._log_info(f"Renamed layer {self.layer_idx} to '{self.new_name}'")
        return CommandResult.ok({"layer_idx": self.layer_idx, "new_name": self.new_name})
    
    def _do_undo(self) -> CommandResult:
        """Restore original layer name."""
        if not self._undo_data:
            return CommandResult.fail("No undo data available")
        
        original = self._undo_data["original_name"]
        self._log_info(f"Restored layer {self.layer_idx} name to '{original}'")
        return CommandResult.ok()
    
    def to_osc(self) -> Tuple[str, List[Any]]:
        """Convert to OSC message format."""
        return (
            self.OSC_ADDRESS,
            [
                self._context.timecode_no,
                self._context.track_group_idx,
                self.layer_idx,
                self.new_name,
            ]
        )
