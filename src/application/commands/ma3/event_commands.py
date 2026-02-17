"""
MA3 Event Commands

Commands for manipulating events in both EchoZero and grandMA3.
Events represent timed triggers (e.g., cue points, lighting changes).
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    BaseMA3Command,
    CommandContext,
    CommandResult,
    ValidationResult,
)


class AddEventCommand(BaseMA3Command):
    """
    Add a new event at a specified time.
    
    Editor: Creates event in the target layer
    MA3: Creates event in the target track
    """
    
    COMMAND_TYPE = "add_event"
    OSC_ADDRESS = "/echozero/timecode/add_event"
    
    def __init__(
        self,
        context: CommandContext,
        time: float,
        classification: str = "",
        duration: float = 0.0,
        event_type: str = "cmd",  # "cmd" or "fader"
        properties: Optional[Dict[str, Any]] = None,
        track_idx: int = 0,
    ):
        super().__init__(context)
        self.time = time
        self.classification = classification
        self.duration = duration
        self.event_type = event_type
        self.properties = properties or {}
        self.track_idx = track_idx
        
        # Will be set after execution for undo
        self._created_event_id: Optional[str] = None
        self._created_event_idx: Optional[int] = None
    
    def validate(self) -> ValidationResult:
        """Validate event parameters."""
        result = ValidationResult.success()
        
        if self.time < 0:
            result.add_error("Event time cannot be negative")
        
        if self.duration < 0:
            result.add_error("Event duration cannot be negative")
        
        if self.event_type not in ("cmd", "fader"):
            result.add_error(f"Invalid event type: {self.event_type}. Use 'cmd' or 'fader'.")
        
        return result
    
    def _do_execute(self) -> CommandResult:
        """Create event in Editor block."""
        if not self._context.editor_block_id:
            return CommandResult.fail("No editor block configured")
        
        # Get editor block and add event
        result = self.facade.describe_block(self._context.editor_block_id)
        if not result.success:
            return CommandResult.fail(f"Failed to get editor block: {result.message}")
        
        # Use the event data service to add event
        from src.domain.entities.event_data_item import Event
        
        # Create event with metadata
        metadata = self.properties.copy()
        metadata["type"] = self.event_type
        
        # Store undo data
        self._undo_data = {
            "time": self.time,
            "classification": self.classification,
        }
        
        # Add event through facade or directly to event data
        # For now, return success - actual implementation depends on Editor block API
        self._log_info(f"Added event at {self.time}s: {self.classification}")
        return CommandResult.ok({"time": self.time, "classification": self.classification})
    
    def _do_undo(self) -> CommandResult:
        """Remove the created event."""
        if not self._undo_data:
            return CommandResult.fail("No undo data available")
        
        # Remove the event that was created
        self._log_info(f"Removed event at {self._undo_data['time']}s")
        return CommandResult.ok()
    
    def to_osc(self) -> Tuple[str, List[Any]]:
        """Convert to OSC message format."""
        props_json = json.dumps(self.properties)
        return (
            self.OSC_ADDRESS,
            [
                self._context.timecode_no,
                self._context.track_group_idx,
                self.track_idx,
                self.time,
                self.event_type,
                props_json,
            ]
        )


class MoveEventCommand(BaseMA3Command):
    """
    Move an existing event to a new time.
    
    Editor: Updates event time in the layer
    MA3: Updates event time in the track
    """
    
    COMMAND_TYPE = "move_event"
    OSC_ADDRESS = "/echozero/timecode/move_event"
    
    def __init__(
        self,
        context: CommandContext,
        event_idx: int,
        new_time: float,
        track_idx: int = 0,
    ):
        super().__init__(context)
        self.event_idx = event_idx
        self.new_time = new_time
        self.track_idx = track_idx
        
        # Original time for undo
        self._original_time: Optional[float] = None
    
    def validate(self) -> ValidationResult:
        """Validate move parameters."""
        result = ValidationResult.success()
        
        if self.event_idx < 0:
            result.add_error("Event index cannot be negative")
        
        if self.new_time < 0:
            result.add_error("New time cannot be negative")
        
        return result
    
    def _do_execute(self) -> CommandResult:
        """Move event in Editor block."""
        if not self._context.editor_block_id:
            return CommandResult.fail("No editor block configured")
        
        # Store original time for undo
        # In real implementation, get this from the event data
        self._original_time = 0.0  # Placeholder
        
        self._undo_data = {
            "event_idx": self.event_idx,
            "original_time": self._original_time,
        }
        
        self._log_info(f"Moved event {self.event_idx} to {self.new_time}s")
        return CommandResult.ok({"event_idx": self.event_idx, "new_time": self.new_time})
    
    def _do_undo(self) -> CommandResult:
        """Move event back to original time."""
        if not self._undo_data:
            return CommandResult.fail("No undo data available")
        
        original_time = self._undo_data["original_time"]
        self._log_info(f"Moved event {self.event_idx} back to {original_time}s")
        return CommandResult.ok()
    
    def to_osc(self) -> Tuple[str, List[Any]]:
        """Convert to OSC message format."""
        return (
            self.OSC_ADDRESS,
            [
                self._context.timecode_no,
                self._context.track_group_idx,
                self.track_idx,
                self.event_idx,
                self.new_time,
            ]
        )


class DeleteEventCommand(BaseMA3Command):
    """
    Delete an existing event.
    
    Editor: Removes event from the layer
    MA3: Removes event from the track
    """
    
    COMMAND_TYPE = "delete_event"
    OSC_ADDRESS = "/echozero/timecode/delete_event"
    
    def __init__(
        self,
        context: CommandContext,
        event_idx: int,
        track_idx: int = 0,
    ):
        super().__init__(context)
        self.event_idx = event_idx
        self.track_idx = track_idx
        
        # Deleted event data for undo
        self._deleted_event: Optional[Dict[str, Any]] = None
    
    def validate(self) -> ValidationResult:
        """Validate delete parameters."""
        result = ValidationResult.success()
        
        if self.event_idx < 0:
            result.add_error("Event index cannot be negative")
        
        return result
    
    def _do_execute(self) -> CommandResult:
        """Delete event from Editor block."""
        if not self._context.editor_block_id:
            return CommandResult.fail("No editor block configured")
        
        # Store deleted event data for undo
        self._deleted_event = {
            "time": 0.0,  # Placeholder - get from actual event
            "classification": "",
            "duration": 0.0,
            "properties": {},
        }
        
        self._undo_data = {
            "event_idx": self.event_idx,
            "deleted_event": self._deleted_event,
        }
        
        self._log_info(f"Deleted event {self.event_idx}")
        return CommandResult.ok({"event_idx": self.event_idx})
    
    def _do_undo(self) -> CommandResult:
        """Restore deleted event."""
        if not self._undo_data or not self._deleted_event:
            return CommandResult.fail("No undo data available")
        
        # Recreate the deleted event at its original position
        self._log_info(f"Restored event {self.event_idx}")
        return CommandResult.ok()
    
    def to_osc(self) -> Tuple[str, List[Any]]:
        """Convert to OSC message format."""
        return (
            self.OSC_ADDRESS,
            [
                self._context.timecode_no,
                self._context.track_group_idx,
                self.track_idx,
                self.event_idx,
            ]
        )


class UpdateEventCommand(BaseMA3Command):
    """
    Update properties of an existing event.
    
    Editor: Updates event properties in the layer
    MA3: Updates event properties in the track
    """
    
    COMMAND_TYPE = "update_event"
    OSC_ADDRESS = "/echozero/timecode/update_event"
    
    def __init__(
        self,
        context: CommandContext,
        event_idx: int,
        track_idx: int = 0,
        classification: Optional[str] = None,
        duration: Optional[float] = None,
        properties: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(context)
        self.event_idx = event_idx
        self.track_idx = track_idx
        self.classification = classification
        self.duration = duration
        self.properties = properties
        
        # Original values for undo
        self._original_values: Optional[Dict[str, Any]] = None
    
    def validate(self) -> ValidationResult:
        """Validate update parameters."""
        result = ValidationResult.success()
        
        if self.event_idx < 0:
            result.add_error("Event index cannot be negative")
        
        if self.duration is not None and self.duration < 0:
            result.add_error("Duration cannot be negative")
        
        # At least one property should be updated
        if self.classification is None and self.duration is None and not self.properties:
            result.add_error("At least one property must be specified for update")
        
        return result
    
    def _do_execute(self) -> CommandResult:
        """Update event in Editor block."""
        if not self._context.editor_block_id:
            return CommandResult.fail("No editor block configured")
        
        # Store original values for undo
        self._original_values = {
            "classification": "",  # Placeholder
            "duration": 0.0,
            "properties": {},
        }
        
        self._undo_data = {
            "event_idx": self.event_idx,
            "original_values": self._original_values,
        }
        
        self._log_info(f"Updated event {self.event_idx}")
        return CommandResult.ok({"event_idx": self.event_idx})
    
    def _do_undo(self) -> CommandResult:
        """Restore original event values."""
        if not self._undo_data or not self._original_values:
            return CommandResult.fail("No undo data available")
        
        self._log_info(f"Restored original values for event {self.event_idx}")
        return CommandResult.ok()
    
    def to_osc(self) -> Tuple[str, List[Any]]:
        """Convert to OSC message format."""
        updates = {}
        if self.classification is not None:
            updates["classification"] = self.classification
        if self.duration is not None:
            updates["duration"] = self.duration
        if self.properties:
            updates["properties"] = self.properties
        
        updates_json = json.dumps(updates)
        return (
            self.OSC_ADDRESS,
            [
                self._context.timecode_no,
                self._context.track_group_idx,
                self.track_idx,
                self.event_idx,
                updates_json,
            ]
        )
