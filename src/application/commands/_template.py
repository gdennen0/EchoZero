"""
Command Template - Copy this file to create new commands

USAGE:
1. Copy this file to a new name (e.g., my_commands.py)
2. Rename the class
3. Implement redo() and undo()
4. Add to __init__.py exports
5. Use via facade.command_bus.execute()
"""

from typing import TYPE_CHECKING, Optional, Any
from .base_command import EchoZeroCommand

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


# =============================================================================
# TEMPLATE: Simple Value Change
# =============================================================================

class ChangeValueCommand(EchoZeroCommand):
    """
    Change a single value.
    
    Redo: Sets value to new_value
    Undo: Restores old_value
    """
    
    COMMAND_TYPE = "template.change_value"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        target_id: str,
        new_value: Any,
        old_value: Optional[Any] = None  # Pre-captured for efficiency
    ):
        super().__init__(facade, f"Change Value")
        self._target_id = target_id
        self._new_value = new_value
        self._old_value = old_value
    
    def redo(self):
        # Capture old value first time
        if self._old_value is None:
            self._old_value = self._get_current_value()
        
        self._set_value(self._new_value)
    
    def undo(self):
        if self._old_value is not None:
            self._set_value(self._old_value)
    
    def _get_current_value(self) -> Any:
        """Get current value from facade."""
        # Example: return self._facade.get_something(self._target_id).data
        raise NotImplementedError("Implement this")
    
    def _set_value(self, value: Any):
        """Set value via facade."""
        # Example: self._facade.set_something(self._target_id, value)
        raise NotImplementedError("Implement this")


# =============================================================================
# TEMPLATE: Create Item
# =============================================================================

class CreateItemCommand(EchoZeroCommand):
    """
    Create a new item.
    
    Redo: Creates the item
    Undo: Deletes the item
    """
    
    COMMAND_TYPE = "template.create_item"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        item_type: str,
        item_name: str,
        **item_properties
    ):
        super().__init__(facade, f"Create {item_name}")
        self._item_type = item_type
        self._item_name = item_name
        self._properties = item_properties
        self._created_id: Optional[str] = None
    
    def redo(self):
        # Create the item
        result = self._facade.create_item(
            self._item_type,
            self._item_name,
            **self._properties
        )
        if result.success and result.data:
            self._created_id = result.data.id
    
    def undo(self):
        if self._created_id:
            self._facade.delete_item(self._created_id)


# =============================================================================
# TEMPLATE: Delete Item (with state preservation)
# =============================================================================

class DeleteItemCommand(EchoZeroCommand):
    """
    Delete an item, preserving state for undo.
    
    Redo: Deletes the item
    Undo: Recreates the item with all original properties
    """
    
    COMMAND_TYPE = "template.delete_item"
    
    def __init__(self, facade: "ApplicationFacade", item_id: str):
        super().__init__(facade, "Delete Item")
        self._item_id = item_id
        self._deleted_data: Optional[dict] = None
    
    def redo(self):
        # Save all state before deletion (first time only)
        if self._deleted_data is None:
            item = self._facade.get_item(self._item_id).data
            self._deleted_data = {
                "type": item.type,
                "name": item.name,
                "properties": item.properties.copy(),
                # Add any other state needed to restore
            }
            # Update description
            self.setText(f"Delete {item.name}")
        
        self._facade.delete_item(self._item_id)
    
    def undo(self):
        if not self._deleted_data:
            self._log_warning("No data stored, cannot undo")
            return
        
        # Recreate with original properties
        result = self._facade.create_item(
            self._deleted_data["type"],
            self._deleted_data["name"],
            **self._deleted_data["properties"]
        )
        
        if result.success and result.data:
            self._item_id = result.data.id  # Update for subsequent redo


# =============================================================================
# TEMPLATE: Move/Position with Merge Support
# =============================================================================

class MoveItemCommand(EchoZeroCommand):
    """
    Move an item to new coordinates.
    
    Supports merging consecutive moves into single undo step.
    
    Redo: Moves to new position
    Undo: Returns to original position
    """
    
    COMMAND_TYPE = "template.move_item"
    MERGE_ID = 9000  # Unique ID for this command type
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        item_id: str,
        new_x: float,
        new_y: float,
        old_x: Optional[float] = None,
        old_y: Optional[float] = None
    ):
        super().__init__(facade, "Move Item")
        self._item_id = item_id
        self._new_x = new_x
        self._new_y = new_y
        self._old_x = old_x
        self._old_y = old_y
    
    def redo(self):
        # Capture old position (first time only)
        if self._old_x is None:
            pos = self._facade.get_item_position(self._item_id)
            self._old_x = pos.x
            self._old_y = pos.y
        
        self._facade.set_item_position(self._item_id, self._new_x, self._new_y)
    
    def undo(self):
        if self._old_x is not None:
            self._facade.set_item_position(self._item_id, self._old_x, self._old_y)
    
    def id(self) -> int:
        """Enable merging."""
        return self.MERGE_ID
    
    def mergeWith(self, other) -> bool:
        """Merge consecutive moves of same item."""
        if not isinstance(other, MoveItemCommand):
            return False
        if other._item_id != self._item_id:
            return False
        
        # Keep our old position, take their new position
        self._new_x = other._new_x
        self._new_y = other._new_y
        return True


# =============================================================================
# TEMPLATE: Batch Operation
# =============================================================================

class BatchUpdateCommand(EchoZeroCommand):
    """
    Update multiple items at once.
    
    Redo: Applies new values to all items
    Undo: Restores all original values
    """
    
    COMMAND_TYPE = "template.batch_update"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        updates: list  # List of {"id": str, "new_value": Any}
    ):
        count = len(updates)
        super().__init__(facade, f"Update {count} Items")
        
        self._updates = updates
        self._old_values: dict = {}  # id -> old_value
    
    def redo(self):
        # Capture old values (first time only)
        if not self._old_values:
            for update in self._updates:
                item_id = update["id"]
                current = self._facade.get_item(item_id).data.value
                self._old_values[item_id] = current
        
        # Apply all updates
        for update in self._updates:
            self._facade.set_item_value(update["id"], update["new_value"])
    
    def undo(self):
        # Restore all old values
        for item_id, old_value in self._old_values.items():
            self._facade.set_item_value(item_id, old_value)



