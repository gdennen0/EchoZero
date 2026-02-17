"""
Action Item Commands - Standardized Undoable Action Item Operations

All action item-related commands that flow through facade.command_bus.

STANDARD ACTION ITEM COMMANDS
=============================

| Command                    | Redo Action           | Undo Action              |
|----------------------------|----------------------|--------------------------|
| AddActionItemCommand       | Creates action item  | Deletes action item      |
| UpdateActionItemCommand    | Updates action item  | Restores old values      |
| DeleteActionItemCommand    | Deletes action item  | Recreates action item    |
| ReorderActionItemsCommand  | Reorders items       | Restores old order       |

USAGE
=====
```python
from src.application.commands import AddActionItemCommand

cmd = AddActionItemCommand(facade, action_set_id, action_item)
facade.command_bus.execute(cmd)
```
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Any
from .base_command import EchoZeroCommand

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade
    from src.features.projects.domain import ActionItem


class AddActionItemCommand(EchoZeroCommand):
    """
    Add an action item to an action set.
    
    Redo: Creates a new action item
    Undo: Deletes the created action item
    
    Args:
        facade: ApplicationFacade instance
        action_set_id: ID of the action set
        action_item: ActionItem entity to add
    """
    
    COMMAND_TYPE = "action_item.add"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        action_set_id: str,
        action_item: "ActionItem"
    ):
        description = f"Add action '{action_item.action_name}'"
        super().__init__(facade, description)
        
        self._action_set_id = action_set_id
        self._action_item = action_item
        self._created_action_item_id: Optional[str] = None
    
    def redo(self):
        """Create the action item."""
        # Set order_index based on current count, but only if not already set
        # This allows callers to explicitly set order_index for insertions
        if self._action_item.order_index is None:
            if hasattr(self._facade, 'action_item_repo') and self._facade.action_item_repo:
                existing_items = self._facade.action_item_repo.list_by_action_set(self._action_set_id)
                self._action_item.order_index = len(existing_items)
            else:
                # Fallback: set order_index to 0 if repo not available
                self._action_item.order_index = 0
        
        result = self._facade.add_action_item(self._action_set_id, self._action_item)
        if result.success and result.data:
            self._created_action_item_id = result.data.id
        else:
            self._log_error(f"Failed to add action item: {result.message if hasattr(result, 'message') else 'Unknown error'}")
    
    def undo(self):
        """Delete the created action item."""
        if self._created_action_item_id:
            self._facade.remove_action_item(self._created_action_item_id)


class UpdateActionItemCommand(EchoZeroCommand):
    """
    Update an action item.
    
    Redo: Updates the action item with new values
    Undo: Restores the old values
    
    Args:
        facade: ApplicationFacade instance
        action_item: ActionItem entity with updated values
        old_action_item: Optional ActionItem with old values (captured if not provided)
    """
    
    COMMAND_TYPE = "action_item.update"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        action_item: "ActionItem",
        old_action_item: Optional["ActionItem"] = None
    ):
        description = f"Update action '{action_item.action_name}'"
        super().__init__(facade, description)
        
        self._action_item = action_item
        self._old_action_item = old_action_item
    
    def redo(self):
        """Update the action item."""
        if not self._old_action_item and self._action_item.id:
            # Capture old state if not provided
            if hasattr(self._facade, 'action_item_repo') and self._facade.action_item_repo:
                old_item = self._facade.action_item_repo.get(self._action_item.id)
                if old_item:
                    from src.features.projects.domain import ActionItem
                    self._old_action_item = ActionItem(
                        id=old_item.id,
                        action_set_id=old_item.action_set_id,
                        project_id=old_item.project_id,
                        action_type=old_item.action_type,
                        block_id=old_item.block_id,
                        block_name=old_item.block_name,
                        action_name=old_item.action_name,
                        action_description=old_item.action_description,
                        action_args=old_item.action_args.copy() if old_item.action_args else {},
                        order_index=old_item.order_index,
                        created_at=old_item.created_at,
                        modified_at=old_item.modified_at,
                        metadata=old_item.metadata.copy() if old_item.metadata else {}
                    )
        
        result = self._facade.update_action_item(self._action_item)
        if not result.success:
            self._log_error(f"Failed to update action item: {result.message if hasattr(result, 'message') else 'Unknown error'}")
    
    def undo(self):
        """Restore the old action item."""
        if self._old_action_item:
            self._facade.update_action_item(self._old_action_item)


class DeleteActionItemCommand(EchoZeroCommand):
    """
    Delete an action item.
    
    Redo: Deletes the action item
    Undo: Recreates the action item with original properties
    
    Args:
        facade: ApplicationFacade instance
        action_item_id: ID of action item to delete
    """
    
    COMMAND_TYPE = "action_item.delete"
    
    def __init__(self, facade: "ApplicationFacade", action_item_id: str):
        # Get action item for description
        action_item = None
        if facade.action_item_repo:
            action_item = facade.action_item_repo.get(action_item_id)
        
        name = action_item.action_name if action_item else action_item_id
        description = f"Delete action '{name}'"
        super().__init__(facade, description)
        
        self._action_item_id = action_item_id
        self._deleted_action_item: Optional["ActionItem"] = None
    
    def redo(self):
        """Delete the action item."""
        # Capture state before deletion
        if not self._deleted_action_item:
            if hasattr(self._facade, 'action_item_repo') and self._facade.action_item_repo:
                action_item = self._facade.action_item_repo.get(self._action_item_id)
                if action_item:
                    from src.features.projects.domain import ActionItem
                    self._deleted_action_item = ActionItem(
                        id=action_item.id,
                        action_set_id=action_item.action_set_id,
                        project_id=action_item.project_id,
                        action_type=action_item.action_type,
                        block_id=action_item.block_id,
                        block_name=action_item.block_name,
                        action_name=action_item.action_name,
                        action_description=action_item.action_description,
                        action_args=action_item.action_args.copy() if action_item.action_args else {},
                        order_index=action_item.order_index,
                        created_at=action_item.created_at,
                        modified_at=action_item.modified_at,
                        metadata=action_item.metadata.copy() if action_item.metadata else {}
                    )
        
        result = self._facade.remove_action_item(self._action_item_id)
        if not result.success:
            self._log_error(f"Failed to delete action item: {result.message if hasattr(result, 'message') else 'Unknown error'}")
    
    def undo(self):
        """Recreate the deleted action item."""
        if self._deleted_action_item:
            self._facade.add_action_item(
                self._deleted_action_item.action_set_id,
                self._deleted_action_item
            )


class ReorderActionItemsCommand(EchoZeroCommand):
    """
    Reorder action items in an action set.
    
    Redo: Reorders action items to new order
    Undo: Restores original order
    
    Args:
        facade: ApplicationFacade instance
        action_set_id: ID of the action set
        new_order: List of action item IDs in new order
    """
    
    COMMAND_TYPE = "action_item.reorder"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        action_set_id: str,
        new_order: List[str]
    ):
        description = f"Reorder {len(new_order)} action{'s' if len(new_order) != 1 else ''}"
        super().__init__(facade, description)
        
        self._action_set_id = action_set_id
        self._new_order = new_order
        self._old_order: Optional[List[str]] = None
    
    def redo(self):
        """Reorder the action items."""
        # Capture old order if not already captured
        if not self._old_order:
            if hasattr(self._facade, 'action_item_repo') and self._facade.action_item_repo:
                items = self._facade.action_item_repo.list_by_action_set(self._action_set_id)
                self._old_order = [item.id for item in sorted(items, key=lambda x: x.order_index)]
        
        # Update order_index for each item
        for new_index, item_id in enumerate(self._new_order):
            if hasattr(self._facade, 'action_item_repo') and self._facade.action_item_repo:
                item = self._facade.action_item_repo.get(item_id)
                if item:
                    item.order_index = new_index
                    self._facade.update_action_item(item)
    
    def undo(self):
        """Restore the original order."""
        if self._old_order:
            for old_index, item_id in enumerate(self._old_order):
                if hasattr(self._facade, 'action_item_repo') and self._facade.action_item_repo:
                    item = self._facade.action_item_repo.get(item_id)
                    if item:
                        item.order_index = old_index
                        self._facade.update_action_item(item)

