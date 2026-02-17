"""
Batch Sync Command

Executes multiple sync operations in macro.
Syncs multiple layers/tracks at once as a single undoable operation.
"""
from typing import TYPE_CHECKING, List, Dict, Any
from dataclasses import dataclass

from src.application.commands.base_command import EchoZeroCommand
from src.utils.message import Log

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


@dataclass
class SyncPair:
    """Defines a sync pair for batch operations."""
    show_manager_block_id: str
    editor_block_id: str
    editor_layer_name: str
    ma3_track_coord: str
    direction: str = "editor_to_ma3"  # or "ma3_to_editor"


class BatchSyncCommand(EchoZeroCommand):
    """
    Sync multiple layers/tracks at once (undoable).
    
    Redo: Executes multiple sync operations in macro
    Undo: Reverses all sync operations
    
    Handles:
    - Executing multiple SyncLayerCommand operations
    - Grouping operations in macro (single undo step)
    - Handling conflicts according to per-entity settings
    
    Args:
        facade: ApplicationFacade instance
        sync_pairs: List of SyncPair objects defining what to sync
    """
    
    COMMAND_TYPE = "layer_sync.batch_sync"
    
    def __init__(
        self,
        facade: "ApplicationFacade",
        sync_pairs: List[SyncPair]
    ):
        entity_count = len(sync_pairs)
        super().__init__(facade, f"Batch Sync: {entity_count} layers")
        
        self._sync_pairs = sync_pairs
        
        # State for undo
        self._sync_commands: List = []
        self._success_count: int = 0
        self._error_count: int = 0
    
    def redo(self):
        """Execute batch sync operations."""
        from .sync_layer_command import SyncLayerCommand
        
        Log.info(f"BatchSyncCommand: Starting batch sync of {len(self._sync_pairs)} pairs")
        
        # Start macro to group all operations as one undo step
        self._facade.command_bus.begin_macro(self.text())
        
        try:
            # Execute each sync command
            for pair in self._sync_pairs:
                try:
                    sync_cmd = SyncLayerCommand(
                        facade=self._facade,
                        show_manager_block_id=pair.show_manager_block_id,
                        editor_block_id=pair.editor_block_id,
                        editor_layer_name=pair.editor_layer_name,
                        ma3_track_coord=pair.ma3_track_coord,
                        direction=pair.direction,
                        clear_target=True
                    )
                    self._facade.command_bus.execute(sync_cmd)
                    self._sync_commands.append(sync_cmd)
                    self._success_count += 1
                except Exception as e:
                    Log.error(f"BatchSyncCommand: Failed to sync {pair.editor_layer_name} <-> {pair.ma3_track_coord}: {e}")
                    self._error_count += 1
        finally:
            # End macro
            self._facade.command_bus.end_macro()
        
        Log.info(f"BatchSyncCommand: Completed {self._success_count} syncs, {self._error_count} errors")
    
    def undo(self):
        """Reverse all sync operations."""
        # Macro undo should reverse all operations automatically
        # Individual commands handle their own undo
        Log.debug(f"BatchSyncCommand: Undoing {len(self._sync_commands)} sync operations")
