"""
Service for managing ShowManager block state independently of UI panel lifecycle.

This service handles:
- Connection status tracking (Editor connection)
- Auto-connect logic
- Connection status polling
- State persistence in block metadata
"""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from PyQt6.QtCore import QTimer, QObject, pyqtSignal
from src.utils.message import Log
from src.application.events.event_bus import EventBus
from src.application.events.events import BlockChanged, BlockRemoved, ConnectionCreated, ConnectionRemoved


@dataclass
class ShowManagerConnectionState:
    """State for a ShowManager block's connection status."""
    block_id: str
    connected_editor_id: Optional[str] = None
    connection_status: str = "unknown"  # "connected", "disconnected", "failed", "unknown"
    connection_error: Optional[str] = None


class ShowManagerStateService(QObject):
    """
    Service for managing ShowManager block state.
    
    This service ensures state persists independently of UI panel lifecycle.
    """
    
    # Signal emitted when connection state changes
    # Note: PyQt signals don't support Optional types, so we use str and pass empty string for None
    connection_state_changed = pyqtSignal(str, str, str, str)  # block_id, status, editor_id (or ""), error (or "")
    
    def __init__(self, event_bus: EventBus, facade):
        super().__init__()
        self._event_bus = event_bus
        self._facade = facade
        
        # Track connection state for each ShowManager block
        self._connection_states: Dict[str, ShowManagerConnectionState] = {}
        
        # Connection status refresh timers (one per block)
        self._status_timers: Dict[str, QTimer] = {}
        
        # Subscribe to connection events
        self._event_bus.subscribe(ConnectionCreated, self._on_connection_created)
        self._event_bus.subscribe(ConnectionRemoved, self._on_connection_removed)
        self._event_bus.subscribe(BlockRemoved, self._on_block_removed)
        
        Log.info("ShowManagerStateService: Initialized")
    
    def get_connection_state(self, block_id: str) -> ShowManagerConnectionState:
        """Get connection state for a block, loading from metadata if not cached."""
        if block_id not in self._connection_states:
            # Load from block metadata
            self._load_state_from_metadata(block_id)
        
        return self._connection_states.get(block_id, ShowManagerConnectionState(block_id=block_id))
    
    def update_connection_state(
        self,
        block_id: str,
        status: str,
        editor_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """Update connection state and persist to metadata."""
        if block_id not in self._connection_states:
            self._connection_states[block_id] = ShowManagerConnectionState(block_id=block_id)
        
        state = self._connection_states[block_id]
        state.connection_status = status
        state.connected_editor_id = editor_id
        state.connection_error = error
        
        # Persist to block metadata
        self._save_state_to_metadata(block_id, state)
        
        # Emit signal (PyQt signals don't support None, use empty string)
        self.connection_state_changed.emit(
            block_id,
            status,
            editor_id or "",
            error or ""
        )
        
        # Polling disabled - connection is now completely manual
        # User must manually check connection status via "Start Listening" button
        self.stop_status_polling(block_id)
    
    def start_status_polling(self, block_id: str) -> None:
        """Start polling connection status for a block."""
        if block_id in self._status_timers:
            return  # Already polling
        
        timer = QTimer()
        timer.timeout.connect(lambda: self._check_connection_status(block_id))
        timer.start(5000)  # Every 5 seconds
        self._status_timers[block_id] = timer
    
    def stop_status_polling(self, block_id: str) -> None:
        """Stop polling connection status for a block."""
        if block_id in self._status_timers:
            self._status_timers[block_id].stop()
            self._status_timers[block_id].deleteLater()
            del self._status_timers[block_id]
    
    def attempt_auto_connect(self, block_id: str) -> bool:
        """
        Attempt to auto-connect ShowManager to an Editor block.
        
        Returns True if connection was successful or already exists.
        Does not retry if status is already "failed" - user must manually fix.
        """
        # Get current state - don't retry if already failed
        state = self.get_connection_state(block_id)
        if state.connection_status == "failed":
            Log.debug(f"ShowManagerStateService: Skipping auto-connect for {block_id} - status is 'failed'. User must manually fix.")
            return False
        
        # Get block
        block_result = self._facade.describe_block(block_id)
        if not block_result.success or not block_result.data:
            return False
        
        block = block_result.data
        if block.type != "ShowManager":
            return False
        
        # Check existing connections
        connections_result = self._facade.list_connections()
        if connections_result.success and connections_result.data:
            for conn in connections_result.data:
                # Check if this ShowManager is already connected to an Editor
                if conn.source_block_id == block_id and conn.source_output_name == "manipulator":
                    target_result = self._facade.describe_block(conn.target_block_id)
                    if target_result.success and target_result.data and target_result.data.type == "Editor":
                        self.update_connection_state(block_id, "connected", conn.target_block_id, None)
                        return True
                elif conn.target_block_id == block_id and conn.target_input_name == "manipulator":
                    source_result = self._facade.describe_block(conn.source_block_id)
                    if source_result.success and source_result.data and source_result.data.type == "Editor":
                        self.update_connection_state(block_id, "connected", conn.source_block_id, None)
                        return True
        
        # Find available Editor blocks
        blocks_result = self._facade.list_blocks()
        if not blocks_result.success or not blocks_result.data:
            return False
        
        editor_blocks = [b for b in blocks_result.data if b.type == "Editor"]
        if not editor_blocks:
            return False
        
        # Try to connect to first available Editor
        target_editor = editor_blocks[0]
        
        # Check if Editor is already connected to another ShowManager
        if connections_result.success and connections_result.data:
            for conn in connections_result.data:
                if (conn.source_block_id == target_editor.id and conn.source_output_name == "manipulator") or \
                   (conn.target_block_id == target_editor.id and conn.target_input_name == "manipulator"):
                    other_block_id = conn.source_block_id if conn.target_block_id == target_editor.id else conn.target_block_id
                    other_result = self._facade.describe_block(other_block_id)
                    if other_result.success and other_result.data and other_result.data.type == "ShowManager":
                        if other_block_id != block_id:
                            self.update_connection_state(
                                block_id,
                                "failed",
                                None,
                                f"Editor '{target_editor.name}' is already connected to another ShowManager"
                            )
                            return False
        
        # Attempt connection
        result = self._facade.connect_blocks(
            source_block_id=block_id,
            source_output="manipulator",
            target_block_id=target_editor.id,
            target_input="manipulator"
        )
        
        if result.success:
            self.update_connection_state(block_id, "connected", target_editor.id, None)
            return True
        else:
            self.update_connection_state(block_id, "failed", None, result.message or "Connection failed")
            return False
    
    def _check_connection_status(self, block_id: str) -> None:
        """Check and update connection status for a block."""
        # Get current state
        state = self.get_connection_state(block_id)
        
        # Don't check if status is "failed" - user must manually retry
        if state.connection_status == "failed":
            # Stop polling on failure - user must manually fix
            self.stop_status_polling(block_id)
            return
        
        # Check if connection still exists
        if state.connected_editor_id:
            connections_result = self._facade.list_connections()
            if connections_result.success and connections_result.data:
                connection_exists = False
                for conn in connections_result.data:
                    if ((conn.source_block_id == block_id and conn.target_block_id == state.connected_editor_id) or
                        (conn.target_block_id == block_id and conn.source_block_id == state.connected_editor_id)):
                        if "manipulator" in (conn.source_output_name, conn.target_input_name):
                            connection_exists = True
                            break
                
                if not connection_exists:
                    # Connection was removed
                    self.update_connection_state(block_id, "disconnected", None, None)
                    return
        
        # Don't auto-retry if status is "disconnected" or "unknown" - user must manually connect
        # Only check existing connections, don't attempt new ones
    
    def _load_state_from_metadata(self, block_id: str) -> None:
        """Load connection state from block metadata."""
        block_result = self._facade.describe_block(block_id)
        if not block_result.success or not block_result.data:
            return
        
        block = block_result.data
        metadata = block.metadata or {}
        
        state = ShowManagerConnectionState(
            block_id=block_id,
            connected_editor_id=metadata.get("connected_editor_id"),
            connection_status=metadata.get("connection_status", "unknown"),
            connection_error=metadata.get("connection_error")
        )
        
        self._connection_states[block_id] = state
        
        # Polling disabled - connection is now completely manual
        # Always stop polling regardless of state
        self.stop_status_polling(block_id)
    
    def _save_state_to_metadata(self, block_id: str, state: ShowManagerConnectionState) -> None:
        """Save connection state to block metadata."""
        if not self._facade.current_project_id:
            Log.warning(f"ShowManagerStateService: Cannot save state - no project loaded")
            return
        
        block_result = self._facade.describe_block(block_id)
        if not block_result.success or not block_result.data:
            Log.warning(f"ShowManagerStateService: Cannot save state - block not found: {block_id}")
            return
        
        block = block_result.data
        metadata = block.metadata or {}
        
        # Check if state actually changed before updating
        current_editor_id = metadata.get("connected_editor_id")
        current_status = metadata.get("connection_status", "unknown")
        current_error = metadata.get("connection_error")
        
        # Convert None to empty string for comparison
        new_editor_id = state.connected_editor_id or ""
        new_status = state.connection_status or "unknown"
        new_error = state.connection_error or ""
        
        if (current_editor_id == new_editor_id and 
            current_status == new_status and 
            current_error == new_error):
            # No changes - skip update to prevent unnecessary events
            Log.debug(f"ShowManagerStateService: State unchanged for {block_id}, skipping update")
            return
        
        metadata["connected_editor_id"] = state.connected_editor_id
        metadata["connection_status"] = state.connection_status
        metadata["connection_error"] = state.connection_error
        
        # Update block using block_service (facade doesn't expose project_repo)
        self._facade.block_service.update_block(
            self._facade.current_project_id,
            block_id,
            block
        )
    
    def _manage_status_timer(self, block_id: str, should_poll: bool) -> None:
        """Start or stop status polling timer based on connection state."""
        if should_poll:
            self.start_status_polling(block_id)
        else:
            self.stop_status_polling(block_id)
    
    def _on_connection_created(self, event: ConnectionCreated) -> None:
        """Handle connection created event."""
        # ConnectionCreated event data is in event.data dict
        if not event.data:
            return
        
        source_block_id = event.data.get("source_block_id")
        target_block_id = event.data.get("target_block_id")
        source_output_name = event.data.get("source_output_name")
        target_input_name = event.data.get("target_input_name")
        
        if not source_block_id or not target_block_id:
            return
        
        # Check if this connection involves a ShowManager
        source_result = self._facade.describe_block(source_block_id)
        target_result = self._facade.describe_block(target_block_id)
        
        if source_result.success and source_result.data and source_result.data.type == "ShowManager":
            if target_result.success and target_result.data and target_result.data.type == "Editor":
                if source_output_name == "manipulator" and target_input_name == "manipulator":
                    self.update_connection_state(
                        source_block_id,
                        "connected",
                        target_block_id,
                        None
                    )
        
        if target_result.success and target_result.data and target_result.data.type == "ShowManager":
            if source_result.success and source_result.data and source_result.data.type == "Editor":
                if source_output_name == "manipulator" and target_input_name == "manipulator":
                    self.update_connection_state(
                        target_block_id,
                        "connected",
                        source_block_id,
                        None
                    )
    
    def _on_connection_removed(self, event: ConnectionRemoved) -> None:
        """Handle connection removed event."""
        # ConnectionRemoved event data is in event.data dict
        if not event.data:
            return
        
        source_block_id = event.data.get("source_block_id")
        target_block_id = event.data.get("target_block_id")
        
        if not source_block_id or not target_block_id:
            return
        
        # Check if this connection involved a ShowManager
        source_result = self._facade.describe_block(source_block_id)
        target_result = self._facade.describe_block(target_block_id)
        
        if source_result.success and source_result.data and source_result.data.type == "ShowManager":
            state = self.get_connection_state(source_block_id)
            if state.connected_editor_id in (source_block_id, target_block_id):
                self.update_connection_state(source_block_id, "disconnected", None, None)
        
        if target_result.success and target_result.data and target_result.data.type == "ShowManager":
            state = self.get_connection_state(target_block_id)
            if state.connected_editor_id in (source_block_id, target_block_id):
                self.update_connection_state(target_block_id, "disconnected", None, None)
    
    def _on_block_removed(self, event: BlockRemoved) -> None:
        """Handle block removed event."""
        if event.block_id in self._connection_states:
            del self._connection_states[event.block_id]
        
        self.stop_status_polling(event.block_id)
    
    def cleanup(self) -> None:
        """Cleanup service on application shutdown."""
        for timer in self._status_timers.values():
            timer.stop()
            timer.deleteLater()
        self._status_timers.clear()
        self._connection_states.clear()
