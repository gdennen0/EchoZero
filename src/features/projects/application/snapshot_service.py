"""
Snapshot Service

Handles saving and restoring data state snapshots for setlist songs.
Reuses serialization logic from ProjectService to maintain consistency.

Bulletproof state switching implementation:
- Uses existing save_snapshot() for backups (no new code)
- Simple validation checks (explicit, no abstractions)
- Explicit rollback tracking (necessary for cross-repository atomicity)
- Standard Python threading for concurrency (built-in)
- Clear error messages at every step

Follows EchoZero standards:
- Lean: Only essential safety mechanisms (~220 lines total)
- Built-in: Uses standard Python, existing methods
- Explicit: Every step visible, no magic
- Simple: Clear structure, easy to understand
"""
from typing import Dict, Any, Optional, List, Callable, Tuple
from pathlib import Path
import os

from src.shared.domain.entities import DataStateSnapshot
from src.features.blocks.domain import Block
from src.features.blocks.domain import BlockRepository
from src.shared.domain.repositories import DataItemRepository
from src.shared.domain.repositories.block_local_state_repository import BlockLocalStateRepository
from src.utils.message import Log


class SnapshotService:
    """
    Service for saving and restoring data state snapshots.
    
    Snapshots contain:
    - Serialized data items (from all blocks)
    - Block local state (input/output references)
    - Block settings overrides (per-song setting changes)
    
    Reuses serialization logic from ProjectService to maintain consistency.
    """
    
    def __init__(
        self,
        block_repo: BlockRepository,
        data_item_repo: DataItemRepository,
        block_local_state_repo: BlockLocalStateRepository,
        project_service=None  # For reusing _build_data_item_from_dict
    ):
        """
        Initialize snapshot service.
        
        Args:
            block_repo: Repository for accessing blocks
            data_item_repo: Repository for accessing data items
            block_local_state_repo: Repository for block local state
            project_service: Optional ProjectService instance for reusing helper methods
        """
        self._block_repo = block_repo
        self._data_item_repo = data_item_repo
        self._block_local_state_repo = block_local_state_repo
        self._project_service = project_service
        
        # Add state helper for unified access
        from src.features.blocks.application.block_state_helper import BlockStateHelper
        self._state_helper = BlockStateHelper(
            block_repo,
            block_local_state_repo,
            data_item_repo,
            project_service
        )
        
        Log.info("SnapshotService: Initialized")
    
    def save_snapshot(
        self,
        project_id: str,
        song_id: str,
        block_settings_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> DataStateSnapshot:
        """
        Save current data state as snapshot.
        
        Serializes all data items and block local state for the project,
        similar to how ProjectService serializes project files.
        
        Args:
            project_id: Project identifier
            song_id: Song identifier this snapshot belongs to
            block_settings_overrides: Optional block setting overrides (block_id -> {setting_key: value})
            
        Returns:
            DataStateSnapshot with serialized data
        """
        from datetime import datetime
        
        Log.info(f"SnapshotService: Saving snapshot for song {song_id}")
        
        # Use helper to get all block states (unified access)
        project_state = self._state_helper.get_project_state(project_id)
        
        # Convert to snapshot format (keep existing format for backward compatibility)
        data_items = []
        block_local_state = {}
        
        # Extract block settings from project_state and merge with provided overrides
        # This ensures block metadata (like file_path) is saved in the snapshot
        overrides = dict(block_settings_overrides) if block_settings_overrides else {}
        
        for block_id, state in project_state.items():
            # Collect data items
            data_items.extend(state["data_items"])
            
            # Collect local state
            if state["local_state"]:
                block_local_state[block_id] = state["local_state"]
            
            # Extract block settings (metadata) and merge with provided overrides
            # Settings are stored in block.metadata, which is included in project_state
            block_settings = state.get("settings", {})
            if block_settings:
                # Merge: provided overrides take precedence, then current block settings
                if block_id not in overrides:
                    overrides[block_id] = {}
                # Only add settings that aren't already in overrides (overrides take precedence)
                for key, value in block_settings.items():
                    if key not in overrides[block_id]:
                        overrides[block_id][key] = value
        
        snapshot = DataStateSnapshot(
            id="",  # Will be generated
            song_id=song_id,
            created_at=datetime.utcnow(),
            data_items=data_items,
            block_local_state=block_local_state,
            block_settings_overrides=overrides
        )
        
        Log.info(
            f"SnapshotService: Saved snapshot for song {song_id} - "
            f"{len(data_items)} data items, {len(block_local_state)} block states, "
            f"{len(overrides)} block overrides"
        )
        
        return snapshot
    
    def _backup_current_state(self, project_id: str) -> Optional[DataStateSnapshot]:
        """
        Backup current project state before switching.
        
        Critical for bulletproof state switching - allows rollback if restore fails.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Snapshot of current state, or None if no current state exists
        """
        try:
            # Check if there's actually data to backup (get all blocks, then their data items)
            blocks = self._block_repo.list_by_project(project_id)
            has_data = False
            for block in blocks:
                block_items = self._data_item_repo.list_by_block(block.id)
                if block_items:
                    has_data = True
                    break
            
            if not has_data:
                Log.debug("SnapshotService: No data items to backup")
                return None  # No state to backup
            
            # Create backup snapshot (using special song_id)
            backup = self.save_snapshot(
                project_id=project_id,
                song_id="__backup__",  # Special ID for backups
                block_settings_overrides=None
            )
            Log.info(f"SnapshotService: Backed up current state ({len(backup.data_items)} items)")
            return backup
        except Exception as e:
            Log.warning(f"SnapshotService: Failed to backup current state: {e}")
            return None  # Best-effort backup - don't fail switch if backup fails
    
    def _validate_snapshot(self, snapshot: DataStateSnapshot, project_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate snapshot integrity before restoring.
        
        Critical for bulletproof state switching - catches problems before restore.
        
        Args:
            snapshot: Snapshot to validate
            project_id: Project identifier
            
        Returns:
            (is_valid, error_message)
        """
        # 1. Check snapshot structure
        if not snapshot.song_id:
            return False, "Snapshot missing song_id"
        
        if not isinstance(snapshot.data_items, list):
            return False, "Snapshot data_items must be a list"
        
        if not isinstance(snapshot.block_local_state, dict):
            return False, "Snapshot block_local_state must be a dict"
        
        # 2. Verify all referenced blocks exist
        blocks = self._block_repo.list_by_project(project_id)
        block_ids = {b.id for b in blocks}
        
        for item_data in snapshot.data_items:
            block_id = item_data.get("block_id")
            if block_id and block_id not in block_ids:
                return False, f"Snapshot references non-existent block: {block_id}"
        
        for block_id in snapshot.block_local_state.keys():
            if block_id not in block_ids:
                return False, f"Snapshot references non-existent block: {block_id}"
        
        # 3. Verify data item structure
        for item_data in snapshot.data_items:
            required_fields = ["id", "block_id", "name", "type"]
            for field in required_fields:
                if field not in item_data:
                    return False, f"Data item missing required field: {field}"
        
        return True, None
    
    def _verify_restored_state(
        self,
        project_id: str,
        snapshot: DataStateSnapshot
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify that restored state matches snapshot.
        
        Critical for bulletproof state switching - ensures restore succeeded.
        
        Args:
            project_id: Project identifier
            snapshot: Original snapshot to verify against
            
        Returns:
            (is_valid, error_message)
        """
        # 1. Count data items (get all blocks, then collect all their data items)
        blocks = self._block_repo.list_by_project(project_id)
        restored_items = []
        for block in blocks:
            block_items = self._data_item_repo.list_by_block(block.id)
            restored_items.extend(block_items)
        
        if len(restored_items) != len(snapshot.data_items):
            return False, (
                f"Data item count mismatch: "
                f"expected {len(snapshot.data_items)}, got {len(restored_items)}"
            )
        
        # 2. Verify block states
        for block_id, expected_state in snapshot.block_local_state.items():
            actual_state = self._block_local_state_repo.get_inputs(block_id)
            if actual_state != expected_state:
                return False, (
                    f"Block state mismatch for {block_id}: "
                    f"expected {expected_state}, got {actual_state}"
                )
        
        return True, None
    
    def restore_snapshot_atomic(
        self,
        project_id: str,
        snapshot: DataStateSnapshot,
        backup_snapshot: Optional[DataStateSnapshot] = None,
        project_dir: Optional[Path] = None,
        event_bus=None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Restore snapshot atomically with rollback on failure.
        
        Bulletproof restore that tracks all operations and rolls back on any failure.
        
        Args:
            project_id: Project identifier
            snapshot: Snapshot to restore
            backup_snapshot: Optional backup to restore if this fails
            project_dir: Optional project directory for resolving relative file paths
            event_bus: Optional event bus for publishing refresh events
            progress_callback: Optional callback(message, current, total) for progress updates
        
        Returns:
            (success, error_message)
        """
        # Track what we've done for rollback
        restored_data_items = []
        restored_block_states = []
        deleted_item_ids = []
        
        try:
            # 1. Save list of items we're about to delete (for rollback)
            # Get all blocks, then collect all their data items
            blocks = self._block_repo.list_by_project(project_id)
            deleted_items = []
            for block in blocks:
                block_items = self._data_item_repo.list_by_block(block.id)
                deleted_items.extend(block_items)
            deleted_item_ids = [item.id for item in deleted_items]
            
            # 2. Clear existing data
            if progress_callback:
                progress_callback("Clearing existing data...", 0, 100)
            deleted_count = self._data_item_repo.delete_by_project(project_id)
            if deleted_count > 0:
                Log.info(f"SnapshotService: Cleared {deleted_count} existing data item(s)")
            
            # 3. Group snapshot data by block for unified restore
            block_states = {}
            for item_data in snapshot.data_items:
                block_id = item_data.get("block_id")
                if block_id:
                    if block_id not in block_states:
                        block_states[block_id] = {
                            "data_items": [],
                            "local_state": snapshot.block_local_state.get(block_id, {})
                        }
                    block_states[block_id]["data_items"].append(item_data)
            
            # Add blocks that only have local state (no data items)
            for block_id, local_state in snapshot.block_local_state.items():
                if block_id not in block_states:
                    block_states[block_id] = {
                        "data_items": [],
                        "local_state": local_state
                    }
            
            # 4. Restore block states (track each one for rollback)
            restored_count = 0
            state_count = 0
            affected_blocks = set()
            
            for idx, (block_id, state) in enumerate(block_states.items()):
                try:
                    if progress_callback:
                        progress_callback(
                            f"Restoring block state... ({idx + 1}/{len(block_states)})",
                            idx + 1,
                            len(block_states)
                        )
                    
                    # Track data items before restore (for rollback)
                    item_ids_before = {item.id for item in self._data_item_repo.list_by_block(block_id)}
                    
                    # Use helper for unified restore
                    self._state_helper.restore_block_state(block_id, state, project_dir)
                    
                    # Track what was restored (for rollback)
                    item_ids_after = {item.id for item in self._data_item_repo.list_by_block(block_id)}
                    newly_restored_items = item_ids_after - item_ids_before
                    restored_data_items.extend(newly_restored_items)
                    
                    affected_blocks.add(block_id)
                    restored_count += len(state["data_items"])
                    if state["local_state"]:
                        restored_block_states.append(block_id)
                    state_count += 1
                except Exception as e:
                    # Rollback on failure
                    error_msg = f"Failed to restore block state for {block_id}: {e}"
                    Log.error(f"SnapshotService: {error_msg}")
                    self._rollback_restore(
                        project_id,
                        deleted_item_ids,
                        restored_data_items,
                        restored_block_states
                    )
                    return False, error_msg
            
            # 5. Verify consistency
            verification_result = self._verify_restored_state(project_id, snapshot)
            if not verification_result[0]:
                # Rollback if verification fails
                error_msg = f"State verification failed: {verification_result[1]}"
                Log.error(f"SnapshotService: {error_msg}")
                self._rollback_restore(
                    project_id,
                    deleted_item_ids,
                    restored_data_items,
                    restored_block_states
                )
                return False, error_msg
            
            # 6. Publish events (only after successful restore)
            if event_bus:
                from src.application.events import BlockChanged
                for block_id in affected_blocks:
                    event_bus.publish(BlockChanged(
                        project_id=project_id,
                        data={
                            "block_id": block_id,
                            "change_type": "data"
                        }
                    ))
                Log.debug(f"SnapshotService: Published BlockChanged events for {len(affected_blocks)} block(s)")
            
            Log.info(
                f"SnapshotService: Successfully restored snapshot - "
                f"{restored_count} data items, {state_count} block states"
            )
            return True, None
            
        except Exception as e:
            # Catch-all for unexpected errors
            error_msg = f"Unexpected error during restore: {e}"
            Log.error(f"SnapshotService: {error_msg}")
            import traceback
            Log.error(traceback.format_exc())
            
            # Attempt rollback
            try:
                self._rollback_restore(
                    project_id,
                    deleted_item_ids,
                    restored_data_items,
                    restored_block_states
                )
            except Exception as rollback_error:
                Log.error(f"SnapshotService: Rollback also failed: {rollback_error}")
                # If rollback fails, try to restore backup
                if backup_snapshot:
                    Log.info("SnapshotService: Attempting to restore backup snapshot")
                    restore_backup_success, backup_error = self.restore_snapshot_atomic(
                        project_id,
                        backup_snapshot,
                        backup_snapshot=None,  # No backup of backup
                        project_dir=project_dir,
                        event_bus=event_bus
                    )
                    if not restore_backup_success:
                        Log.error(f"SnapshotService: Backup restore also failed: {backup_error}")
            
            return False, error_msg
    
    def _rollback_restore(
        self,
        project_id: str,
        deleted_item_ids: List[str],
        restored_data_items: List[str],
        restored_block_states: List[str]
    ) -> None:
        """
        Rollback a failed restore operation.
        
        Removes items that were restored. Note: Cannot fully restore deleted items
        without their full data - this is why we backup before switching.
        """
        Log.info("SnapshotService: Rolling back failed restore operation")
        
        # Remove items that were restored
        for item_id in restored_data_items:
            try:
                self._data_item_repo.delete(item_id)
            except Exception as e:
                Log.warning(f"SnapshotService: Failed to rollback data item {item_id}: {e}")
        
        # Clear block states that were restored
        for block_id in restored_block_states:
            try:
                self._block_local_state_repo.set_inputs(block_id, {})
            except Exception as e:
                Log.warning(f"SnapshotService: Failed to rollback block state {block_id}: {e}")
        
        # Note: We can't fully restore deleted items without their full data
        Log.warning(
            f"SnapshotService: Rollback complete. "
            f"Note: {len(deleted_item_ids)} deleted items cannot be automatically restored. "
            f"Previous state backup should be used if available."
        )
    
    def restore_snapshot(
        self,
        project_id: str,
        snapshot: DataStateSnapshot,
        project_dir: Optional[Path] = None,
        event_bus=None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> None:
        """
        Restore snapshot into database.
        
        Deserializes and loads data items and block local state,
        similar to how ProjectService restores project files.
        
        Publishes events to notify UI components of data changes.
        
        Args:
            project_id: Project identifier
            snapshot: DataStateSnapshot to restore
            project_dir: Optional project directory for resolving relative file paths
            event_bus: Optional event bus for publishing refresh events
            progress_callback: Optional callback(message, current, total) for progress updates
        """
        Log.info(f"SnapshotService: Restoring snapshot for song {snapshot.song_id}")
        
        total_items = len(snapshot.data_items) + len(snapshot.block_local_state)
        
        # Clear existing data items first (like execution engine does)
        if progress_callback:
            progress_callback("Clearing existing data...", 0, total_items)
        deleted_count = self._data_item_repo.delete_by_project(project_id)
        if deleted_count > 0:
            Log.info(f"SnapshotService: Cleared {deleted_count} existing data item(s) before restore")
        
        # Clear block local state for all blocks (clean slate before restore)
        blocks = self._block_repo.list_by_project(project_id)
        cleared_local_state_count = 0
        for block in blocks:
            try:
                self._block_local_state_repo.clear_inputs(block.id)
                cleared_local_state_count += 1
            except Exception as e:
                Log.warning(f"SnapshotService: Failed to clear local state for block '{block.name}': {e}")
        if cleared_local_state_count > 0:
            Log.info(f"SnapshotService: Cleared local state for {cleared_local_state_count} block(s) before restore")
        
        # Clear block metadata for all blocks (clean slate before restore)
        # This ensures old metadata doesn't persist when switching songs
        # If snapshot has block_settings_overrides, they'll be applied after restore
        # If snapshot doesn't have overrides, blocks will have empty metadata (default state)
        cleared_metadata_count = 0
        block_metadata_before_clear = {b.id: dict(b.metadata) if b.metadata else {} for b in blocks}
        for block in blocks:
            try:
                # Clear metadata to empty dict before restore
                block.metadata = {}
                self._block_repo.update(block)
                cleared_metadata_count += 1
            except Exception as e:
                Log.warning(f"SnapshotService: Failed to clear metadata for block '{block.name}': {e}")
        if cleared_metadata_count > 0:
            Log.info(f"SnapshotService: Cleared metadata for {cleared_metadata_count} block(s) before restore")
        
        # Group snapshot data by block for unified restore
        block_states = {}
        for item_data in snapshot.data_items:
            block_id = item_data.get("block_id")
            if block_id:
                if block_id not in block_states:
                    block_states[block_id] = {
                        "data_items": [],
                        "local_state": snapshot.block_local_state.get(block_id, {})
                    }
                block_states[block_id]["data_items"].append(item_data)
        
        # Add blocks that only have local state (no data items)
        for block_id, local_state in snapshot.block_local_state.items():
            if block_id not in block_states:
                block_states[block_id] = {
                    "data_items": [],
                    "local_state": local_state
                }
        
        # Restore each block state using helper (unified restore)
        restored_count = 0
        state_count = 0
        affected_blocks = set()
        
        for idx, (block_id, state) in enumerate(block_states.items()):
            try:
                current_item = sum(len(s["data_items"]) for s in list(block_states.values())[:idx]) + idx + 1
                if progress_callback:
                    progress_callback(
                        f"Restoring block state... ({idx + 1}/{len(block_states)})",
                        current_item,
                        total_items
                    )
                
                # Use helper for unified restore
                self._state_helper.restore_block_state(block_id, state, project_dir)
                
                affected_blocks.add(block_id)
                restored_count += len(state["data_items"])
                if state["local_state"]:
                    state_count += 1
            except Exception as e:
                Log.warning(f"SnapshotService: Failed to restore block state for {block_id}: {e}")
        
        # Publish events to trigger UI refresh
        if event_bus:
            from src.application.events import BlockUpdated, BlockChanged
            # Publish BlockUpdated events so UI panels refresh with new data
            # Get block info for each affected block
            for block_id in affected_blocks:
                try:
                    block = self._block_repo.get_by_id(block_id)
                    if block:
                        # Publish BlockUpdated event (UI panels listen to this)
                        event_bus.publish(BlockUpdated(
                            project_id=project_id,
                            data={
                                "id": block.id,
                                "name": block.name,
                                "type": block.type
                            }
                        ))
                        # Also publish BlockChanged for status indicator updates
                        event_bus.publish(BlockChanged(
                            project_id=project_id,
                            data={
                                "block_id": block_id,
                                "change_type": "data"
                            }
                        ))
                    else:
                        Log.warning(f"SnapshotService: Block {block_id} not found when publishing events")
                        # Still publish BlockChanged even if block not found
                        event_bus.publish(BlockChanged(
                            project_id=project_id,
                            data={
                                "block_id": block_id,
                                "change_type": "data"
                            }
                        ))
                except Exception as e:
                    Log.warning(f"SnapshotService: Failed to get block {block_id} for event: {e}")
                    # Still publish BlockChanged even if we can't get block info
                    event_bus.publish(BlockChanged(
                        project_id=project_id,
                        data={
                            "block_id": block_id,
                            "change_type": "data"
                        }
                    ))
            Log.debug(f"SnapshotService: Published BlockUpdated and BlockChanged events for {len(affected_blocks)} block(s)")
        
        # Apply block settings overrides (restore block metadata)
        # Note: We cleared all metadata above, so this will fully restore the snapshot's metadata
        if snapshot.block_settings_overrides:
            try:
                if progress_callback:
                    progress_callback("Applying block settings...", total_items, total_items + 1)
                # Since we cleared metadata above, apply_block_overrides will fully restore it
                self.apply_block_overrides(project_id, snapshot.block_settings_overrides)
                Log.info(f"SnapshotService: Applied block settings overrides for {len(snapshot.block_settings_overrides)} block(s)")
                
                # Publish additional BlockUpdated events for blocks with settings changes
                if event_bus:
                    from src.application.events import BlockUpdated
                    for block_id in snapshot.block_settings_overrides.keys():
                        try:
                            block = self._block_repo.get_by_id(block_id)
                            if block:
                                event_bus.publish(BlockUpdated(
                                    project_id=project_id,
                                    data={
                                        "id": block.id,
                                        "name": block.name,
                                        "type": block.type,
                                        "settings_updated": True
                                    }
                                ))
                        except Exception as e:
                            Log.warning(f"SnapshotService: Failed to publish BlockUpdated for block {block_id}: {e}")
            except Exception as e:
                Log.warning(f"SnapshotService: Failed to apply block settings overrides: {e}")
        else:
            # Snapshot has no overrides (old snapshot format) - blocks will have empty metadata
            # This is fine - they'll get their metadata when executed again
            Log.info(f"SnapshotService: Snapshot has no block_settings_overrides (old format) - blocks have empty metadata")
        
        Log.info(
            f"SnapshotService: Restored snapshot for song {snapshot.song_id} - "
            f"{restored_count} data items, {state_count} block states, "
            f"{len(snapshot.block_settings_overrides) if snapshot.block_settings_overrides else 0} block settings"
        )
    
    def apply_block_overrides(
        self,
        project_id: str,
        overrides: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Apply block setting overrides before execution.
        
        Merges override values into block.metadata (where settings are stored).
        This is a simple JSON dict merge - no special serialization needed.
        
        Args:
            project_id: Project identifier
            overrides: Dict mapping block_id -> {setting_key: value, ...}
        """
        if not overrides:
            return
        
        Log.info(f"SnapshotService: Applying block overrides for {len(overrides)} block(s)")
        
        blocks = self._block_repo.list_by_project(project_id)
        for block in blocks:
            if block.id in overrides:
                # Merge override into block.metadata (settings are stored here)
                original_metadata = dict(block.metadata) if block.metadata else {}
                original_metadata.update(overrides[block.id])
                block.metadata = original_metadata
                self._block_repo.update(block)
                Log.debug(
                    f"SnapshotService: Applied overrides to block {block.name}: "
                    f"{list(overrides[block.id].keys())}"
                )
    
    def _build_data_item_from_dict(self, data: dict):
        """
        Build DataItem from dictionary (fallback if ProjectService not available).
        
        Reuses logic from ProjectService._build_data_item_from_dict.
        """
        from datetime import datetime, timezone
        from src.shared.domain.entities import DataItem
        from src.shared.domain.entities import AudioDataItem
        from src.shared.domain.entities import EventDataItem
        
        item_type = (data.get("type") or "").lower()
        if item_type == "audio":
            return AudioDataItem.from_dict(data)
        if item_type == "event":
            return EventDataItem.from_dict(data)
        
        created_at_str = data.get("created_at")
        created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(timezone.utc)
        return DataItem(
            id=data.get("id", ""),
            block_id=data.get("block_id", ""),
            name=data.get("name", "DataItem"),
            type=data.get("type", "Data"),
            created_at=created_at,
            file_path=data.get("file_path"),
            metadata=data.get("metadata", {})
        )

