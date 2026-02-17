# Final Implementation Plan: Unified Block State Helper

## Goal
Create a unified access point for block state (both read AND restore) to make setlist functionality feel built-in, without changing storage.

## Core Decision: Keep Current Storage
- ✅ Metadata stays as JSON in `blocks.metadata` (no DB changes)
- ✅ `block_local_state` table unchanged
- ✅ `data_items` table unchanged
- ✅ No schema migrations needed

## Implementation: Unified Helper Pattern (Read + Restore)

### Phase 1: Create BlockStateHelper (~150 LOC)

**File:** `src/application/services/block_state_helper.py`

**Purpose:** Unified access to block state - both reading AND restoring

**Read Methods:**
- `get_block_state(block_id)` → Returns dict with:
  - `block_id`
  - `settings` (from `block.metadata`)
  - `local_state` (from `block_local_state` repo)
  - `data_items` (from `data_item` repo)
- `get_project_state(project_id)` → Returns dict of all block states

**Restore Methods:**
- `restore_block_state(block_id, state_dict, project_dir=None)` → Restores:
  - Block local state (via `block_local_state_repo.set_inputs()`)
  - Data items (via `data_item_repo.create()`)
  - Note: Metadata overrides handled separately (via commands for undo support)
- `restore_project_state(project_id, project_state_dict, project_dir=None)` → Restores all blocks

**Helper Methods:**
- `_build_data_item_from_dict(item_dict)` → Deserializes data items (reuses ProjectService logic)
- `_resolve_file_paths(data_items, project_dir)` → Resolves relative paths

**Design:**
- Simple helper (not a manager)
- Returns/accepts dicts (no new entities)
- Handles both read and restore
- ~150 LOC (includes restore logic)

### Phase 2: Refactor SnapshotService

**File:** `src/application/services/snapshot_service.py`

**Changes:**

1. **Add BlockStateHelper to __init__:**
```python
def __init__(self, ...):
    # ... existing init ...
    
    # Add state helper for unified access
    from src.application.services.block_state_helper import BlockStateHelper
    self._state_helper = BlockStateHelper(
        block_repo,
        block_local_state_repo,
        data_item_repo,
        project_service  # For _build_data_item_from_dict
    )
```

2. **Refactor save_snapshot() to use helper:**
```python
def save_snapshot(...) -> DataStateSnapshot:
    # Use helper to get all block states (unified access)
    project_state = self._state_helper.get_project_state(project_id)
    
    # Convert to snapshot format (keep existing format for backward compatibility)
    data_items = []
    block_local_state = {}
    
    for block_id, state in project_state.items():
        data_items.extend(state["data_items"])
        if state["local_state"]:
            block_local_state[block_id] = state["local_state"]
    
    # Create snapshot (keep existing format)
    return DataStateSnapshot(...)
```

3. **Refactor restore_snapshot() to use helper:**
```python
def restore_snapshot(...) -> None:
    # Clear existing data items first
    self._data_item_repo.delete_by_project(project_id)
    
    # Group snapshot data by block
    block_states = {}
    for item_data in snapshot.data_items:
        block_id = item_data["block_id"]
        if block_id not in block_states:
            block_states[block_id] = {
                "data_items": [],
                "local_state": snapshot.block_local_state.get(block_id, {})
            }
        block_states[block_id]["data_items"].append(item_data)
    
    # Restore each block state using helper
    for block_id, state in block_states.items():
        self._state_helper.restore_block_state(
            block_id,
            state,
            project_dir=project_dir
        )
    
    # Apply metadata overrides (still handled separately for undo support)
    if snapshot.block_settings_overrides:
        self.apply_block_overrides(project_id, snapshot.block_settings_overrides)
    
    # Publish events...
```

**Benefits:**
- Unified access (one place to get AND restore state)
- Less duplication
- Clearer code
- Easier to test

### Phase 3: Testing & Documentation

**Tests:**
- Unit tests for `BlockStateHelper`:
  - `get_block_state()` - returns correct structure
  - `get_project_state()` - returns all blocks
  - `restore_block_state()` - restores correctly
  - `restore_project_state()` - restores all blocks
- Integration tests for `SnapshotService`:
  - Save/restore cycle works
  - Backward compatibility maintained
  - File path resolution works

**Documentation:**
- Document unified state access pattern
- Update architecture docs
- Document restore process

## What We're NOT Changing

- ✅ Database schema (no migrations)
- ✅ Metadata storage (stays as JSON)
- ✅ BlockSettingsManager (works as-is)
- ✅ Existing snapshot format (backward compatible)
- ✅ Metadata override mechanism (still uses commands for undo)

## Success Criteria

- [ ] `BlockStateHelper` provides unified read access
- [ ] `BlockStateHelper` provides unified restore access
- [ ] `SnapshotService` uses helper for both save AND restore
- [ ] Existing snapshots still work
- [ ] Setlist song switching still works
- [ ] No performance regression
- [ ] Code is simple and maintainable (~150 LOC helper)

## Key Principles

1. **"Best part is no part"** - Minimal new code (~150 LOC)
2. **"Simplicity"** - Simple helper, not complex abstractions
3. **Leverage existing storage** - No DB changes needed
4. **Unified access** - One place to get AND restore block state
5. **Backward compatible** - Existing code still works

## Files to Create/Modify

**Create:**
- `src/application/services/block_state_helper.py` (~150 LOC)

**Modify:**
- `src/application/services/snapshot_service.py`:
  - Refactor `save_snapshot()` to use helper
  - Refactor `restore_snapshot()` to use helper

**No Changes:**
- Database schema
- Metadata storage
- BlockSettingsManager
- Other services (optional later)

## Implementation Details

### BlockStateHelper.restore_block_state()

```python
def restore_block_state(
    self,
    block_id: str,
    state: Dict[str, Any],
    project_dir: Optional[Path] = None
) -> None:
    """
    Restore block state from unified state dict.
    
    Restores:
    - Block local state (via block_local_state_repo.set_inputs())
    - Data items (via data_item_repo.create())
    
    Note: Metadata overrides are handled separately (via commands for undo support).
    
    Args:
        block_id: Block identifier
        state: State dict with 'local_state' and 'data_items' keys
        project_dir: Optional project directory for resolving file paths
    """
    # Restore local state
    if state.get("local_state"):
        self._block_local_state_repo.set_inputs(block_id, state["local_state"])
    
    # Restore data items
    for item_dict in state.get("data_items", []):
        # Deserialize data item
        data_item = self._build_data_item_from_dict(item_dict)
        
        # Resolve file paths if project_dir provided
        if project_dir and data_item.file_path:
            if not os.path.isabs(data_item.file_path):
                resolved_path = project_dir / data_item.file_path
                data_item.file_path = str(resolved_path)
        
        # Create data item
        self._data_item_repo.create(data_item)
```

### BlockStateHelper._build_data_item_from_dict()

```python
def _build_data_item_from_dict(self, data: dict) -> DataItem:
    """
    Build DataItem from dict (reuses ProjectService logic).
    
    Delegates to ProjectService if available, otherwise uses fallback.
    """
    if self._project_service:
        return self._project_service._build_data_item_from_dict(data)
    else:
        # Fallback implementation (same as SnapshotService currently has)
        item_type = (data.get("type") or "").lower()
        if item_type == "audio":
            from src.domain.entities.audio_data_item import AudioDataItem as ExportedAudioDataItem
            return ExportedAudioDataItem.from_dict(data)
        if item_type == "event":
            from src.domain.entities.event_data_item import EventDataItem as ExportedEventDataItem
            return ExportedEventDataItem.from_dict(data)
        
        # Default DataItem
        from datetime import datetime, timezone
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
```

## Integration with Project Save/Load

### Current Project Save/Load Flow

**Project Save (`_write_project_file`):**
1. Queries blocks manually (line 624)
2. Iterates blocks to get data_items (line 635)
3. Packages files (copies to project data directory) - lines 639-707
4. Queries block_local_state manually (line 737)
5. Serializes everything to JSON file

**Project Load (`import_project_from_file`):**
1. Deserializes blocks from JSON
2. Restores data_items manually (line 346)
3. Resolves file paths (line 352-354)
4. Validates files exist (line 357)
5. Restores block_local_state manually (line 430)

### How BlockStateHelper Fits

**Project Save - Can Use Helper (with file packaging):**
```python
def _write_project_file(self, project: Project) -> Optional[str]:
    # Use helper to get project state (unified access)
    project_state = self._state_helper.get_project_state(project.id)
    
    # Extract blocks (already have from helper's block query)
    blocks = [block.to_dict() for block in self._block_repo.list_by_project(project.id)]
    
    # Extract data items and package files (keep file packaging logic)
    data_items = []
    for block_id, state in project_state.items():
        for item_dict in state["data_items"]:
            # Package file if it exists (keep existing file packaging logic)
            # ... existing file packaging code ...
            data_items.append(item_dict)
    
    # Extract block local state
    block_local_state_serialized = {}
    for block_id, state in project_state.items():
        if state["local_state"]:
            block_local_state_serialized[block_id] = state["local_state"]
    
    # ... rest of existing code ...
```

**Project Load - Can Use Helper (with file validation):**
```python
def import_project_from_file(self, file_path: str) -> Project:
    # ... existing project/block/connection restore ...
    
    # Group data items by block for helper
    block_states = {}
    for item_data in data.get("data_items", []):
        block_id = item_data["block_id"]
        if block_id not in block_states:
            block_states[block_id] = {
                "data_items": [],
                "local_state": data.get("block_local_state", {}).get(block_id, {})
            }
        block_states[block_id]["data_items"].append(item_data)
    
    # Restore block states using helper (handles file path resolution)
    project_dir = Path(file_path).parent
    for block_id, state in block_states.items():
        self._state_helper.restore_block_state(
            block_id,
            state,
            project_dir=project_dir
        )
    
    # Note: File validation still needed (check if files exist after restore)
    # This could be added to helper or kept separate
```

### Key Differences: Project vs Snapshot

| Aspect | SnapshotService | ProjectService |
|--------|----------------|----------------|
| **File Packaging** | No (snapshots store references) | Yes (copies files to project dir) |
| **File Validation** | No | Yes (validates files exist) |
| **File Path Resolution** | Simple (relative to project_dir) | Complex (handles missing files) |
| **Use Helper** | ✅ Full integration | ⚠️ Partial (file ops separate) |

### Recommendation: Optional Integration

**Phase 1 (Required):** Use helper in `SnapshotService` (no file packaging needed)

**Phase 2 (Optional):** Use helper in `ProjectService`:
- Use `get_project_state()` for save (simplifies data gathering)
- Use `restore_project_state()` for load (simplifies restore)
- Keep file packaging/validation logic separate (different concern)

**Benefits of Optional Integration:**
- Unified access pattern across services
- Less duplication
- Easier to maintain

**Why Optional:**
- ProjectService has complex file operations (packaging, validation)
- File operations are separate concern from state access
- Current code works fine

## Timeline

- **Week 1:** Create `BlockStateHelper` with read + restore methods + tests
- **Week 2:** Refactor `SnapshotService` to use helper + integration tests
- **Week 3:** (Optional) Refactor `ProjectService` to use helper
- **Week 4:** Documentation + verification

## Unified Save/Load/State Switching Pattern

### Core Insight

**All three operations (save, load, state switching) follow the same pattern:**

1. **Get state** → `BlockStateHelper.get_project_state()`
2. **Transform** → File operations (if needed for project save)
3. **Serialize/store** → Snapshot/Project/State switch format
4. **Deserialize/load** → Load from storage
5. **Transform** → File operations (if needed for project load)
6. **Restore state** → `BlockStateHelper.restore_project_state()`

### Unified Services

**SnapshotService:**
- Save: Get state → Serialize to snapshot
- Restore: Deserialize snapshot → Restore state

**ProjectService:**
- Save: Get state → Package files → Serialize to project file
- Load: Deserialize project file → Resolve paths → Restore state

**SetlistService:**
- Switch: Get current state → Save backup → Load target state → Restore target state

**All use the same core pattern!**

### File Operations (Separate Concern)

**File operations are separate from state operations:**
- **Package files** (copy to project directory) - for project save
- **Resolve file paths** (relative to project_dir) - for project/snapshot load
- **Validate files exist** - for project load

These can be:
- Separate helper (`FileOperationHelper`)
- Or methods on `BlockStateHelper` (simpler)
- Or callbacks/strategies (more flexible)

**Recommendation:** Keep file operations as methods on `BlockStateHelper` for simplicity, but clearly separated.

## Summary

This plan provides unified access for both reading AND restoring block state, making setlist functionality feel built-in while keeping storage unchanged. The helper handles the complexity of restoring state, making `SnapshotService`, `ProjectService`, and `SetlistService` all use the same clean pattern.

**Key Benefits:**
- ✅ Unified pattern across all services
- ✅ Less duplication
- ✅ Easier to maintain
- ✅ Clearer separation of concerns (state vs files)

**Integration Points:**
- ✅ **SnapshotService** - Full integration (no file operations)
- ⚠️ **ProjectService** - Optional integration (file operations separate)

