# BlockStateHelper Integration with Project Save/Load

## Overview

`BlockStateHelper` provides unified access to block state (read + restore). This document explains how it integrates with existing project save/load functionality.

## Current Project Save/Load Flow

### Project Save (`_write_project_file`)

**Current Implementation:**
1. Queries blocks manually: `block_repo.list_by_project()`
2. Iterates blocks to get data_items: `data_item_repo.list_by_block()`
3. **Packages files** (copies to project data directory) - unique to project save
4. Queries block_local_state: `block_local_state_repo.get_inputs()`
5. Serializes everything to JSON file

**What Gets Saved:**
- Blocks (with metadata as JSON)
- Connections
- Data items (with file references)
- Block local state
- UI state
- Setlists/songs
- Snapshots

### Project Load (`import_project_from_file`)

**Current Implementation:**
1. Deserializes blocks from JSON
2. Restores data_items manually: `data_item_repo.create()`
3. **Resolves file paths** (relative to project directory)
4. **Validates files exist** (marks missing files)
5. Restores block_local_state: `block_local_state_repo.set_inputs()`

**What Gets Loaded:**
- Same as what gets saved (above)

## How BlockStateHelper Fits

### SnapshotService (Full Integration)

**Why Full Integration Works:**
- ✅ No file packaging needed (snapshots store references only)
- ✅ Simple file path resolution (relative to project_dir)
- ✅ No file validation needed
- ✅ Perfect match for helper's restore methods

**Integration:**
```python
# Save: Use helper to get state
project_state = self._state_helper.get_project_state(project_id)

# Restore: Use helper to restore state
for block_id, state in block_states.items():
    self._state_helper.restore_block_state(block_id, state, project_dir)
```

### ProjectService (Optional Integration)

**Why Optional:**
- ⚠️ Has complex file packaging logic (copies files to project directory)
- ⚠️ Has file validation logic (checks if files exist, marks missing)
- ⚠️ File operations are separate concern from state access

**Can Use Helper For:**
- ✅ Getting project state (simplifies data gathering)
- ✅ Restoring block state (simplifies restore logic)

**Must Keep Separate:**
- ⚠️ File packaging (copying files to project data directory)
- ⚠️ File validation (checking if files exist after restore)
- ⚠️ Waveform restoration (special handling for audio items)

**Example Integration:**

**Save:**
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
            # ... existing file packaging code (lines 639-707) ...
            data_items.append(item_dict)
    
    # Extract block local state
    block_local_state_serialized = {}
    for block_id, state in project_state.items():
        if state["local_state"]:
            block_local_state_serialized[block_id] = state["local_state"]
    
    # ... rest of existing code ...
```

**Load:**
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

## Key Differences

| Aspect | SnapshotService | ProjectService |
|--------|----------------|----------------|
| **File Packaging** | ❌ No (snapshots store references) | ✅ Yes (copies files to project dir) |
| **File Validation** | ❌ No | ✅ Yes (validates files exist) |
| **File Path Resolution** | ✅ Simple (relative to project_dir) | ⚠️ Complex (handles missing files) |
| **Use Helper** | ✅ Full integration | ⚠️ Partial (file ops separate) |
| **Priority** | ✅ **Required** (Phase 2) | ⚠️ **Optional** (Phase 3) |

## Implementation Phases

### Phase 1: Create BlockStateHelper
- Read methods (`get_block_state`, `get_project_state`)
- Restore methods (`restore_block_state`, `restore_project_state`)
- Helper methods (`_build_data_item_from_dict`)

### Phase 2: Integrate with SnapshotService (Required)
- Refactor `save_snapshot()` to use helper
- Refactor `restore_snapshot()` to use helper
- Full integration (no file operations needed)

### Phase 3: Integrate with ProjectService (Optional)
- Refactor `_write_project_file()` to use helper for state gathering
- Refactor `import_project_from_file()` to use helper for restore
- Keep file packaging/validation logic separate

## Benefits

**For SnapshotService:**
- ✅ Unified access (one place to get/restore state)
- ✅ Less duplication
- ✅ Clearer code
- ✅ Easier to test

**For ProjectService (if integrated):**
- ✅ Unified access pattern (same as SnapshotService)
- ✅ Less duplication
- ⚠️ File operations still separate (different concern)

## Summary

**BlockStateHelper** provides unified access to block state that works perfectly for:
- ✅ **SnapshotService** - Full integration (no file operations)
- ⚠️ **ProjectService** - Optional integration (file operations separate)

The helper handles the common pattern (get/restore state), while file operations remain in ProjectService where they belong (different concern).

