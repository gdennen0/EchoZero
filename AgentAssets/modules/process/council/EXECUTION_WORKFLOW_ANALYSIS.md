# Execution Workflow Analysis: Setlist Support via Data State Snapshots

## Executive Summary

**Question:** Do we need to refactor the block/execution workflow before implementing setlist support?

**Answer:** **NO** - We can use a snapshot/checkpoint approach: save data states separately, restore when needed. Core execution engine doesn't need to change.

## Reframed Approach: Data State Snapshots

### Key Insight

**The core app can clear and reload normally. We just save data states (snapshots) separately, then restore them when switching songs.**

This is much simpler than execution context! The execution engine doesn't need to know about setlists at all.

### Current Execution Model (Unchanged)

**Current Flow:**
```
1. execute_project(project_id)
2. Clear all data items for project_id (delete_by_project)
3. Execute blocks in topological order
4. Store outputs as data items (scoped to block_id/project_id)
5. Update block_local_state (per block_id)
```

**This stays exactly the same.** We don't change the execution engine at all.

### Snapshot Approach

**New Flow for Setlists:**
```
1. Load template project (blocks, connections)
2. For each song:
   a. Run normal execution (clears data, executes, stores results)
   b. After execution completes, save "data state snapshot"
   c. Snapshot contains: data_items + block_local_state
3. To switch songs:
   a. Clear current data from database (normal clear)
   b. Restore saved snapshot (load data_items + block_local_state into DB)
   c. UI updates to show restored data
```

**Benefits:**
- Execution engine unchanged - just runs normally
- Uses existing serialization pattern (projects already serialize data_items + block_local_state)
- Clean separation: execution (transient) vs snapshots (persistent)
- Simple mental model: save results, restore later

### Existing Serialization Pattern (Already Works!)

**Projects already serialize data states!**

**1. Project Service Serializes Data Items:**
```python
# src/application/services/project_service.py:584
# Serialize data items for project file
data_items = []
for block_entity in block_entities:
    items = self._data_item_repo.list_by_block(block_entity.id)
    for item in items:
        data_items.append(item.to_dict())
```

**2. Project Service Serializes Block Local State:**
```python
# src/application/services/project_service.py:586
block_local_state_serialized = {}
for block_entity in block_entities:
    local_state = self._block_local_state_repo.get_inputs(block_entity.id)
    if local_state:
        block_local_state_serialized[block_entity.id] = local_state
```

**3. Projects Restore Data States:**
```python
# src/application/services/project_service.py:284
# Restore data items from project file
for item_data in data.get("data_items", []):
    data_item = self._build_data_item_from_dict(item_data)
    self._data_item_repo.create(data_item)
```

**This pattern already exists!** We just need to:
- Use it for setlist song snapshots (not just project files)
- Store snapshots separately (per song)
- Restore snapshots on demand (when switching songs)

## Setlist Requirements (Reframed)

### What Setlists Need (Simplified)

**Save and Restore Data States:**
- Same project template (blocks, connections)
- Different data states (one per song)
- Save data state after each execution
- Restore data state when switching songs

**Setlist Flow (Simplified):**
```
1. Load template project (blocks, connections)
2. For each song:
   a. Run normal execution (clears data, executes blocks)
   b. After execution, save "data state snapshot" (data_items + block_local_state)
   c. Snapshot stored separately (per song)
3. To switch songs:
   a. Clear current data from database
   b. Restore saved snapshot (load data_items + block_local_state)
   c. UI updates to show restored data
```

### No Conflicts with Current Architecture!

| **Current Design** | **Setlist Need** | **Solution** |
|---|---|---|
| `execute_project(project_id)` clears all data | Need to save data before clearing | Save snapshot AFTER execution |
| Data items scoped to `block_id` | Need multiple data states | Save snapshots separately (not in DB) |
| Execution engine clears data | Need to preserve song data | Snapshots stored outside DB |
| No snapshot concept | Need to save/restore states | Use existing serialization pattern |

**No execution context needed!** Just save snapshots, restore when needed.

## Recommended Approach: Data State Snapshots

### Option A: Snapshot/Checkpoint Pattern (Recommended)

**Approach:**
1. Don't change execution engine at all
2. After each execution, save "data state snapshot"
3. Snapshot contains: serialized data_items + block_local_state
4. Store snapshots separately (in .ezs file or separate storage)
5. When switching songs, restore snapshot into database

**Implementation:**
```python
class SetlistService:
    def process_song(self, setlist_id: str, song_id: str):
        # 1. Load template project (blocks, connections)
        template = self.load_template(setlist_id)
        self.facade.load_project(template.path)
        
        # 2. Set audio path for this song
        self.facade.execute_block_command(song.audio_path, ...)
        
        # 3. Run normal execution (clears data, executes)
        result = self.facade.execute_project()
        
        # 4. Save snapshot AFTER execution
        snapshot = self._save_data_state_snapshot()
        self._store_snapshot(setlist_id, song_id, snapshot)
    
    def switch_active_song(self, setlist_id: str, song_id: str):
        # 1. Clear current data (normal clear)
        self.database.clear_runtime_tables()
        
        # 2. Load template (blocks, connections)
        template = self.load_template(setlist_id)
        self.facade.load_project(template.path)
        
        # 3. Restore snapshot (load data_items + block_local_state)
        snapshot = self._load_snapshot(setlist_id, song_id)
        self._restore_data_state_snapshot(snapshot)
```

**Database Schema:**
```sql
-- Store snapshots in .ezs file (JSON), not in database
-- Or store snapshot references:
CREATE TABLE setlist_song_snapshots (
    song_id TEXT PRIMARY KEY,
    setlist_id TEXT NOT NULL,
    snapshot_path TEXT NOT NULL,  -- Path to snapshot file, or JSON blob
    created_at TEXT NOT NULL,
    FOREIGN KEY (setlist_id) REFERENCES setlists(id)
);
```

**Snapshot Format (JSON):**
```json
{
    "data_items": [
        // ... serialized data items (same format as .ez file)
    ],
    "block_local_state": {
        "block_id_1": { "input_port": "data_item_id", ... },
        // ... (same format as .ez file)
    },
    "block_settings_overrides": {
        "block_id_1": { "onset_threshold": 0.7, ... },  // Per-song setting overrides
        // ... (subset of block.metadata, serialized JSON)
    }
}
```

**Note:** Block settings are already serialized in `block.metadata` (JSON). Per-song overrides are just another JSON blob to save/restore - very simple!

**Pros:**
- **No execution engine changes** - core app runs normally
- **Uses existing serialization pattern** - projects already serialize this data
- **Clean separation** - execution (transient) vs snapshots (persistent)
- **Simple mental model** - save results, restore later
- **No database schema changes** - snapshots stored separately
- **Aligns with "best part is no part"** - minimal changes, maximum value

**Cons:**
- Need to implement snapshot save/restore logic
- Need to decide storage location (file-based vs DB blob)

**Effort:** ~600-800 LOC, 1 week

**Verdict:** **Recommended** - Simplest approach, uses existing patterns, no architectural changes needed

---

## Recommendation

### Recommended Approach: Data State Snapshots (No Refactor Needed!)

**Rationale:**
1. **No execution engine changes** - core app runs normally
2. **Uses existing patterns** - projects already serialize/deserialize data states
3. **Simplest approach** - save snapshots, restore when needed
4. **Perfect alignment with core values** - "best part is no part" (no changes to core), "simplicity" (save/restore pattern)

**Implementation Plan:**

**Phase 1: Snapshot Infrastructure (Week 1)**
1. Create `DataStateSnapshot` entity/class
2. Implement snapshot save logic (serialize data_items + block_local_state + block_settings_overrides)
3. Implement snapshot restore logic (deserialize and load into DB)
4. Implement block override application (merge overrides into block.metadata)
5. Reuse existing ProjectService serialization methods

**Note:** Block settings are already serialized (block.metadata is JSON), so overrides are just JSON dicts - no special handling needed!

**Phase 2: Setlist Implementation (Week 1-2)**
1. Create setlist entities and repositories
2. Implement SetlistService with snapshot save/restore
3. Integrate with execution flow (save snapshot after execution)
4. Implement song switching (restore snapshot)
5. UI for setlist management and song switching
6. .ezs file format (includes snapshots)

**Implementation Details:**

**Snapshot Save (After Execution):**
```python
def _save_data_state_snapshot(self, project_id: str, block_overrides: Dict[str, Dict] = None) -> Dict[str, Any]:
    """Save current data state as snapshot"""
    # Serialize data items (reuse ProjectService logic)
    data_items = []
    blocks = self._block_repo.list_by_project(project_id)
    for block in blocks:
        items = self._data_item_repo.list_by_block(block.id)
        data_items.extend([item.to_dict() for item in items])
    
    # Serialize block local state (reuse ProjectService logic)
    block_local_state = {}
    for block in blocks:
        state = self._block_local_state_repo.get_inputs(block.id)
        if state:
            block_local_state[block.id] = state
    
    # Serialize block settings overrides (already JSON - just save the overrides!)
    # block_overrides = { "block_id": { "setting_key": value, ... }, ... }
    # This is just a JSON dict - already serialized!
    
    return {
        "data_items": data_items,
        "block_local_state": block_local_state,
        "block_settings_overrides": block_overrides or {}
    }
```

**Note:** Block settings are stored in `block.metadata` (already JSON). Per-song overrides are just a JSON dict - no special serialization needed!

**Snapshot Restore (When Switching Songs):**
```python
def _restore_data_state_snapshot(self, project_id: str, snapshot: Dict[str, Any]):
    """Restore snapshot into database"""
    # Restore data items (reuse ProjectService logic)
    for item_data in snapshot.get("data_items", []):
        data_item = self._build_data_item_from_dict(item_data)
        self._data_item_repo.create(data_item)
    
    # Restore block local state (reuse ProjectService logic)
    for block_id, state in snapshot.get("block_local_state", {}).items():
        self._block_local_state_repo.set_inputs(block_id, state)
    
    # Block settings overrides are applied BEFORE execution (merge into block.metadata)
    # So they're already "baked in" to the snapshot - no restore needed!
    # (Or restore them if we want to show what overrides were used)
```

**Applying Overrides (Before Execution):**
```python
def _apply_block_overrides(self, project_id: str, overrides: Dict[str, Dict]):
    """Apply block setting overrides before execution"""
    # Overrides = { "block_id": { "setting_key": value, ... }, ... }
    blocks = self._block_repo.list_by_project(project_id)
    for block in blocks:
        if block.id in overrides:
            # Merge override into block.metadata (settings are stored here)
            block.metadata.update(overrides[block.id])
            self._block_repo.update(block)
```

**Note:** Block settings are already in `block.metadata` (JSON dict). Applying overrides = merging JSON dicts. Very simple!

**Benefits:**
- **No architectural changes** - execution engine untouched
- **Leverages existing code** - reuse ProjectService serialization
- **Simple mental model** - save results, restore later
- **Clean separation** - execution (transient) vs snapshots (persistent)
- **Block settings are already JSON** - overrides are just JSON dicts, no special handling needed
- **Fast to implement** - most logic already exists (serialization, block.metadata)
- **Backwards compatible** - single projects unchanged

**Block Settings Overrides:**
- Block settings stored in `block.metadata` (already JSON)
- Per-song overrides = JSON dict of setting key/value pairs
- Apply overrides = merge dict into `block.metadata` before execution
- Store overrides in snapshot = just save the JSON dict
- No special serialization needed - it's all JSON already!

**Storage Options:**
- **Option 1: Store in .ezs file** (JSON, like .ez files)
- **Option 2: Store as separate files** (one snapshot file per song)
- **Option 3: Store as DB blob** (in setlist_song_snapshots table)

**Recommendation:** Store in .ezs file (consistent with .ez file format, easy to inspect/debug)

---

## Summary

**Answer to Original Question:** Do we need to refactor the execution workflow?

**NO!** Use snapshot/checkpoint pattern instead:
- Core app runs normally (clears, executes, stores)
- Save data state snapshot after execution
- Restore snapshot when switching songs
- No changes to execution engine needed

This aligns perfectly with core values:
- **"Best part is no part"** - No changes to core execution engine
- **"Simplicity and refinement"** - Simple save/restore pattern
- **Use existing patterns** - Leverage ProjectService serialization
- **Minimal changes, maximum value** - Reuse existing code, add snapshot layer

