# Unified Save/Load/State Switching Approach

## Clarification: "References Only"

**What "snapshots store references only" means:**
- Snapshots store file **paths** (strings like `/path/to/file.wav`)
- Snapshots **DO NOT** copy the actual files
- When restoring, uses existing files from project directory
- Files are managed by the project/template

**Project save is different:**
- Project save **COPIES** files into project's `data/` directory
- Files are packaged/archived with the project
- Project can be moved/shared with all files included

**Why the difference:**
- Snapshots are temporary (per-song state within a project)
- Projects are permanent (complete archive)
- Snapshots assume files already exist in project directory
- Projects need to be self-contained

## Current Duplication

**Three similar patterns:**

1. **SnapshotService.save_snapshot()** - Gets state, creates snapshot
2. **ProjectService._write_project_file()** - Gets state, packages files, writes JSON
3. **SetlistService.switch_active_song()** - Switches between saved states

**All three:**
- Get block state (blocks, data_items, block_local_state)
- Serialize/restore state
- Handle file paths differently

## Unified Approach

### Core Pattern: State Operations

**Unified State Operations:**
```python
class BlockStateHelper:
    """Unified helper for all state operations"""
    
    # Read operations
    def get_block_state(self, block_id: str) -> Dict[str, Any]
    def get_project_state(self, project_id: str) -> Dict[str, Dict[str, Any]]
    
    # Restore operations
    def restore_block_state(self, block_id: str, state: Dict[str, Any], project_dir: Optional[Path] = None)
    def restore_project_state(self, project_id: str, state: Dict[str, Dict[str, Any]], project_dir: Optional[Path] = None)
    
    # Save operations (serialize to dict)
    def serialize_block_state(self, block_id: str) -> Dict[str, Any]
    def serialize_project_state(self, project_id: str) -> Dict[str, Dict[str, Any]]
```

### Unified Save Pattern

**All save operations use same pattern:**
```python
# 1. Get state (unified)
project_state = state_helper.get_project_state(project_id)

# 2. Transform for specific use case
#    - Snapshot: Keep as-is (references only)
#    - Project: Package files (copy to data/)
#    - State switch: Keep as-is (references only)

# 3. Serialize/store
#    - Snapshot: Create DataStateSnapshot
#    - Project: Write JSON file
#    - State switch: Store in memory/DB
```

### Unified Load/Restore Pattern

**All restore operations use same pattern:**
```python
# 1. Deserialize/load state
#    - Snapshot: Load from DataStateSnapshot
#    - Project: Load from JSON file
#    - State switch: Load from stored state

# 2. Transform if needed
#    - Snapshot: Resolve file paths (relative to project_dir)
#    - Project: Resolve file paths + validate files exist
#    - State switch: Resolve file paths (relative to project_dir)

# 3. Restore state (unified)
state_helper.restore_project_state(project_id, project_state, project_dir)
```

### Unified State Switching Pattern

**State switching becomes:**
```python
class StateSwitchService:
    """Unified service for state switching"""
    
    def save_current_state(self, project_id: str, state_id: str) -> Dict[str, Any]:
        """Save current state (for snapshots, projects, or state switching)"""
        return self._state_helper.serialize_project_state(project_id)
    
    def restore_state(self, project_id: str, state: Dict[str, Any], project_dir: Optional[Path] = None) -> None:
        """Restore state (unified restore)"""
        self._state_helper.restore_project_state(project_id, state, project_dir)
    
    def switch_state(self, project_id: str, from_state_id: str, to_state_id: str) -> None:
        """Switch between two saved states"""
        # Load target state
        target_state = self._load_state(to_state_id)
        
        # Restore target state
        self.restore_state(project_id, target_state)
```

## Proposed Unified Architecture

### BlockStateHelper (Core)

**Handles all state operations:**
- Get state (read from DB)
- Restore state (write to DB)
- Serialize state (to dict)

**No file operations** - just state access

### FileOperationHelper (Separate)

**Handles file operations:**
- Package files (copy to project directory)
- Resolve file paths (relative to project_dir)
- Validate files exist

**Can be used by:**
- ProjectService (for project save/load)
- BlockStateHelper (for file path resolution during restore)

### Unified Services

**SnapshotService:**
```python
def save_snapshot(self, project_id: str, song_id: str) -> DataStateSnapshot:
    # Get state (unified)
    project_state = self._state_helper.get_project_state(project_id)
    
    # Serialize to snapshot format
    return self._serialize_to_snapshot(project_id, project_state, song_id)

def restore_snapshot(self, project_id: str, snapshot: DataStateSnapshot, project_dir: Optional[Path] = None):
    # Deserialize from snapshot
    project_state = self._deserialize_from_snapshot(snapshot)
    
    # Restore state (unified)
    self._state_helper.restore_project_state(project_id, project_state, project_dir)
```

**ProjectService:**
```python
def _write_project_file(self, project: Project) -> Optional[str]:
    # Get state (unified)
    project_state = self._state_helper.get_project_state(project.id)
    
    # Package files (separate concern)
    packaged_state = self._file_helper.package_files(project_state, project.save_directory)
    
    # Serialize to project file
    return self._serialize_to_project_file(project, packaged_state)

def import_project_from_file(self, file_path: str) -> Project:
    # Deserialize from project file
    project_data = self._deserialize_from_project_file(file_path)
    
    # Extract state
    project_state = self._extract_state_from_project_data(project_data)
    
    # Resolve file paths (separate concern)
    resolved_state = self._file_helper.resolve_file_paths(project_state, Path(file_path).parent)
    
    # Restore state (unified)
    self._state_helper.restore_project_state(project.id, resolved_state, Path(file_path).parent)
    
    # Validate files (separate concern)
    self._file_helper.validate_files(project_state)
```

**SetlistService:**
```python
def switch_active_song(self, project_id: str, song_id: str) -> None:
    # Get current state (unified)
    current_state = self._state_helper.get_project_state(project_id)
    
    # Save current state as backup
    self._save_state_backup(project_id, current_state)
    
    # Load target song state
    target_snapshot = self._get_snapshot_for_song(song_id)
    target_state = self._deserialize_from_snapshot(target_snapshot)
    
    # Restore target state (unified)
    project_dir = self._get_project_directory(project_id)
    self._state_helper.restore_project_state(project_id, target_state, project_dir)
```

## Benefits of Unified Approach

1. **Single source of truth** - BlockStateHelper handles all state operations
2. **Less duplication** - Save/load/switch all use same pattern
3. **Easier to maintain** - Fix bugs in one place
4. **Clearer separation** - State operations vs file operations
5. **Easier to test** - Test state operations independently

## Implementation Plan

### Phase 1: Create BlockStateHelper
- Read methods (`get_block_state`, `get_project_state`)
- Restore methods (`restore_block_state`, `restore_project_state`)
- Serialize methods (`serialize_block_state`, `serialize_project_state`)

### Phase 2: Create FileOperationHelper (Optional)
- Package files (copy to project directory)
- Resolve file paths (relative to project_dir)
- Validate files exist

### Phase 3: Refactor Services
- SnapshotService: Use BlockStateHelper
- ProjectService: Use BlockStateHelper + FileOperationHelper
- SetlistService: Use BlockStateHelper for state switching

## Summary

**Unified Pattern:**
1. **Get state** → BlockStateHelper.get_project_state()
2. **Transform** → File operations (if needed)
3. **Serialize/store** → Snapshot/Project/State switch format
4. **Deserialize/load** → Load from storage
5. **Transform** → File operations (if needed)
6. **Restore state** → BlockStateHelper.restore_project_state()

**Key Insight:**
- State operations are unified (get/restore/serialize)
- File operations are separate (packaging/validation)
- Each service uses same pattern, just different transformations

This creates a clean, unified approach where save/load/state switching all use the same core pattern, with file operations as a separate, composable concern.

