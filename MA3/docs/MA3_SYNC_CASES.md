# MA3 Sync Cases and Flow Documentation

## Overview

This document describes all the sync cases between EchoZero Editor layers and grandMA3 (MA3) tracks. The goal is to establish a clear, simplified flow that handles all cases consistently.

## Key Concepts

### Entities and Sources

- **SyncLayerEntity**: Represents a link between an Editor layer and an MA3 track
- **SyncSource.EDITOR**: Entity was created by syncing Editor -> MA3
- **SyncSource.MA3**: Entity was created by syncing MA3 -> Editor

### Identifiers

- **editor_layer_id**: Display name of the layer (e.g., "clap", "kick")
- **editor_block_id**: UUID of the Editor block containing the layer
- **editor_data_item_id**: UUID of the EventDataItem containing the layer's events
- **ma3_coord**: MA3 track coordinate (e.g., "tc101_tg1_tr5")
- **track_name**: Name shown in MA3 (e.g., "ez_clap")

### Data Item Ownership (CRITICAL PATTERN)

The **editor_data_item_id** field determines which EventDataItem receives sync updates.
This pattern ensures updates modify the correct layer, not create duplicates.

| Source | Data Item | Behavior |
|--------|-----------|----------|
| **EDITOR** | Use ORIGINAL | When syncing Editor->MA3, we store the original data item ID. MA3 updates modify the SAME layer the user selected. |
| **MA3** | Create NEW | When syncing MA3->Editor, we create a new data item since no Editor layer exists yet. |

**Why this matters:**
- Without storing the original data item ID, MA3 updates would create new `ma3_sync_*` layers
- This caused the bug where "clap" layer spawned a duplicate "ma3_sync_ez_clap" layer
- The fix: `_get_editor_layer_info()` returns the original `data_item_id`, stored in entity at creation

### Hooks

When a track is "hooked", MA3 sends event updates to EchoZero whenever events change in that track.

---

## Sync Cases

### CASE 1: New Editor Layer -> New MA3 Track

**Scenario**: User checks sync checkbox for an Editor layer. No matching MA3 track exists.

**Flow**:
1. Create new track in MA3 with name `ez_{layer_name}`
2. Create sequence if needed, assign to track
3. Push all Editor events to MA3
4. Create SyncLayerEntity (source=EDITOR)
5. Hook the track

**Result**: Editor layer is now synced with a new MA3 track. Changes in either direction will sync.

---

### CASE 2: New Editor Layer -> Existing MA3 Track (Name Match)

**Scenario**: User checks sync checkbox for Editor layer "clap". MA3 already has track "ez_clap".

**Sub-cases**:

#### 2A: User chooses "Create New"
- Generate unique name (e.g., "ez_clap_2")
- Proceed as CASE 1

#### 2B: User chooses "Use Existing" -> Reconciliation Required
When both Editor and MA3 have events, user must choose:

##### 2B-i: Keep Editor Events
1. Clear MA3 track events
2. Push Editor events to MA3
3. Create entity, hook track
4. Skip initial hook response (we just pushed)

##### 2B-ii: Keep MA3 Events  
1. Store original data_item_id from Editor layer
2. Create entity (with editor_data_item_id set)
3. Hook track
4. Receive MA3 events via hook
5. Push MA3 events to Editor's ORIGINAL data item (same layer, no duplicate)

#### 2C: User chooses "Cancel"
- No action taken

---

### CASE 3: MA3 Track -> New Editor Layer

**Scenario**: User initiates sync from MA3 tab, selecting an MA3 track to sync to Editor.

**Flow**:
1. Create SyncLayerEntity (source=MA3)
2. Hook the MA3 track
3. Receive events from MA3
4. Create EventDataItem in Editor with MA3 events
5. Display as new layer in Editor

**Result**: MA3 track is now synced with Editor. Changes in either direction will sync.

---

### CASE 4: Reconnecting After Restart

**Scenario**: EchoZero or MA3 restarts. Previously synced entities exist in settings.

**Flow**:
1. On MA3 connection, load saved SyncLayerEntities
2. For each entity with valid ma3_coord:
   - Re-hook the MA3 track
   - Receive current events from MA3
   - Compare with Editor events
   - If diverged, apply based on settings or prompt user

---

### CASE 5: Ongoing Bidirectional Sync

**Scenario**: Entity is synced and hooked. User makes changes.

#### 5A: User Changes Events in MA3
1. Hook sends updated events to EchoZero
2. `on_track_events_received()` called
3. Entity found by ma3_coord
4. Compare MA3 vs Editor events
5. If apply_updates_enabled: push MA3 events to Editor
6. Else: mark as diverged

#### 5B: User Changes Events in Editor
1. `BlockUpdated` event fires with `events_updated=True`
2. `_on_editor_block_updated()` called
3. Find matching entities by block_id
4. Schedule push from Editor to MA3

---

## Fixed Issues

### Issue 1: Undefined Variable
- Line 1592: Used `editor_name` but should use `raw_name`
- **Status**: FIXED

### Issue 2: Complex Multi-Stage Flow
- `sync_editor_to_ma3` called multiple times with different strategies
- **Status**: FIXED - Now single atomic call with predetermined action

### Issue 3: Skip Flag Race Condition
- `_skip_initial_hook` can be set but hook response may already be queued
- **Status**: PARTIALLY FIXED - Entity now created before hook

### Issue 4: Inconsistent Entity Creation Timing
- Entity created after event push in some cases
- **Status**: FIXED - Entity always created before hook

### Issue 5: Wrong Event Count Used for Divergence Check
- `_get_ma3_events()` returns empty for unhooked tracks
- Reconciliation dialog never shown because `ma3_event_count = 0`
- **Status**: FIXED - Now uses `track_info.event_count` from track metadata

---

## Implemented Solution

### Principle: Single-Pass with UI-Side Dialogs

The sync operation now follows a clean pattern:
1. UI layer gathers all information
2. UI layer shows dialogs and makes all decisions
3. `sync_editor_to_ma3` is called ONCE with a predetermined action
4. The sync function executes atomically with no prompts

### Method Signature (Implemented)

```python
def sync_editor_to_ma3(
    self,
    editor_layer_id: str,
    target_timecode: int,
    target_track_group: int,
    target_sequence: int,
    action: str = "create_new",
    existing_track_no: Optional[int] = None,
) -> Optional[str]:
    """
    Atomic sync operation. All decisions made before calling.
    
    Actions:
    - "create_new": Create new MA3 track, push Editor events
    - "use_existing_keep_editor": Use existing track, push Editor events
    - "use_existing_keep_ma3": Use existing track, receive MA3 events
    """
```

### UI Flow (Implemented in ShowManagerPanel)

```python
def _sync_editor_layer_with_dialogs(self, editor_layer_id: str):
    """
    Handles all dialog logic before calling sync_editor_to_ma3.
    
    Flow:
    1. Gather layer info, Editor events, check for existing MA3 track
    2. If no conflict -> action = "create_new"
    3. If conflict -> show "Use Existing / Create New / Cancel" dialog
    4. If "Use Existing" with divergence -> show "Keep Editor / Keep MA3" dialog
    5. Call sync_editor_to_ma3 ONCE with determined action
    """
```

---

## Implementation Status

- [x] Refactor `sync_editor_to_ma3` to accept single `action` parameter
- [x] Move dialog logic to UI layer (`_sync_editor_layer_with_dialogs`)
- [x] Ensure entity is created BEFORE any hook is established
- [x] Add skip_initial_hook guard for Editor->MA3 sync
- [x] Add comprehensive logging at each stage
- [x] Data item ownership pattern for Editor-sourced entities
- [ ] Write unit tests for each case (future)

---

## Data Item Pattern Implementation

### Finding Original Data Item

```python
def _get_editor_layer_info(self, layer_id: str) -> Optional[Dict[str, Any]]:
    """Get Editor layer info, including the data item ID."""
    layer = editor_api.get_layer(layer_id)
    if layer:
        # Find the data item that contains this layer's events
        data_item_id = self._find_data_item_for_layer(layer.group_name)
        return {
            "layer_id": layer.name,
            "name": layer.name,
            "block_id": self._get_editor_block_id(),
            "group_name": layer.group_name,
            "event_count": layer.event_count,
            "data_item_id": data_item_id,  # CRITICAL: Original data item
        }

def _find_data_item_for_layer(self, group_name: str) -> Optional[str]:
    """Find the data item ID that matches a layer's group_name."""
    items = self._facade.data_item_repo.list_by_block(editor_block_id)
    for item in items:
        if isinstance(item, EventDataItem):
            if item.name == group_name:
                return item.id
    return None
```

### Storing Original Data Item in Entity

```python
# In sync_editor_to_ma3(), after creating entity:
original_data_item_id = layer_info.get("data_item_id")
if original_data_item_id:
    entity.editor_data_item_id = original_data_item_id
```

### Using Original Data Item for Updates

```python
def _get_or_create_sync_data_item(self, entity: SyncLayerEntity) -> Optional[str]:
    """Get the EventDataItem for synced layer events."""
    
    # Priority 1: Use stored data_item_id (original layer)
    if entity.editor_data_item_id:
        existing = self._facade.data_item_repo.get(entity.editor_data_item_id)
        if existing:
            return entity.editor_data_item_id
    
    # Priority 2: Find by group_name (Editor-sourced)
    if entity.source == SyncSource.EDITOR and entity.group_name:
        data_item_id = self._find_data_item_for_layer(entity.group_name)
        if data_item_id:
            return data_item_id
    
    # Priority 3: Create new (MA3-sourced only)
    if entity.source == SyncSource.MA3:
        # Create new data item for MA3 track
        return self._create_sync_data_item(entity)
    
    return None
```

### Pattern Summary

| Step | Editor->MA3 | MA3->Editor |
|------|-------------|-------------|
| 1. Find data item | `_find_data_item_for_layer(group_name)` | N/A |
| 2. Store in entity | `entity.editor_data_item_id = original_id` | N/A |
| 3. On MA3 update | Use stored `editor_data_item_id` | Create new data item |
| 4. Result | Same layer updates | New layer created |

---

## Signal Reference

| Signal | Args | Purpose |
|--------|------|---------|
| `entities_changed` | - | Entity list changed |
| `entity_updated` | entity_id | Specific entity updated |
| `sync_status_changed` | entity_id, status | Entity sync status changed |
| `divergence_detected` | entity_id, comparison | Divergence found |
| `error_occurred` | entity_id, message | Error during sync |
| `ma3_connection_changed` | is_connected | MA3 connection state |
| `track_conflict_prompt` | (deprecated) | Use UI-side dialog instead |
| `existing_track_prompt` | (deprecated) | Use UI-side dialog instead |
| `reconciliation_prompt` | (deprecated) | Use UI-side dialog instead |

---

## Summary

The implementation has been refactored to follow a clean separation of concerns:

**UI Layer (ShowManagerPanel):**
- Gathers all necessary information (Editor events, MA3 tracks)
- Shows dialogs for conflict resolution and reconciliation
- Determines the appropriate action before calling sync

**Application Layer (SyncSystemManager):**
- `sync_editor_to_ma3()` is now atomic - no prompts, no multi-stage calls
- Accepts predetermined `action` parameter
- Executes the sync operation in one pass
- Creates entity BEFORE hooking to prevent race conditions

**Key Actions:**
1. `create_new` - Create new MA3 track with Editor events
2. `use_existing_keep_editor` - Link to existing track, push Editor events
3. `use_existing_keep_ma3` - Link to existing track, receive MA3 events

**Deprecated Signals:**
- `existing_track_prompt` - No longer emitted
- `reconciliation_prompt` - No longer emitted
- Dialog logic now handled entirely in UI layer
