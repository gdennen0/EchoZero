# ShowManager Sync System

This document describes the clean-sheet redesign of the EchoZero-MA3 layer synchronization system.

## Design Philosophy

**Core principle: One entity type, one manager, one source of truth.**

The system was redesigned from the ground up to be simple and maintainable:
- Single `SyncLayerEntity` type replaces separate `EditorLayerEntity` and `MA3TrackEntity`
- Single `SyncSystemManager` orchestrates all sync operations
- The `source` field prevents duplicate entries by design
- UI is thin - just calls manager methods and updates based on signals

## Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │     ShowManagerPanel (UI)       │
                    │  - Synced tab                   │
                    │  - Available tab                │
                    └─────────────┬───────────────────┘
                                  │ (thin UI, signals)
                    ┌─────────────▼───────────────────┐
                    │     SyncSystemManager           │
                    │  - sync_layer()                 │
                    │  - unsync_layer()               │
                    │  - resync_layer()               │
                    │  - check_divergences()          │
                    │  - resolve_divergence()         │
                    └─────────────┬───────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
    ┌─────────▼─────────┐  ┌──────▼──────┐  ┌────────▼────────┐
    │ MA3CommService    │  │ EditorAPI   │  │ Settings        │
    │ (OSC to MA3)      │  │ (Layers)    │  │ Manager         │
    └───────────────────┘  └─────────────┘  └─────────────────┘
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Source of truth | `synced_layers` only | Eliminates `layer_mappings` confusion |
| Default sync direction | Bidirectional | Most common use case |
| Auto-create behavior | Prompt user first | Prevents accidental creation |
| Reconnect strategy | Prompt user per divergent layer | User controls resolution |
| UI table scope | Split tabs: Synced + Available | Clear separation |
| Naming convention | Always prefix (ma3_, ez_) | Visual distinction |
| Hook management | Lazy hook (on first sync) | Reduces overhead |
| Unsync behavior | Asymmetric (see below) | Matches mental model |

### Asymmetric Unsync Behavior

The unsync behavior differs based on where the layer originated:

- **MA3-sourced layer unchecked**: Delete the synced copy from Editor
  - Rationale: MA3 is the authoritative source, the Editor copy is derived
  
- **Editor-sourced layer unchecked**: Leave the synced copy in MA3
  - Rationale: May want to resync later, don't delete user work in MA3

## Unified Entity Model

The `SyncLayerEntity` combines both MA3 track and Editor layer concepts:

```python
@dataclass
class SyncLayerEntity:
    # Identity
    id: str                    # UUID
    source: SyncSource         # "ma3" or "editor" - WHERE IT ORIGINATED
    name: str                  # Prefixed (ma3_Kick or ez_Kick)
    
    # MA3-side identity (populated when synced to MA3)
    ma3_coord: Optional[str]
    ma3_timecode_no: Optional[int]
    ma3_track_group: Optional[int]
    ma3_track: Optional[int]
    
    # Editor-side identity (populated when synced to Editor)
    editor_layer_id: Optional[str]
    editor_block_id: Optional[str]
    
    # State
    sync_status: SyncStatus    # unmapped, synced, diverged, error
    event_count: int
    last_sync_time: Optional[datetime]
    error_message: Optional[str]
    
    # Settings
    settings: SyncLayerSettings
    group_name: Optional[str]
```

The `source` field is the key innovation - it prevents duplicates by design. Each sync layer appears exactly once, in the section corresponding to its source.

## SyncSystemManager API

### Core Operations

```python
# Start syncing a layer
entity_id = manager.sync_layer(source="ma3", source_id="tc1_tg1_tr1")

# Stop syncing (applies asymmetric delete rules)
success = manager.unsync_layer(entity_id)

# Re-run sync for existing layer
success = manager.resync_layer(entity_id)
```

### Reconnection Handling

```python
# Called when MA3 connection established
manager.on_ma3_connected()

# Check all synced layers for divergence
diverged_ids = manager.check_divergences()

# Resolve divergence: "ma3_wins", "ez_wins", or "merge"
success = manager.resolve_divergence(entity_id, "ma3_wins")
```

### Data Access for UI

```python
# Get all synced layers
layers = manager.get_synced_layers()

# Get available (unsynced) items for each tab
available_ma3 = manager.get_available_ma3_tracks()
available_editor = manager.get_available_editor_layers()
```

### Signals

```python
# Entity list changed (add/remove)
entities_changed = pyqtSignal()

# Specific entity updated
entity_updated = pyqtSignal(str)  # entity_id

# Status changed
sync_status_changed = pyqtSignal(str, str)  # entity_id, new_status

# Divergence detected
divergence_detected = pyqtSignal(str, object)  # entity_id, comparison

# Error occurred
error_occurred = pyqtSignal(str, str)  # entity_id, message

# MA3 connection state
ma3_connection_changed = pyqtSignal(bool)
```

## Sync Flows

### Sync MA3 Track to Editor

1. User checks MA3 track in Available tab
2. UI calls `manager.sync_layer("ma3", coord)`
3. Manager checks if Editor layer exists
4. If missing: prompts user or auto-creates
5. Manager hooks MA3 track for changes
6. Manager fetches MA3 events and pushes to Editor
7. Manager saves to `synced_layers` settings
8. Manager emits `entities_changed` signal
9. Layer moves to Synced tab

### Unsync MA3-sourced Layer

1. User unchecks MA3-sourced layer in Synced tab
2. UI calls `manager.unsync_layer(entity_id)`
3. Manager unhooks MA3 track
4. Manager deletes Editor layer (asymmetric behavior)
5. Manager removes from `synced_layers`
6. Manager emits `entities_changed` signal
7. Track returns to Available tab (MA3 side)

### Reconnect with Divergence

1. MA3 connection established
2. `manager.on_ma3_connected()` called
3. Manager hooks all synced MA3 tracks
4. Manager calls `check_divergences()`
5. For each diverged layer:
   - Manager emits `divergence_detected` signal
   - UI shows resolution dialog
   - User chooses strategy
   - `manager.resolve_divergence(id, strategy)` called
6. Layer status updated to SYNCED

## Migration from Legacy System

The system automatically migrates legacy entity formats:

- `MA3TrackEntity` dict -> `SyncLayerEntity` with `source=MA3`
- `EditorLayerEntity` dict -> `SyncLayerEntity` with `source=EDITOR`
- `layer_mappings` dict -> (removed, not migrated)

## File Structure

```
src/features/show_manager/
  domain/
    sync_layer_entity.py     # Unified entity
    layer_sync_types.py      # Enums
    sync_state.py            # Fingerprinting
  application/
    sync_system_manager.py   # Single orchestration point
    sync_layer_manager.py    # Comparison utility
    commands/                # Reused by SyncSystemManager
```

**Removed legacy files:**
- `editor_layer_entity.py` - replaced by `SyncLayerEntity`
- `ma3_track_entity.py` - replaced by `SyncLayerEntity`
- `layer_sync_controller.py` - replaced by `SyncSystemManager`
- `sync_engine.py` - merged into `SyncSystemManager`
- `sync_registry.py` - utilities inlined where needed

## Settings Storage

Synced layers are stored in ShowManager block metadata:

```json
{
  "synced_layers": [
    {
      "id": "uuid-1",
      "source": "ma3",
      "name": "ma3_Kick",
      "ma3_coord": "tc1_tg1_tr1",
      "ma3_timecode_no": 1,
      "ma3_track_group": 1,
      "ma3_track": 1,
      "editor_layer_id": "ma3_Kick",
      "editor_block_id": "editor-block-id",
      "sync_status": "synced",
      "event_count": 42,
      "settings": {
        "direction": "bidirectional",
        "conflict_strategy": "prompt_user",
        "sequence_no": 1
      }
    }
  ]
}
```

## UI Implementation Notes

The UI should be thin - no orchestration logic:

1. **Synced Tab**: Shows `manager.get_synced_layers()`
2. **Available Tab**: Shows `manager.get_available_ma3_tracks()` and `manager.get_available_editor_layers()`
3. **Checkbox toggle**: Calls `sync_layer()` or `unsync_layer()`
4. **Status indicator**: Bound to `sync_status_changed` signal
5. **Reconnect**: Calls `check_divergences()`, shows dialog on `divergence_detected`

## Testing Checklist

- [ ] Sync MA3 track creates Editor layer
- [ ] Sync Editor layer creates MA3 copy
- [ ] Unsync MA3-sourced deletes from Editor
- [ ] Unsync Editor-sourced leaves MA3 copy
- [ ] Reconnect prompts for diverged layers
- [ ] Resolution strategies work correctly
- [ ] Sequence changes update MA3 track.target
- [ ] No duplicate layers in any view
- [ ] Settings persist and load correctly
- [ ] Migration from legacy format works
