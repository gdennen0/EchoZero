# CmdSubTrack Hooking Implementation

## Overview

This document describes the implementation of CmdSubTrack hooking for real-time MA3 → Editor layer synchronization, including event fingerprinting, change detection, and OSC traffic management.

## Problem Statement

1. **High-frequency callbacks**: CmdSubTrack hooks fire on every event movement, creating excessive OSC traffic
2. **No event UUIDs**: Events are identified only by index, which shifts when events are deleted
3. **Index instability**: When event at index 2 is deleted, event at index 3 becomes index 2
4. **Change detection**: Need to identify what actually changed (added, deleted, modified, moved)

## Solution Architecture

### 1. Event Fingerprinting

Events are identified by content-based fingerprints instead of indices:

```lua
function eventFingerprint(evt)
    return string.format("%.6f|%s|%s", evt.time or 0, evt.cmd or "", evt.name or "")
end
```

**Fingerprint Format**: `{time}|{cmd}|{name}`
- Time: 6 decimal places for precision
- Cmd: Command string (empty if none)
- Name: Event name (empty if none)

**Why this works**:
- Time + cmd + name uniquely identifies an event
- Survives index shifts (fingerprint doesn't change when index does)
- Allows matching events across sync operations

### 2. Change Detection

The system compares last known state vs current state to detect:

- **Added**: Events in current but not in last state
- **Deleted**: Events in last but not in current state  
- **Moved**: Same fingerprint but different index
- **Modified**: Same index but different fingerprint (rare, usually means deletion + addition)

```lua
function detectEventChanges(oldEvents, newEvents)
    -- Build fingerprint maps
    -- Compare to find added/deleted/moved
    -- Return structured change list
end
```

### 3. Debouncing

To limit OSC traffic during event dragging:

```lua
EZ._debounceDelay = 0.5  -- 500ms default

-- In callback:
if (now - lastProcess) < debounceDelay then
    return  -- Skip this update
end
```

**Note**: MA3 Lua doesn't have proper timers, so we use simple time-based checks. The change detection also helps filter redundant updates.

### 4. State Tracking

Each hooked track maintains:

```lua
EZ._lastEventStates[key] = {
    events = {...},      -- Last known event list with fingerprints
    timestamp = os.time()
}
```

This allows:
- Change detection by comparing old vs new
- Recovery if sync gets out of sync
- Debugging and logging

## Implementation Details

### Lua Plugin (echozero.lua)

#### HookCmdSubTrack Function

```lua
function EZ.HookCmdSubTrack(tcNo, tgNo, trackNo, timeRangeIdx, subTrackIdx)
    -- 1. Find CmdSubTrack
    -- 2. Hook it with global callback
    -- 3. Initialize last known state
    -- 4. Store hook info
end
```

#### Change Detection Flow

1. **Callback fires** → `EZ._onTrackChange(obj)`
2. **Debounce check** → Skip if too soon since last process
3. **Get current events** → `getTrackEvents(track)`
4. **Compare with last state** → `detectEventChanges(old, new)`
5. **Send only if changes** → OSC message with change details
6. **Update last state** → Store for next comparison

#### OSC Message Format

```json
{
  "type": "track",
  "change": "changed",
  "tc": 101,
  "tg": 1,
  "track": 1,
  "events": [...],
  "changes": {
    "added_count": 1,
    "deleted_count": 0,
    "moved_count": 2
  },
  "added": [...],
  "deleted": [...],
  "moved": [...]
}
```

## Python Integration

### Event Matching Strategy

When receiving change notifications from MA3:

1. **Use fingerprints** to match events, not indices
2. **Handle index shifts** by matching on fingerprint
3. **Apply changes** based on detected change types:
   - Added → Create new Editor events
   - Deleted → Remove Editor events (match by fingerprint)
   - Moved → Update Editor event times (match by fingerprint)

### Integration Points

1. **LayerSyncController**: When mapping MA3 track → Editor layer
   - Call `EZ.HookCmdSubTrack()` via OSC
   - Store mapping with track coordinates

2. **MA3EventHandler**: Handle "track/changed" messages
   - Parse change detection results
   - Match events by fingerprint
   - Apply changes to Editor layer

3. **Event Matching Service**: New service to match events
   - Match MA3 events to Editor events by fingerprint
   - Handle index shifts gracefully
   - Maintain sync state

## Usage

### Hooking a Track for Sync

```lua
-- In MA3 plugin
EZ.HookCmdSubTrack(101, 1, 1)  -- TC101, TG1, Track 1
```

### Setting Debounce Delay

```lua
EZ.SetDebounceDelay(0.3)  -- 300ms debounce
```

### Unhooking

```lua
EZ.UnhookTrack(101, 1, 1)
```

## Edge Cases

### Event Deletion

When event at index 2 is deleted:
- Old: [evt1@idx1, evt2@idx2, evt3@idx3]
- New: [evt1@idx1, evt3@idx2]
- Detection: evt2 deleted, evt3 moved (idx3→idx2)

### Event Addition

When event added at index 2:
- Old: [evt1@idx1, evt3@idx2]
- New: [evt1@idx1, evt2@idx2, evt3@idx3]
- Detection: evt2 added, evt3 moved (idx2→idx3)

### Rapid Dragging

During event dragging:
- Multiple callbacks fire rapidly
- Debouncing limits processing
- Change detection filters redundant updates
- Final state sent when dragging stops

## CRITICAL: MA3 Lua State Persistence

### The Problem

**MA3's Lua plugin state persists across EchoZero restarts, but EchoZero's state does not persist across restarts.**

This architectural asymmetry causes subtle but critical bugs:

```
Session 1:
  EchoZero: "Hook track 101.1.1"
  MA3: "OK, hooked" → Sends events to EchoZero ✓
  EchoZero: Receives events, displays them ✓

Session 2 (EchoZero restarted, MA3 still running):
  EchoZero: "Hook track 101.1.1"  
  MA3: "Already hooked, returning true" → NO EVENTS SENT ✗
  EchoZero: Waits for events... nothing arrives ✗
```

### Why This Happens

1. **MA3 Lua plugin state is in-memory**: The `EZ._hooks` table persists as long as MA3 is running
2. **EchoZero state is ephemeral**: When EchoZero restarts, it loses all knowledge of previous hooks
3. **Early-return optimization**: The original code returned early if already hooked, skipping the event send

### The Fix (Implemented)

The `HookCmdSubTrack()` function now ALWAYS sends events, even when already hooked:

```lua
if alreadyHooked then
    log(string.format("  Already hooked: %s (will re-send initial events)", subTrackKey))
    
    -- Get existing hook info and current events
    local hookInfo = EZ._hooks[subTrackKey]
    local currentEvents = getTrackEvents(hookInfo.track)
    
    -- Send confirmation WITH events (re-sync)
    sendMessage("subtrack", "hooked", {
        tc = tcNo, tg = tgNo, track = trackNo,
        event_count = #currentEvents,
        events = currentEvents,
        resync = true  -- Flag indicates this is a re-sync
    })
    
    return true
end
```

### Design Principle for AI Agents

**When designing EchoZero ↔ MA3 interactions, always assume:**

1. **MA3 may have stale state** from previous EchoZero sessions
2. **EchoZero always starts fresh** with no knowledge of MA3's current state
3. **Requests should be idempotent** - calling the same operation twice should work correctly
4. **Always send data on request** - never assume the other side already has it

### Other Manifestations

This pattern can appear in other places:

| Operation | Wrong Approach | Correct Approach |
|-----------|---------------|------------------|
| Hook track | Skip if already hooked | Always send current events |
| Unhook track | Assume it was hooked | Check existence, handle gracefully |
| Get status | Return cached state | Always query current state |
| Clear data | Assume data exists | Handle empty/missing gracefully |

### Testing for State Persistence Bugs

To reproduce state persistence issues:

1. Start MA3 and EchoZero
2. Set up a hook (should work)
3. Restart EchoZero only (keep MA3 running)
4. Try to set up the same hook again
5. If events don't appear → state persistence bug

### Related: MA3 Plugin Reload

When the MA3 plugin is reloaded (`Lua "dofile(...)"` or plugin menu):
- `EZ._hooks` is cleared (empty table)
- All hooks are lost
- No unhook callbacks fire
- EchoZero won't know hooks are gone

EchoZero should handle the case where MA3 thinks nothing is hooked but EchoZero thinks tracks are hooked. The `subtrack.hooked` message with `resync=true` helps with this.

## Future Improvements

1. **Proper debouncing**: Use MA3 timer mechanism if available
2. **Batch updates**: Group multiple changes into single OSC message
3. **Conflict resolution**: Handle simultaneous MA3 + Editor changes
4. **Performance**: Optimize fingerprint calculation for large event lists
5. **Recovery**: Handle cases where sync state gets corrupted

## Testing

See `@echozero_debug.lua` for testing CmdSubTrack hooking:
- Hooks CmdSubTrack
- Monitors change callbacks
- Verifies fingerprint generation
- Tests change detection

## Related Files

- `@echozero.lua` - Main plugin implementation
- `@echozero_debug.lua` - Test implementation
- `src/features/show_manager/application/layer_sync_controller.py` - Python integration point
- `src/features/ma3/application/ma3_event_handler.py` - OSC message handler
