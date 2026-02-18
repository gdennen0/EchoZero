# MA3 Integration Pitfalls for AI Agents

This document captures hard-won knowledge about MA3 ↔ EchoZero integration issues that are easy to miss during development.

## 1. State Persistence Asymmetry (CRITICAL)

### The Problem

**MA3's Lua plugin state persists across EchoZero restarts, but EchoZero's state does not.**

This is the #1 source of subtle bugs in MA3 integration.

### Example Scenario

```
Session 1:
  EchoZero: "Hook track 101.1.1"
  MA3: "OK, hooked" → Sends events to EchoZero
  EchoZero: Receives events ✓

Session 2 (EchoZero restarted, MA3 still running):
  EchoZero: "Hook track 101.1.1"  
  MA3: "Already hooked" → Returns true, NO EVENTS SENT
  EchoZero: Waits forever for events that never arrive ✗
```

### Root Cause

- MA3 Lua plugin: State stored in `EZ._hooks`, `EZ._lastEventStates`, etc. - **persists while MA3 runs**
- EchoZero: All state in memory - **lost on restart**

### Solution Pattern

**Always respond with current data, even if state indicates "already done":**

```lua
-- WRONG: Early return without data
if EZ._hooks[key] then
    return true  -- No events sent!
end

-- CORRECT: Always send current state
if EZ._hooks[key] then
    local currentEvents = getTrackEvents(...)
    sendMessage("subtrack", "hooked", {events = currentEvents, resync = true})
    return true
end
```

### Design Principle

**All MA3 → EchoZero operations should be idempotent and always include current state data.**

---

## 2. OSC Message Queue Timing

### The Problem

OSC messages from MA3 arrive asynchronously. If EchoZero's listener isn't ready, messages are lost.

### Example

```
EchoZero: Sends HookCmdSubTrack command
EchoZero: Immediately starts waiting for response
MA3: Processes command, sends subtrack.hooked
[MESSAGE LOST - EchoZero listener not polling]
EchoZero: Times out waiting
```

### Solution

- EchoZero must have OSC listener active BEFORE sending commands
- Use message queue with polling (not synchronous wait)
- Log all received messages for debugging

---

## 3. MA3 Track Indexing

### The Problem

MA3 tracks are 1-indexed, but Track 1 is always the "Markers" track (system track). User tracks start at index 2.

### Confusion Points

| Concept | MA3 Internal | EchoZero API | User-Visible |
|---------|--------------|--------------|--------------|
| Markers track | Index 1 | N/A (hidden) | Hidden |
| First user track | Index 2 | Track 1 | "Track 1" |
| Second user track | Index 3 | Track 2 | "Track 2" |

### Solution

The Lua plugin handles this offset internally:

```lua
-- User-visible track 1 → MA3 internal track 2
local ma3TrackNo = trackNo + 1
```

**Never expose MA3 internal indices to EchoZero. Always use user-visible indices (1-based, excluding Markers).**

---

## 4. Hook Callback Function Lifetime

### The Problem

MA3's `HookObjectChange()` requires the callback function to persist. If the function is garbage collected, the hook silently stops working.

### Example

```lua
-- WRONG: Local function will be garbage collected
local function myCallback(obj)
    -- handle change
end
HookObjectChange(myCallback, target, plugin)  -- Hook may break later

-- CORRECT: Store callback in persistent table
EZ._onTrackChange = function(obj)
    -- handle change
end
HookObjectChange(EZ._onTrackChange, target, plugin)  -- Persists
```

### Solution

Store all hook callbacks in the global `EZ` table to prevent garbage collection.

---

## 5. MA3 Plugin Reload Clears State

### The Problem

When the MA3 plugin is reloaded (via `dofile()` or plugin menu), all Lua state is cleared, including:
- Hook registrations (`EZ._hooks`)
- Event state tracking (`EZ._lastEventStates`)
- Socket handles

EchoZero won't know the hooks are gone.

### Detection

No reliable way to detect plugin reload from EchoZero side.

### Mitigation

- EchoZero should periodically verify hooks are active (ping/status)
- Re-establish hooks if status indicates none active
- Handle `subtrack.hooked` with `resync=true` gracefully

---

## 6. Time Unit Conversion

### The Problem

MA3 uses internal time units, not seconds:

```
16,777,216 internal units = 1 second
```

### Conversion

```lua
local TIME_UNITS_PER_SECOND = 16777216

function ma3TimeToSeconds(ma3Time)
    return (ma3Time or 0) / TIME_UNITS_PER_SECOND
end

function secondsToMa3Time(seconds)
    return math.floor((seconds or 0) * TIME_UNITS_PER_SECOND)
end
```

### Gotcha

The conversion can lose precision. For event matching, use fingerprints (time + cmd + name) rather than exact time comparison.

---

## 7. Synced Layer Data Flow

### The Problem

EchoZero has two distinct data flow patterns for layers:

1. **Standard Layers**: Data flows through execution pipeline (upstream blocks → downstream blocks)
2. **Synced Layers**: Data comes from MA3 via OSC (external source)

If code treats synced layers the same as standard layers, synced events get cleared during execution.

### Solution

The `TimelineWidget.set_events()` method now uses **selective clearing**:

```python
# Get synced layer IDs
synced_layer_ids = {layer.id for layer in layer_manager.get_synced_layers()}

if synced_layer_ids:
    # Only clear non-synced layer events
    scene.clear_events_except_layers(synced_layer_ids)
else:
    # No synced layers, clear all
    scene.clear_events()
```

**Never clear synced layer events during execution pipeline updates. Only the ShowManager should modify synced layers via EditorAPI.**

---

## Quick Reference: Do's and Don'ts

### DO

- Always send current state data with responses, even if "already done"
- Use user-visible track indices (1-based, excluding Markers)
- Store hook callbacks in persistent tables (prevent GC)
- Log all OSC messages for debugging
- Handle re-sync scenarios gracefully
- Preserve synced layer events during execution

### DON'T

- Return early without data just because state says "already done"
- Expose MA3 internal track indices to EchoZero
- Use local functions for hook callbacks
- Assume EchoZero and MA3 have synchronized state
- Clear synced layer events during execution pipeline updates
- Use exact time comparison for event matching (use fingerprints)

---

## Related Documentation

- `ma3_plugins/docs/CMDSUBTrack_HOOKING_IMPLEMENTATION.md` - Detailed hooking implementation
- `ma3_plugins/docs/architecture/ARCHITECTURE.md` - Overall OSC architecture
- `ma3_plugins/docs/TIMECODE_STRUCTURE.md` - MA3 timecode data model
- `@echozero.lua` - Main Lua plugin
- `@echozero_debug.lua` - Debug/test Lua plugin
