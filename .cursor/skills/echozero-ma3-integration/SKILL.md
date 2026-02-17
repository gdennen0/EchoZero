---
name: echozero-ma3-integration
description: MA3 GrandMA3 integration for EchoZero - Lua plugins, DataPool access, OSC, hooks, data structures. Use when working on MA3 sync, Lua plugins, echozero.lua, show manager, track hooks, DataPool, timecodes, or MA3 data access.
---

# MA3 Integration

## Plugin Location

`ma3_plugins/plugins/echozero_spine/` - echozero.lua, echozero_osc.lua, timecode.lua, Sequence.lua

Load path (MA3): `~/MALightingTechnology/gma3_library/datapools/plugins/`

Reload after edit: `RP` (ReloadAllPlugins) in MA3 console.

## DataPool Hierarchy

```
DataPool().Timecodes[tcNo]
  └── TrackGroups (tc:Children())
      └── TrackGroup[tgNo]
          └── :Children() -> [0]=Marker (SKIP), [1+]=Track
              └── Track (user track, trackNo+1 for MA3 internal)
                  └── :Children() -> TimeRanges
                      └── TimeRange
                          └── :Children() -> SubTracks
                              └── CmdSubTrack or FaderSubTrack
                                  └── :Children() -> Events (CmdEvent, FaderEvent)
```

**CRITICAL: Use :Children() for traversal. Direct property access (tc.TrackGroups) does NOT work.**

## Key Lua Helpers (echozero.lua)

```lua
EZ.getDP()                    -- DataPool()
EZ.getTC(tcNo)                 -- Timecode by number
EZ.getTG(tcNo, tgNo)           -- TrackGroup
EZ.getTrack(tcNo, tgNo, trackNo)  -- trackNo is user-visible; internally uses trackNo+1
EZ.getCmdSubTrack(tcNo, tgNo, trackNo, timeRangeIdx, subTrackIdx)
EZ.getTrackEvents(track)       -- All events from track
```

## Track Indexing

- **User-visible Track 1** = MA3 index 2 (index 1 is Marker)
- Lua: `EZ.getTrack(tcNo, tgNo, trackNo)` - trackNo is user-visible; internally `trackNo + 1`
- Never expose MA3 internal indices to EchoZero.

## Accessing Events

```lua
-- Track -> TimeRanges -> SubTracks -> Events
local timeRanges = track:Children()
for trIdx = 1, #timeRanges do
    local subTracks = timeRanges[trIdx]:Children()
    for stIdx = 1, #subTracks do
        local subTrack = subTracks[stIdx]
        local cls = subTrack:GetClass and subTrack:GetClass() or "unknown"
        if cls == "CmdSubTrack" or cls == "FaderSubTrack" then
            local evts = subTrack:Children()
            for _, evt in ipairs(evts) do
                -- evt: no, time, duration, cmd, name
            end
        end
    end
end
```

## Time Conversion

16,777,216 internal units = 1 second

```lua
local TIME_UNITS_PER_SECOND = 16777216
function ma3TimeToSeconds(ma3Time)
    return (ma3Time or 0) / TIME_UNITS_PER_SECOND
end
```

Use fingerprints (time + cmd + name) for event matching, not exact time.

## Building Lua Plugins

### Plugin Handle (Required for Hooks)

```lua
local luaComponentHandle = select(4, ...)
local function getPluginHandle()
    return luaComponentHandle and luaComponentHandle:Parent() or nil
end
```

### Hook Pattern

```lua
-- Store callback in EZ table (persist - prevent GC)
EZ._onTrackChange = function(obj) ... end
local hookId = HookObjectChange(EZ._onTrackChange, targetObject, pluginHandle)
-- Cleanup: Unhook(hookId)
```

Local functions get garbage collected - hooks break. Store in EZ.

### Safe Access (pcall)

```lua
local ok, children = pcall(function() return obj:Children() end)
```

### Filter Marker

TrackGroup Children: index 0 or name "Marker" is system track - skip when iterating user tracks.

## State Persistence (CRITICAL)

MA3 Lua state persists across EchoZero restarts. EchoZero state does not.

Always send current data with responses, even if "already done":

```lua
if EZ._hooks[key] then
    local currentEvents = getTrackEvents(...)
    sendMessage("subtrack", "hooked", {events = currentEvents, resync = true})
    return true
end
```

## OSC

- EchoZero listener must be active BEFORE MA3 sends
- Use OSC.jsonEncode for structured data
- EZ.sendMessage(msgType, changeType, data)

## Synced Layers (Python)

Synced layers get data from MA3. Never clear during execution:

```python
synced_ids = {l.id for l in layer_manager.get_synced_layers()}
scene.clear_events_except_layers(synced_ids)
```

## Reference

- Pitfalls: `AgentAssets/MA3_INTEGRATION_PITFALLS.md`
- Data structures: `ma3_plugins/docs/MA3_DATA_STRUCTURES.md`
- Plugin guide: `ma3_plugins/docs/MA3_PLUGIN_DEVELOPMENT_GUIDE.md`
- Timecode: `ma3_plugins/docs/TIMECODE_STRUCTURE.md`
