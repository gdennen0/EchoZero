# grandMA3 Lua Development Learnings

Practical knowledge and gotchas discovered while building EchoZero's MA3 integration.

---

## Critical Discoveries

### 1. Children() is the ONLY way to traverse hierarchy

**Problem:** Direct property access returns nil for child collections.

```lua
-- DOES NOT WORK
local trackGroups = timecode.TrackGroups  -- nil
local tracks = trackGroup.Tracks          -- nil
local events = subTrack.Events            -- nil

-- WORKS
local trackGroups = timecode:Children()
local tracks = trackGroup:Children()
local events = subTrack:Children()
```

**Why:** MA3's object model doesn't expose child collections as properties. The `:Children()` method is required.

---

### 2. Marker Track at Index 1

**Problem:** Track index 1 is reserved for the system "Marker" track.

```lua
-- Track group children
local children = trackGroup:Children()
-- children[1] = "Marker" (system track)
-- children[2] = First user track
-- children[3] = Second user track
```

**Solution:** Always filter or offset:

```lua
-- Filter approach
for i = 1, #children do
    local track = children[i]
    if track.name ~= "Marker" then
        -- Process user track
    end
end

-- Offset approach (EchoZero uses this)
local userTrackNo = 1  -- User-visible track number
local ma3Index = userTrackNo + 1  -- Skip Marker
local track = children[ma3Index]
```

---

### 3. Time Format Conversion

**Problem:** MA3 stores time in a non-obvious format.

```lua
-- Internal format: 1 second = 16777216 units
-- This is 2^24 (likely for sub-frame precision)

local ONE_SECOND = 16777216  -- 0x1000000

-- Reading time
local timeRaw = event.time
if timeRaw > 86400 then
    -- Raw internal format
    local seconds = timeRaw / ONE_SECOND
else
    -- Already in seconds (some contexts)
    local seconds = timeRaw
end

-- Writing time
local seconds = 5.0
event:Set("Time", math.floor(seconds * ONE_SECOND))
```

**Additional Issue:** Frame rate may affect time display. If events appear at 2x expected time, check `EZ.config.timeScaleFactor`.

---

### 4. Hook Plugin Handle Requirement

**Problem:** HookObjectChange requires a plugin handle, but it's not obvious how to get it.

```lua
-- The plugin handle is passed as argument 4 when loading
local luaComponentHandle = select(4, ...)

-- WRONG: Use luaComponentHandle directly
HookObjectChange(callback, target, luaComponentHandle)  -- May work but unreliable

-- CORRECT: Get parent to get plugin handle
local pluginHandle = luaComponentHandle:Parent()
HookObjectChange(callback, target, pluginHandle)  -- Reliable
```

**Why:** The 4th argument is the LuaComponent, not the Plugin itself. Calling `:Parent()` gets the actual Plugin handle.

---

### 5. Hook Callbacks Must Be Global

**Problem:** Local callback functions get garbage collected.

```lua
-- FAILS: Callback gets garbage collected
function EZ.HookTrack(...)
    local function onTrackChange(obj)  -- Local function
        -- This callback may stop working!
    end
    HookObjectChange(onTrackChange, track, pluginHandle)
end

-- WORKS: Global callback persists
function EZ._onTrackChange(obj)  -- Global function
    -- This callback persists
end

function EZ.HookTrack(...)
    HookObjectChange(EZ._onTrackChange, track, pluginHandle)
end
```

---

### 6. Delete() Works on Parent

**Problem:** Deleting a child object uses parent method with index.

```lua
-- WRONG: No :Delete() method on child
event:Delete()  -- Error!

-- CORRECT: Call on parent with child index
local events = cmdSubTrack:Children()
cmdSubTrack:Delete(3)  -- Delete 3rd event (1-based index)
```

**Pattern:**
```lua
function EZ.DeleteEvent(tcNo, tgNo, trackNo, eventIdx)
    local cmdSubTrack = getCmdSubTrack(tcNo, tgNo, trackNo)
    if cmdSubTrack then
        pcall(function() cmdSubTrack:Delete(eventIdx) end)
    end
end
```

---

### 7. Set() vs Direct Assignment

**Problem:** Some properties require `:Set()`, direct assignment may silently fail.

```lua
-- May or may not work depending on property
event.cmd = "Go+ Seq 1"
track.name = "My Track"

-- Always works
event:Set("Cmd", "Go+ Seq 1")
track:Set("Name", "My Track")
event:Set("Time", time_units)
```

**Best Practice:** Use `:Set()` for writes, dot notation for reads.

---

### 8. pcall() Everything

**Problem:** MA3 Lua can throw errors unexpectedly on object access.

```lua
-- FRAGILE: May crash
local tc = DataPool().Timecodes[101]
local children = tc:Children()

-- ROBUST: Handle errors gracefully
local ok, tc = pcall(function() return DataPool().Timecodes[101] end)
if not ok or not tc then
    log("Timecode 101 not found")
    return nil
end

local childrenOk, children = pcall(function() return tc:Children() end)
if not childrenOk or not children then
    log("Could not get children")
    return nil
end
```

---

### 9. Pool Numbers vs Indices

**Problem:** UI shows pool numbers, Lua uses indices.

```lua
-- UI shows "Timecode 101"
-- Lua: DataPool().Timecodes[101] works (direct pool number access)

-- UI shows "Track Group 1"
-- Lua: tc:Children()[1] (1-based index into children)

-- For track groups and tracks, index = pool number in simple cases
-- For timecodes, the pool number is used directly
```

**Recommendation:** Always document whether a parameter is "pool number" or "index".

---

### 10. Hook CmdSubTrack, Not Events

**Problem:** Hooking individual events hits hook limits quickly.

```lua
-- BAD: One hook per event (hits limits)
for _, event in ipairs(events) do
    HookObjectChange(callback, event, pluginHandle)
end

-- GOOD: One hook on CmdSubTrack (efficient)
local cmdSubTrack = getCmdSubTrack(tcNo, tgNo, trackNo)
HookObjectChange(callback, cmdSubTrack, pluginHandle)
-- Callback fires when ANY event in the subtrack changes
```

---

### 11. socket.core is Available

**Discovery:** MA3 includes LuaSocket, but only the core module.

```lua
-- WORKS
local socket = require("socket.core")
local udp = socket.udp()
udp:setpeername("127.0.0.1", 9000)
udp:send(data)
udp:close()

-- MAY NOT WORK (higher-level wrappers)
local socket = require("socket")  -- May fail
local http = require("socket.http")  -- May not exist
```

---

### 12. Reload Plugins with RP

**Discovery:** After editing plugin files, reload with:

```
RP
```

This is short for `ReloadAllPlugins`. The show file does NOT save Lua memory state - plugins are always reloaded from disk.

---

## Object Hierarchy Cheat Sheet

```
DataPool()
├── Timecodes
│   └── [tcNo] (Timecode)
│       └── :Children() → [TrackGroup, ...]
│           └── :Children() → [Marker, Track1, Track2, ...]
│               │              ^^^^^^ Always index 1!
│               └── :Children() → [TimeRange, ...]
│                   └── :Children() → [CmdSubTrack, FaderSubTrack, ...]
│                       └── :Children() → [Event, ...]
├── Sequences
│   └── [seqNo] (Sequence)
│       └── ... (cues, etc.)
└── ... (other pools)
```

---

## Common Gotchas Checklist

When debugging issues:

- [ ] Using `:Children()` not property access?
- [ ] Filtering out "Marker" track?
- [ ] Using `:Set()` for writes?
- [ ] Converting time with `/ 16777216`?
- [ ] Got plugin handle via `select(4, ...):Parent()`?
- [ ] Callback function is global, not local?
- [ ] Calling `Delete()` on parent with index?
- [ ] Wrapped object access in `pcall()`?
- [ ] Reloaded plugin with `RP` after changes?

---

## Patterns That Work

### Safe Object Access

```lua
local function safeGet(func)
    local ok, result = pcall(func)
    return ok and result or nil
end

local tc = safeGet(function() return DataPool().Timecodes[101] end)
local children = tc and safeGet(function() return tc:Children() end)
```

### Track Iteration with Marker Filter

```lua
local function forEachTrack(tg, callback)
    local children = tg:Children() or {}
    local userTrackNo = 0
    for i = 1, #children do
        if children[i].name ~= "Marker" then
            userTrackNo = userTrackNo + 1
            callback(children[i], userTrackNo, i)
        end
    end
end
```

### Event Extraction

```lua
local function getTrackEvents(track)
    local events = {}
    for _, timeRange in ipairs(track:Children() or {}) do
        for _, subTrack in ipairs(timeRange:Children() or {}) do
            local class = subTrack:GetClass()
            if class == "CmdSubTrack" or class == "FaderSubTrack" then
                for idx, evt in ipairs(subTrack:Children() or {}) do
                    table.insert(events, {
                        no = evt.no or idx,
                        time = convertTime(evt.time),
                        cmd = evt.cmd or "",
                        name = evt.name or ""
                    })
                end
            end
        end
    end
    return events
end
```

---

## Things Still Unknown

- [ ] How to receive commands FROM EchoZero (MA3 doesn't have TCP server)
- [ ] Best way to handle bidirectional sync conflicts
- [ ] Performance limits of hook system
- [ ] How to persist plugin config across sessions
- [ ] Full list of hookable object types
- [ ] How to detect MA3 version/features programmatically

---

## Resources

- [MA3 Lua API](https://help.malighting.com/grandMA3/2.0/HTML/lua_objectfree_hookobjectchange.html)
- [MA3 Keyword Reference](https://help.malighting.com/grandMA3/2.0/HTML/keyword_reloadallplugins.html)
- EchoZero Plugin: `ma3_plugins/plugins/echozero_spine/echozero.lua`

---

*Last Updated: January 2026*
