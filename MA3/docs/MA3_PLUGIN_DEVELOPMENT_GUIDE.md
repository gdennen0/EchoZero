# MA3 Plugin Development Guide

Comprehensive guide for developing grandMA3 Lua plugins that integrate with EchoZero.

---

## Table of Contents

1. [Plugin Architecture](#plugin-architecture)
2. [DataPool Structure](#datapool-structure)
3. [Accessing Data Objects](#accessing-data-objects)
4. [Socket Communication](#socket-communication)
5. [OSC Protocol](#osc-protocol)
6. [Hook System](#hook-system)
7. [Common Patterns](#common-patterns)
8. [Debugging](#debugging)
9. [Error Handling](#error-handling)
10. [Lessons Learned](#lessons-learned)

---

## Plugin Architecture

### Plugin File Structure

```
ma3_plugins/
  plugins/
    echozero.lua        # Main plugin module
    echozero_debug.lua  # Debug/test plugin
```

### Loading a Plugin

Plugins are loaded from the grandMA3's plugin folder:
- **macOS**: `~/MALightingTechnology/gma3_library/datapools/plugins/`
- **Windows**: `%USERPROFILE%\MALightingTechnology\gma3_library\datapools\plugins\`

### Plugin Entry Point

```lua
-- Load the main plugin
Lua "dofile('/path/to/echozero.lua')"
```

### Module Pattern

```lua
-- echozero.lua (main module)
local EZ = {
    config = {
        ip = "127.0.0.1",
        port = 9000,
        debug = false
    },
    _socket = nil,
    _socketOk = false,
}

-- Initialize socket on module load
local ok, socket = pcall(function() return require("socket.core") end)
if ok and socket then
    EZ._socket = socket
    EZ._socketOk = true
end

-- Expose functions
function EZ.GetTimecodes() ... end
function EZ.GetTrackGroups(tcNo) ... end

return EZ
```

---

## DataPool Structure

### Hierarchy

```
DataPool
  └── Timecodes (indexed 1-N in Lua, but accessed by pool number in UI)
        └── TrackGroups (children of Timecode)
              └── Tracks (children of TrackGroup, filter out "Marker")
                    └── TimeRanges (children of Track)
                          └── SubTracks (CmdSubTrack, FaderSubTrack)
                                └── Events (children of SubTrack)
```

### Important Notes

1. **Pool Numbers vs Indices**: MA3 UI shows pool numbers (e.g., TC 101), but Lua accesses by index
2. **Children Access**: Always use `:Children()` method, NOT direct property access
3. **Marker Tracks**: TrackGroups contain a "Marker" child that should be filtered out

### Finding Objects by Pool Number

```lua
local function getTC(tcNo)
    local dp = DataPool()
    if not dp or not dp.Timecodes then return nil end
    
    -- Iterate all timecodes to find by pool number
    for i = 1, dp.Timecodes:Count() do
        local tc = dp.Timecodes[i]
        if tc and tc.no == tcNo then
            return tc
        end
    end
    return nil
end
```

---

## Accessing Data Objects

### CRITICAL: Use :Children() for Hierarchy

**WRONG** (will return nil):
```lua
local trackGroups = tc.TrackGroups  -- DOES NOT WORK
local tracks = tg.Tracks            -- DOES NOT WORK
```

**CORRECT**:
```lua
local trackGroups = tc:Children()   -- Returns all children (TrackGroups)
local tracks = tg:Children()        -- Returns all children (Tracks + Marker)
```

### Filtering Out Marker

```lua
local tracks = tg:Children()
for i = 1, #tracks do
    local track = tracks[i]
    if track and track.name ~= "Marker" then
        -- Process actual track
    end
end
```

### Accessing Events

Events are deeply nested:

```lua
local function getTrackEvents(track)
    local events = {}
    if not track then return events end
    
    -- Track -> TimeRanges
    local timeRanges = track:Children()
    if not timeRanges then return events end
    
    for trIdx = 1, #timeRanges do
        local timeRange = timeRanges[trIdx]
        if timeRange then
            -- TimeRange -> SubTracks
            local subTracks = timeRange:Children()
            if subTracks then
                for stIdx = 1, #subTracks do
                    local subTrack = subTracks[stIdx]
                    if subTrack then
                        -- Check subtrack type
                        local subTrackClass = subTrack:GetClass and subTrack:GetClass() or "unknown"
                        
                        if subTrackClass == "CmdSubTrack" or subTrackClass == "FaderSubTrack" then
                            -- SubTrack -> Events
                            local evts = subTrack:Children()
                            if evts then
                                for evIdx = 1, #evts do
                                    local evt = evts[evIdx]
                                    if evt then
                                        table.insert(events, {
                                            no = evt.no or evIdx,
                                            time = evt.time or 0,
                                            duration = evt.duration or 0,
                                            cmd = evt.cmd or "",
                                            name = evt.name or "",
                                            cue = evt.cue,
                                            subtrack_type = subTrackClass
                                        })
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    return events
end
```

### Time Value Conversion

MA3 stores time in a special format:

```lua
local function convertTime(timeRaw)
    if type(timeRaw) == "number" then
        -- Large values are in internal format (divide by 16777216)
        if timeRaw > 86400 then
            return timeRaw / 16777216
        else
            return timeRaw
        end
    elseif type(timeRaw) == "string" then
        return tonumber(timeRaw) or 0
    end
    return 0
end
```

---

## Socket Communication

### Loading socket.core

```lua
-- Must use pcall to handle potential load failure
local ok, socket = pcall(function()
    return require("socket.core")
end)

if ok and socket then
    EZ._socket = socket
    EZ._socketOk = true
    
    -- Verify udp() is available
    if socket.udp then
        Printf("[EZ] socket.core loaded with UDP support")
    end
else
    Printf("[EZ] socket.core failed to load: %s", tostring(socket))
end
```

### Sending UDP Data

```lua
local function sendUDP(data, ip, port)
    if not EZ._socketOk then return false end
    
    local ok, err = pcall(function()
        local udp = EZ._socket.udp()
        udp:setpeername(ip, port)
        udp:send(data)
        udp:close()
    end)
    
    return ok
end
```

---

## OSC Protocol

### OSC Packet Format

OSC packets have this structure:
1. **Address** (null-terminated, padded to 4 bytes)
2. **Type Tag String** (comma + type chars, null-terminated, padded to 4 bytes)
3. **Arguments** (each padded to 4 bytes)

### Type Tags
- `i` = 32-bit integer (big-endian)
- `f` = 32-bit float (big-endian)
- `s` = string (null-terminated, padded to 4 bytes)
- `b` = blob (length-prefixed binary)

### Padding Function

```lua
local function oscPad(s)
    -- OSC strings must be null-terminated and padded to 4-byte boundary
    local len = #s + 1  -- +1 for null terminator
    local padded = math.ceil(len / 4) * 4
    return s .. string.rep('\0', padded - #s)
end
```

### Packing Integers and Floats

```lua
local function packInt(n)
    -- 32-bit big-endian integer
    local b1 = math.floor(n / 16777216) % 256
    local b2 = math.floor(n / 65536) % 256
    local b3 = math.floor(n / 256) % 256
    local b4 = n % 256
    return string.char(b1, b2, b3, b4)
end

local function packFloat(f)
    -- IEEE 754 single precision
    -- (Implementation depends on available libraries)
    return string.pack(">f", f)  -- If available
end
```

### Building OSC Message

```lua
local function sendOSC(address, types, ...)
    local args = {...}
    local typeTag = "," .. (types or "")
    local data = oscPad(address) .. oscPad(typeTag)
    
    local i = 1
    for c in types:gmatch(".") do
        local v = args[i]
        if c == "i" then
            data = data .. packInt(v or 0)
        elseif c == "f" then
            data = data .. packFloat(v or 0)
        elseif c == "s" then
            data = data .. oscPad(tostring(v or ""))
        end
        i = i + 1
    end
    
    return sendUDP(data, EZ.config.ip, EZ.config.port)
end
```

### EchoZero Message Format

EchoZero uses a pipe-delimited string format within OSC:

```
Address: /ez/message
Type: s (string)
Payload: type=X|change=Y|timestamp=Z|field1=value1|field2=[json_array]
```

Example:
```
type=trackgroups|change=list|timestamp=1234567890|tc=101|count=2|trackgroups=[{"no":1,"name":"Group1","track_count":5}]
```

---

## Hook System

### Purpose

Hooks allow MA3 to notify EchoZero when data changes, avoiding constant polling.

### Official MA3 Hook API

Based on the [official MA3 documentation](https://help.malighting.com/grandMA3/2.0/HTML/lua_objectfree_hookobjectchange.html):

**HookObjectChange(function, handle, handle[, handle])**
- First arg: Callback function triggered on change
- Second arg: Handle to the object to monitor
- Third arg: Handle to the plugin (REQUIRED)
- Fourth arg: Optional additional object for callback
- Returns: Integer hook ID

**Unhook(integer)**
- Removes a hook by its ID
- See: https://help.malighting.com/grandMA3/2.0/HTML/lua_objectfree_unhook.html

**DumpAllHooks()**
- Prints all hooks to System Monitor for debugging
- See: https://help.malighting.com/grandMA3/2.0/HTML/lua_objectfree_dumpallhooks.html

### Capturing the Plugin Handle

**CRITICAL**: The plugin handle must be captured at module load time:

```lua
-- At the TOP of your plugin file (global scope)
local luaComponentHandle = select(4, ...)
local pluginHandle = nil
if luaComponentHandle then
    pluginHandle = luaComponentHandle:Parent()
end
```

### Implementing Hooks

```lua
-- Store for active hooks
EZ._hooks = {}
EZ._pluginHandle = pluginHandle

function EZ.HookTrack(tcNo, tgNo, trackNo)
    local key = string.format("%d.%d.%d", tcNo, tgNo, trackNo)
    
    -- Check if already hooked
    if EZ._hooks[key] then
        return true
    end
    
    -- Get the track object
    local track = getTrack(tcNo, tgNo, trackNo)
    if not track then return false end
    
    -- Verify HookObjectChange is available
    if not HookObjectChange then
        Printf("HookObjectChange not available")
        return false
    end
    
    -- Create callback
    local function onTrackChange(obj, changeType)
        Printf("Track %s changed!", key)
        -- Send notification to EchoZero
        sendMessage("track", "changed", {
            tc = tcNo, tg = tgNo, track = trackNo,
            changeType = changeType or "modified"
        })
    end
    
    -- Register the hook
    local hookId = HookObjectChange(onTrackChange, track, EZ._pluginHandle)
    
    if hookId then
        EZ._hooks[key] = {
            id = hookId,
            track = track,
            tc = tcNo,
            tg = tgNo,
            trackNo = trackNo
        }
        return true
    end
    
    return false
end

function EZ.UnhookTrack(tcNo, tgNo, trackNo)
    local key = string.format("%d.%d.%d", tcNo, tgNo, trackNo)
    
    local hookInfo = EZ._hooks[key]
    if not hookInfo then return false end
    
    -- Use Unhook(id) to remove
    if Unhook then
        pcall(function() Unhook(hookInfo.id) end)
    end
    
    EZ._hooks[key] = nil
    return true
end

function EZ.UnhookAll()
    for key, hookInfo in pairs(EZ._hooks) do
        if Unhook then
            pcall(function() Unhook(hookInfo.id) end)
        end
    end
    EZ._hooks = {}
end
```

### Reloading Plugins

After editing the Lua plugin, you must reload it in MA3. Use the `RP` command (shortcut for `ReloadAllPlugins`):

```
RP
```

See: https://help.malighting.com/grandMA3/2.0/HTML/keyword_reloadallplugins.html

**Important**: The show file does NOT save Lua memory state. When the show loads, all plugins are reloaded from disk.

### Testing Hooks

To test hook functionality:

1. **Reload the plugin** (if you made changes):
   ```
   RP
   ```

2. **In MA3 Console**:
   ```
   EZ.TestHook(101, 1, 1)
   ```

2. **Expected Output**:
   ```
   === HOOK TEST ===
   Testing hook for TC101.TG1.TR1
     HookObjectChange: available
     Plugin handle: captured
     Track: found ('TrackName')
     Attempting to create hook...
     SUCCESS: Hook created!
   === END HOOK TEST ===
   ```

3. **In EchoZero Console**:
   ```
   HOOK TEST SUCCESS: TC101.TG1.TR1
     Hook created - modify track to test
   ```

4. **Modify the track in MA3** - add/delete/move an event

5. **In EchoZero Console**, you should see:
   ```
   Track changed: tc101_tg1_tr1 (modified) - X events
   ```

### Debugging Hooks

```lua
-- List all EchoZero hooks
EZ.ListHooks()

-- Dump all MA3 hooks to System Monitor
EZ.DumpHooks()
```

### Hook State Management

EchoZero should track:
1. Which tracks are currently hooked
2. When to unhook (project close, block deleted, etc.)
3. Hook failures and retry logic

---

## Common Patterns

### Safe Object Access with pcall

```lua
local function safeGetChildren(obj)
    if not obj then return nil end
    
    local ok, children = pcall(function()
        return obj:Children()
    end)
    
    if ok then
        return children
    else
        return nil
    end
end
```

### Sending Structured Messages

```lua
local function sendMessage(msgType, changeType, data)
    local timestamp = os.time()
    local parts = {
        "type=" .. msgType,
        "change=" .. changeType,
        "timestamp=" .. timestamp
    }
    
    for key, value in pairs(data) do
        if type(value) == "table" then
            -- Encode as JSON
            table.insert(parts, key .. "=" .. encodeJSON(value))
        else
            table.insert(parts, key .. "=" .. tostring(value))
        end
    end
    
    local message = table.concat(parts, "|")
    return sendOSC("/ez/message", "s", message)
end
```

### Simple JSON Encoding

```lua
local function encodeJSON(tbl)
    if type(tbl) ~= "table" then
        return tostring(tbl)
    end
    
    -- Check if array
    local isArray = (#tbl > 0)
    
    if isArray then
        local items = {}
        for _, v in ipairs(tbl) do
            if type(v) == "table" then
                table.insert(items, encodeJSON(v))
            elseif type(v) == "string" then
                table.insert(items, '"' .. v .. '"')
            else
                table.insert(items, tostring(v))
            end
        end
        return "[" .. table.concat(items, ",") .. "]"
    else
        local items = {}
        for k, v in pairs(tbl) do
            local valStr
            if type(v) == "table" then
                valStr = encodeJSON(v)
            elseif type(v) == "string" then
                valStr = '"' .. v .. '"'
            else
                valStr = tostring(v)
            end
            table.insert(items, '"' .. k .. '":' .. valStr)
        end
        return "{" .. table.concat(items, ",") .. "}"
    end
end
```

---

## Debugging

### Reloading After Changes

After editing plugin files, reload them in MA3:

```
RP
```

This is a shortcut for `ReloadAllPlugins`. See the [official docs](https://help.malighting.com/grandMA3/2.0/HTML/keyword_reloadallplugins.html).

### Printf for Console Output

```lua
Printf("[EZ] Debug message: %s", someValue)
```

### Debug Mode Toggle

```lua
local function dbg(msg)
    if EZ.config.debug then
        Printf("[EZ DEBUG] %s", msg)
    end
end
```

### Testing Socket Connectivity

```lua
function EZ.TestSocket()
    if not EZ._socketOk then
        Printf("[EZ] Socket not available")
        return false
    end
    
    Printf("[EZ] Testing socket to %s:%d", EZ.config.ip, EZ.config.port)
    
    local ok = sendOSC("/ez/test", "s", "ping")
    
    if ok then
        Printf("[EZ] Socket test: SUCCESS")
    else
        Printf("[EZ] Socket test: FAILED")
    end
    
    return ok
end
```

### Dumping DataPool Structure

```lua
function EZ.DebugTimecode(tcNo)
    local tc = getTC(tcNo)
    if not tc then
        Printf("[EZ] TC %d not found", tcNo)
        return
    end
    
    Printf("[EZ] TC %d: '%s'", tcNo, tc.name or "")
    
    local tgs = tc:Children()
    if tgs then
        Printf("[EZ]   %d track groups", #tgs)
        for i = 1, #tgs do
            local tg = tgs[i]
            if tg then
                Printf("[EZ]   TG[%d]: '%s'", i, tg.name or "")
                
                local tracks = tg:Children()
                if tracks then
                    for j = 1, #tracks do
                        local track = tracks[j]
                        if track and track.name ~= "Marker" then
                            Printf("[EZ]     Track[%d]: '%s'", j, track.name or "")
                        end
                    end
                end
            end
        end
    end
end
```

---

## Error Handling

### Always Send Error Responses

```lua
function EZ.GetTrackGroups(tcNo)
    local tc = getTC(tcNo)
    if not tc then
        sendMessage("trackgroups", "error", {
            tc = tcNo,
            error = "Timecode not found"
        })
        return nil
    end
    
    -- ... process and send success response
end
```

### Graceful Degradation

```lua
-- If socket fails, continue without crashing
if not EZ._socketOk then
    Printf("[EZ] WARNING: OSC disabled, socket not available")
    -- Plugin can still function for local operations
end
```

---

## Lessons Learned

### Critical Discoveries

1. **:Children() is Required**: Direct property access (tc.TrackGroups) does NOT work. Must use tc:Children() method.

2. **Filter Out Marker**: TrackGroups contain a "Marker" child that must be filtered when counting/iterating tracks.

3. **Time Format**: Event times may be in internal format (large numbers). Divide by 16777216 if > 86400.

4. **Pool Numbers != Indices**: UI shows pool numbers, Lua uses 1-based indices. Must iterate to find by pool number.

5. **pcall Everything**: MA3 Lua environment can throw errors unexpectedly. Wrap object access in pcall.

6. **OSC Padding**: All OSC strings must be null-terminated AND padded to 4-byte boundaries.

7. **UDP is Unreliable**: Messages can be lost. For critical operations, implement acknowledgment or polling.

### Performance Considerations

1. **Minimize GetTracks Calls**: Cache track lists, only refresh when needed.

2. **Use Hooks Over Polling**: Once hooks are implemented, prefer them over repeated data fetches.

3. **Batch Messages**: When possible, send multiple data items in one message.

### EchoZero Integration Notes

1. **Connection Flow**:
   - EchoZero starts OSC listener
   - User triggers Lua command in MA3: `EZ.GetTrackGroups(101)`
   - MA3 sends response via UDP to EchoZero
   - EchoZero parses and updates UI

2. **Recommended Flow**:
   ```
   1. Check connection: EZ.ping()
   2. Get structure: EZ.GetTimecodes()
   3. For each TC: EZ.GetTrackGroups(tcNo)
   4. For each TG with tracks: EZ.GetTracks(tcNo, tgNo)
   5. Hook tracks of interest: EZ.HookTrack(...)
   6. On done: EZ.UnhookAllTracks()
   ```

3. **State to Track in EchoZero**:
   - MA3 connection status (last ping time)
   - Which timecode is selected
   - Which tracks are hooked
   - Cached track/event data

---

## Quick Reference

### Function Signatures

```lua
-- Connection
EZ.Ping()                           -- Send ping, check connection
EZ.Status()                         -- Get status info
EZ.SetTarget(ip, port)              -- Set EchoZero address

-- Data Queries
EZ.GetTimecodes()                   -- List all timecodes
EZ.GetTrackGroups(tcNo)             -- List track groups for timecode
EZ.GetTracks(tcNo, tgNo)            -- List tracks in track group
EZ.GetEvents(tcNo, tgNo, tr)        -- Get events for specific track
EZ.GetAllEvents(tcNo)               -- Get all events for timecode

-- Hooks (Real-time Monitoring)
EZ.TestHook(tcNo, tgNo, tr)         -- Test hook functionality
EZ.HookTrack(tcNo, tgNo, tr)        -- Start monitoring track
EZ.UnhookTrack(tcNo, tgNo, tr)      -- Stop monitoring track
EZ.HookTrackGroup(tcNo, tgNo)       -- Hook all tracks in group
EZ.UnhookTrackGroup(tcNo, tgNo)     -- Unhook all tracks in group
EZ.UnhookAll()                      -- Remove all hooks
EZ.ListHooks()                      -- List active EchoZero hooks
EZ.DumpHooks()                      -- Dump all MA3 hooks to System Monitor

-- Debugging
EZ.DebugTimecode(tcNo)              -- Debug print timecode structure
EZ.TestSocket()                     -- Test socket connectivity
EZ.SetDebug(bool)                   -- Enable/disable debug output
```

### Message Types

| Type        | Change Types              | Description                |
|-------------|---------------------------|----------------------------|
| connection  | ping, status              | Connection health          |
| timecodes   | list, error               | Timecode listing           |
| trackgroups | list, error               | Track group listing        |
| trackgroup  | hooked, unhooked, error   | Track group hook status    |
| tracks      | list, error, unhooked_all | Track listing, bulk unhook |
| track       | hooked, unhooked, changed | Single track hook status   |
| events      | list, all, error          | Event data                 |
| hooks       | list                      | Active hook listing        |
| hook_test   | success, failed, error    | Hook test results          |

---

*Last Updated: January 2026*
