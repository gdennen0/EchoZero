# grandMA3 Lua API Reference

Complete reference for the grandMA3 Lua API, organized by purpose for EchoZero development.

**Legend:**
- [USED] = Currently used in EchoZero plugin
- [TESTED] = Tested but not in production
- [UNTESTED] = Not yet tested

---

## Table of Contents

1. [Core Data Access](#core-data-access)
2. [Object Manipulation](#object-manipulation)
3. [Object Properties](#object-properties)
4. [Object Navigation](#object-navigation)
5. [Hooks & Change Notifications](#hooks--change-notifications)
6. [Console Output & Debugging](#console-output--debugging)
7. [Command Execution](#command-execution)
8. [File System](#file-system)
9. [Variables & State](#variables--state)
10. [UI & Dialogs](#ui--dialogs)
11. [Fixture & DMX](#fixture--dmx)
12. [Utility Functions](#utility-functions)
13. [EchoZero-Specific Patterns](#echozero-specific-patterns)

---

## Core Data Access

### Pool Access Functions [USED]

These functions return handles to major show data pools.

| Function | Returns | Description | EchoZero Usage |
|----------|---------|-------------|----------------|
| `DataPool()` | handle | Root data pool | [USED] Primary entry point for all data |
| `ShowData()` | handle | Show data root | [UNTESTED] |
| `ShowSettings()` | handle | Show settings | [UNTESTED] |
| `Root()` | handle | Root object | [UNTESTED] |
| `Pult()` | handle | Console root | [UNTESTED] |
| `Programmer()` | handle | Programmer content | [UNTESTED] |
| `Patch()` | handle | Patch information | [UNTESTED] |
| `CurrentUser()` | handle | Current user | [UNTESTED] |
| `CurrentProfile()` | handle | Current profile | [UNTESTED] |
| `CurrentExecPage()` | handle | Current executor page | [UNTESTED] |
| `SelectedSequence()` | handle | Currently selected sequence | [UNTESTED] |
| `SelectedTimecode()` | handle | Currently selected timecode | [UNTESTED] |

### DataPool Children (accessed via `DataPool().PoolName`)

| Pool Name | Index | Description | EchoZero Usage |
|-----------|-------|-------------|----------------|
| `Timecodes` | #0014 | Timecode tracks | [USED] Core for sync |
| `Sequences` | #0006 | Sequence/cue lists | [USED] Track assignment |
| `Plugins` | #0007 | Lua plugins | [UNTESTED] |
| `Macros` | #0008 | Macros | [UNTESTED] |
| `Groups` | #0005 | Fixture groups | [UNTESTED] |
| `Worlds` | #0001 | World definitions | [UNTESTED] |
| `Pages` | #0012 | Executor pages | [UNTESTED] |
| `Layouts` | #0013 | Layouts | [UNTESTED] |

### Object Retrieval [USED]

```lua
-- Get object by address string
GetObject(string:address) -> handle

-- Get object list by address
ObjectList(string:address[, {options}]) -> {handles}

-- Convert address to handle
FromAddr(string:address[, base_handle]) -> handle

-- Convert handle to address
ToAddr(handle, bool:with_name[, bool:visible_addr]) -> string
```

**EchoZero Usage:**
```lua
-- We use direct DataPool access, not GetObject
local tc = DataPool().Timecodes[tcNo]
local seq = DataPool().Sequences[seqNo]
```

---

## Object Manipulation

### Creating Objects [USED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `Create(handle, index[, class[, undo]])` | Create child at index | [UNTESTED] |
| `Append(handle[, class[, undo[, count]]])` | Append child | [UNTESTED] |
| `Acquire(handle[, class[, undo]])` | Get or create child | [USED] Track/event creation |
| `Insert(handle, index[, class[, undo[, count]]])` | Insert child at index | [UNTESTED] |

**EchoZero Pattern for Creating:**
```lua
-- Create a new track
local track = trackGroup:Acquire()
track.name = "My Track"

-- Create TimeRange in track
local timeRange = track:Acquire('TimeRange')

-- Create CmdSubTrack in TimeRange
local cmdSubTrack = timeRange:Acquire('CmdSubTrack')

-- Create event in CmdSubTrack
local event = cmdSubTrack:Acquire()
event:Set("Time", time_units)
event:Set("Cmd", "Go+ Seq 1")
```

### Deleting Objects [USED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `Delete(handle, index[, undo])` | Delete child at index | [USED] Event deletion |
| `Remove(handle, index[, undo])` | Remove child at index | [UNTESTED] |

**CRITICAL:** `Delete()` is called on the PARENT with the child index:
```lua
-- Delete event at index 3 from CmdSubTrack
cmdSubTrack:Delete(3)  -- NOT event:Delete()
```

### Copying Objects

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `Copy(dst_handle, src_handle[, undo])` | Copy object | [UNTESTED] |

---

## Object Properties

### Getting/Setting Properties [USED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `Get(handle, property_name[, role])` | Get property value | [USED] Implicit via dot notation |
| `Set(handle, property_name, value[, change_level])` | Set property value | [USED] Event time/cmd |
| `SetChildren(handle, property, value[, recursive])` | Set on all children | [UNTESTED] |

**EchoZero Property Access:**
```lua
-- Dot notation (uses Get internally)
local name = track.name
local time = event.time
local cmd = event.cmd

-- Explicit Set for reliable writes
event:Set("Time", time_units)
event:Set("Cmd", "Go+ Seq 1")
track:Set("Target", sequence)
```

### Property Introspection

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `PropertyCount(handle)` | Number of properties | [UNTESTED] |
| `PropertyName(handle, index)` | Property name by index | [UNTESTED] |
| `PropertyType(handle, index)` | Property type | [UNTESTED] |
| `PropertyInfo(handle, index)` | Property metadata | [UNTESTED] |
| `Dump(handle)` | Debug dump object info | [TESTED] Exploration |

### Object Class/Type [USED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `GetClass(handle)` | Get object class name | [USED] Identify CmdSubTrack |
| `IsClass(handle)` | Check if is class | [UNTESTED] |
| `GetChildClass(handle)` | Expected child class | [UNTESTED] |
| `IsValid(handle)` | Check handle valid | [UNTESTED] |

**EchoZero Class Checking:**
```lua
local subTrackClass = subTrack:GetClass()
if subTrackClass == "CmdSubTrack" or subTrackClass == "FaderSubTrack" then
    -- Process this subtrack
end
```

---

## Object Navigation

### Hierarchy Navigation [USED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `Children(handle)` | Get all children as array | [USED] Core navigation |
| `Count(handle)` | Number of children | [USED] |
| `MaxCount(handle)` | Max possible children | [UNTESTED] |
| `Parent(handle)` | Get parent handle | [USED] Plugin handle |
| `Index(handle)` | Get object index | [UNTESTED] |
| `Ptr(handle, index)` | Get child by index (1-based) | [UNTESTED] |
| `CurrentChild(handle)` | Currently selected child | [UNTESTED] |

**CRITICAL: Always use `:Children()` for hierarchy traversal:**
```lua
-- CORRECT
local trackGroups = timecode:Children()
local tracks = trackGroup:Children()
local timeRanges = track:Children()
local subTracks = timeRange:Children()
local events = subTrack:Children()

-- WRONG (these don't work)
local tgs = timecode.TrackGroups  -- Returns nil
local tracks = trackGroup.Tracks  -- Returns nil
```

### Finding Objects

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `Find(handle, name[, class])` | Find child by name | [UNTESTED] |
| `FindRecursive(handle, name[, class])` | Find in all descendants | [UNTESTED] |
| `FindWild(handle, name)` | Find with wildcards | [UNTESTED] |
| `FindParent(handle, class)` | Find ancestor by class | [UNTESTED] |

---

## Hooks & Change Notifications

### Hook Functions [USED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `HookObjectChange(callback, handle, plugin_handle[, target])` | Register change callback | [USED] Track monitoring |
| `Unhook(hook_id)` | Remove hook by ID | [USED] |
| `UnhookMultiple(callback, target, context)` | Remove multiple hooks | [UNTESTED] |
| `DumpAllHooks()` | Debug: list all hooks | [USED] Debugging |
| `PrepareWaitObjectChange(handle[, threshold])` | Prepare for wait | [UNTESTED] |

**EchoZero Hook Pattern:**
```lua
-- Capture plugin handle at load time (arg 4)
local luaComponentHandle = select(4, ...)
local pluginHandle = luaComponentHandle and luaComponentHandle:Parent() or nil

-- Register hook
local function onTrackChange(obj)
    -- obj is the changed object
    Printf("Track changed: %s", obj.name or "?")
    -- Send events to EchoZero
end

local hookId = HookObjectChange(onTrackChange, cmdSubTrack, pluginHandle)

-- Store for cleanup
EZ._hooks[key] = { id = hookId, track = track, ... }

-- Cleanup
Unhook(hookId)
```

**Hook Important Notes:**
1. Plugin handle MUST be captured at module load (arg 4)
2. Callbacks MUST be global functions (not local) to avoid garbage collection
3. Hook the CmdSubTrack, not individual events (avoids hook limits)
4. Always store hook ID for cleanup

---

## Console Output & Debugging

### Output Functions [USED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `Printf(format, ...)` | Print to command line | [USED] Primary logging |
| `Echo(format, ...)` | Print to echo area | [UNTESTED] |
| `ErrPrintf(format, ...)` | Print as error | [UNTESTED] |
| `ErrEcho(format, ...)` | Echo as error | [UNTESTED] |

**EchoZero Logging:**
```lua
local function log(msg) Printf("[EZ] %s", msg) end
local function dbg(msg) if EZ.config.debug then Printf("[EZ DBG] %s", msg) end end
```

### Debugging

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `Dump(handle)` | Object API: dump info | [TESTED] Exploration |
| `DumpAllHooks()` | List all MA3 hooks | [USED] Hook debugging |

---

## Command Execution

### Command Functions [USED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `Cmd(command[, undo], ...)` | Execute MA3 command | [USED] Store/Delete |
| `CmdIndirect(command[, undo[, target]])` | Execute without blocking | [UNTESTED] |
| `CmdIndirectWait(command[, undo[, target]])` | Execute and wait | [UNTESTED] |
| `CmdObj()` | Get command line object | [UNTESTED] |

**EchoZero Command Usage:**
```lua
-- Create sequence
Cmd(string.format('Store Sequence %d /name="%s" /nc', seqNo, name))

-- Delete events (alternative to API)
Cmd(string.format('Delete Timecode %d.%d.%d.1.1 /nc', tcNo, tgNo, trackNo))
```

### Undo Management

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `CreateUndo(text)` | Start undo group | [UNTESTED] |
| `CloseUndo(handle)` | End undo group | [UNTESTED] |

---

## File System

### Path Functions

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `GetPath(type[, create])` | Get system path | [UNTESTED] |
| `GetPathType(object[, content_type])` | Get path type name | [UNTESTED] |
| `GetPathSeparator()` | Get OS path separator | [UNTESTED] |
| `FileExists(path)` | Check file exists | [UNTESTED] |
| `DirList(path[, filter])` | List directory | [UNTESTED] |
| `CopyFile(src, dst)` | Copy file | [UNTESTED] |
| `CreateDirectoryRecursive(path)` | Create directories | [UNTESTED] |

### Import/Export

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `Import(filename)` | Import from file | [UNTESTED] |
| `Export(filename, data)` | Export to file | [UNTESTED] |
| `ExportJson(filename, data)` | Export as JSON | [UNTESTED] |
| `ExportCSV(filename, data)` | Export as CSV | [UNTESTED] |

---

## Variables & State

### Variable Storage [UNTESTED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `GlobalVars()` | Global variables handle | [UNTESTED] |
| `UserVars()` | User variables handle | [UNTESTED] |
| `PluginVars([plugin_name])` | Plugin preferences | [UNTESTED] |
| `SetVar(vars, name, value)` | Set variable | [UNTESTED] |
| `GetVar(vars, name)` | Get variable | [UNTESTED] |
| `DelVar(vars, name)` | Delete variable | [UNTESTED] |

**Potential Usage:** Store EchoZero config persistently:
```lua
-- Set config
SetVar(PluginVars("EchoZero"), "ip", "192.168.1.100")
SetVar(PluginVars("EchoZero"), "port", 9000)

-- Get config
local ip = GetVar(PluginVars("EchoZero"), "ip")
```

---

## UI & Dialogs

### Dialog Functions [UNTESTED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `Confirm([title, message, display, showCancel])` | Confirmation dialog | [UNTESTED] |
| `TextInput([title, value, x, y])` | Text input dialog | [UNTESTED] |
| `PopupInput({options})` | Popup input | [UNTESTED] |
| `MessageBox({options})` | Message box | [UNTESTED] |

### Progress Bar [UNTESTED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `StartProgress(name)` | Start progress bar | [UNTESTED] |
| `StopProgress(index)` | Stop progress bar | [UNTESTED] |
| `SetProgress(index, value)` | Set progress value | [UNTESTED] |
| `SetProgressText(index, text)` | Set progress text | [UNTESTED] |
| `SetProgressRange(index, start, end)` | Set progress range | [UNTESTED] |
| `IncProgress(index[, delta])` | Increment progress | [UNTESTED] |

---

## Fixture & DMX

### Selection Functions [UNTESTED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `SelectionTable()` | Get selection as table | [UNTESTED] |
| `SelectionCount()` | Number of selected | [UNTESTED] |
| `SelectionFirst()` | First selected subfixture | [UNTESTED] |
| `SelectionNext(current)` | Next selected subfixture | [UNTESTED] |

### DMX Functions [UNTESTED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `GetDMXValue(address[, universe, percent])` | Get DMX value | [UNTESTED] |
| `GetDMXUniverse(universe[, percent])` | Get full universe | [UNTESTED] |
| `CheckDMXCollision(mode, address[, count])` | Check for collisions | [UNTESTED] |

---

## Utility Functions

### System Information [UNTESTED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `HostOS()` | Operating system | [UNTESTED] |
| `HostType()` | Host type | [UNTESTED] |
| `Version()` | MA3 version | [UNTESTED] |
| `ReleaseType()` | Release type | [UNTESTED] |
| `SerialNumber()` | Serial number | [UNTESTED] |
| `Time()` | Current time | [USED] Timestamps |

### Handle Conversion [UNTESTED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `IntToHandle(int)` | Integer to handle | [UNTESTED] |
| `HandleToInt(handle)` | Handle to integer | [UNTESTED] |
| `StrToHandle(str)` | String to handle | [UNTESTED] |
| `HandleToStr(handle)` | Handle to string | [UNTESTED] |
| `IsObjectValid(handle)` | Check handle valid | [UNTESTED] |

### Timer [UNTESTED]

| Function | Description | EchoZero Usage |
|----------|-------------|----------------|
| `Timer(func, delay, max_count[, cleanup[, context]])` | Scheduled callback | [UNTESTED] |

---

## EchoZero-Specific Patterns

### Time Conversion [USED]

MA3 stores time in a special internal format:

```lua
-- Internal time unit = 1 second * 16777216
local ONE_SECOND = 16777216

-- Convert seconds to MA3 time units
local function secondsToUnits(seconds)
    return math.floor((seconds or 0) * ONE_SECOND)
end

-- Convert MA3 time units to seconds
local function unitsToSeconds(units)
    if type(units) == "number" and units > 86400 then
        return units / ONE_SECOND
    end
    return units  -- Already in seconds
end

-- Usage
event:Set("Time", secondsToUnits(5.0))  -- Set to 5 seconds
local seconds = unitsToSeconds(event.time)
```

### Timecode Hierarchy [USED]

```
DataPool()
  └── Timecodes[tcNo]
        └── TrackGroup (via :Children()[tgNo])
              └── Track (via :Children()[trackNo])
                    └── TimeRange (via :Children()[1])
                          └── CmdSubTrack / FaderSubTrack (via :Children()[n])
                                └── Event (via :Children()[n])
```

**IMPORTANT:** Track index 1 is reserved for "Marker" (system track). User tracks start at index 2.

### Track Numbering Convention [USED]

```lua
-- User-visible track number (1-based, excludes Marker)
local userTrackNo = 1  -- First user track

-- MA3 internal index (includes Marker at 1)
local ma3Index = userTrackNo + 1  -- = 2

-- Access track
local track = DataPool().Timecodes[tcNo][tgNo][ma3Index]
```

### Socket Communication [USED]

```lua
-- Load socket.core (MA3 includes LuaSocket)
local ok, socket = pcall(function() return require("socket.core") end)
if ok and socket and socket.udp then
    EZ._socket = socket
    EZ._socketOk = true
end

-- Send UDP
local function sendUDP(data, ip, port)
    local udp = EZ._socket.udp()
    udp:setpeername(ip, port)
    udp:send(data)
    udp:close()
end
```

### OSC Message Building [USED]

```lua
-- Null-terminate and pad to 4 bytes
local function oscPad(s)
    local p = s .. "\0"
    return p .. string.rep("\0", (4 - #p % 4) % 4)
end

-- Pack 32-bit big-endian integer
local function packInt(n)
    n = math.floor(n or 0)
    return string.char(
        math.floor(n / 16777216) % 256,
        math.floor(n / 65536) % 256,
        math.floor(n / 256) % 256,
        n % 256
    )
end

-- Build OSC message
local function buildOSC(address, types, ...)
    local args = {...}
    local data = oscPad(address) .. oscPad("," .. (types or ""))
    
    local i = 1
    for c in types:gmatch(".") do
        if c == "i" then data = data .. packInt(args[i])
        elseif c == "s" then data = data .. oscPad(tostring(args[i] or ""))
        end
        i = i + 1
    end
    return data
end
```

---

## API Discovery Commands

These commands help explore the API from within MA3:

```lua
-- Get API function list
GetApiDescriptor()  -- Object-free API

-- Get Object API function list
GetObjApiDescriptor()  -- Object methods

-- Dump object properties
DataPool().Timecodes[101]:Dump()

-- List all hooks
DumpAllHooks()
```

---

## Quick Reference: EchoZero Function Mapping

| EchoZero Function | MA3 APIs Used |
|-------------------|---------------|
| `EZ.GetTimecodes()` | `DataPool().Timecodes:Children()` |
| `EZ.GetTrackGroups(tc)` | `tc:Children()` |
| `EZ.GetTracks(tc, tg)` | `tg:Children()`, filter `!= "Marker"` |
| `EZ.GetEvents(tc, tg, tr)` | Nested `:Children()` traversal |
| `EZ.CreateTrack(tc, tg, name)` | `tg:Acquire()`, `.name =` |
| `EZ.AddEvent(tc, tg, tr, time, cmd)` | `cmdSubTrack:Acquire()`, `:Set()` |
| `EZ.DeleteEvent(tc, tg, tr, idx)` | `cmdSubTrack:Delete(idx)` |
| `EZ.HookTrack(tc, tg, tr)` | `HookObjectChange()` |
| `EZ.UnhookTrack(tc, tg, tr)` | `Unhook(hookId)` |
| `EZ.CreateSequence(no, name)` | `Cmd("Store Sequence ...")` |
| `EZ.AssignTrackSequence(tc, tg, tr, seq)` | `track:Set("Target", seq)` |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2026 | Initial comprehensive API reference |

---

*Generated from grandMA3_lua_functions.txt for EchoZero development*
