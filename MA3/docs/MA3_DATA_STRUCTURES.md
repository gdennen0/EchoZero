# grandMA3 Data Structures Reference

This document provides comprehensive documentation of grandMA3 data structures for AI agents working on EchoZero-MA3 integrations.

## Overview

grandMA3 uses a hierarchical object model accessed through `DataPool()`. All show data is organized into pools and sub-pools.

## DataPool Structure

### Root Access

```lua
DataPool()  -- Returns the root DataPool object
```

### DataPool Properties

| Property | Type | Access | Description |
|----------|------|--------|-------------|
| `NAME` | string | Read/Write | Pool name |
| `INDEX` | string | Read Only | Pool index number |
| `COUNT` | string | Read Only | Number of child pools |
| `NO` | string | Read Only | Number and count format |
| `LOCK` | string | Read/Write | Lock status (e.g., "PL") |
| `IGNORENETWORK` | string | Read Only | Network ignore flag |
| `STRUCTURELOCKED` | string | Read Only | Structure lock status |
| `SYSTEMLOCKED` | string | Read Only | System lock status |
| `USEREXPANDED` | string | Read/Write | UI expansion state |
| `FADERENABLED` | string | Read Only | Fader enabled flag |
| `OWNED` | string | Read Only | Ownership flag |
| `HIDDEN` | string | Read/Write | Visibility flag |
| `MEMORYFOOTPRINT` | string | Read Only | Memory usage |
| `GUID` | string | Read Only | Unique identifier |
| `SCRIBBLE` | string | Read/Write | User notes |
| `APPEARANCE` | string | Read/Write | Appearance settings |
| `NOTE` | string | Read/Write | Notes |
| `TAGS` | string | Read/Write | Tags |

### Child Pools

The default DataPool contains 15 child pools:

| Index | Name | Class | Access Pattern | Description |
|-------|------|-------|----------------|-------------|
| #0001 | Worlds | Worlds | `DataPool().Worlds` | World definitions |
| #0002 | Filters | Filters | `DataPool().Filters` | Filter definitions |
| #0003 | GeneratorTypes | GeneratorTypes | `DataPool().GeneratorTypes` | Generator type definitions |
| #0004 | PresetPools | PresetPools | `DataPool().PresetPools` | Preset pool collections |
| #0005 | Groups | Groups | `DataPool().Groups` | Fixture groups |
| #0006 | Sequences | Sequences | `DataPool().Sequences` | Sequence (cue list) definitions |
| #0007 | Plugins | Plugins | `DataPool().Plugins` | Lua plugin definitions |
| #0008 | Macros | Macros | `DataPool().Macros` | Macro definitions |
| #0009 | Quickeys | Quickeys | `DataPool().Quickeys` | Quick key definitions |
| #0010 | MAtricks | MAtricks | `DataPool().MAtricks` | Matrix/transform definitions |
| #0011 | Configurations | Configurations | `DataPool().Configurations` | Configuration definitions |
| #0012 | Pages | Pages | `DataPool().Pages` | Executor page definitions |
| #0013 | Layouts | Layouts | `DataPool().Layouts` | Layout definitions |
| #0014 | Timecodes | Timecodes | `DataPool().Timecodes` | Timecode track definitions |
| #0015 | Timers | Timers | `DataPool().Timers` | Timer definitions |

## Access Patterns

### Accessing Objects

```lua
-- Access a pool
local sequences = DataPool().Sequences

-- Access by index
local timecode = DataPool().Timecodes[101]

-- Access nested object
local timecode_cue = DataPool().Timecodes[101][1]  -- Timecode 101, Cue 1

-- Access by name (if available)
local sequence = DataPool().Sequences["My Sequence"]
```

### Common Object Properties

Most MA3 objects share common properties:

```lua
local obj = DataPool().Sequences[1]

obj.name      -- Object name (string)
obj.no        -- Object number (number)
obj.index     -- Object index (number)
obj:Children() -- Get child objects (returns array)
```

## Hooking into Objects

### HookObjectChange Pattern

```lua
-- Standard hook pattern
local pluginHandle = luaComponentHandle:Parent()
local targetObject = DataPool().Sequences[1]
local hookId = HookObjectChange(callbackFunction, targetObject, pluginHandle)

-- Callback function signature
local function callbackFunction(obj)
    -- obj is the object that changed
    Printf("Object changed: " .. obj.name)
end

-- Cleanup
Unhook(hookId)
```

### Hookable Objects

Objects that support `HookObjectChange()`:
- Sequences
- Timecodes
- Timecode cues
- Macros
- Groups
- Presets
- Most pool objects

## Timecode Structure

### Access Pattern

```lua
-- Timecode pool
local timecodes = DataPool().Timecodes

-- Specific timecode track
local tc = DataPool().Timecodes[101]

-- Specific cue in timecode
local cue = DataPool().Timecodes[101][1]
```

### Timecode Object Properties

```lua
local tc = DataPool().Timecodes[101]

tc.name       -- Timecode name
tc.no         -- Timecode number
tc.index      -- Timecode index
tc:Children() -- Get cues in this timecode
```

### Timecode Cue Properties

```lua
local cue = DataPool().Timecodes[101][1]

cue.name      -- Cue name
cue.no        -- Cue number
cue.index     -- Cue index
-- Additional timecode-specific properties
```

## Sequence Structure

### Access Pattern

```lua
-- Sequence pool
local sequences = DataPool().Sequences

-- Specific sequence
local seq = DataPool().Sequences[1]

-- Sequence properties
seq.name      -- Sequence name
seq.no        -- Sequence number
seq:Children() -- Get cues in sequence
```

## Common Operations

### Iterating Objects

```lua
-- Iterate all sequences
local sequences = DataPool().Sequences:Children()
for i = 1, #sequences do
    local seq = sequences[i]
    Printf("Sequence: " .. seq.name)
end
```

### Finding Objects

```lua
-- Find sequence by name
local sequences = DataPool().Sequences:Children()
for i = 1, #sequences do
    if sequences[i].name == "My Sequence" then
        return sequences[i]
    end
end
```

### Checking Object Existence

```lua
-- Check if timecode exists
local tc = DataPool().Timecodes[101]
if tc then
    Printf("Timecode 101 exists: " .. tc.name)
else
    Printf("Timecode 101 does not exist")
end
```

## Example: Hooking into Timecode

```lua
-- Extract plugin parameters
local luaComponentHandle = select(4, ...)

-- Store hook ID
local hook_id = nil

-- Callback function
local function on_timecode_change(obj)
    Printf("Timecode changed: " .. obj.name)
    Printf("Timecode number: " .. tostring(obj.no))
end

-- Start hook
function StartHook()
    local pluginHandle = luaComponentHandle:Parent()
    local timecode = DataPool().Timecodes[101][1]
    
    if not timecode then
        Echo("Error: Timecode 101[1] not found")
        return false
    end
    
    hook_id = HookObjectChange(on_timecode_change, timecode, pluginHandle)
    
    if hook_id then
        Echo("Hooked into Timecode 101[1] (Hook ID: " .. hook_id .. ")")
        return true
    else
        Echo("Error: Failed to create hook")
        return false
    end
end

-- Stop hook
function StopHook()
    if hook_id then
        Unhook(hook_id)
        hook_id = nil
        Echo("Hook stopped")
    end
end
```

## Notes for AI Agents

1. **Always check for nil**: Objects may not exist, always validate before use
2. **Use proper handles**: `HookObjectChange()` requires the plugin handle
3. **Clean up hooks**: Always store hook IDs and provide cleanup functions
4. **Property access**: Use dot notation for properties, colon for methods
5. **Index vs Number**: `index` is 0-based array index, `no` is the object number
6. **Children() method**: Returns array of child objects, use `#array` for length

## Related Documentation

- [MA3 Lua API Documentation](https://help.malighting.com/grandMA3/)
- [HookObjectChange Reference](https://help.malighting.com/grandMA3/2.0/HTML/lua_objectfree_hookobjectchange.html)
- EchoZero MA3 Integration: `docs/MA3_INTEGRATION.md`

