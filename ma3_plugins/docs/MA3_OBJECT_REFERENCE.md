# grandMA3 Object Reference

Quick reference for common MA3 objects and their access patterns.

## Quick Access Patterns

### Sequences (Cue Lists)

```lua
DataPool().Sequences           -- All sequences
DataPool().Sequences[1]        -- Sequence number 1
DataPool().Sequences[1].name  -- Sequence name
DataPool().Sequences[1]:Children()  -- Cues in sequence
```

### Timecodes

```lua
DataPool().Timecodes           -- All timecode tracks
DataPool().Timecodes[101]      -- Timecode track 101
DataPool().Timecodes[101][1]  -- Cue 1 in timecode 101
DataPool().Timecodes[101]:Children()  -- All cues in timecode 101
```

### Groups

```lua
DataPool().Groups             -- All groups
DataPool().Groups[1]          -- Group number 1
DataPool().Groups[1].name     -- Group name
```

### Macros

```lua
DataPool().Macros             -- All macros
DataPool().Macros[1]          -- Macro number 1
DataPool().Macros[1].name     -- Macro name
```

### Presets

```lua
DataPool().PresetPools        -- Preset pools
DataPool().PresetPools[1]     -- Preset pool 1
```

### Pages

```lua
DataPool().Pages              -- All executor pages
DataPool().Pages[1]           -- Page number 1
DataPool().Pages[1]:Children()  -- Executors on page
```

## Object Properties

### Common Properties (Most Objects)

| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Object name |
| `no` | number | Object number |
| `index` | number | Array index (0-based) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `:Children()` | array | Get child objects |
| `:Parent()` | object | Get parent object |

## Hook Patterns

### Basic Hook Setup

```lua
-- 1. Get plugin handle
local luaComponentHandle = select(4, ...)
local pluginHandle = luaComponentHandle:Parent()

-- 2. Get target object
local target = DataPool().Sequences[1]

-- 3. Define callback
local function on_change(obj)
    Printf("Changed: " .. obj.name)
end

-- 4. Create hook
local hook_id = HookObjectChange(on_change, target, pluginHandle)

-- 5. Cleanup (when done)
Unhook(hook_id)
```

### Hookable Object Types

- ✅ Sequences
- ✅ Timecodes
- ✅ Timecode cues
- ✅ Groups
- ✅ Macros
- ✅ Presets
- ✅ Pages
- ✅ Most pool objects

## Common Patterns

### Check if Object Exists

```lua
local obj = DataPool().Sequences[999]
if obj then
    Printf("Exists: " .. obj.name)
else
    Printf("Does not exist")
end
```

### Iterate All Objects

```lua
local sequences = DataPool().Sequences:Children()
for i = 1, #sequences do
    Printf(sequences[i].name)
end
```

### Find Object by Name

```lua
local sequences = DataPool().Sequences:Children()
for i = 1, #sequences do
    if sequences[i].name == "My Sequence" then
        return sequences[i]
    end
end
```

## Integration with EchoZero

### Sending Changes to EchoZero

```lua
local function send_to_echozero(message)
    local socket = require("socket.core")
    local udp = socket.udp()
    udp:setpeername("127.0.0.1", 9000)
    udp:send(message .. "\r\n")
    udp:close()
end

local function on_sequence_change(obj)
    local msg = "type=sequence|name=" .. obj.name .. "|change=changed"
    send_to_echozero(msg)
end
```

### Formatting Messages

Use pipe-delimited format for EchoZero:

```
type=<object_type>|name=<object_name>|change=<change_type>|timestamp=<unix_time>|key=value
```

Example:
```
type=sequence|name=Song1|change=changed|timestamp=1234567890|no=1
```

