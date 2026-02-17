# MA3 Lua and Data Structures - Reference

## DataPool Child Pools

| Index | Name | Access |
|-------|------|--------|
| #0014 | Timecodes | DataPool().Timecodes |
| #0007 | Plugins | DataPool().Plugins |
| #0006 | Sequences | DataPool().Sequences |

## Object Properties

Most MA3 objects: `obj.name`, `obj.no`, `obj.index`, `obj:Children()`

## HookObjectChange

```
HookObjectChange(callback, targetObject, pluginHandle)
Returns: hook ID (integer)
Unhook(hookId)  -- cleanup
```

## Event Properties

CmdEvent/FaderEvent: `no`, `time`, `duration`, `cmd`, `name`, `cue`

## EZ.getTrack Internal Offset

```lua
-- User track 1 -> MA3 index 2
local track = EZ.getTrack(tcNo, tgNo, trackNo)
-- Uses: DataPool().Timecodes[tcNo][tgNo][trackNo + 1]
```

## EZ Module Loading Order

echozero_init.lua -> echozero.lua -> echozero_osc.lua -> timecode.lua -> Sequence.lua

OSC is late-bound - order independent for sendMessage.
