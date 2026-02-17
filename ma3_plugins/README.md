# MA3 Plugins

This directory contains grandMA3 Lua plugins for EchoZero integration.

## Quick Start for Developers

1. **API Reference**: `docs/MA3_LUA_API_REFERENCE.md` - Complete Lua API organized by purpose
2. **Learnings**: `docs/MA3_LEARNINGS.md` - Practical gotchas and patterns discovered
3. **Plugin Guide**: `docs/MA3_PLUGIN_DEVELOPMENT_GUIDE.md` - Full development guide

## Directory Structure

```
ma3_plugins/
  README.md           # This file
  docs/               # Documentation
  plugins/            # Production plugins
  dev/                # Development/exploration tools
  grandMA3_lua_functions.txt  # Raw MA3 Lua API export
```

## Production Plugin

### echozero_spine

The main EchoZero integration plugin for grandMA3.

- **Location**: `plugins/echozero_spine/`
- **Main File**: `echozero.lua` - Core plugin with all EZ.* functions
- **Debug File**: `echozero_debug.lua` - Diagnostic and testing utilities

**Key Functions:**
```lua
-- Connection
EZ.Ping()                           -- Test connection
EZ.SetTarget(ip, port)              -- Set EchoZero address

-- Queries
EZ.GetTimecodes()                   -- List all timecodes
EZ.GetTrackGroups(tcNo)             -- List track groups
EZ.GetTracks(tcNo, tgNo)            -- List tracks
EZ.GetEvents(tcNo, tgNo, trackNo)   -- Get track events

-- Manipulation
EZ.CreateTrack(tcNo, tgNo, name)    -- Create new track
EZ.AddEvent(tcNo, tgNo, trackNo, time, cmd)  -- Add event
EZ.DeleteEvent(tcNo, tgNo, trackNo, idx)     -- Delete event

-- Real-time Sync
EZ.HookTrack(tcNo, tgNo, trackNo)   -- Start monitoring
EZ.UnhookTrack(tcNo, tgNo, trackNo) -- Stop monitoring
EZ.UnhookAll()                      -- Remove all hooks
```

## Development Tools

Located in `dev/`:

| File | Purpose |
|------|---------|
| `test.lua` | Test plugin for hooking into MA3 objects |
| `timecode_explorer.lua` | Systematic exploration of timecode structure |
| `timecode_helper.lua` | Standalone helper functions for timecode work |
| `TrackDump.txt` | Example dump output from MA3 |

## Documentation

### Core References

| Document | Description |
|----------|-------------|
| `MA3_LUA_API_REFERENCE.md` | Complete API reference, organized by category |
| `MA3_LEARNINGS.md` | Practical gotchas and working patterns |
| `MA3_PLUGIN_DEVELOPMENT_GUIDE.md` | Full development walkthrough |

### Technical Details

| Document | Description |
|----------|-------------|
| `ARCHITECTURE.md` | EchoZero <-> MA3 OSC architecture |
| `MA3_DATA_STRUCTURES.md` | MA3 object model documentation |
| `TIMECODE_STRUCTURE.md` | Timecode-specific hierarchy |
| `EVENT_PROPERTIES.md` | Event object properties |
| `MA3_CONNECTION_FLOW.md` | Communication protocol |
| `DataPool/` | Raw MA3 structure dumps |

## Installation

1. Copy plugin to MA3:
   ```
   [MA3 Show]/datapools/plugins/echozero_spine/echozero.lua
   ```

2. Load in grandMA3: **Setup > Plugins**

3. Test connection:
   ```
   Lua "EZ.Ping()"
   ```

## For AI Agents

When working on MA3 integrations:

1. **API Reference**: Start with `docs/MA3_LUA_API_REFERENCE.md`
   - Functions organized by category
   - [USED] markers show what EchoZero actively uses
   
2. **Learnings**: Check `docs/MA3_LEARNINGS.md` before coding
   - 12 critical discoveries
   - Common gotchas checklist
   - Patterns that work

3. **Deep Dive**: For new features, read:
   - `docs/MA3_PLUGIN_DEVELOPMENT_GUIDE.md` - Complete walkthrough
   - `docs/TIMECODE_STRUCTURE.md` - Hierarchy details
   - `docs/DataPool/` - Raw structure dumps

4. **Examples**: Study `plugins/echozero_spine/echozero.lua`
   - 1600+ lines of working patterns
   - Hook system implementation
   - OSC communication

## Testing

Listen for MA3 messages:
```bash
python ma3_plugins/listen_for_ma3.py
```

Quick OSC test:
```bash
python ma3_plugins/quick_osc_test.py
```

Integration test:
```bash
python ma3_plugins/test_ma3_integration.py
```

## Key Learnings Summary

| Issue | Solution |
|-------|----------|
| Property access returns nil | Use `:Children()` for hierarchy |
| Track 1 not working | Index 1 is "Marker", user tracks start at 2 |
| Time values wrong | Divide by 16777216 for large values |
| Hooks stop working | Callbacks must be global functions |
| Delete() fails | Call on parent: `subTrack:Delete(index)` |
| Writes don't stick | Use `:Set("Prop", value)` not assignment |

See `docs/MA3_LEARNINGS.md` for complete details.
