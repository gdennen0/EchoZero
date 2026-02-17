# Timecode Structure Documentation

This document captures the discovered structure of grandMA3 timecodes through systematic exploration.

## Hierarchy

**CORRECTED STRUCTURE** (as of latest discovery):

```
DataPool().Timecodes
  └── [timecode_no] - Timecode Track
      └── [track_group_no] - Track Group
          └── :Children() - Array of children
              └── [0] - "Marker" (SKIP THIS)
              └── [1+] - Track
                  └── :Children() - TimeRanges
                      └── [time_range_index] - TimeRange
                          └── :Children() - SubTracks
                              └── [subtrack_index] - CmdSubTrack or FaderSubTrack
                                  └── :Children() - Actual Events
                                      └── [event_index] - CmdEvent or FaderEvent
```

**Important**: 
- Track Group's `Children()` array has "Marker" at index 0. Always skip index 0 and start from index 1.
- **Tracks contain TimeRanges**, not event layers directly
- **TimeRanges contain SubTracks** (CmdSubTrack for commands, FaderSubTrack for faders)
- **SubTracks contain the actual Events** (CmdEvent or FaderEvent)
- Use `subtrack:GetClass()` to determine if it's "CmdSubTrack" or "FaderSubTrack"

## Access Patterns

### Level 1: Timecode Pool

```lua
local timecodes = DataPool().Timecodes
local children = timecodes:Children()  -- All timecode tracks
```

### Level 2: Timecode Track

```lua
-- Direct access
local tc = DataPool().Timecodes[101]

-- Via Children()
local timecodes = DataPool().Timecodes:Children()
local tc = timecodes[1]  -- First timecode

-- Properties
tc.name   -- Timecode name
tc.no     -- Timecode number
tc.index  -- Array index
```

### Level 3: Track Group

```lua
-- Direct access
local track_group = DataPool().Timecodes[101][1]

-- Via Children()
local track_groups = tc:Children()
local track_group = track_groups[1]

-- Get track group children (array)
local children = track_group:Children()
-- children[0] is "Marker" - SKIP IT
-- children[1] is first actual child
```

**Critical**: Track Group's `Children()` has "Marker" at index 0. Always skip it!

### Level 4: Layer/Child (after skipping Marker)

```lua
-- Get track group children
local children = track_group:Children()
-- Skip index 0 (Marker)
local layer = children[1]  -- First actual layer

-- Or iterate (skipping Marker)
for i = 1, #children do
    local child = children[i]
    -- Skip if name is "Marker"
    if child.name ~= "Marker" then
        -- Process child
    end
end
```

### Level 5: Event Layer

```lua
-- Get layer's children (event layers, not events!)
local event_layers = layer:Children()
local event_layer = event_layers[1]  -- First event layer

-- Full path example
local tc = DataPool().Timecodes[101]
local track_group = tc[1]
local children = track_group:Children()
local layer = children[1]  -- Skip index 0 Marker
local event_layers = layer:Children()
local event_layer = event_layers[1]
```

### Level 6: Actual Event

```lua
-- Get event layer's children (actual events)
local events = event_layer:Children()
local event = events[1]  -- First actual event

-- Full path example
local tc = DataPool().Timecodes[101]
local track_group = tc[1]
local children = track_group:Children()
local layer = children[1]  -- Skip index 0 Marker
local event_layers = layer:Children()
local event_layer = event_layers[1]
local events = event_layer:Children()
local event = events[1]  -- Actual event!
```

## Exploration Results

### Discovered Structure

**Critical Finding**: Track Group's `Children()` array has "Marker" at index 0. Always skip index 0 when iterating!

```
Timecode > Track Group > :Children() > [0]="Marker" (SKIP) > [1+]=Track > :Children() > TimeRange > :Children() > CmdSubTrack/FaderSubTrack > :Children() > CmdEvent/FaderEvent
```

**Key Points**:
- Tracks contain **TimeRanges**, not event layers directly
- TimeRanges contain **SubTracks** (CmdSubTrack for commands, FaderSubTrack for faders)
- SubTracks contain the **actual Events** (CmdEvent or FaderEvent)
- Use `subtrack:GetClass()` to determine if it's "CmdSubTrack" or "FaderSubTrack"

### Test Results

Run `timecode_explorer.lua` and document findings here:

```lua
-- Test commands to run in MA3:
Lua "FullExploration()"
Lua "ExploreTimecodeByNumber(101)"
Lua "ExplorePath(101, 1, 1, 1)"
```

### Discovered Properties

#### Timecode Track Properties
- `name` - Timecode name
- `no` - Timecode number
- `index` - Array index
- `:Children()` - Returns array of track groups

#### Track Group Properties
- `name` - Track group name
- `no` - Track group number
- `:Children()` - Returns array where:
  - `[0]` = "Marker" (SKIP THIS!)
  - `[1+]` = Actual layers/children

#### Track Properties
- `name` - Track name
- `no` - Track number
- `:Children()` - Returns array of TimeRanges

#### TimeRange Properties
- `name` - TimeRange name
- `:Children()` - Returns array of SubTracks (CmdSubTrack and FaderSubTrack)

#### SubTrack Properties (CmdSubTrack/FaderSubTrack)
- `name` - SubTrack name
- `:GetClass()` - Returns "CmdSubTrack" or "FaderSubTrack"
- `:Children()` - Returns array of Events (CmdEvent or FaderEvent)

#### Event Properties (CmdEvent/FaderEvent)
- `name` - Event name
- `Time` - Event time (in seconds)
- `Type` - Event type ("cmd" or "fader")
- `CMD` - Command string (for CmdEvent)
- `Fade` - Fade time (if applicable)
- `Delay` - Delay time (if applicable)
- `Value` - Fader value (for FaderEvent)

## Common Operations

### Get All Events in Timecode

```lua
function GetAllEvents(tc_no)
    local events = {}
    local tc = DataPool().Timecodes[tc_no]
    if not tc then return events end
    
    local track_groups = tc:Children()
    for tg_idx = 1, #track_groups do
        local tg = track_groups[tg_idx]
        local children = tg:Children()
        
        -- Skip index 0 (Marker), start from 1
        for child_idx = 1, #children do
            local child = children[child_idx]
            -- Double-check it's not Marker
            if child.name ~= "Marker" then
                -- Get event layers
                local event_layers = child:Children()
                for event_layer_idx = 1, #event_layers do
                    local event_layer = event_layers[event_layer_idx]
                    -- Get actual events from event layer
                    local actual_events = event_layer:Children()
                    for event_idx = 1, #actual_events do
                        table.insert(events, actual_events[event_idx])
                    end
                end
            end
        end
    end
    
    return events
end
```

### Find Event by Time

```lua
function FindEventByTime(tc_no, time_value)
    -- Implementation depends on event properties
    -- To be discovered through exploration
end
```

## Testing Checklist

- [ ] Verify timecode pool access
- [ ] Verify timecode track access
- [ ] Verify track group access
- [ ] Verify layer access
- [ ] Verify event access
- [ ] Document all properties at each level
- [ ] Document Children() behavior at each level
- [ ] Test edge cases (missing objects, empty structures)
- [ ] Compare direct index vs Children() access

## Next Steps

1. Run `timecode_explorer.lua` plugin
2. Document all discovered properties
3. Test different timecode numbers
4. Explore event properties (time, command, etc.)
5. Create helper functions for common operations

