# Timecode Event Properties

This document captures discovered properties of timecode event objects.

## Accessing Events

### Path to Event

```lua
-- Full path to a live command event
local tc = DataPool().Timecodes[101]
local track_group = tc[1]
local ordered_children = track_group:Children()
local track = ordered_children[2]  -- ordered child 1 is MarkerTrack
local time_range = track:Children()[1]
local cmd_subtrack = time_range:Children()[1]
local event = cmd_subtrack:Children()[1]
```

### Using Helper Functions

```lua
-- Recommended access tuple:
-- tc_no, tg_no, ordered_track_child_idx, time_range_idx, subtrack_idx, event_idx
local event = GetEvent(101, 1, 2, 1, 1, 1)
ExploreEvent(101, 1, 2, 1, 1, 1)
```

## Event Properties

### Observed CmdEvent Properties

From the live terminal sample path
`Timecode -> TrackGroup -> Track -> TimeRange -> CmdSubTrack -> CmdEvent`:

- **Time Properties**:
  - `TIME` - event time in MA time format
  - `ABSTIME` - resolved absolute time
  - `RAWTIME` - raw internal MA time units

- **Command Properties**:
  - `TOKEN` - command token such as `Goto`
  - `CUEDESTINATION` - target cue
  - `EXECUTECOMMAND` - whether the command executes
  - `REALTIMECMD` - realtime command text

- **Identification**:
  - `NAME` - event label
  - `INDEX` - child index
  - `CLASS` - `CmdEvent`

## Exploration Functions

### GetEventProperties(event)

Returns a table of all discovered properties:

```lua
local event = GetEvent(101, 1, 1, 1)
local props = GetEventProperties(event)
for prop_name, value in pairs(props) do
    Printf(prop_name .. " = " .. tostring(value))
end
```

### PrintEventProperties(event, label)

Prints all properties of an event:

```lua
local event = GetEvent(101, 1, 1, 1)
PrintEventProperties(event, "My Event")
```

### ExploreEvent(tc_no, tg_no, child_idx, event_layer_idx, event_idx)

Gets event and prints all its properties (now requires event_layer_index):

```lua
ExploreEvent(101, 1, 1, 1, 1)  -- tc, tg, child, event_layer, event
```

### ExploreEventLayers(tc_no, tg_no, child_idx)

Explores all event layers in a layer:

```lua
ExploreEventLayers(101, 1, 1)
```

### ExploreEventsInLayer(tc_no, tg_no, child_idx, event_layer_idx)

Explores all events in a specific event layer:

```lua
ExploreEventsInLayer(101, 1, 1, 1)  -- tc, tg, child, event_layer
```

### ExploreEventLayer(tc_no, tg_no, child_idx, event_layer_idx)

Explores a specific event layer and its events:

```lua
ExploreEventLayer(101, 1, 1, 1)
```

### ExploreFirstEvents(tc_no, max_events)

Explores first N events in timecode:

```lua
ExploreFirstEvents(101, 5)  -- First 5 events
```

## Safe Property Access

### GetEventProperty(event, property_name)

Safely get a property value:

```lua
local event = GetEvent(101, 1, 1, 1)
local time = GetEventProperty(event, "time")
local command = GetEventProperty(event, "command")
```

### Helper Functions

```lua
-- Get event time (tries multiple property names)
local time = GetEventTime(event)

-- Get event command (tries multiple property names)
local cmd = GetEventCommand(event)
```

## Comparing Events

### CompareEvents(event1, event2)

Compare properties of two events:

```lua
local event1 = GetEvent(101, 1, 1, 1)
local event2 = GetEvent(101, 1, 1, 2)
CompareEvents(event1, event2)
```

## Discovery Workflow

1. **Load both plugins**:
   ```lua
   -- Load timecode_helper.lua
   -- Load timecode_explorer.lua
   ```

2. **Show full hierarchy** (RECOMMENDED FIRST):
   ```lua
   Lua "PrintHierarchy(101)"  -- Compact view
   Lua "PrintHierarchyWithProperties(101)"  -- With event properties
   ```

3. **Explore first events**:
   ```lua
   Lua "ExploreFirstEvents(101, 3)"
   ```

4. **Explore event layers**:
   ```lua
   Lua "ExploreEventLayers(101, 1, 1)"
   ```

5. **Explore events in event layer**:
   ```lua
   Lua "ExploreEventsInLayer(101, 1, 1, 1)"
   ```

6. **Explore specific event**:
   ```lua
   Lua "ExploreEvent(101, 1, 1, 1, 1)"  -- Now requires event_layer_index
   ```

7. **Compare events**:
   ```lua
   local e1 = GetEvent(101, 1, 1, 1, 1)  -- tc, tg, child, layer, event
   local e2 = GetEvent(101, 1, 1, 1, 2)
   CompareEvents(e1, e2)
   ```

6. **Document findings** in this file

## Discovered Properties

*Document discovered properties here as you explore*

### Example Event Structure

```lua
-- After exploration, document like this:
Event Properties:
  name = "Event 1"
  no = 1
  index = 1
  time = 0.0
  command = "Go Sequence 1"
  fade = 0.5
  delay = 0.0
```

## Notes

- Properties may vary by event type
- Some properties may be read-only
- Use `GetEventProperty()` for safe access
- Always check for nil before using properties

## Related Documentation

- `TIMECODE_STRUCTURE.md` - Timecode hierarchy
- `MA3_NESTED_ACCESS.md` - Access patterns
- `timecode_helper.lua` - Helper functions
- `timecode_explorer.lua` - Exploration functions
