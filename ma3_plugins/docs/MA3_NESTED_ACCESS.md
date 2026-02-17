# grandMA3 Nested Object Access Patterns

This document explains how to access nested objects in grandMA3's hierarchical structure.

## Basic Pattern

```lua
DataPool().DataObject[child_key][grandchild_key][great_grandchild_key]
```

## Structure Hierarchy

### General Pattern

```
DataPool()
  └── DataObject (e.g., Timecodes, Sequences)
      └── [index] - Child object (e.g., Timecode 101)
          └── [index] - Grandchild (e.g., Track 1)
              └── [index] - Great-grandchild (e.g., Layer 1)
                  └── [index] - Event (e.g., Event 1)
```

## Access Methods

### Method 1: Direct Index Access

```lua
-- Access nested objects using array-style indexing
local timecode = DataPool().Timecodes[101]
local track = timecode[1]
local layer = track[1]
local event = layer[1]
```

### Method 2: Children() Method

```lua
-- Get all children as an array
local timecode = DataPool().Timecodes[101]
local children = timecode:Children()  -- Returns array of children
local first_track = children[1]

-- Iterate children
for i = 1, #children do
    Printf("Child " .. i .. ": " .. children[i].name)
end
```

### Method 3: Combined Approach

```lua
-- Use Children() to discover structure, then use direct index
local timecode = DataPool().Timecodes[101]
local children = timecode:Children()

if #children > 0 then
    -- First child exists
    local first_track = children[1]
    -- Or use direct access
    local first_track_direct = timecode[1]
end
```

## Timecode Structure

### Actual Hierarchy

```
DataPool().Timecodes
  └── [timecode_no] - Timecode Track (e.g., 101)
      └── [track_group_no] - Track Group (e.g., 1)
          └── :Children() - Array of children
              └── [0] - "Marker" (SKIP THIS!)
              └── [1] - First Track
              └── [2] - Second Track
              └── ...
                  └── :Children() - TimeRanges
                      └── [time_range_index] - TimeRange
                          └── :Children() - SubTracks
                              └── [subtrack_index] - CmdSubTrack or FaderSubTrack
                                  └── :Children() - Actual Events
                                      └── [event_index] - CmdEvent or FaderEvent
```

**Critical**: 
- Track Group's `Children()` array has "Marker" at index 0. Always skip index 0!
- **Tracks contain TimeRanges**, not event layers directly
- **TimeRanges contain SubTracks** (CmdSubTrack for commands, FaderSubTrack for faders)
- **SubTracks contain the actual Events** (CmdEvent or FaderEvent)
- Use `subtrack:GetClass()` to determine if it's "CmdSubTrack" or "FaderSubTrack"

### Access Examples

```lua
-- Level 1: Timecode Pool
local timecodes = DataPool().Timecodes

-- Level 2: Specific Timecode
local tc = DataPool().Timecodes[101]

-- Level 3: Track Group
local track_group = DataPool().Timecodes[101][1]

-- Level 4: Get Track Group Children (skip index 0 Marker)
local children = track_group:Children()
local track = children[1]  -- First actual track (skip [0] Marker)

-- Level 5: Get TimeRanges from Track
local time_ranges = track:Children()
local time_range = time_ranges[1]  -- First TimeRange

-- Level 6: Get SubTracks from TimeRange
local sub_tracks = time_range:Children()
local sub_track = sub_tracks[1]  -- First SubTrack
local sub_track_class = sub_track:GetClass()  -- "CmdSubTrack" or "FaderSubTrack"

-- Level 7: Get Actual Events from SubTrack
local events = sub_track:Children()
local event = events[1]  -- First actual event (CmdEvent or FaderEvent)
```

**Important Pattern**: 
- Always use `:Children()` for track groups and skip index 0!
- Track's children are **TimeRanges**, not events directly
- TimeRange's children are **SubTracks** (CmdSubTrack or FaderSubTrack)
- SubTrack's children are the **actual Events** (CmdEvent or FaderEvent)
- Use `subtrack:GetClass()` to check if it's "CmdSubTrack" or "FaderSubTrack"

### Using Children() to Explore

```lua
-- Explore timecode structure
local tc = DataPool().Timecodes[101]
local track_groups = tc:Children()  -- Get all track groups

for tg_idx = 1, #track_groups do
    Printf("Track Group " .. tg_idx .. ": " .. track_groups[tg_idx].name)
    local children = track_groups[tg_idx]:Children()  -- Get track group children
    
    -- CRITICAL: Skip index 0 (Marker)
    for track_idx = 1, #children do
        local track = children[track_idx]
        -- Double-check it's not Marker
        if track.name ~= "Marker" then
            Printf("  Track " .. track_idx .. ": " .. track.name)
            local time_ranges = track:Children()  -- Get TimeRanges
            
            for tr_idx = 1, #time_ranges do
                local time_range = time_ranges[tr_idx]
                Printf("    TimeRange " .. tr_idx .. ": " .. (time_range.name or "unnamed"))
                local sub_tracks = time_range:Children()  -- Get SubTracks
                
                for st_idx = 1, #sub_tracks do
                    local sub_track = sub_tracks[st_idx]
                    local sub_track_class = sub_track:GetClass()
                    Printf("      SubTrack " .. st_idx .. " (" .. sub_track_class .. "): " .. sub_track.name)
                    local events = sub_track:Children()  -- Get actual events
                    
                    for event_idx = 1, #events do
                        local event = events[event_idx]
                        Printf("        Event " .. event_idx .. ": " .. (event.name or "unnamed") .. " @ " .. (event.Time or 0))
                    end
                end
            end
        end
    end
end
```

## Safe Access Pattern

Always check for nil when accessing nested objects:

```lua
function GetTimecodeEvent(tc_no, track_no, layer_no, event_no)
    local tc = DataPool().Timecodes[tc_no]
    if not tc then
        return nil, "Timecode not found"
    end
    
    local track = tc[track_no]
    if not track then
        return nil, "Track not found"
    end
    
    local layer = track[layer_no]
    if not layer then
        return nil, "Layer not found"
    end
    
    local event = layer[event_no]
    if not event then
        return nil, "Event not found"
    end
    
    return event
end
```

## Index vs Number

Important distinction:

- **Array Index**: 1-based in Lua, used for `[index]` access
- **Object Number**: The actual object number (may differ from index)
- **Children()**: Returns array indexed from 1

```lua
local tc = DataPool().Timecodes[101]
local children = tc:Children()

-- children[1] is the first child (by index)
-- children[1].no is the object number (may be different)
Printf("First child index: 1, number: " .. children[1].no)
```

## Common Patterns

### Pattern 1: Iterate All Levels (Correct)

```lua
function IterateTimecodeStructure(tc_no)
    local tc = DataPool().Timecodes[tc_no]
    if not tc then return end
    
    local track_groups = tc:Children()
    for tg_idx = 1, #track_groups do
        local tg = track_groups[tg_idx]
        Printf("Track Group: " .. tg.name)
        
        local children = tg:Children()
        -- Skip index 0 (Marker)
        for child_idx = 1, #children do
            local child = children[child_idx]
            -- Double-check it's not Marker
            if child.name ~= "Marker" then
                Printf("  Layer: " .. child.name)
                
                -- Get event layers (not events!)
                local event_layers = child:Children()
                for event_layer_idx = 1, #event_layers do
                    local event_layer = event_layers[event_layer_idx]
                    Printf("    Event Layer: " .. event_layer.name)
                    
                    -- Get actual events
                    local events = event_layer:Children()
                    for event_idx = 1, #events do
                        local event = events[event_idx]
                        Printf("      Event: " .. event.name)
                    end
                end
            end
        end
    end
end
```

### Pattern 2: Find by Name

```lua
function FindTimecodeByName(name)
    local timecodes = DataPool().Timecodes:Children()
    for i = 1, #timecodes do
        if timecodes[i].name == name then
            return timecodes[i]
        end
    end
    return nil
end
```

### Pattern 3: Get Path to Object

```lua
function GetObjectPath(obj, parent_pool)
    local path = {}
    local current = obj
    
    while current do
        table.insert(path, 1, current.name or current.no)
        -- Try to get parent (implementation depends on MA3 API)
        -- This is a conceptual example
    end
    
    return table.concat(path, " > ")
end
```

## Testing and Exploration

Use the `timecode_explorer.lua` plugin to systematically explore structures:

```lua
-- In MA3 command line
Lua "FullExploration()"           -- Complete exploration
Lua "QuickTest()"                  -- Quick test
Lua "ExploreTimecodeByNumber(101)" -- Specific timecode
Lua "ExplorePath(101, 1, 1, 1)"   -- Specific path
Lua "TestAccessPatterns(101)"      -- Test access methods
```

## Notes

1. **Indexing is 1-based**: Lua arrays start at 1, not 0
2. **Children() returns array**: Use `#array` to get length
3. **Always check for nil**: Objects may not exist
4. **Use pcall for safety**: Wrap access in pcall to catch errors
5. **Direct index vs Children()**: Both work, but Children() is safer for iteration

## Related Documentation

- `MA3_DATA_STRUCTURES.md` - Complete data structure reference
- `MA3_OBJECT_REFERENCE.md` - Quick object reference
- `timecode_explorer.lua` - Exploration plugin

