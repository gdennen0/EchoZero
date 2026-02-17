-- Timecode Structure Explorer
-- Systematically explores and documents the timecode hierarchy
-- Use this to understand: DataPool().Timecodes[timecode_no][track_no][layer_no][event_no]

-- Extract plugin parameters
local pluginName = select(1, ...)
local componentName = select(2, ...)
local signalTable = select(3, ...)
local luaComponentHandle = select(4, ...)

-- Exploration results storage
local exploration_results = {}

-- Helper function to safely get object info
local function get_object_info(obj, label)
    if not obj then
        return label .. ": nil"
    end
    
    local info = {}
    table.insert(info, label .. ":")
    
    -- Try common properties
    if obj.name then
        table.insert(info, "  name = " .. tostring(obj.name))
    end
    if obj.no then
        table.insert(info, "  no = " .. tostring(obj.no))
    end
    if obj.index then
        table.insert(info, "  index = " .. tostring(obj.index))
    end
    
    -- Try to get children count
    local children = nil
    local success, result = pcall(function()
        return obj:Children()
    end)
    
    if success and result then
        table.insert(info, "  children_count = " .. tostring(#result))
    end
    
    -- Try to get class/type
    local obj_type = type(obj)
    table.insert(info, "  type = " .. obj_type)
    
    return table.concat(info, "\n")
end

-- Function to dump all properties of an object
local function dump_object_properties(obj, label, depth)
    depth = depth or 0
    local indent = string.rep("  ", depth)
    
    if not obj then
        Printf(indent .. label .. ": nil")
        return
    end
    
    Printf(indent .. label .. ":")
    Printf(indent .. "  type: " .. type(obj))
    
    -- Try to get name
    local success, name = pcall(function() return obj.name end)
    if success and name then
        Printf(indent .. "  name: " .. tostring(name))
    end
    
    -- Try to get number
    success, num = pcall(function() return obj.no end)
    if success and num then
        Printf(indent .. "  no: " .. tostring(num))
    end
    
    -- Try to get index
    success, idx = pcall(function() return obj.index end)
    if success and idx then
        Printf(indent .. "  index: " .. tostring(idx))
    end
    
    -- Try to get children
    success, children = pcall(function() return obj:Children() end)
    if success and children then
        Printf(indent .. "  children_count: " .. tostring(#children))
        if depth < 5 then  -- Limit recursion
            for i = 1, math.min(#children, 3) do  -- Only show first 3
                dump_object_properties(children[i], "child[" .. i .. "]", depth + 1)
            end
            if #children > 3 then
                Printf(indent .. "  ... (" .. (#children - 3) .. " more children)")
            end
        end
    end
end

-- Explore timecode pool
function ExploreTimecodePool()
    Printf("=== Exploring Timecode Pool ===")
    
    local timecodes = DataPool().Timecodes
    if not timecodes then
        Printf("ERROR: Timecodes pool not found")
        return
    end
    
    Printf("Timecodes pool found")
    dump_object_properties(timecodes, "Timecodes Pool", 0)
    
    -- Try to get children
    local success, children = pcall(function() return timecodes:Children() end)
    if success and children then
        Printf("\nTimecode pool has " .. #children .. " timecode tracks")
        
        -- Explore first few timecodes
        for i = 1, math.min(#children, 3) do
            Printf("\n--- Timecode Track " .. i .. " ---")
            ExploreTimecodeTrack(children[i], i)
        end
    else
        Printf("Could not get timecode pool children")
    end
end

-- Explore a specific timecode track
function ExploreTimecodeTrack(tc, tc_index)
    if not tc then
        Printf("Timecode track is nil")
        return
    end
    
    dump_object_properties(tc, "Timecode Track", 1)
    
    -- Try to access by index
    local success, children = pcall(function() return tc:Children() end)
    if success and children then
        Printf("  Track has " .. #children .. " children")
        
        -- Explore first child (might be track group or layer)
        if #children > 0 then
            Printf("\n  --- First Child ---")
            ExploreTimecodeChild(children[1], 1, 1)
        end
    else
        Printf("  Could not get timecode track children")
    end
    
    -- Also try direct index access
    Printf("\n  Trying direct index access:")
    for i = 1, 5 do
        local success, child = pcall(function() return tc[i] end)
        if success and child then
            Printf("    tc[" .. i .. "] exists")
            dump_object_properties(child, "tc[" .. i .. "]", 2)
        else
            Printf("    tc[" .. i .. "] = nil")
        end
    end
end

-- Explore a timecode child (track group, layer, etc.)
function ExploreTimecodeChild(child, child_index, depth)
    depth = depth or 1
    local indent = string.rep("  ", depth)
    
    if not child then
        Printf(indent .. "Child is nil")
        return
    end
    
    dump_object_properties(child, "Child[" .. child_index .. "]", depth)
    
    -- Try to get its children
    local success, grandchildren = pcall(function() return child:Children() end)
    if success and grandchildren then
        Printf(indent .. "  Has " .. #grandchildren .. " children")
        
        -- IMPORTANT: Skip index 0 which is "Marker"
        Printf(indent .. "  Note: Index 0 is 'Marker' - skip it")
        
        -- Explore children starting from index 1
        for i = 1, math.min(#grandchildren, 5) do
            local child_name = "Unknown"
            local success_name, name = pcall(function() return grandchildren[i].name end)
            if success_name and name then
                child_name = name
            end
            Printf(indent .. "  child[" .. i .. "]: " .. child_name)
        end
        
        if #grandchildren > 1 and depth < 4 then
            Printf(indent .. "  --- First Non-Marker Child (index 1) ---")
            ExploreTimecodeChild(grandchildren[1], 1, depth + 1)
        end
    end
    
    -- Try direct index access (skip 0)
    for i = 1, 3 do
        local success, indexed = pcall(function() return child[i] end)
        if success and indexed then
            Printf(indent .. "  child[" .. i .. "] exists")
        end
    end
end

-- Explore specific timecode by number
function ExploreTimecodeByNumber(tc_no)
    Printf("=== Exploring Timecode " .. tc_no .. " ===")
    
    local tc = DataPool().Timecodes[tc_no]
    if not tc then
        Printf("Timecode " .. tc_no .. " not found")
        return
    end
    
    ExploreTimecodeTrack(tc, tc_no)
end

-- Comprehensive exploration of a specific path
function ExplorePath(tc_no, track_group_no, child_index, event_no)
    Printf("=== Exploring Path: Timecode[" .. tc_no .. "][" .. track_group_no .. "][child " .. child_index .. "][" .. event_no .. "] ===")
    Printf("Note: Track group children - index 0 is 'Marker', start from index 1")
    
    -- Level 1: Timecode pool
    local timecodes = DataPool().Timecodes
    Printf("Level 1: Timecodes pool exists: " .. tostring(timecodes ~= nil))
    
    -- Level 2: Specific timecode
    local tc = timecodes[tc_no]
    Printf("Level 2: Timecode[" .. tc_no .. "] exists: " .. tostring(tc ~= nil))
    if tc then
        dump_object_properties(tc, "Timecode[" .. tc_no .. "]", 0)
        
        -- Level 3: Track Group
        if track_group_no then
            local track_group = tc[track_group_no]
            Printf("Level 3: Timecode[" .. tc_no .. "][" .. track_group_no .. "] (Track Group) exists: " .. tostring(track_group ~= nil))
            if track_group then
                dump_object_properties(track_group, "TrackGroup[" .. track_group_no .. "]", 1)
                
                -- Get track group children
                local success, children = pcall(function() return track_group:Children() end)
                if success and children then
                    Printf("  Track Group has " .. #children .. " children")
                    Printf("  Index 0: 'Marker' (skip this)")
                    
                    -- Level 4: Track Group Child (skip index 0)
                    if child_index and child_index > 0 then
                        local child = children[child_index]
                        Printf("Level 4: TrackGroup[" .. track_group_no .. "].Children()[" .. child_index .. "] exists: " .. tostring(child ~= nil))
                        if child then
                            local child_name = "Unknown"
                            local success_name, name = pcall(function() return child.name end)
                            if success_name and name then
                                child_name = name
                            end
                            Printf("  Child name: " .. child_name)
                            dump_object_properties(child, "Child[" .. child_index .. "]", 2)
                            
                            -- Level 5: Event (if child has events)
                            if event_no then
                                local success_events, events = pcall(function() return child:Children() end)
                                if success_events and events and events[event_no] then
                                    local event = events[event_no]
                                    Printf("Level 5: Child[" .. child_index .. "].Children()[" .. event_no .. "] exists: " .. tostring(event ~= nil))
                                    if event then
                                        dump_object_properties(event, "Event[" .. event_no .. "]", 3)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

-- Test different access patterns
function TestAccessPatterns(tc_no)
    Printf("=== Testing Access Patterns for Timecode " .. tc_no .. " ===")
    
    local tc = DataPool().Timecodes[tc_no]
    if not tc then
        Printf("Timecode " .. tc_no .. " not found")
        return
    end
    
    Printf("\nPattern 1: Direct index")
    for i = 1, 5 do
        local success, obj = pcall(function() return tc[i] end)
        Printf("  tc[" .. i .. "]: " .. (success and obj and "exists" or "nil"))
    end
    
    Printf("\nPattern 2: Children() method")
    local success, children = pcall(function() return tc:Children() end)
    if success and children then
        Printf("  Children count: " .. #children)
        for i = 1, math.min(#children, 5) do
            Printf("  children[" .. i .. "]: " .. (children[i] and "exists" or "nil"))
        end
    end
    
    Printf("\nPattern 3: Compare direct index vs Children()")
    local direct = {}
    local via_children = {}
    
    for i = 1, 5 do
        local success, obj = pcall(function() return tc[i] end)
        direct[i] = success and obj ~= nil
    end
    
    success, children = pcall(function() return tc:Children() end)
    if success and children then
        for i = 1, math.min(#children, 5) do
            via_children[i] = children[i] ~= nil
        end
    end
    
    for i = 1, 5 do
        Printf("  Index " .. i .. ": direct=" .. tostring(direct[i]) .. ", children=" .. tostring(via_children[i]))
    end
end

-- Helper: Get event (if helper plugin not loaded) - now goes through event layers
local function GetEventHelper(tc_no, track_group_no, child_index, event_layer_index, event_index)
    local tc = DataPool().Timecodes[tc_no]
    if not tc then return nil end
    
    local track_group = tc[track_group_no]
    if not track_group then return nil end
    
    local success, children = pcall(function() return track_group:Children() end)
    if not success or not children or child_index < 1 or child_index > #children then
        return nil
    end
    
    local child = children[child_index]
    if not child then return nil end
    
    -- Get event layers (not events!)
    local success_event_layers, event_layers = pcall(function() return child:Children() end)
    if not success_event_layers or not event_layers or event_layer_index < 1 or event_layer_index > #event_layers then
        return nil
    end
    
    local event_layer = event_layers[event_layer_index]
    if not event_layer then return nil end
    
    -- Get actual events from event layer
    local success_events, events = pcall(function() return event_layer:Children() end)
    if not success_events or not events or event_index < 1 or event_index > #events then
        return nil
    end
    
    return events[event_index]
end

-- Helper: Get event layers (if helper plugin not loaded)
local function GetEventLayersHelper(tc_no, track_group_no, child_index)
    local tc = DataPool().Timecodes[tc_no]
    if not tc then return nil end
    
    local track_group = tc[track_group_no]
    if not track_group then return nil end
    
    local success, children = pcall(function() return track_group:Children() end)
    if not success or not children or child_index < 1 or child_index > #children then
        return nil
    end
    
    local child = children[child_index]
    if not child then return nil end
    
    -- These are event layers, not events!
    local success_event_layers, event_layers = pcall(function() return child:Children() end)
    if success_event_layers and event_layers then
        return event_layers
    end
    return nil
end

-- Helper: Get events in event layer (if helper plugin not loaded)
local function GetEventsInLayerHelper(tc_no, track_group_no, child_index, event_layer_index)
    local event_layers = GetEventLayersHelper(tc_no, track_group_no, child_index)
    if not event_layers or event_layer_index < 1 or event_layer_index > #event_layers then
        return nil
    end
    
    local event_layer = event_layers[event_layer_index]
    if not event_layer then return nil end
    
    local success_events, events = pcall(function() return event_layer:Children() end)
    if success_events and events then
        return events
    end
    return nil
end

-- Explore event properties in detail (now requires event_layer_index)
function ExploreEventProperties(tc_no, track_group_no, child_index, event_layer_index, event_index)
    Printf("=== Exploring Event Properties ===")
    Printf("Path: Timecode[" .. tc_no .. "][" .. track_group_no .. "][child " .. child_index .. "][layer " .. event_layer_index .. "][event " .. event_index .. "]")
    Printf("Note: Events are children of event layers!")
    
    -- Try to use helper function if available, otherwise use local helper
    local event = nil
    if GetEvent then
        event = GetEvent(tc_no, track_group_no, child_index, event_layer_index, event_index)
    else
        event = GetEventHelper(tc_no, track_group_no, child_index, event_layer_index, event_index)
    end
    
    if not event then
        Printf("Event not found")
        return
    end
    
    -- Use helper function if available
    if PrintEventProperties then
        PrintEventProperties(event, "Event")
    else
        -- Fallback: manual exploration
        dump_object_properties(event, "Event", 0)
        
        -- Try to get common event properties
        Printf("\nTrying common event properties:")
        local common_props = {"name", "no", "index", "time", "timecode", "command", "fade", "delay", "duration", "type", "value", "timestamp", "start", "end", "position", "data", "cmd"}
        for _, prop in ipairs(common_props) do
            local success, value = pcall(function() return event[prop] end)
            if success and value ~= nil then
                local value_str = tostring(value)
                if type(value) == "table" then
                    value_str = "table (" .. #value .. " items)"
                end
                Printf("  " .. prop .. " = " .. value_str)
            end
        end
        
        -- Try to iterate all properties
        Printf("\nTrying to enumerate all properties:")
        local success_iter, _ = pcall(function()
            local count = 0
            for k, v in pairs(event) do
                if type(k) == "string" and count < 20 then
                    count = count + 1
                    local value_str = tostring(v)
                    if type(v) == "table" then
                        value_str = "table"
                    end
                    Printf("  " .. k .. " = " .. value_str)
                end
            end
        end)
    end
end

-- Explore event layers in a specific layer
function ExploreEventLayers(tc_no, track_group_no, child_index)
    Printf("=== Exploring Event Layers in Layer ===")
    Printf("Path: Timecode[" .. tc_no .. "][" .. track_group_no .. "][child " .. child_index .. "]")
    Printf("Note: These are event layers, not events. Events are children of event layers!")
    
    -- Try to use helper function if available, otherwise use local helper
    local event_layers = nil
    if GetEventLayers then
        event_layers = GetEventLayers(tc_no, track_group_no, child_index)
    else
        event_layers = GetEventLayersHelper(tc_no, track_group_no, child_index)
    end
    
    if not event_layers then
        Printf("No event layers found or layer doesn't exist")
        return
    end
    
    Printf("Found " .. #event_layers .. " event layers")
    
    for i = 1, math.min(#event_layers, 5) do
        local event_layer = event_layers[i]
        local el_name = "Unknown"
        local success, name = pcall(function() return event_layer.name end)
        if success and name then
            el_name = name
        end
        
        Printf("\n--- Event Layer " .. i .. ": " .. el_name .. " ---")
        dump_object_properties(event_layer, "Event Layer", 0)
        
        -- Get events in this event layer
        local success_events, events = pcall(function() return event_layer:Children() end)
        if success_events and events then
            Printf("  Contains " .. #events .. " events:")
            for j = 1, math.min(#events, 3) do
                Printf("    Event[" .. j .. "]")
                ExploreEventProperties(tc_no, track_group_no, child_index, i, j)
            end
            if #events > 3 then
                Printf("    ... (" .. (#events - 3) .. " more events)")
            end
        end
    end
    
    if #event_layers > 5 then
        Printf("\n... (" .. (#event_layers - 5) .. " more event layers)")
    end
end

-- Explore events in a specific event layer
function ExploreEventsInLayer(tc_no, track_group_no, child_index, event_layer_index)
    Printf("=== Exploring Events in Event Layer ===")
    Printf("Path: Timecode[" .. tc_no .. "][" .. track_group_no .. "][child " .. child_index .. "][layer " .. event_layer_index .. "]")
    
    -- Try to use helper function if available, otherwise use local helper
    local events = nil
    if GetEventsInLayer then
        events = GetEventsInLayer(tc_no, track_group_no, child_index, event_layer_index)
    else
        events = GetEventsInLayerHelper(tc_no, track_group_no, child_index, event_layer_index)
    end
    
    if not events then
        Printf("No events found or event layer doesn't exist")
        return
    end
    
    Printf("Found " .. #events .. " events")
    
    for i = 1, math.min(#events, 5) do
        Printf("\n--- Event " .. i .. " ---")
        ExploreEventProperties(tc_no, track_group_no, child_index, event_layer_index, i)
    end
    
    if #events > 5 then
        Printf("\n... (" .. (#events - 5) .. " more events)")
    end
end

-- Main exploration function
function FullExploration()
    Printf("========================================")
    Printf("TIMECODE STRUCTURE EXPLORATION")
    Printf("========================================")
    
    -- Start with pool
    ExploreTimecodePool()
    
    Printf("\n\n========================================")
    Printf("Testing specific timecode (101)")
    Printf("========================================")
    ExploreTimecodeByNumber(101)
    
    Printf("\n\n========================================")
    Printf("Testing access patterns")
    Printf("========================================")
    TestAccessPatterns(101)
    
    Printf("\n\n========================================")
    Printf("Testing specific path")
    Printf("========================================")
    ExplorePath(101, 1, 1, 1)
    
    Printf("\n\n========================================")
    Printf("Exploring event layers")
    Printf("========================================")
    ExploreEventLayers(101, 1, 1)
    
    Printf("\n\n========================================")
    Printf("Exploring events in event layer")
    Printf("========================================")
    ExploreEventsInLayer(101, 1, 1, 1)
    
    Printf("\n\n========================================")
    Printf("Exploring event properties")
    Printf("========================================")
    ExploreEventProperties(101, 1, 1, 1, 1)
    
    Printf("\n\nExploration complete!")
end

-- Quick test function
function QuickTest()
    Printf("=== Quick Timecode Test ===")
    
    local tc = DataPool().Timecodes[101]
    if tc then
        Printf("Timecode 101 exists: " .. (tc.name or "unnamed"))
        
        -- Try first child
        local success, children = pcall(function() return tc:Children() end)
        if success and children and #children > 0 then
            Printf("First child exists")
            dump_object_properties(children[1], "First Child", 0)
        end
        
        -- Try direct index
        if tc[1] then
            Printf("tc[1] exists")
            dump_object_properties(tc[1], "tc[1]", 0)
        end
    else
        Printf("Timecode 101 not found")
    end
end

-- Main function
return function()
    Echo('Timecode Explorer Plugin loaded')
    Echo('Available functions:')
    Echo('  FullExploration() - Complete structure exploration')
    Echo('  QuickTest() - Quick test of timecode 101')
    Echo('  ExploreTimecodeByNumber(101) - Explore specific timecode')
    Echo('  ExplorePath(101, 1, 1, 1) - Explore specific path')
    Echo('  TestAccessPatterns(101) - Test different access methods')
    Echo('  ExploreEventLayers(101, 1, 1) - Explore event layers')
    Echo('  ExploreEventsInLayer(101, 1, 1, 1) - Explore events in event layer')
    Echo('  ExploreEventProperties(101, 1, 1, 1, 1) - Explore event properties')
    Echo('')
    Echo('Note: Events are children of event layers!')
    Echo('      Path: tc > tg > child > event_layer > event')
    Echo('')
    Echo('Note: Load timecode_helper.lua for additional helper functions')
    
    -- Uncomment to auto-run
    -- QuickTest()
end

