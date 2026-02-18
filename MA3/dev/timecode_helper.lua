local pluginName = select(1, ...)
local componentName = select(2, ...)
local signalTable = select(3, ...)
local luaComponentHandle = select(4, ...)

-- Get timecode track
function GetTimecode(tc_no)
    return DataPool().Timecodes[tc_no]
end

-- Show full hierarchy - ONE SIMPLE FUNCTION
function ShowHierarchy(tc_no)
    local tc = GetTimecode(tc_no)
    if not tc then
        Printf("Timecode " .. tc_no .. " not found")
        return
    end
    
    local tc_name = "Unknown"
    local success, name = pcall(function() return tc.name end)
    if success and name then
        tc_name = name
    end
    
    Printf("========================================")
    Printf("TIMECODE HIERARCHY: " .. tc_no .. " - " .. tc_name)
    Printf("========================================")
    Printf("")
    
    -- Get track groups
    local success, track_groups = pcall(function() return tc:Children() end)
    if not success or not track_groups then
        Printf("  No track groups found")
        return
    end
    
    Printf("Track Groups: " .. #track_groups)
    Printf("")
    
    for tg_idx = 1, #track_groups do
        local tg = track_groups[tg_idx]
        local tg_name = "Unknown"
        local success_name, name = pcall(function() return tg.name end)
        if success_name and name then
            tg_name = name
        end
        
        Printf("  Track Group " .. tg_idx .. ": " .. tg_name)
        
        -- Get track group children
        local success_children, children = pcall(function() return tg:Children() end)
        if success_children and children then
            Printf("    Children: " .. #children)
            Printf("      [0] Marker (skip)")
            Printf("")
            
            -- Iterate children (skip index 0)
            for child_idx = 1, #children do
                local child = children[child_idx]
                local child_name = "Unknown"
                local success_child_name, child_name_val = pcall(function() return child.name end)
                if success_child_name and child_name_val then
                    child_name = child_name_val
                end
                
                Printf("      [" .. child_idx .. "] " .. child_name)
                
                -- Get event layers
                local success_event_layers, event_layers = pcall(function() return child:Children() end)
                if success_event_layers and event_layers then
                    Printf("        Event Layers: " .. #event_layers)
                    
                    for event_layer_idx = 1, #event_layers do
                        local event_layer = event_layers[event_layer_idx]
                        local el_name = "Unknown"
                        local success_el_name, el_name_val = pcall(function() return event_layer.name end)
                        if success_el_name and el_name_val then
                            el_name = el_name_val
                        end
                        
                        Printf("          [" .. event_layer_idx .. "] " .. el_name)
                        
                        -- Get actual events
                        local success_events, events = pcall(function() return event_layer:Children() end)
                        if success_events and events then
                            Printf("            Events: " .. #events)
                            
                            -- Show first few events
                            for event_idx = 1, math.min(#events, 3) do
                                local event = events[event_idx]
                                local event_name = "Unknown"
                                local success_event_name, event_name_val = pcall(function() return event.name end)
                                if success_event_name and event_name_val then
                                    event_name = event_name_val
                                end
                                Printf("              [" .. event_idx .. "] " .. event_name)
                            end
                            
                            if #events > 3 then
                                Printf("              ... (" .. (#events - 3) .. " more events)")
                            end
                        else
                            Printf("            No events")
                        end
                    end
                else
                    Printf("        No event layers")
                end
                Printf("")
            end
        else
            Printf("    No children found")
        end
        Printf("")
    end
    
    Printf("========================================")
end

-- Explore cmd_subtrack access - check the actual CmdSubTrack events
function ExploreCmdSubtracks(tc_no)
    local tc = GetTimecode(tc_no)
    if not tc then
        Printf("Timecode " .. tc_no .. " not found")
        return
    end
    
    Printf("========================================")
    Printf("EXPLORING cmd_subtrack ACCESS")
    Printf("Note: CmdSubTrack appears to be an event name, checking events for cmd_subtrack property")
    Printf("========================================")
    Printf("")
    
    -- Get track groups
    local success, track_groups = pcall(function() return tc:Children() end)
    if not success or not track_groups then
        Printf("  No track groups found")
        return
    end
    
    for tg_idx = 1, math.min(#track_groups, 2) do
        local tg = track_groups[tg_idx]
        local success_children, children = pcall(function() return tg:Children() end)
        if success_children and children then
            -- Skip index 0 (Marker)
            for child_idx = 1, math.min(#children, 3) do
                local child = children[child_idx]
                local child_name = "Unknown"
                local success_name, name = pcall(function() return child.name end)
                if success_name and name then
                    child_name = name
                end
                
                Printf("--- Layer [" .. child_idx .. "]: " .. child_name .. " ---")
                
                -- Get event layers
                local success_event_layers, event_layers = pcall(function() return child:Children() end)
                if success_event_layers and event_layers and #event_layers > 0 then
                    Printf("  Event Layers: " .. #event_layers)
                    
                    for el_idx = 1, math.min(#event_layers, 2) do
                        local event_layer = event_layers[el_idx]
                        local el_name = "Unknown"
                        local success_el_name, el_name_val = pcall(function() return event_layer.name end)
                        if success_el_name and el_name_val then
                            el_name = el_name_val
                        end
                        
                        Printf("    Event Layer [" .. el_idx .. "]: " .. el_name)
                        
                        -- Get actual events (these are the CmdSubTrack objects)
                        local success_events, events = pcall(function() return event_layer:Children() end)
                        if success_events and events and #events > 0 then
                            Printf("      Events: " .. #events)
                            
                            for event_idx = 1, math.min(#events, 2) do
                                local event = events[event_idx]
                                local event_name = "Unknown"
                                local success_event_name, event_name_val = pcall(function() return event.name end)
                                if success_event_name and event_name_val then
                                    event_name = event_name_val
                                end
                                
                                Printf("        Event [" .. event_idx .. "]: " .. event_name)
                                
                                -- NOW check for cmd_subtrack on the actual event (CmdSubTrack object)
                                local success_cmd, cmd_subtrack = pcall(function() return event.cmd_subtrack end)
                                if success_cmd and cmd_subtrack then
                                    Printf("          SUCCESS: event.cmd_subtrack EXISTS!")
                                    Printf("          Type: " .. type(cmd_subtrack))
                                    Printf("          Value: " .. tostring(cmd_subtrack))
                                    
                                    -- Try to dump it
                                    local success_dump = pcall(function() cmd_subtrack:Dump() end)
                                    if success_dump then
                                        Printf("          Dump() called on cmd_subtrack")
                                    end
                                else
                                    Printf("          ERROR: event.cmd_subtrack = nil")
                                end
                                
                                -- Try as method
                                local success_method, method_result = pcall(function() return event:cmd_subtrack() end)
                                if success_method and method_result then
                                    Printf("          SUCCESS: event:cmd_subtrack() method exists!")
                                    Printf("          Result type: " .. type(method_result))
                                end
                                
                                -- Try CmdSubtrack (capitalized)
                                local success_cap, cmd_subtrack_cap = pcall(function() return event.CmdSubtrack end)
                                if success_cap and cmd_subtrack_cap then
                                    Printf("          SUCCESS: event.CmdSubtrack exists!")
                                end
                                
                                -- Try cmdSubtrack (camelCase)
                                local success_camel, cmd_subtrack_camel = pcall(function() return event.cmdSubtrack end)
                                if success_camel and cmd_subtrack_camel then
                                    Printf("          SUCCESS: event.cmdSubtrack exists!")
                                end
                                
                                Printf("")
                            end
                        else
                            Printf("      No events in this layer")
                        end
                    end
                else
                    Printf("  No event layers")
                end
                Printf("")
            end
        end
    end
    
    Printf("========================================")
end

-- Dump subtracks (CmdSubTrack events) from tracks (event layers)
function DumpSubtracks(tc_no)
    local tc = GetTimecode(tc_no)
    if not tc then
        Printf("Timecode " .. tc_no .. " not found")
        return
    end
    
    local tc_name = "Unknown"
    local success, name = pcall(function() return tc.name end)
    if success and name then
        tc_name = name
    end
    
    Printf("========================================")
    Printf("DUMPING SUBTRACKS: Timecode " .. tc_no .. " - " .. tc_name)
    Printf("Note: Event Layers are 'tracks', Events are 'CmdSubTrack' subtracks")
    Printf("========================================")
    Printf("")
    
    -- Get track groups
    local success, track_groups = pcall(function() return tc:Children() end)
    if not success or not track_groups then
        Printf("  No track groups found")
        return
    end
    
    for tg_idx = 1, #track_groups do
        local tg = track_groups[tg_idx]
        local tg_name = "Unknown"
        local success_name, name = pcall(function() return tg.name end)
        if success_name and name then
            tg_name = name
        end
        
        Printf("--- Track Group " .. tg_idx .. ": " .. tg_name .. " ---")
        Printf("")
        
        -- Get track group children (layers)
        local success_children, children = pcall(function() return tg:Children() end)
        if success_children and children then
            -- Skip index 0 (Marker)
            for child_idx = 1, #children do
                local child = children[child_idx]
                local child_name = "Unknown"
                local success_child_name, child_name_val = pcall(function() return child.name end)
                if success_child_name and child_name_val then
                    child_name = child_name_val
                end
                
                Printf("========================================")
                Printf("LAYER [" .. child_idx .. "]: " .. child_name)
                Printf("========================================")
                
                -- Get tracks (event layers)
                local success_tracks, tracks = pcall(function() return child:Children() end)
                if success_tracks and tracks and #tracks > 0 then
                    Printf("Tracks (Event Layers): " .. #tracks)
                    Printf("")
                    
                    for track_idx = 1, #tracks do
                        local track = tracks[track_idx]
                        local track_name = "Unknown"
                        local success_track_name, track_name_val = pcall(function() return track.name end)
                        if success_track_name and track_name_val then
                            track_name = track_name_val
                        end
                        
                        Printf("--- Track [" .. track_idx .. "]: " .. track_name .. " ---")
                        
                        -- Get subtracks (CmdSubTrack events)
                        local success_subtracks, subtracks = pcall(function() return track:Children() end)
                        if success_subtracks and subtracks and #subtracks > 0 then
                            Printf("  Subtracks (CmdSubTrack events): " .. #subtracks)
                            Printf("")
                            
                            for subtrack_idx = 1, #subtracks do
                                local subtrack = subtracks[subtrack_idx]
                                local subtrack_name = "Unknown"
                                local success_subtrack_name, subtrack_name_val = pcall(function() return subtrack.name end)
                                if success_subtrack_name and subtrack_name_val then
                                    subtrack_name = subtrack_name_val
                                end
                                
                                Printf("========================================")
                                Printf("SUBTRACK [" .. subtrack_idx .. "]: " .. subtrack_name)
                                Printf("========================================")
                                
                                -- Dump the subtrack (MA3 will print it automatically)
                                local success_dump = pcall(function() subtrack:Dump() end)
                                if not success_dump then
                                    Printf("Error: Dump() failed or not available")
                                end
                                
                                Printf("")
                                Printf("")
                            end
                        else
                            Printf("  No subtracks in this track")
                        end
                        Printf("")
                    end
                else
                    Printf("No tracks (event layers) found")
                end
                Printf("")
            end
        else
            Printf("  No children found")
        end
        Printf("")
    end
    
    Printf("========================================")
    Printf("Dump complete")
    Printf("========================================")
end

-- Dump each layer to understand structure
function DumpLayers(tc_no)
    local tc = GetTimecode(tc_no)
    if not tc then
        Printf("Timecode " .. tc_no .. " not found")
        return
    end
    
    local tc_name = "Unknown"
    local success, name = pcall(function() return tc.name end)
    if success and name then
        tc_name = name
    end
    
    Printf("========================================")
    Printf("DUMPING LAYERS: Timecode " .. tc_no .. " - " .. tc_name)
    Printf("========================================")
    Printf("")
    
    -- Get track groups
    local success, track_groups = pcall(function() return tc:Children() end)
    if not success or not track_groups then
        Printf("  No track groups found")
        return
    end
    
    for tg_idx = 1, #track_groups do
        local tg = track_groups[tg_idx]
        local tg_name = "Unknown"
        local success_name, name = pcall(function() return tg.name end)
        if success_name and name then
            tg_name = name
        end
        
        Printf("--- Track Group " .. tg_idx .. ": " .. tg_name .. " ---")
        Printf("")
        
        -- Get track group children
        local success_children, children = pcall(function() return tg:Children() end)
        if success_children and children then
            Printf("Children count: " .. #children)
            Printf("")
            
            -- Iterate children (skip index 0)
            for child_idx = 1, #children do
                local child = children[child_idx]
                local child_name = "Unknown"
                local success_child_name, child_name_val = pcall(function() return child.name end)
                if success_child_name and child_name_val then
                    child_name = child_name_val
                end
                
                Printf("========================================")
                Printf("LAYER [" .. child_idx .. "]: " .. child_name)
                Printf("========================================")
                
                -- Dump the layer (MA3 will print it automatically)
                local success_dump = pcall(function() child:Dump() end)
                if not success_dump then
                    Printf("Error: Dump() failed or not available")
                end
                
                Printf("")
                Printf("")
            end
        else
            Printf("  No children found")
        end
        Printf("")
    end
    
    Printf("========================================")
    Printf("Dump complete")
    Printf("========================================")
end

-- Recursive function to explore all events in a track (using direct indexing pattern)
local function ExploreEventsRecursive(dir, prefix, depth, maxDepth)
    local i = 1
    if maxDepth and depth > maxDepth then
        return
    end
    
    while dir[i] do
        local content = dir[i]
        local class = "Unknown"
        local success_class, class_val = pcall(function() return content:GetClass() end)
        if success_class and class_val then
            class = class_val
        end
        
        local name = "Unknown"
        local success_name, name_val = pcall(function() return content.name end)
        if success_name and name_val then
            name = name_val
        end
        
        local index = "?"
        local success_index, index_val = pcall(function() return content.index end)
        if success_index and index_val then
            index = tostring(index_val)
        end
        
        Printf(prefix .. '|---' .. index .. ': ' .. name .. ': ' .. class)
        
        -- If it's an event, show its properties
        if class == "FaderEvent" or class == "CmdEvent" or class == "CmdSubTrack" then
            -- Try to get TIME property
            local success_time, time_val = pcall(function() return content:Get("TIME") end)
            if success_time and time_val then
                Printf(prefix .. '|   TIME: ' .. tostring(time_val))
            end
            
            -- Try to get other common properties
            local props_to_check = {"CMD", "FADER", "VALUE", "SUBTrack"}
            for _, prop in ipairs(props_to_check) do
                local success_prop, prop_val = pcall(function() return content:Get(prop) end)
                if success_prop and prop_val then
                    Printf(prefix .. '|   ' .. prop .. ': ' .. tostring(prop_val))
                end
            end
        end
        
        -- Recurse into children
        ExploreEventsRecursive(content, prefix .. '|   ', depth + 1, maxDepth)
        i = i + 1
    end
end

-- Explore all events in a track using recursive direct indexing
-- Hierarchy: Timecode > Track Group > Track
function ExploreTrackEvents(tc_no, track_group_idx, track_idx)
    local tc = GetTimecode(tc_no)
    if not tc then
        Printf("Timecode " .. tc_no .. " not found")
        return
    end
    
    Printf("========================================")
    Printf("EXPLORING TRACK EVENTS (Direct Indexing Pattern)")
    Printf("Hierarchy: Timecode > Track Group > Track")
    Printf("========================================")
    Printf("")
    
    -- Navigate to the specific track
    local success, track_groups = pcall(function() return tc:Children() end)
    if not success or not track_groups then
        Printf("  No track groups found")
        return
    end
    
    if not track_groups[track_group_idx] then
        Printf("  Track Group " .. track_group_idx .. " not found")
        return
    end
    
    local tg = track_groups[track_group_idx]
    local tg_name = "Unknown"
    local success_tg_name, tg_name_val = pcall(function() return tg.name end)
    if success_tg_name and tg_name_val then
        tg_name = tg_name_val
    end
    Printf("Track Group: " .. track_group_idx .. " - " .. tg_name)
    
    local success_children, tracks = pcall(function() return tg:Children() end)
    if not success_children or not tracks then
        Printf("  No tracks found")
        return
    end
    
    -- Skip index 0 (Marker) if it exists
    -- Tracks start at index 1
    if not tracks[track_idx] then
        Printf("  Track " .. track_idx .. " not found (total tracks: " .. #tracks .. ")")
        Printf("  Note: Index 0 is Marker, tracks start at index 1")
        return
    end
    
    local track = tracks[track_idx]
    if not track then
        Printf("  Track is nil")
        return
    end
    
    local track_name = "Unknown"
    local success_track_name, track_name_val = pcall(function() return track.name end)
    if success_track_name and track_name_val then
        track_name = track_name_val
    end
    
    Printf("Track: " .. track_idx .. " - " .. track_name)
    Printf("Using recursive direct indexing pattern (dir[i])")
    Printf("")
    
    -- Use the recursive exploration pattern
    ExploreEventsRecursive(track, '', 1, 10)
    
    Printf("========================================")
end

-- List all events in a track with their classes and properties
-- Hierarchy: Timecode > Track Group > Track
function ListTrackEvents(tc_no, track_group_idx, track_idx)
    local tc = GetTimecode(tc_no)
    if not tc then
        Printf("Timecode " .. tc_no .. " not found")
        return
    end
    
    -- Get timecode name
    local tc_name = "Unknown"
    local success_tc_name, tc_name_val = pcall(function() return tc.name end)
    if success_tc_name and tc_name_val then
        tc_name = tc_name_val
    end
    
    Printf("========================================")
    Printf("LISTING TRACK EVENTS (Using :GetClass() and :Get())")
    Printf("Hierarchy: Timecode > Track Group > Track")
    Printf("========================================")
    Printf("")
    Printf("Timecode: " .. tc_no .. " - " .. tc_name)
    
    -- Navigate to the specific track
    local success, track_groups = pcall(function() return tc:Children() end)
    if not success or not track_groups then
        Printf("  No track groups found")
        return
    end
    
    if not track_groups[track_group_idx] then
        Printf("  Track Group " .. track_group_idx .. " not found")
        return
    end
    
    local tg = track_groups[track_group_idx]
    local tg_name = "Unknown"
    local success_tg_name, tg_name_val = pcall(function() return tg.name end)
    if success_tg_name and tg_name_val then
        tg_name = tg_name_val
    end
    Printf("Track Group: " .. track_group_idx .. " - " .. tg_name)
    
    local success_children, tracks = pcall(function() return tg:Children() end)
    if not success_children or not tracks then
        Printf("  No tracks found")
        return
    end
    
    -- Skip index 0 (Marker) if it exists
    -- Tracks start at index 1
    if not tracks[track_idx] then
        Printf("  Track " .. track_idx .. " not found (total tracks: " .. #tracks .. ")")
        Printf("  Note: Index 0 is Marker, tracks start at index 1")
        return
    end
    
    local track = tracks[track_idx]
    if not track then
        Printf("  Track is nil")
        return
    end
    
    local track_name = "Unknown"
    local success_track_name, track_name_val = pcall(function() return track.name end)
    if success_track_name and track_name_val then
        track_name = track_name_val
    end
    
    Printf("Track: " .. track_idx .. " - " .. track_name)
    Printf("")
    Printf("Hierarchy: Track > TimeRange > CmdSubTrack > Event")
    Printf("")
    
    -- Get TimeRanges from track
    local success_tr_children, tr_children = pcall(function() return track:Children() end)
    if not success_tr_children or not tr_children then
        Printf("  No TimeRanges found")
        return
    end
    
    for time_range_idx = 1, #tr_children do
        local time_range = tr_children[time_range_idx]
        local time_range_name = "Unknown"
        local success_tr_name, tr_name_val = pcall(function() return time_range.name end)
        if success_tr_name and tr_name_val then
            time_range_name = tr_name_val
        end
        
        local time_range_class = "Unknown"
        local success_tr_class, tr_class_val = pcall(function() return time_range:GetClass() end)
        if success_tr_class and tr_class_val then
            time_range_class = tr_class_val
        end
        
        Printf("========================================")
        Printf("TIMERANGE [" .. time_range_idx .. "]: " .. time_range_name .. " (" .. time_range_class .. ")")
        Printf("========================================")
        Printf("")
        
        -- Dump the TimeRange
        Printf("TIMERANGE DUMP:")
        Printf("========================================")
        local success_dump = pcall(function() time_range:Dump() end)
        if not success_dump then
            Printf("Error: Dump() failed or not available")
        end
        Printf("========================================")
        Printf("")
        
        -- Get CmdSubTrack objects from TimeRange
        Printf("Getting CmdSubTrack objects from TimeRange:")
        local success_tr_children, tr_children_inner = pcall(function() return time_range:Children() end)
        if success_tr_children and tr_children_inner and #tr_children_inner > 0 then
            Printf("  CmdSubTrack count: " .. #tr_children_inner)
            Printf("")
            
            -- Iterate through CmdSubTrack objects
            for cmd_subtrack_idx = 1, #tr_children_inner do
                local cmd_subtrack = tr_children_inner[cmd_subtrack_idx]
                local cmd_subtrack_name = "Unknown"
                local success_cst_name, cst_name_val = pcall(function() return cmd_subtrack.name end)
                if success_cst_name and cst_name_val then
                    cmd_subtrack_name = cst_name_val
                end
                
                local cmd_subtrack_class = "Unknown"
                local success_cst_class, cst_class_val = pcall(function() return cmd_subtrack:GetClass() end)
                if success_cst_class and cst_class_val then
                    cmd_subtrack_class = cst_class_val
                end
                
                Printf("--- CmdSubTrack [" .. cmd_subtrack_idx .. "]: " .. cmd_subtrack_name .. " (" .. cmd_subtrack_class .. ") ---")
                
                -- Get Events from CmdSubTrack
                local success_cst_children, cst_children = pcall(function() return cmd_subtrack:Children() end)
                if success_cst_children and cst_children and #cst_children > 0 then
                    Printf("  Events count: " .. #cst_children)
                    Printf("")
                    
                    -- List all events
                    for event_idx = 1, #cst_children do
                        local event = cst_children[event_idx]
                        local event_name = "Unknown"
                        local success_event_name, event_name_val = pcall(function() return event.name end)
                        if success_event_name and event_name_val then
                            event_name = event_name_val
                        end
                        
                        local event_class = "Unknown"
                        local success_event_class, event_class_val = pcall(function() return event:GetClass() end)
                        if success_event_class and event_class_val then
                            event_class = event_class_val
                        end
                        
                        Printf("  Event [" .. event_idx .. "]: " .. event_name .. " (" .. event_class .. ")")
                        
                        -- Try to get TIME property
                        local success_time, time_val = pcall(function() return event:Get("TIME") end)
                        if success_time and time_val then
                            Printf("    TIME: " .. tostring(time_val))
                        end
                        
                        -- Try common properties
                        local props_to_check = {"CMD", "FADER", "VALUE", "SUBTrack", "SUBTrackIndex", "CUEDESTINATION"}
                        for _, prop in ipairs(props_to_check) do
                            local success_prop, prop_val = pcall(function() return event:Get(prop) end)
                            if success_prop and prop_val then
                                Printf("    " .. prop .. ": " .. tostring(prop_val))
                            end
                        end
                        
                        -- Check for unique identifiers
                        Printf("    --- Unique Identifiers ---")
                        local unique_id_props = {"GUID", "UDID", "ID", "NO", "INDEX", "UNIQUEID"}
                        for _, prop in ipairs(unique_id_props) do
                            local success_prop, prop_val = pcall(function() return event:Get(prop) end)
                            if success_prop and prop_val then
                                Printf("    " .. prop .. ": " .. tostring(prop_val))
                            end
                        end
                        
                        -- Also try direct property access
                        local success_guid_direct, guid_direct = pcall(function() return event.GUID end)
                        if success_guid_direct and guid_direct then
                            Printf("    GUID (direct): " .. tostring(guid_direct))
                        end
                        
                        local success_no_direct, no_direct = pcall(function() return event.no end)
                        if success_no_direct and no_direct then
                            Printf("    no (direct): " .. tostring(no_direct))
                        end
                        
                        local success_index_direct, index_direct = pcall(function() return event.index end)
                        if success_index_direct and index_direct then
                            Printf("    index (direct): " .. tostring(index_direct))
                        end
                        
                        -- Dump the event to see all properties
                        Printf("    --- EVENT DUMP ---")
                        local success_event_dump = pcall(function() event:Dump() end)
                        if not success_event_dump then
                            Printf("    Error: Dump() failed or not available")
                        end
                        
                        Printf("")
                    end
                else
                    Printf("  No events found in this CmdSubTrack")
                end
                Printf("")
            end
        else
            Printf("  No CmdSubTrack objects found in TimeRange")
            Printf("")
            
            -- Try direct indexing on TimeRange
            Printf("Trying direct indexing on TimeRange (time_range[i]):")
            local i = 1
            while time_range[i] do
                local content = time_range[i]
                local content_name = "Unknown"
                local success_content_name, content_name_val = pcall(function() return content.name end)
                if success_content_name and content_name_val then
                    content_name = content_name_val
                end
                
                local content_class = "Unknown"
                local success_content_class, content_class_val = pcall(function() return content:GetClass() end)
                if success_content_class and content_class_val then
                    content_class = content_class_val
                end
                
                Printf("  [" .. i .. "] " .. content_name .. " (" .. content_class .. ")")
                i = i + 1
            end
            Printf("")
        end
    end
    
    Printf("========================================")
end

-- Set event time - test function to change event TIME property
-- Hierarchy: Timecode > Track Group > Track > TimeRange > CmdSubTrack > Event
-- new_time: time in internal units (16777216 = 1 second)
function SetEventTime(tc_no, track_group_idx, track_idx, time_range_idx, cmd_subtrack_idx, event_idx, new_time)
    local tc = GetTimecode(tc_no)
    if not tc then
        Printf("Timecode " .. tc_no .. " not found")
        return false
    end
    
    Printf("========================================")
    Printf("SETTING EVENT TIME")
    Printf("========================================")
    Printf("")
    
    -- Navigate to track
    local success, track_groups = pcall(function() return tc:Children() end)
    if not success or not track_groups or not track_groups[track_group_idx] then
        Printf("  Track Group " .. track_group_idx .. " not found")
        return false
    end
    
    local tg = track_groups[track_group_idx]
    local success_children, tracks = pcall(function() return tg:Children() end)
    if not success_children or not tracks or not tracks[track_idx] then
        Printf("  Track " .. track_idx .. " not found")
        return false
    end
    
    local track = tracks[track_idx]
    
    -- Get TimeRange
    local success_tr_children, tr_children = pcall(function() return track:Children() end)
    if not success_tr_children or not tr_children or not tr_children[time_range_idx] then
        Printf("  TimeRange " .. time_range_idx .. " not found")
        return false
    end
    
    local time_range = tr_children[time_range_idx]
    
    -- Get CmdSubTrack
    local success_cst_children, cst_children = pcall(function() return time_range:Children() end)
    if not success_cst_children or not cst_children or not cst_children[cmd_subtrack_idx] then
        Printf("  CmdSubTrack " .. cmd_subtrack_idx .. " not found")
        return false
    end
    
    local cmd_subtrack = cst_children[cmd_subtrack_idx]
    
    -- Get Event
    local success_event_children, event_children = pcall(function() return cmd_subtrack:Children() end)
    if not success_event_children or not event_children or not event_children[event_idx] then
        Printf("  Event " .. event_idx .. " not found")
        return false
    end
    
    local event = event_children[event_idx]
    
    -- Get current TIME
    local success_get_time, current_time = pcall(function() return event:Get("TIME") end)
    if success_get_time and current_time then
        Printf("Current TIME: " .. tostring(current_time))
    else
        Printf("Could not get current TIME")
    end
    
    Printf("Setting TIME to: " .. tostring(new_time))
    
    -- Try to set TIME
    local success_set = pcall(function() event:Set("TIME", new_time) end)
    if success_set then
        Printf("SUCCESS: :Set('TIME', " .. tostring(new_time) .. ") succeeded")
        
        -- Verify the change
        local success_get_new_time, new_time_val = pcall(function() return event:Get("TIME") end)
        if success_get_new_time and new_time_val then
            Printf("New TIME: " .. tostring(new_time_val))
            if new_time_val == new_time then
                Printf("SUCCESS: TIME successfully changed!")
            else
                Printf("WARNING: TIME value differs (expected: " .. tostring(new_time) .. ", got: " .. tostring(new_time_val) .. ")")
            end
        end
    else
        Printf("ERROR: :Set('TIME', ...) failed")
    end
    
    Printf("========================================")
    return success_set
end

-- List all events with their unique identifiers
-- Hierarchy: Timecode > Track Group > Track > TimeRange > CmdSubTrack > Event
-- Note: Uses :Get(propertyName) as primary method for property access
--       :Get(propertyName, Enums.Roles.Edit) can be used to get string representation
function ListEventsWithIDs(tc_no, track_group_idx, track_idx)
    local tc = GetTimecode(tc_no)
    if not tc then
        Printf("Timecode " .. tc_no .. " not found")
        return
    end
    
    -- Get timecode name
    local tc_name = "Unknown"
    local success_tc_name, tc_name_val = pcall(function() return tc.name end)
    if success_tc_name and tc_name_val then
        tc_name = tc_name_val
    end
    
    Printf("========================================")
    Printf("EVENTS WITH UNIQUE IDENTIFIERS")
    Printf("Timecode: " .. tc_no .. " - " .. tc_name)
    Printf("========================================")
    Printf("")
    
    -- Navigate to track
    local success, track_groups = pcall(function() return tc:Children() end)
    if not success or not track_groups or not track_groups[track_group_idx] then
        Printf("  Track Group " .. track_group_idx .. " not found")
        return
    end
    
    local tg = track_groups[track_group_idx]
    local tg_name = "Unknown"
    local success_tg_name, tg_name_val = pcall(function() return tg.name end)
    if success_tg_name and tg_name_val then
        tg_name = tg_name_val
    end
    Printf("Track Group: " .. track_group_idx .. " - " .. tg_name)
    
    local success_children, tracks = pcall(function() return tg:Children() end)
    if not success_children or not tracks or not tracks[track_idx] then
        Printf("  Track " .. track_idx .. " not found")
        return
    end
    
    local track = tracks[track_idx]
    local track_name = "Unknown"
    local success_track_name, track_name_val = pcall(function() return track.name end)
    if success_track_name and track_name_val then
        track_name = track_name_val
    end
    Printf("Track: " .. track_idx .. " - " .. track_name)
    Printf("")
    
    -- Get TimeRanges
    local success_tr_children, tr_children = pcall(function() return track:Children() end)
    if not success_tr_children or not tr_children then
        Printf("  No TimeRanges found")
        return
    end
    
    local total_events = 0
    
    for time_range_idx = 1, #tr_children do
        local time_range = tr_children[time_range_idx]
        local time_range_name = "Unknown"
        local success_tr_name, tr_name_val = pcall(function() return time_range.name end)
        if success_tr_name and tr_name_val then
            time_range_name = tr_name_val
        end
        
        Printf("--- TimeRange [" .. time_range_idx .. "]: " .. time_range_name .. " ---")
        
        -- Get CmdSubTracks
        local success_cst_children, cst_children = pcall(function() return time_range:Children() end)
        if success_cst_children and cst_children then
            for cmd_subtrack_idx = 1, #cst_children do
                local cmd_subtrack = cst_children[cmd_subtrack_idx]
                local cmd_subtrack_name = "Unknown"
                local success_cst_name, cst_name_val = pcall(function() return cmd_subtrack.name end)
                if success_cst_name and cst_name_val then
                    cmd_subtrack_name = cst_name_val
                end
                
                Printf("  CmdSubTrack [" .. cmd_subtrack_idx .. "]: " .. cmd_subtrack_name)
                
                -- Get Events
                local success_event_children, event_children = pcall(function() return cmd_subtrack:Children() end)
                if success_event_children and event_children then
                    for event_idx = 1, #event_children do
                        local event = event_children[event_idx]
                        total_events = total_events + 1
                        
                        local event_name = "Unknown"
                        local success_event_name, event_name_val = pcall(function() return event.name end)
                        if success_event_name and event_name_val then
                            event_name = event_name_val
                        end
                        
                        local event_class = "Unknown"
                        local success_event_class, event_class_val = pcall(function() return event:GetClass() end)
                        if success_event_class and event_class_val then
                            event_class = event_class_val
                        end
                        
                        Printf("    Event [" .. event_idx .. "]: " .. event_name .. " (" .. event_class .. ")")
                        
                        -- Get TIME using :Get()
                        local success_time, time_val = pcall(function() return event:Get("TIME") end)
                        if success_time and time_val then
                            Printf("      TIME: " .. tostring(time_val))
                        end
                        
                        -- Check for unique identifiers using :Get() as primary method
                        Printf("      Unique IDs:")
                        
                        -- GUID via :Get()
                        local success_guid, guid_val = pcall(function() return event:Get("GUID") end)
                        if success_guid and guid_val then
                            Printf("        GUID: " .. tostring(guid_val))
                        end
                        
                        -- NO via :Get()
                        local success_no, no_val = pcall(function() return event:Get("NO") end)
                        if success_no and no_val then
                            Printf("        NO: " .. tostring(no_val))
                        end
                        
                        -- INDEX via :Get()
                        local success_index, index_val = pcall(function() return event:Get("INDEX") end)
                        if success_index and index_val then
                            Printf("        INDEX: " .. tostring(index_val))
                        end
                        
                        -- ID via :Get()
                        local success_id, id_val = pcall(function() return event:Get("ID") end)
                        if success_id and id_val then
                            Printf("        ID: " .. tostring(id_val))
                        end
                        
                        -- UDID via :Get()
                        local success_udid, udid_val = pcall(function() return event:Get("UDID") end)
                        if success_udid and udid_val then
                            Printf("        UDID: " .. tostring(udid_val))
                        end
                        
                        Printf("")
                        
                        -- Try to get handle using FromAddr
                        Printf("        --- HANDLE VIA FromAddr ---")
                        
                        local handle_found = false
                        local handle_result = nil
                        local handle_method = ""
                        
                        -- Try to get address from the event object itself
                        local success_addr, addr_val = pcall(function() return event:Addr() end)
                        if success_addr and addr_val then
                            Printf("        Event :Addr(): " .. tostring(addr_val))
                            
                            -- Try to create handle from this address
                            local success_fromaddr, handle_fromaddr = pcall(function() return FromAddr(addr_val) end)
                            if success_fromaddr and handle_fromaddr then
                                handle_found = true
                                handle_result = handle_fromaddr
                                handle_method = "FromAddr(event:Addr())"
                                Printf("        SUCCESS: Got handle via " .. handle_method)
                                Printf("        handle: " .. tostring(handle_fromaddr))
                                
                                -- Try to get address back from handle
                                local success_handle_addr, handle_addr = pcall(function() return handle_fromaddr:Addr() end)
                                if success_handle_addr and handle_addr then
                                    Printf("        handle:Addr(): " .. tostring(handle_addr))
                                end
                                
                                -- Try to get native address
                                local success_handle_addr_native, handle_addr_native = pcall(function() return handle_fromaddr:AddrNative() end)
                                if success_handle_addr_native and handle_addr_native then
                                    Printf("        handle:AddrNative(): " .. tostring(handle_addr_native))
                                end
                            else
                                Printf("        FAILED: FromAddr() failed for address: " .. tostring(addr_val))
                            end
                        else
                            Printf("        FAILED: Event :Addr() not available")
                        end
                        
                        -- Try constructing address manually based on hierarchy
                        if not handle_found then
                            -- Format: Timecodes.{tc_no}.{track_group_idx}.{track_idx}.{time_range_idx}.{cmd_subtrack_idx}.{event_idx}
                            local manual_addr = "Timecodes." .. tc_no .. "." .. track_group_idx .. "." .. track_idx .. "." .. time_range_idx .. "." .. cmd_subtrack_idx .. "." .. event_idx
                            Printf("        Trying manual address: " .. manual_addr)
                            local success_manual_fromaddr, handle_manual = pcall(function() return FromAddr(manual_addr) end)
                            if success_manual_fromaddr and handle_manual then
                                handle_found = true
                                handle_result = handle_manual
                                handle_method = "FromAddr(manual address)"
                                Printf("        SUCCESS: Got handle via " .. handle_method)
                                Printf("        handle: " .. tostring(handle_manual))
                                
                                -- Verify it's the same event
                                local success_manual_addr, manual_addr_back = pcall(function() return handle_manual:Addr() end)
                                if success_manual_addr and manual_addr_back then
                                    Printf("        handle:Addr(): " .. tostring(manual_addr_back))
                                end
                            else
                                Printf("        FAILED: FromAddr() failed for manual address")
                            end
                        end
                        
                        -- Try with DataPool as base handle
                        if not handle_found then
                            local success_dp, datapool = pcall(function() return DataPool() end)
                            if success_dp and datapool then
                                local named_addr = "Timecodes." .. tc_name .. ".TrackGroup" .. track_group_idx .. ".Track" .. track_idx .. ".TimeRange" .. time_range_idx .. ".CmdSubTrack" .. cmd_subtrack_idx .. ".Event" .. event_idx
                                Printf("        Trying named address with DataPool base: " .. named_addr)
                                local success_named_fromaddr, handle_named = pcall(function() return FromAddr(named_addr, datapool) end)
                                if success_named_fromaddr and handle_named then
                                    handle_found = true
                                    handle_result = handle_named
                                    handle_method = "FromAddr(named address, DataPool)"
                                    Printf("        SUCCESS: Got handle via " .. handle_method)
                                    Printf("        handle: " .. tostring(handle_named))
                                else
                                    Printf("        FAILED: FromAddr() failed for named address")
                                end
                            end
                        end
                        
                        -- Final summary
                        if handle_found then
                            Printf("        ========================================")
                            Printf("        HANDLE OBTAINED: YES")
                            Printf("        Method: " .. handle_method)
                            Printf("        Handle type: " .. type(handle_result))
                            Printf("        Handle tostring: " .. tostring(handle_result))
                            
                            -- Try to get more information about the handle
                            local success_handle_addr, handle_addr = pcall(function() return handle_result:Addr() end)
                            if success_handle_addr and handle_addr then
                                Printf("        Handle:Addr(): " .. tostring(handle_addr))
                            end
                            
                            local success_handle_addr_native, handle_addr_native = pcall(function() return handle_result:AddrNative() end)
                            if success_handle_addr_native and handle_addr_native then
                                Printf("        Handle:AddrNative(): " .. tostring(handle_addr_native))
                            end
                            
                            -- Try to get class
                            local success_handle_class, handle_class = pcall(function() return handle_result:GetClass() end)
                            if success_handle_class and handle_class then
                                Printf("        Handle:GetClass(): " .. tostring(handle_class))
                            end
                            
                            -- Try to get name
                            local success_handle_name, handle_name = pcall(function() return handle_result:Get("name") end)
                            if success_handle_name and handle_name then
                                Printf("        Handle:Get('name'): " .. tostring(handle_name))
                            end
                            
                            -- Dump the handle to see all its properties
                            Printf("        --- HANDLE DUMP ---")
                            local success_handle_dump = pcall(function() handle_result:Dump() end)
                            if not success_handle_dump then
                                Printf("        Handle:Dump() not available")
                            end
                            Printf("        --- END HANDLE DUMP ---")
                            
                            Printf("        ========================================")
                        else
                            Printf("        ========================================")
                            Printf("        HANDLE OBTAINED: NO")
                            Printf("        All methods failed")
                            Printf("        ========================================")
                        end
                        
                        Printf("        --- END HANDLE ---")
                        Printf("")
                        
                        -- Dump the event to see all properties
                        Printf("        --- EVENT DUMP ---")
                        local success_dump = pcall(function() event:Dump() end)
                        if not success_dump then
                            Printf("        Error: Dump() failed or not available")
                        end
                        Printf("        --- END DUMP ---")
                        Printf("")
                    end
                end
            end
        end
    end
    
    Printf("========================================")
    Printf("Total events found: " .. total_events)
    Printf("========================================")
end

-- Explore event handles and persistent identifiers
-- Check if events have handles, UDID, GUID, or other persistent references
function ExploreEventHandles(tc_no, track_group_idx, track_idx, event_index)
    local tc = GetTimecode(tc_no)
    if not tc then
        Printf("Timecode " .. tc_no .. " not found")
        return
    end
    
    Printf("========================================")
    Printf("EXPLORING EVENT HANDLES AND IDENTIFIERS")
    Printf("========================================")
    Printf("")
    
    -- Navigate to track
    local success, track_groups = pcall(function() return tc:Children() end)
    if not success or not track_groups or not track_groups[track_group_idx] then
        Printf("  Track Group " .. track_group_idx .. " not found")
        return
    end
    
    local tg = track_groups[track_group_idx]
    local success_children, tracks = pcall(function() return tg:Children() end)
    if not success_children or not tracks or not tracks[track_idx] then
        Printf("  Track " .. track_idx .. " not found")
        return
    end
    
    local track = tracks[track_idx]
    
    -- Find the event
    local success_tr_children, tr_children = pcall(function() return track:Children() end)
    if not success_tr_children or not tr_children then
        Printf("  No TimeRanges found")
        return
    end
    
    local current_event_index = 0
    local target_event = nil
    local found_path = ""
    
    for time_range_idx = 1, #tr_children do
        local time_range = tr_children[time_range_idx]
        local success_cst_children, cst_children = pcall(function() return time_range:Children() end)
        if success_cst_children and cst_children then
            for cmd_subtrack_idx = 1, #cst_children do
                local cmd_subtrack = cst_children[cmd_subtrack_idx]
                local success_event_children, event_children = pcall(function() return cmd_subtrack:Children() end)
                if success_event_children and event_children then
                    for event_idx = 1, #event_children do
                        current_event_index = current_event_index + 1
                        if current_event_index == event_index then
                            target_event = event_children[event_idx]
                            local time_range_name = "Unknown"
                            local success_tr_name, tr_name_val = pcall(function() return time_range.name end)
                            if success_tr_name and tr_name_val then
                                time_range_name = tr_name_val
                            end
                            local cst_name = "Unknown"
                            local success_cst_name, cst_name_val = pcall(function() return cmd_subtrack.name end)
                            if success_cst_name and cst_name_val then
                                cst_name = cst_name_val
                            end
                            found_path = "TimeRange[" .. time_range_idx .. "]:" .. time_range_name .. " > CmdSubTrack[" .. cmd_subtrack_idx .. "]:" .. cst_name .. " > Event[" .. event_idx .. "]"
                            break
                        end
                    end
                    if target_event then break end
                end
                if target_event then break end
            end
        end
        if target_event then break end
    end
    
    if not target_event then
        Printf("  Event index " .. event_index .. " not found (total events: " .. current_event_index .. ")")
        return
    end
    
    -- Get event info
    local event_name = "Unknown"
    local success_event_name, event_name_val = pcall(function() return target_event.name end)
    if success_event_name and event_name_val then
        event_name = event_name_val
    end
    
    local event_class = "Unknown"
    local success_event_class, event_class_val = pcall(function() return target_event:GetClass() end)
    if success_event_class and event_class_val then
        event_class = event_class_val
    end
    
    Printf("Event: " .. event_name .. " (" .. event_class .. ")")
    Printf("Path: " .. found_path)
    Printf("")
    
    -- Check for GUID/UDID via :Get()
    Printf("=== Checking for GUID/UDID via :Get() ===")
    local guid_props = {"GUID", "UDID", "ID", "UNIQUEID", "HANDLE", "HANDLEID"}
    for _, prop in ipairs(guid_props) do
        local success_get, val = pcall(function() return target_event:Get(prop) end)
        if success_get and val then
            Printf("  :Get('" .. prop .. "'): " .. tostring(val) .. " (type: " .. type(val) .. ")")
        end
    end
    Printf("")
    
    -- Check for GUID/UDID via direct property access
    Printf("=== Checking for GUID/UDID via direct property access ===")
    local direct_props = {"GUID", "UDID", "ID", "guid", "udid", "id", "no", "NO", "index", "INDEX", "handle", "HANDLE"}
    for _, prop in ipairs(direct_props) do
        local success_direct, val = pcall(function() return target_event[prop] end)
        if success_direct and val then
            Printf("  ." .. prop .. ": " .. tostring(val) .. " (type: " .. type(val) .. ")")
        end
    end
    Printf("")
    
    -- Check if the event object itself can be used as a handle
    Printf("=== Testing event object as handle ===")
    Printf("Event object type: " .. type(target_event))
    Printf("Event object tostring: " .. tostring(target_event))
    
    -- Try to store the event and access it later
    Printf("")
    Printf("Testing if event can be stored and retrieved:")
    local stored_event = target_event
    local success_stored_name, stored_name = pcall(function() return stored_event.name end)
    if success_stored_name and stored_name then
        Printf("  SUCCESS: Event object can be stored and accessed")
        Printf("  Stored event name: " .. tostring(stored_name))
    else
        Printf("  ERROR: Event object cannot be stored/accessed")
    end
    Printf("")
    
    -- Check if there's a way to get a handle via path
    Printf("=== Testing handle creation methods ===")
    
    -- Check if event has a :Handle() method
    local success_handle_method, handle_result = pcall(function() return target_event:Handle() end)
    if success_handle_method and handle_result then
        Printf("  :Handle() method exists, returned: " .. tostring(handle_result))
    end
    
    -- Check if event has a :GetHandle() method
    local success_get_handle_method, get_handle_result = pcall(function() return target_event:GetHandle() end)
    if success_get_handle_method and get_handle_result then
        Printf("  :GetHandle() method exists, returned: " .. tostring(get_handle_result))
    end
    
    Printf("")
    
    -- Full dump to see all available properties and methods
    Printf("=== FULL EVENT DUMP ===")
    local success_dump = pcall(function() target_event:Dump() end)
    if not success_dump then
        Printf("  Dump() failed or not available")
    end
    Printf("")
    
    -- Try to see if we can access the event via its parent's children
    Printf("=== Testing persistent access via parent ===")
    Printf("If we store the parent (CmdSubTrack) and index, can we retrieve the event?")
    local parent_cmd_subtrack = nil
    local parent_event_index = nil
    
    -- Find the parent again
    for time_range_idx = 1, #tr_children do
        local time_range = tr_children[time_range_idx]
        local success_cst_children, cst_children = pcall(function() return time_range:Children() end)
        if success_cst_children and cst_children then
            for cmd_subtrack_idx = 1, #cst_children do
                local cmd_subtrack = cst_children[cmd_subtrack_idx]
                local success_event_children, event_children = pcall(function() return cmd_subtrack:Children() end)
                if success_event_children and event_children then
                    for event_idx = 1, #event_children do
                        if event_children[event_idx] == target_event then
                            parent_cmd_subtrack = cmd_subtrack
                            parent_event_index = event_idx
                            break
                        end
                    end
                    if parent_cmd_subtrack then break end
                end
                if parent_cmd_subtrack then break end
            end
        end
        if parent_cmd_subtrack then break end
    end
    
    if parent_cmd_subtrack and parent_event_index then
        Printf("  Parent CmdSubTrack found, event index: " .. parent_event_index)
        Printf("  Testing retrieval via parent:Children()[index]")
        local success_retrieve, retrieved_events = pcall(function() return parent_cmd_subtrack:Children() end)
        if success_retrieve and retrieved_events and retrieved_events[parent_event_index] then
            local retrieved_event = retrieved_events[parent_event_index]
            local success_retrieved_name, retrieved_name = pcall(function() return retrieved_event.name end)
            if success_retrieved_name and retrieved_name == event_name then
                Printf("  SUCCESS: Event can be retrieved via parent and index")
                Printf("  This suggests index-based access is persistent")
            else
                Printf("  ERROR: Event retrieval failed or name mismatch")
            end
        end
    end
    
    Printf("========================================")
end

-- Test hooking individual events
-- Check if we can use HookObjectChange() on events
-- TEMPORARILY DISABLED - Removed to fix compilation issues
function TestEventHook(tc_no, track_group_idx, track_idx, event_index)
    Printf("TestEventHook is temporarily disabled")
    Printf("This function will be re-enabled once compilation issues are resolved")
    return false
end

-- Move all events in a track forward by one second
function MoveTrackEventsForward(tc_no, track_group_idx, track_idx, seconds)
    seconds = seconds or 1
    local one_second_internally = 16777216
    local time_offset = seconds * one_second_internally
    
    local tc = GetTimecode(tc_no)
    if not tc then
        Printf("Timecode " .. tc_no .. " not found")
        return false
    end
    
    local success, track_groups = pcall(function() return tc:Children() end)
    if not success or not track_groups or not track_groups[track_group_idx] then
        Printf("Track Group " .. track_group_idx .. " not found")
        return false
    end
    
    local tg = track_groups[track_group_idx]
    local success_children, tracks = pcall(function() return tg:Children() end)
    if not success_children or not tracks or not tracks[track_idx] then
        Printf("Track " .. track_idx .. " not found")
        return false
    end
    
    local track = tracks[track_idx]
    
    local success_tr_children, tr_children = pcall(function() return track:Children() end)
    if not success_tr_children or not tr_children then
        Printf("No TimeRanges found")
        return false
    end
    
    local moved_events = 0
    
    for time_range_idx = 1, #tr_children do
        local time_range = tr_children[time_range_idx]
        local success_cst_children, cst_children = pcall(function() return time_range:Children() end)
        if success_cst_children and cst_children then
            for cmd_subtrack_idx = 1, #cst_children do
                local cmd_subtrack = cst_children[cmd_subtrack_idx]
                local success_event_children, event_children = pcall(function() return cmd_subtrack:Children() end)
                if success_event_children and event_children then
                    for event_idx = 1, #event_children do
                        local event = event_children[event_idx]
                        local success_get_time, current_time = pcall(function() return event:Get("TIME") end)
                        if success_get_time and current_time then
                            local new_time = current_time + time_offset
                            local success_set_time = pcall(function() event:Set("TIME", new_time) end)
                            if success_set_time then
                                moved_events = moved_events + 1
                            end
                        end
                    end
                end
            end
        end
    end
    
    Printf("Moved " .. moved_events .. " events")
    return moved_events > 0
end

-- Main function
return function()
    Echo('Timecode Helper loaded')
    Echo('Available functions:')
    Echo('  GetTimecode(tc_no) - Get timecode object')
    Echo('  ShowHierarchy(tc_no) - Show full hierarchy')
    Echo('  ExploreCmdSubtracks(tc_no) - Explore cmd_subtrack access')
    Echo('  DumpSubtracks(tc_no) - Dump all subtracks')
    Echo('  DumpLayers(tc_no) - Dump all layers')
    Echo('  ExploreTrackEvents(tc_no, track_group_idx, track_idx) - Explore events recursively')
    Echo('  ListTrackEvents(tc_no, track_group_idx, track_idx) - List all events with properties')
    Echo('  SetEventTime(tc_no, tg_idx, track_idx, tr_idx, cst_idx, event_idx, new_time) - Set event time')
    Echo('  ListEventsWithIDs(tc_no, track_group_idx, track_idx) - List events with unique IDs')
    Echo('  ExploreEventHandles(tc_no, tg_idx, track_idx, event_index) - Explore event handles')
    Echo('  TestEventHook(tc_no, tg_idx, track_idx, event_index) - Test event hooks (disabled)')
    Echo('')
    Echo('Example usage:')
    Echo('  ShowHierarchy(101)')
    Echo('  ListTrackEvents(101, 1, 2)')
    Echo('  ListEventsWithIDs(101, 1, 2)')
end
