EZ = EZ or {}

-- PUBLIC API: TRACK DIAGNOSTICS
-- Get detailed track info for debugging
function EZ.GetTrackInfo(tcNo, tgNo, trackNo)
    local ma3TrackNo = (trackNo or 0) + 1
    local track = DataPool().Timecodes[tcNo][tgNo][ma3TrackNo]
    
    if not track then
        EZ.log(string.format("GetTrackInfo: Track %d.%d.%d not found", tcNo, tgNo, trackNo))
        EZ.sendMessage("track", "info", {
            tc = tcNo, tg = tgNo, track = trackNo,
            exists = false
        })
        return nil
    end
    
    -- Get track properties
    local hasTarget = track.target ~= nil
    local targetName = hasTarget and tostring(track.target) or "NONE"
    
    -- Count children by class
    local children = track:Children() or {}
    local timeRangeCount = 0
    local childInfo = {}
    
    for i = 1, #children do
        local child = children[i]
        local cls = "unknown"
        if child and child.GetClass then
            local ok, c = pcall(function() return child:GetClass() end)
            if ok then 
                cls = c
                if cls == "TimeRange" then timeRangeCount = timeRangeCount + 1 end
            end
        end
        table.insert(childInfo, {idx = i, class = cls})
    end
    
    EZ.log(string.format("GetTrackInfo: Track %d.%d.%d - target=%s, children=%d, timeRanges=%d",
        tcNo, tgNo, trackNo, targetName, #children, timeRangeCount))
    
    EZ.sendMessage("track", "info", {
        tc = tcNo, tg = tgNo, track = trackNo,
        exists = true,
        hasTarget = hasTarget,
        targetName = targetName,
        childCount = #children,
        timeRangeCount = timeRangeCount,
        children = childInfo
    })
    
    return {
        hasTarget = hasTarget,
        targetName = targetName,
        childCount = #children,
        timeRangeCount = timeRangeCount
    }
end

-- =============================================================================
-- PUBLIC API: TIMECODE QUERIES
-- =============================================================================
function EZ.GetTimecodes()
    EZ.log("GetTimecodes() called")
    local dp = EZ.getDP()
    if not dp then
        EZ.log("ERROR: GetTimecodes() - DataPool not accessible")
        -- Send error response so EchoZero knows something went wrong
        EZ.sendMessage("timecodes", "error", {error = "DataPool not accessible"})
        return nil
    end
    if not dp.Timecodes then
        EZ.log("ERROR: GetTimecodes() - DataPool.Timecodes not found (no timecodes in show?)")
        EZ.sendMessage("timecodes", "error", {error = "No Timecodes in DataPool"})
        return nil
    end
    local result = {}
    local ok, children = pcall(function() return dp.Timecodes:Children() end)
    if ok and children then
        for _, tc in ipairs(children) do
            table.insert(result, {
                no = tc.no,
                name = tc.name or ""
            })
        end
        EZ.log(string.format("GetTimecodes() -> Found %d timecodes", #result))
    else
        EZ.log("ERROR: GetTimecodes() - Failed to get Timecodes children")
        EZ.sendMessage("timecodes", "error", {error = "Failed to enumerate timecodes"})
        return nil
    end
    
    EZ.sendMessage("timecodes", "list", {count = #result, timecodes = result})
    return result
end
-- EZ.DebugTimecode moved to echozero_debug.lua
function EZ.GetTrackGroups(tcNo)
    EZ.log(string.format("GetTrackGroups(%s) called", tostring(tcNo)))
    
    local tc = EZ.getTC(tcNo)
    if not tc then
        EZ.log(string.format("ERROR: TC %d not found", tcNo))
        EZ.sendMessage("trackgroups", "error", {tc = tcNo, error = "Timecode not found"})
        return nil
    end
    EZ.log(string.format("  TC %d found: '%s'", tcNo, tc.name or ""))
    
    local result = {}
    -- Track groups are direct children of timecode (use :Children())
    local ok, children = pcall(function() return tc:Children() end)
    if ok and children then
        EZ.log(string.format("  Found %d children (track groups)", #children))
        for i = 1, #children do
            local tg = children[i]
            if tg then
                -- Count tracks in this group (for info)
                local trackCount = 0
                local tgChildren = tg:Children()
                if tgChildren then
                    for j = 1, #tgChildren do
                        if tgChildren[j] and tgChildren[j].name ~= "Marker" then
                            trackCount = trackCount + 1
                        end
                    end
                end
                
                table.insert(result, {
                    no = i,  -- Use index as "no" since that's how we access it
                    name = tg.name or "",
                    track_count = trackCount
                })
                EZ.log(string.format("    TG[%d]: '%s' (%d tracks)", i, tg.name or "", trackCount))
            end
        end
    else
        EZ.log("  WARNING: tc:Children() failed or returned nil")
    end
    EZ.log(string.format("GetTrackGroups(%d) -> %d groups, sending to EchoZero...", tcNo, #result))
    local sendOk = EZ.sendMessage("trackgroups", "list", {tc = tcNo, count = #result, trackgroups = result})
    if sendOk then
        EZ.log("  Message sent successfully")
    else
        EZ.log("  ERROR: Failed to send message!")
    end
    return result
end
-- Get user-visible tracks (excludes system Marker track at index 1)
-- Returns tracks with 1-based indexing (1 = first user track, which is MA3 index 2)
function EZ.GetTracks(tcNo, tgNo)
    EZ.log(string.format("GetTracks(%d, %d) called", tcNo, tgNo))
    local tg = EZ.getTG(tcNo, tgNo)
    if not tg then
        EZ.log(string.format("ERROR: TG %d.%d not found", tcNo, tgNo))
        EZ.sendMessage("tracks", "error", {tc = tcNo, tg = tgNo, error = "Track group not found"})
        return nil
    end
    EZ.log(string.format("  TG %d.%d found: '%s'", tcNo, tgNo, tg.name or ""))
    local result = {}
    -- Tracks are children of track group (use :Children())
    -- NOTE: Index 1 is reserved for system 'Marker' track, user tracks start at index 2
    local ok, children = pcall(function() return tg:Children() end)
    if ok and children then
        local trackIdx = 0
        for i = 1, #children do
            local track = children[i]
            -- Skip "Marker" entries (always at index 1)
            if track and track.name ~= "Marker" then
                trackIdx = trackIdx + 1
                -- Count events in this track
                local eventCount = 0
                local timeRange = track:Children() and track:Children()[1]
                if timeRange and string.find(timeRange.name or "", "^TimeRange") then
                    local cmdSubTrack = timeRange:Children() and timeRange:Children()[1]
                    if cmdSubTrack and string.find(cmdSubTrack.name or "", "^CmdSubTrack") then
                        local events = cmdSubTrack:Children()
                        if events then
                            eventCount = #events
                        end
                    end
                end
                -- Extract sequence number from track.target
                local sequenceNo = nil
                if track.target then
                    local targetStr = tostring(track.target)
                    -- Parse "Sequence 123" format to extract number
                    local seqMatch = string.match(targetStr, "Sequence%s+(%d+)")
                    if seqMatch then
                        sequenceNo = tonumber(seqMatch)
                    end
                end
                table.insert(result, {
                    no = trackIdx,  -- User-visible track number (1-based, excluding Marker)
                    name = track.name or "",
                    event_count = eventCount,
                    sequence_no = sequenceNo,  -- Sequence number if assigned, nil otherwise
                    note = track.note or ""  -- EZ identity anchor (persists across reorder)
                })
                EZ.log(string.format("    Track[%d]: '%s' (MA3 index %d, events=%d, seq=%s, note='%s')", trackIdx, track.name or "", i, eventCount, tostring(sequenceNo or "none"), track.note or ""))
            end
        end
        EZ.log(string.format("  Found %d tracks (excluded Marker entries)", #result))
    else
        EZ.log("  WARNING: tg:Children() failed or returned nil")
    end
    EZ.log(string.format("GetTracks(%d,%d) -> %d tracks, sending to EchoZero...", tcNo, tgNo, #result))
    local sendOk = EZ.sendMessage("tracks", "list", {tc = tcNo, tg = tgNo, count = #result, tracks = result})
    if sendOk then
        EZ.log("  Message sent successfully")
    else
        EZ.log("  ERROR: Failed to send message!")
    end
    return result
end
-- Set the .note property on an MA3 track (used for EZ identity anchoring)
-- trackNo is user-visible track number (1-based, excluding Marker)
function EZ.SetTrackNote(tcNo, tgNo, trackNo, noteValue)
    EZ.log(string.format("SetTrackNote(%d, %d, %d, '%s') called", tcNo, tgNo, trackNo, tostring(noteValue)))
    
    local tg = EZ.getTG(tcNo, tgNo)
    if not tg then
        EZ.log(string.format("ERROR: TG %d.%d not found", tcNo, tgNo))
        return false
    end
    
    local ok, children = pcall(function() return tg:Children() end)
    if not ok or not children then
        EZ.log("ERROR: tg:Children() failed")
        return false
    end
    
    -- Find the track by user-visible index (skip Marker entries)
    local userIdx = 0
    for i = 1, #children do
        if children[i] and children[i].name ~= "Marker" then
            userIdx = userIdx + 1
            if userIdx == trackNo then
                children[i].note = noteValue or ""
                EZ.log(string.format("  Set .note='%s' on track '%s' (MA3 index %d)", tostring(noteValue), children[i].name or "", i))
                return true
            end
        end
    end
    
    EZ.log(string.format("ERROR: Track %d not found in TG %d.%d", trackNo, tcNo, tgNo))
    return false
end

-- Get events from a track
-- NOTE: trackNo is user-visible track number (1-based, excluding Marker).
-- Internally adds +1 offset to access MA3 track (trackNo 1 -> MA3 index 2, etc.)
local function compute_events_checksum(events)
    local sum = 0
    if not events then
        return sum
    end
    for i, evt in ipairs(events) do
        local time_val = tonumber(evt.time) or 0
        local idx_val = tonumber(evt.idx or i) or 0
        sum = sum + math.floor((time_val * 1000) + 0.5) + idx_val
    end
    return sum
end

function EZ.GetEvents(tcNo, tgNo, trackNo, request_id)
    EZ.log(string.format("GetEvents(%d, %d, %d) called", tcNo, tgNo, trackNo))
    EZ.sendMessage("events", "debug", {tc = tcNo, tg = tgNo, track = trackNo, request_id = request_id, msg = "get_events_enter"})
    if OSC and OSC.sendOSC then
        local dbg = string.format(
            "type=debug|change=get_events_enter|timestamp=%s|tc=%s|tg=%s|track=%s|request_id=%s",
            tostring(os.time()),
            tostring(tcNo),
            tostring(tgNo),
            tostring(trackNo),
            tostring(request_id or "")
        )
        OSC.sendOSC("/ez/message", "s", dbg)
    end
    
    -- Convert user-visible track number to MA3 index (add +1 to skip Marker at index 1)
    local ma3TrackNo = trackNo + 1
    local track = EZ.getTrack(tcNo, tgNo, ma3TrackNo)
    if not track then
        EZ.log(string.format("  ERROR: Track not found (MA3 index %d)", ma3TrackNo))
        EZ.sendMessage("events", "error", {tc = tcNo, tg = tgNo, track = trackNo, error = "Track not found"})
        return nil
    end
    
    EZ.log(string.format("  Track found: '%s'", track.name or "unnamed"))
    
    local events = EZ.getTrackEvents(track)
    
    -- Add coordinate info to each event
    for i, evt in ipairs(events) do
        evt.tc = tcNo
        evt.tg = tgNo
        evt.track = trackNo
        evt.idx = i
    end
    
    -- Display events in MA3 feedback/chat
    EZ.log(string.format("=== Events in TC%d.TG%d.TR%d ===", tcNo, tgNo, trackNo))
    EZ.log(string.format("  Track: '%s' (%d events)", track.name or "unnamed", #events))
    for i, evt in ipairs(events) do
        local timeStr = string.format("%.3f", evt.time or 0)
        local cmdStr = evt.cmd or ""
        local nameStr = evt.name or ""
        EZ.log(string.format("  [%d] time=%ss, cmd='%s', name='%s'", i, timeStr, cmdStr, nameStr))
    end
    EZ.log("=============================")
    
    local maxPer = EZ.config.maxChangeEvents or 40
    local checksum = compute_events_checksum(events)
    local total = #events
    local sent_ok = true
    local last_len = 0
    if maxPer and total > maxPer then
        local totalChunks = math.ceil(total / maxPer)
        for chunkIdx = 1, totalChunks do
            local startIdx = (chunkIdx - 1) * maxPer + 1
            local endIdx = math.min(startIdx + maxPer - 1, total)
            local chunk = {}
            for i = startIdx, endIdx do
                table.insert(chunk, events[i])
            end
            local ok = EZ.sendMessage("events", "list", {
                tc = tcNo,
                tg = tgNo,
                track = trackNo,
                count = total,
                offset = startIdx,
                chunk_index = chunkIdx,
                total_chunks = totalChunks,
                request_id = request_id,
                checksum = checksum,
                events = chunk
            })
            sent_ok = sent_ok and ok
            last_len = OSC and OSC._last_send_len or 0
        end
    else
        local ok = EZ.sendMessage("events", "list", {
            tc = tcNo, 
            tg = tgNo, 
            track = trackNo, 
            count = total, 
            request_id = request_id,
            checksum = checksum,
            events = events
        })
        sent_ok = ok
        last_len = OSC and OSC._last_send_len or 0
    end
    EZ.log(string.format("GetEvents: send events.list ok=%s len=%d total=%d", tostring(sent_ok), tonumber(last_len) or 0, total))
    return events
end

-- Get all events from all user-visible tracks in a timecode (excludes Marker track)
-- Returns tracks with events, using user-visible track numbers (1-based, excluding Marker)
function EZ.GetAllEvents(tcNo)
    EZ.log(string.format("GetAllEvents(%d) called", tcNo))
    
    local tc = EZ.getTC(tcNo)
    if not tc then
        EZ.log(string.format("ERROR: TC %d not found", tcNo))
        EZ.sendMessage("events", "error", {tc = tcNo, error = "Timecode not found"})
        return nil
    end
    
    local allEvents = {}
    
    -- Track groups are children of timecode
    local ok, tgChildren = pcall(function() return tc:Children() end)
    if ok and tgChildren then
        EZ.log(string.format("  Found %d track groups", #tgChildren))
        for tgIdx = 1, #tgChildren do
            local tg = tgChildren[tgIdx]
            if tg then
                -- Tracks are children of track group
                -- NOTE: Index 1 is reserved for Marker, user tracks start at index 2
                local trackOk, trackChildren = pcall(function() return tg:Children() end)
                if trackOk and trackChildren then
                    local trackIdx = 0  -- User-visible track number (1-based, excluding Marker)
                    for j = 1, #trackChildren do
                        local track = trackChildren[j]
                        -- Skip "Marker" entries (always at index 1)
                        if track and track.name ~= "Marker" then
                            trackIdx = trackIdx + 1
                            local events = EZ.getTrackEvents(track)
                            for k, evt in ipairs(events) do
                                evt.tc = tcNo
                                evt.tg = tgIdx
                                evt.track = trackIdx  -- User-visible track number
                                evt.idx = k
                            end
                            table.insert(allEvents, {
                                tc = tcNo,
                                tg = tgIdx,
                                track = trackIdx,  -- User-visible track number
                                track_name = track.name or "",
                                events = events
                            })
                        end
                    end
                end
            end
        end
    end
    
    EZ.log(string.format("GetAllEvents(%d) -> %d track entries", tcNo, #allEvents))
    for i, trackEntry in ipairs(allEvents) do
        EZ.log(string.format("Track %d (%s):", trackEntry.track, trackEntry.track_name or ""))
        if trackEntry.events and #trackEntry.events > 0 then
            for j, evt in ipairs(trackEntry.events) do
                EZ.log(string.format("  Event %d: time=%.3f, cmd=%s, idx=%d", j, tonumber(evt.time) or -1, tostring(evt.cmd), evt.idx or j))
            end
        else
            EZ.log("  (No events)")
        end
    end
    EZ.sendMessage("events", "all", {tc = tcNo, count = #allEvents, tracks = allEvents})
    return allEvents
end

-- =============================================================================
-- PUBLIC API: MANIPULATION
-- =============================================================================

-- Create a new track in a track group
-- NOTE: track name is user-provided; MA3 will assign a new track number
function EZ.CreateTrack(tcNo, tgNo, trackName)
    EZ.log(string.format("CreateTrack(%s, %s, %s) called", tostring(tcNo), tostring(tgNo), tostring(trackName)))
    local tg = EZ.getTG(tcNo, tgNo)
    if not tg then
        EZ.log(string.format("ERROR: TG %d.%d not found", tcNo, tgNo))
        EZ.sendMessage("track", "error", {tc = tcNo, tg = tgNo, error = "Track group not found"})
        return nil
    end

    local desired = tostring(trackName or ""):gsub('^%s+', ''):gsub('%s+$', '')
    if desired == "" then
        EZ.log("ERROR: Track name is empty")
        EZ.sendMessage("track", "error", {tc = tcNo, tg = tgNo, error = "Track name required"})
        return nil
    end

    -- Check if track already exists by name
    local ok_children, children = pcall(function() return tg:Children() end)
    if ok_children and children then
        local existing_idx = 0
        for i = 1, #children do
            local track = children[i]
            if track and track.name ~= "Marker" then
                existing_idx = existing_idx + 1
                if track.name and tostring(track.name):lower() == desired:lower() then
                    EZ.sendMessage("track", "exists", {tc = tcNo, tg = tgNo, track = existing_idx, name = track.name})
                    EZ.log(string.format("Track already exists: %s (track %d)", track.name, existing_idx))
                    return existing_idx
                end
            end
        end
    end
    -- Create track via Acquire
    local ok_create, track = pcall(function() return tg:Acquire() end)
    
    -- Check if the track was created successfully
    if not ok_create or not track then
        EZ.log("Failed to create track via Acquire")
        EZ.sendMessage("track", "error", {tc = tcNo, tg = tgNo, error = "Track acquire failed"})
        return false
    end

    local ok_name = pcall(function() track.name = desired end)
    if not ok_name then
        pcall(function() track:Set("name", desired) end)
    end

    EZ.sendMessage("track", "created", {tc = tcNo, tg = tgNo, name = track.name})
    EZ.log(string.format("Created track: %s", track.name or desired))
    return true
end

-- Add event to a track
function EZ.AddEvent(tcNo, tgNo, trackNo, time, cmd)
    local one_second_internally = 16777216
    local ma3TrackNo = (trackNo or 0) + 1
    
    -- Step 1: Get the track
    local track = DataPool().Timecodes[tcNo][tgNo][ma3TrackNo]
    if not track then
        EZ.log(string.format("FATAL: AddEvent - Track not found at %d.%d.%d (MA3 index %d)", tcNo, tgNo, trackNo, ma3TrackNo))
        EZ.sendMessage("event", "error", {tc = tcNo, tg = tgNo, track = trackNo, error = "Track not found"})
        return false
    end
    -- Step 2: Check if time range exists
    local time_range = EZ.GetFirstTimeRange(track)
    if not time_range then
        EZ.log(string.format("FATAL: AddEvent - No TimeRange found in track %d.%d.%d", tcNo, tgNo, trackNo))
        EZ.sendMessage("event", "error", {tc = tcNo, tg = tgNo, track = trackNo, error = "No TimeRange - CreateTrack may have failed"})
        return false
    end
    -- Step 3: Check if CmdSubTrack exists
    local cmd_subtrack = EZ.GetFirstCmdSubTrack(time_range)
    if not cmd_subtrack then
        EZ.log(string.format("FATAL: AddEvent - No CmdSubTrack found in TimeRange (track %d.%d.%d)", tcNo, tgNo, trackNo))
        EZ.sendMessage("event", "error", {tc = tcNo, tg = tgNo, track = trackNo, error = "No CmdSubTrack - Attempting to Aquire() CmdSubTrack"})
        return false
    end
    -- Step 4: Create the event
    local time_units = math.floor((tonumber(time) or 0) * one_second_internally)
    local event = cmd_subtrack:Acquire()
    event:Set("Time", time_units)
    event:Set("Cmd", cmd or "")
    return true
end

function EZ.AssignTrackSequence(tcNo, tgNo, trackNo, seqNo)
    -- Track numbers are user-visible (1-based), MA3 index includes Marker at 1
    local ma3TrackNo = (trackNo or 0) + 1
    local track = DataPool().Timecodes[tcNo][tgNo][ma3TrackNo]
    if not track then
        EZ.log(string.format("ERROR: Track not found"))
        return false
    end
    local sequence = DataPool().Sequences[seqNo]
    if not sequence then
        EZ.log(string.format("ERROR: Sequence not found"))
        return false
    end
    EZ.log("AssignTrackSequence: seq found: " .. sequence.name)
    track:Set('Target', sequence)
    if tostring(track.target) == "Sequence " .. tostring(sequence.no) then
        EZ.log(string.format("Assigned sequence %d to track %d", seqNo, trackNo))
        EZ.sendMessage("track", "assigned", {
            tc = tcNo, tg = tgNo, track = trackNo, seq = seqNo
        })
        return true
    end
    EZ.log(string.format("ERROR: Failed to assign track %d to sequence %d", trackNo, seqNo))
    EZ.sendMessage("track", "error", {
        tc = tcNo, tg = tgNo, track = trackNo, error = "Failed to assign track to sequence"
    })
    return false
end

function EZ.CreateTimeRange(tcNo, tgNo, trackNo)
    -- Creates a new TimeRange in the track
    local ma3TrackNo = trackNo + 1
    local track = DataPool().Timecodes[tcNo][tgNo][ma3TrackNo]
    local time_range = track:Acquire("TimeRange")
    if not time_range then
        EZ.log(string.format("FATAL: CreateTimeRange - Failed to Acquire TimeRange in track %d.%d.%d", tcNo, tgNo, trackNo))
        return false
    end
    return true
end

function EZ.DeleteTimeRange(tcNo, tgNo, trackNo, time_range_idx)
    local ma3TrackNo = trackNo + 1
    local track = DataPool().Timecodes[tcNo][tgNo][ma3TrackNo]
    track:Delete(time_range_idx)
    return true
end

function EZ.DeleteCmdSubTrack(tcNo, tgNo, trackNo, time_range_idx, cmd_subtrack_idx)
    local ma3TrackNo = trackNo + 1
    local track = DataPool().Timecodes[tcNo][tgNo][ma3TrackNo]
    local time_ranges = track:Children()
    local time_range = time_ranges[time_range_idx]
    local cmd_subtracks = time_range:Children()
    local cmd_subtrack = cmd_subtracks[cmd_subtrack_idx]
    if not cmd_subtrack then
        EZ.log(string.format("FATAL: DeleteCmdSubTrack - CmdSubTrack %d does not exist in TimeRange %d in track %d.%d.%d", cmd_subtrack_idx, time_range_idx, tcNo, tgNo, trackNo))
        return false
    end
    cmd_subtrack:Delete()
    return true
end

function EZ.CreateCmdSubTrack(tcNo, tgNo, trackNo, time_range_idx)
    -- Creates a new CmdSubTrack in the TimeRange using Acquire(), note it appears that this doesnt usually work without a sequence being assigned already
    local ma3TrackNo = trackNo + 1
    local track = DataPool().Timecodes[tcNo][tgNo][ma3TrackNo]
    local time_ranges = track:Children()
    local time_range = time_ranges[time_range_idx]
    local expectedChildClass = EZ.GetExpectedChildClass(time_range)
    EZ.log("Expected child class: " .. expectedChildClass)
    EZ.log("Trying CreateCmdSubTrack Method 1")
    time_range:Append('CmdSubTrack')
    if #time_range:Children() > 0 then
        EZ.log("CmdSubTrack created successfully")
        return true
    end 
    EZ.log("CmdSubTrack could not be created, ensure sequence is assigned")
    return false
end

function EZ.GetExpectedChildClass(parent)
    if not parent then return nil end
    local ok, childClass = pcall(function() return parent:GetChildClass() end)
    return ok and childClass or nil
end

function EZ.GetFirstTimeRange(track)
    -- Pass the track object itself
    local track_children = track:Children()
    for i = 1, #track_children do
        local child = track_children[i]
        if child and child.GetClass then
            local ok, cls = pcall(function() return child:GetClass() end)
            if ok and cls == "TimeRange" then
                return child
            end
        end
    end
    return false
end

function EZ.GetFirstCmdSubTrack(time_range)
    local time_range_children = time_range:Children()
    for i = 1, #time_range_children do
        local child = time_range_children[i]
        if child and child.GetClass then
            local ok, cls = pcall(function() return child:GetClass() end)
            if ok and cls == "CmdSubTrack" then
                return child
            end
        end
    end
    return false
end

-- Verify track is ready for events (has sequence and TimeRange/CmdSubTrack)
function EZ.VerifyTrackReady(tcNo, tgNo, trackNo)
    local ma3TrackNo = (trackNo or 0) + 1
    local track = DataPool().Timecodes[tcNo][tgNo][ma3TrackNo]
    
    if not track then
        EZ.log(string.format("VerifyTrackReady: FAIL - Track %d.%d.%d not found", tcNo, tgNo, trackNo))
        return false, "Track not found"
    end
    
    if not track.target then
        EZ.log(string.format("VerifyTrackReady: FAIL - Track %d.%d.%d has no sequence", tcNo, tgNo, trackNo))
        return false, "No sequence assigned"
    end
    
    -- Find TimeRange
    local children = track:Children() or {}
    local timeRange = nil
    for i = 1, #children do
        local child = children[i]
        if child and child.GetClass then
            local ok, cls = pcall(function() return child:GetClass() end)
            if ok and cls == "TimeRange" then
                timeRange = child
                break
            end
        end
    end
    
    if not timeRange then
        EZ.log(string.format("VerifyTrackReady: FAIL - Track %d.%d.%d has no TimeRange", tcNo, tgNo, trackNo))
        return false, "No TimeRange"
    end
    
    -- Find CmdSubTrack
    local tr_children = timeRange:Children() or {}
    local cmdSubTrack = nil
    for i = 1, #tr_children do
        local child = tr_children[i]
        if child and child.GetClass then
            local ok, cls = pcall(function() return child:GetClass() end)
            if ok and cls == "CmdSubTrack" then
                cmdSubTrack = child
                break
            end
        end
    end
    
    if not cmdSubTrack then
        EZ.log(string.format("VerifyTrackReady: FAIL - Track %d.%d.%d has no CmdSubTrack", tcNo, tgNo, trackNo))
        return false, "No CmdSubTrack"
    end
    
    EZ.log(string.format("VerifyTrackReady: OK - Track %d.%d.%d is ready for events", tcNo, tgNo, trackNo))
    return true, "Ready"
end

-- PUBLIC API: TRACK NAME CHECKING
-- Check if a track with a given name already exists

-- Check if a track with a specific name exists in a track group
function EZ.TrackNameExists(tcNo, tgNo, trackName)
    EZ.log(string.format("TrackNameExists(%d, %d, '%s') called", tcNo, tgNo, trackName))
    local tg = EZ.getTG(tcNo, tgNo)
    if not tg then
        EZ.log(string.format("  ERROR: Track group %d.%d not found", tcNo, tgNo))
        EZ.sendMessage("track", "name_check", {tc = tcNo, tg = tgNo, name = trackName, exists = false, error = "Track group not found"})
        return false, nil
    end
    
    local ok, children = pcall(function() return tg:Children() end)
    if ok and children then
        for i, track in ipairs(children) do
            if track and track.name and track.name ~= "Marker" then
                if track.name == trackName then
                    local userTrackNo = i - 1  -- Convert MA3 index to user-visible (1-based, excluding Marker)
                    EZ.log(string.format("  Found track '%s' at position %d", trackName, userTrackNo))
                    EZ.sendMessage("track", "name_check", {
                        tc = tcNo, tg = tgNo, name = trackName, 
                        exists = true, track_no = userTrackNo
                    })
                    return true, userTrackNo
                end
            end
        end
    end
    
    EZ.log(string.format("  Track '%s' not found in %d.%d", trackName, tcNo, tgNo))
    EZ.sendMessage("track", "name_check", {tc = tcNo, tg = tgNo, name = trackName, exists = false})
    return false, nil
end

-- Check if a track name exists in any track group of a timecode
function EZ.TrackNameExistsInTimecode(tcNo, trackName)
    EZ.log(string.format("TrackNameExistsInTimecode(%d, '%s') called", tcNo, trackName))
    local tc = EZ.getTC(tcNo)
    if not tc then
        EZ.log(string.format("  ERROR: Timecode %d not found", tcNo))
        EZ.sendMessage("track", "name_check_all", {tc = tcNo, name = trackName, exists = false, error = "Timecode not found"})
        return false, nil, nil
    end
    
    local ok, trackGroups = pcall(function() return tc:Children() end)
    if ok and trackGroups then
        for tgIdx, tg in ipairs(trackGroups) do
            local tgOk, tracks = pcall(function() return tg:Children() end)
            if tgOk and tracks then
                for trackIdx, track in ipairs(tracks) do
                    if track and track.name and track.name ~= "Marker" then
                        if track.name == trackName then
                            local userTrackNo = trackIdx - 1
                            EZ.log(string.format("  Found track '%s' at TC%d.TG%d.TR%d", trackName, tcNo, tgIdx, userTrackNo))
                            EZ.sendMessage("track", "name_check_all", {
                                tc = tcNo, name = trackName, 
                                exists = true, tg = tgIdx, track_no = userTrackNo
                            })
                            return true, tgIdx, userTrackNo
                        end
                    end
                end
            end
        end
    end
    
    EZ.log(string.format("  Track '%s' not found in timecode %d", trackName, tcNo))
    EZ.sendMessage("track", "name_check_all", {tc = tcNo, name = trackName, exists = false})
    return false, nil, nil
end

-- Update event in a track
-- NOTE: trackNo is user-visible track number (1-based, excluding Marker).
-- MA3 command automatically handles the +1 offset for Marker track.
function EZ.UpdateEvent(tcNo, tgNo, trackNo, eventIdx, time, cmd)
    -- Update event properties using CmdIndirect (silent) to keep MA3 console clean
    -- These are frequent sync operations that would flood the console with Cmd()
    local cmdStr
    if time then
        cmdStr = string.format(
            'Set Timecode %d.%d.%d.1.1.%d Property "Time" "%s"',
            tcNo, tgNo, trackNo, eventIdx, tostring(time)
        )
        EZ.RunCommand(cmdStr, "silent")
    end
    
    if cmd then
        cmdStr = string.format(
            'Set Timecode %d.%d.%d.1.1.%d Property "Cmd" "%s"',
            tcNo, tgNo, trackNo, eventIdx, cmd
        )
        EZ.RunCommand(cmdStr, "silent")
    end
    
    EZ.sendMessage("event", "updated", {
        tc = tcNo, tg = tgNo, track = trackNo, idx = eventIdx, time = time, cmd = cmd
    })
    EZ.log(string.format("Updated event %d in %d.%d.%d", eventIdx, tcNo, tgNo, trackNo))
    return true
end

-- Delete event from a track using subTrack:Delete(eventIdx)
-- NOTE: trackNo is user-visible track number (1-based, excluding Marker).
function EZ.DeleteEvent(tcNo, tgNo, trackNo, eventIdx)
    EZ.log(string.format("DeleteEvent(%d, %d, %d, %d) called", tcNo, tgNo, trackNo, eventIdx))
    
    -- Use getCmdSubTrack which handles the +1 offset for Marker track
    local subTrack = EZ.getCmdSubTrack(tcNo, tgNo, trackNo)
    if not subTrack then
        EZ.log(string.format("DeleteEvent: CmdSubTrack not found for %d.%d.%d", tcNo, tgNo, trackNo))
        return false
    end
    
    local events = subTrack:Children() or {}
    EZ.log(string.format("DeleteEvent: Found CmdSubTrack with %d events", #events))
    
    -- Check if event index is valid (1-based)
    if eventIdx < 1 or eventIdx > #events then
        EZ.log(string.format("DeleteEvent: Event index %d out of range (1-%d)", eventIdx, #events))
        return false
    end
    
    -- Use subTrack:Delete(eventIdx) method (confirmed working via TestMethod2)
    local ok, err = pcall(function() subTrack:Delete(eventIdx) end)
    if ok then
        EZ.log(string.format("Deleted event %d from %d.%d.%d via subTrack:Delete(%d)", eventIdx, tcNo, tgNo, trackNo, eventIdx))
        EZ.sendMessage("event", "deleted", {
            tc = tcNo, tg = tgNo, track = trackNo, idx = eventIdx
        })
        return true
    else
        EZ.log(string.format("subTrack:Delete(%d) failed: %s", eventIdx, tostring(err)))
        return false
    end
end

-- Clear all events from a track
-- Clear all events from a track (delete events one by one, preserve structure)
-- NOTE: trackNo is user-visible track number (1-based, excluding Marker).
-- IMPORTANT: Do NOT use 'Delete Timecode X.Y.Z.1.1' as that deletes the CmdSubTrack container!
function EZ.ClearTrack(tcNo, tgNo, trackNo)
    EZ.log(string.format("ClearTrack(%d, %d, %d) called", tcNo, tgNo, trackNo))

    -- Get the CmdSubTrack using our helper (handles +1 offset for Marker track)
    local subTrack = EZ.getCmdSubTrack(tcNo, tgNo, trackNo)
    if not subTrack then
        EZ.log(string.format("ClearTrack: CmdSubTrack not found for %d.%d.%d", tcNo, tgNo, trackNo))
        return false
    end

    local events = subTrack:Children() or {}
    local eventCount = #events
    EZ.log(string.format("ClearTrack: Found %d events to delete", eventCount))

    if eventCount == 0 then
        EZ.sendMessage("track", "cleared", {tc = tcNo, tg = tgNo, track = trackNo, count = 0})
        return true
    end

    -- Delete from highest index to lowest to avoid index shifting issues
    local deletedCount = 0
    for i = eventCount, 1, -1 do
        local ok, err = pcall(function() subTrack:Delete(i) end)
        if ok then
            deletedCount = deletedCount + 1
        else
            EZ.log(string.format("ClearTrack: Failed to delete event %d: %s", i, tostring(err)))
        end
    end
    EZ.log(string.format("ClearTrack: Deleted %d/%d events from track %d.%d.%d", deletedCount, eventCount, tcNo, tgNo, trackNo))
    EZ.sendMessage("track", "cleared", {tc = tcNo, tg = tgNo, track = trackNo, count = deletedCount})
    return deletedCount == eventCount
end