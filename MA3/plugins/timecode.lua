EZ = EZ or {}

local function safeChildren(handle)
    if not handle then
        return {}
    end
    local ok, children = pcall(function() return handle:Children() end)
    if ok and children then
        return children
    end
    return {}
end

local function safeStringProperty(handle, propertyName)
    if not handle or not propertyName then
        return ""
    end
    local ok, value = pcall(function() return handle[propertyName] end)
    if ok and value ~= nil then
        return tostring(value)
    end
    return ""
end

local function safeSequenceNo(track)
    if not track then
        return nil
    end
    local okTarget, target = pcall(function() return track.target end)
    if not okTarget or not target then
        return nil
    end
    local okNo, seqNo = pcall(function() return target.no end)
    if not okNo then
        return nil
    end
    return tonumber(seqNo) or nil
end

local function countUserTracks(trackGroup)
    local count = 0
    local tracks = safeChildren(trackGroup)
    for i = 1, #tracks do
        local track = tracks[i]
        if track and safeStringProperty(track, "name") ~= "Marker" then
            count = count + 1
        end
    end
    return count
end

local function countTrackEventsForBrowse(track)
    local eventCount = 0
    local timeRanges = safeChildren(track)
    for trIdx = 1, #timeRanges do
        local timeRange = timeRanges[trIdx]
        local subTracks = safeChildren(timeRange)
        for stIdx = 1, #subTracks do
            local subTrack = subTracks[stIdx]
            local subTrackName = safeStringProperty(subTrack, "name")
            if string.find(subTrackName, "^CmdSubTrack") or string.find(subTrackName, "^FaderSubTrack") then
                eventCount = eventCount + #safeChildren(subTrack)
            end
        end
    end
    return eventCount
end

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
            local tcNo = tonumber(tc and tc.no)
            if tcNo and tcNo > 0 then
                table.insert(result, {
                    no = tcNo,
                    name = safeStringProperty(tc, "name")
                })
            end
        end
        table.sort(result, function(left, right)
            return (left.no or 0) < (right.no or 0)
        end)
        EZ.log(string.format("GetTimecodes() -> Found %d timecodes", #result))
    else
        EZ.log("ERROR: GetTimecodes() - Failed to get Timecodes children")
        EZ.sendMessage("timecodes", "error", {error = "Failed to enumerate timecodes"})
        return nil
    end
    
    EZ.sendMessage("timecodes", "list", {
        count = #result,
        plugin_version = EZ._version,
        timecodes = result
    })
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
                local trackCount = countUserTracks(tg)
                local tgName = safeStringProperty(tg, "name")
                table.insert(result, {
                    no = i,  -- Use index as "no" since that's how we access it
                    name = tgName,
                    track_count = trackCount
                })
                EZ.log(string.format("    TG[%d]: '%s' (%d tracks)", i, tgName, trackCount))
            end
        end
    else
        EZ.log("  WARNING: tc:Children() failed or returned nil")
    end
    EZ.log(string.format("GetTrackGroups(%d) -> %d groups, sending to EchoZero...", tcNo, #result))
    local sendOk = EZ.sendMessage("trackgroups", "list", {
        tc = tcNo,
        count = #result,
        plugin_version = EZ._version,
        trackgroups = result
    })
    if sendOk then
        EZ.log("  Message sent successfully")
    else
        EZ.log("  ERROR: Failed to send message!")
    end
    return result
end
-- Get user-visible tracks (excludes system Marker track at index 1)
-- Returns tracks with 1-based indexing (1 = first user track, which is MA3 index 2)
function EZ.GetTracks(tcNo, tgNo, request_id)
    EZ.log(string.format("GetTracks(%d, %d, %s) called", tcNo, tgNo, tostring(request_id)))
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
            local trackName = safeStringProperty(track, "name")
            -- Skip "Marker" entries (always at index 1)
            if track and trackName ~= "Marker" then
                trackIdx = trackIdx + 1
                local eventCount = countTrackEventsForBrowse(track)
                local sequenceNo = safeSequenceNo(track)
                local noteValue = safeStringProperty(track, "note")
                table.insert(result, {
                    no = trackIdx,  -- User-visible track number (1-based, excluding Marker)
                    name = trackName,
                    event_count = eventCount,
                    sequence_no = sequenceNo,  -- Sequence number if assigned, nil otherwise
                    note = noteValue  -- EZ identity anchor (persists across reorder)
                })
                EZ.log(string.format("    Track[%d]: '%s' (MA3 index %d, events=%d, seq=%s, note='%s')", trackIdx, trackName, i, eventCount, tostring(sequenceNo or "none"), noteValue))
            end
        end
        EZ.log(string.format("  Found %d tracks (excluded Marker entries)", #result))
    else
        EZ.log("  WARNING: tg:Children() failed or returned nil")
    end
    local total = #result
    local maxPer = 10
    EZ.log(string.format("GetTracks(%d,%d) -> %d tracks, sending to EchoZero...", tcNo, tgNo, total))
    if total > maxPer then
        local totalChunks = math.ceil(total / maxPer)
        for chunkIdx = 1, totalChunks do
            local startIdx = (chunkIdx - 1) * maxPer + 1
            local endIdx = math.min(startIdx + maxPer - 1, total)
            local chunk = {}
            for index = startIdx, endIdx do
                table.insert(chunk, result[index])
            end
            local payload = {
                tc = tcNo,
                tg = tgNo,
                count = total,
                plugin_version = EZ._version,
                offset = startIdx,
                chunk_index = chunkIdx,
                total_chunks = totalChunks,
                tracks = chunk
            }
            if request_id ~= nil then
                payload.request_id = request_id
            end
            local sendChunkOk = EZ.sendMessage("tracks", "list", payload)
            if sendChunkOk then
                EZ.log(string.format("  Chunk %d/%d sent successfully", chunkIdx, totalChunks))
            else
                EZ.log(string.format("  ERROR: Failed to send chunk %d/%d", chunkIdx, totalChunks))
            end
        end
        return result
    end

    local payload = {
        tc = tcNo,
        tg = tgNo,
        count = total,
        plugin_version = EZ._version,
        tracks = result
    }
    if request_id ~= nil then
        payload.request_id = request_id
    end
    local sendOk = EZ.sendMessage("tracks", "list", payload)
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

-- Create a new timecode pool in the next available slot
function EZ.CreateTimecode(timecodeName)
    EZ.log(string.format("CreateTimecode(%s) called", tostring(timecodeName)))

    local dp = EZ.getDP()
    if not dp or not dp.Timecodes then
        EZ.sendMessage("timecode", "error", {error = "Timecode pool unavailable"})
        return nil
    end

    local desired = tostring(timecodeName or ""):gsub("^%s+", ""):gsub("%s+$", "")
    local children = {}
    local ok_children, resolved_children = pcall(function() return dp.Timecodes:Children() end)
    if ok_children and resolved_children then
        children = resolved_children
    end

    local used = {}
    for _, tc in ipairs(children) do
        local no = tonumber(tc and tc.no)
        if no and no > 0 then
            used[no] = true
            if desired ~= "" and tc.name and tostring(tc.name):lower() == desired:lower() then
                EZ.sendMessage("timecode", "exists", {no = no, name = tc.name})
                return no
            end
        end
    end

    local next_no = 1
    while used[next_no] do
        next_no = next_no + 1
    end

    local created = nil
    local ok_create = pcall(function() created = dp.Timecodes:Acquire() end)
    if not ok_create or not created then
        EZ.sendMessage("timecode", "error", {error = "Timecode create failed"})
        return nil
    end

    local resolved_no = tonumber(created.no) or next_no
    local resolved_name = desired ~= "" and desired or string.format("TC %d", resolved_no)
    local ok_name = pcall(function() created.name = resolved_name end)
    if not ok_name then
        pcall(function() created:Set("name", resolved_name) end)
    end

    EZ.sendMessage("timecode", "created", {no = resolved_no, name = created.name or resolved_name})
    return resolved_no
end

-- Create a new track group in a timecode pool
function EZ.CreateTrackGroup(tcNo, trackGroupName)
    EZ.log(string.format("CreateTrackGroup(%s, %s) called", tostring(tcNo), tostring(trackGroupName)))
    local tc = EZ.getTC(tcNo)
    if not tc then
        EZ.sendMessage("trackgroup", "error", {tc = tcNo, error = "Timecode not found"})
        return nil
    end

    local desired = tostring(trackGroupName or ""):gsub("^%s+", ""):gsub("%s+$", "")
    local ok_children, children = pcall(function() return tc:Children() end)
    if not ok_children or not children then
        EZ.sendMessage("trackgroup", "error", {tc = tcNo, error = "Track group list unavailable"})
        return nil
    end

    local used = {}
    for tg_idx = 1, #children do
        local tg = children[tg_idx]
        used[tg_idx] = true
        if desired ~= "" and tg and tg.name and tostring(tg.name):lower() == desired:lower() then
            EZ.sendMessage("trackgroup", "exists", {tc = tcNo, tg = tg_idx, name = tg.name})
            return tg_idx
        end
    end

    local next_tg_no = 1
    while used[next_tg_no] do
        next_tg_no = next_tg_no + 1
    end

    local created = nil
    local ok_create = pcall(function() created = tc:Acquire() end)
    if not ok_create or not created then
        EZ.sendMessage("trackgroup", "error", {tc = tcNo, error = "Track group create failed"})
        return nil
    end

    local resolved_name = desired ~= "" and desired or string.format("Group %d", next_tg_no)
    local ok_name = pcall(function() created.name = resolved_name end)
    if not ok_name then
        pcall(function() created:Set("name", resolved_name) end)
    end

    local created_tg_no = next_tg_no
    local ok_refresh, refreshed = pcall(function() return tc:Children() end)
    if ok_refresh and refreshed then
        for tg_idx = 1, #refreshed do
            local tg = refreshed[tg_idx]
            if tg and tg.name and tostring(tg.name) == (created.name or resolved_name) then
                created_tg_no = tg_idx
            end
        end
    end

    EZ.sendMessage(
        "trackgroup",
        "created",
        {tc = tcNo, tg = created_tg_no, name = created.name or resolved_name}
    )
    return created_tg_no
end

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

local CURRENT_SONG_SEQUENCE_RANGE_SIZE = 99

local function getChildrenSafe(handle)
    if not handle then
        return {}
    end
    local ok, children = pcall(function() return handle:Children() end)
    if ok and children then
        return children
    end
    return {}
end

local function getClassSafe(handle)
    if not handle or not handle.GetClass then
        return nil
    end
    local ok, cls = pcall(function() return handle:GetClass() end)
    if ok then
        return cls
    end
    return nil
end

local function trimString(value)
    return tostring(value or ""):gsub("^%s+", ""):gsub("%s+$", "")
end

local function escapeCmdString(value)
    return tostring(value or ""):gsub('"', '\\"')
end

local function mergePayload(base, extra)
    if not extra then
        return base
    end
    for k, v in pairs(extra) do
        base[k] = v
    end
    return base
end

local function findChildByClass(parent, className, ordinal)
    local children = getChildrenSafe(parent)
    local targetOrdinal = ordinal or 1
    local seen = 0

    for childIdx = 1, #children do
        local child = children[childIdx]
        if getClassSafe(child) == className then
            seen = seen + 1
            if seen == targetOrdinal then
                return child, seen, childIdx
            end
        end
    end

    return nil, nil, nil
end

local function getTrackHandle(tcNo, tgNo, trackNo)
    local ma3TrackNo = (trackNo or 0) + 1
    return EZ.getTrack(tcNo, tgNo, ma3TrackNo), ma3TrackNo
end

local function getAssignedSequenceNo(track)
    if not track or not track.target then
        return nil
    end

    local okNo, directNo = pcall(function() return tonumber(track.target.no) end)
    if okNo and directNo then
        return directNo
    end

    local targetStr = tostring(track.target)
    local seqMatch = string.match(targetStr, "Sequence%s+(%d+)")
    if seqMatch then
        return tonumber(seqMatch)
    end

    return nil
end

local function getSequenceHandle(seqNo)
    local dp = EZ.getDP()
    if not dp or not dp.Sequences then
        return nil
    end
    local ok, sequence = pcall(function() return dp.Sequences[seqNo] end)
    if ok then
        return sequence
    end
    return nil
end

local function buildTrackError(tcNo, tgNo, trackNo, errorCode, extra, emitReply)
    local payload = mergePayload({
        tc = tcNo,
        tg = tgNo,
        track = trackNo,
        error = errorCode
    }, extra)
    if emitReply ~= false then
        EZ.sendMessage("track", "error", payload)
    end
    return false, payload
end

local function buildSequenceError(changeType, errorCode, extra)
    local payload = mergePayload({error = errorCode}, extra)
    EZ.sendMessage("sequence", changeType or "error", payload)
    return nil, payload
end

local function buildSequenceRangeError(errorCode, extra)
    local payload = mergePayload({error = errorCode}, extra)
    EZ.sendMessage("sequence_range", "error", payload)
    return nil, payload
end

local function countSequenceCues(sequence)
    local cues = getChildrenSafe(sequence)
    return #cues
end

local function listSequencesInRange(startNo, endNo)
    local dp = EZ.getDP()
    if not dp then
        return nil, "datapool_unavailable"
    end

    if not dp.Sequences then
        return {}, nil
    end

    local lower = tonumber(startNo)
    local upper = tonumber(endNo)
    if lower and upper and lower > upper then
        lower, upper = upper, lower
    end

    local ok, children = pcall(function() return dp.Sequences:Children() end)
    if not ok or not children then
        return nil, "sequence_enumeration_failed"
    end

    local result = {}
    for _, sequence in ipairs(children) do
        local no = tonumber(sequence and sequence.no)
        if no and (not lower or no >= lower) and (not upper or no <= upper) then
            table.insert(result, {
                no = no,
                name = sequence.name or "",
                cue_count = countSequenceCues(sequence)
            })
        end
    end

    table.sort(result, function(a, b) return (a.no or 0) < (b.no or 0) end)
    return result, nil
end

local function resolveCurrentSongSequenceRange()
    local okVars, globalVars = pcall(GlobalVars)
    if not okVars or not globalVars then
        return nil, "global_vars_unavailable"
    end

    local okSong, songLabel = pcall(function() return GetVar(globalVars, "song") end)
    if not okSong then
        return nil, "song_global_lookup_failed"
    end

    songLabel = trimString(songLabel)
    if songLabel == "" then
        return nil, "song_global_missing"
    end

    local sequences, listErr = listSequencesInRange()
    if not sequences then
        return nil, listErr
    end

    for _, sequence in ipairs(sequences) do
        if trimString(sequence.name) == songLabel then
            return {
                song_label = songLabel,
                start = sequence.no,
                ["end"] = sequence.no + CURRENT_SONG_SEQUENCE_RANGE_SIZE
            }, nil
        end
    end

    return nil, "song_anchor_sequence_not_found"
end

local function findAvailableSequenceNo(startNo, endNo)
    local dp = EZ.getDP()
    if not dp then
        return nil, "datapool_unavailable"
    end

    local lower = math.max(1, tonumber(startNo) or 1)
    local upper = tonumber(endNo) or 9999
    if lower > upper then
        lower, upper = upper, lower
    end

    for seqNo = lower, upper do
        if not getSequenceHandle(seqNo) then
            return seqNo, nil
        end
    end

    if endNo then
        return nil, "no_free_sequence_in_range"
    end

    return nil, "no_available_sequence_numbers"
end

local function createSequenceAtNumber(seqNo, preferredName, mode)
    local numericSeqNo = tonumber(seqNo)
    if not numericSeqNo then
        return buildSequenceError("error", "invalid_sequence_number", {no = seqNo, mode = mode})
    end

    if getSequenceHandle(numericSeqNo) then
        return buildSequenceError("error", "sequence_already_exists", {no = numericSeqNo, mode = mode})
    end

    local cmdStr = string.format("Store Sequence %d", numericSeqNo)
    local name = trimString(preferredName)
    if name ~= "" then
        cmdStr = cmdStr .. string.format(' /name="%s"', escapeCmdString(name))
    end
    cmdStr = cmdStr .. " /nc"

    local ok, result = EZ.RunCommand(cmdStr, "feedback")
    if not ok then
        return buildSequenceError("error", "sequence_create_command_failed", {
            no = numericSeqNo,
            name = name,
            mode = mode,
            detail = result
        })
    end

    local sequence = getSequenceHandle(numericSeqNo)
    if not sequence then
        return buildSequenceError("error", "sequence_create_verification_failed", {
            no = numericSeqNo,
            name = name,
            mode = mode,
            detail = result
        })
    end

    local payload = {
        no = numericSeqNo,
        name = sequence.name or name,
        mode = mode
    }
    EZ.sendMessage("sequence", "created", payload)
    return payload
end

local function ensureTimeRange(track)
    local timeRange, ordinal = findChildByClass(track, "TimeRange", 1)
    local created = false

    if not timeRange then
        local okAcquire = pcall(function() track:Acquire("TimeRange") end)
        if not okAcquire then
            return nil, nil, "time_range_create_failed"
        end
        timeRange, ordinal = findChildByClass(track, "TimeRange", 1)
        created = timeRange ~= nil
    end

    if not timeRange then
        return nil, nil, "time_range_create_failed"
    end

    return timeRange, ordinal, nil, created
end

local function ensureCmdSubTrack(timeRange)
    local cmdSubTrack = findChildByClass(timeRange, "CmdSubTrack", 1)
    local created = false

    if not cmdSubTrack then
        local okAcquire = pcall(function() timeRange:Acquire("CmdSubTrack") end)
        if not okAcquire then
            pcall(function() timeRange:Append("CmdSubTrack") end)
        end
        cmdSubTrack = findChildByClass(timeRange, "CmdSubTrack", 1)
        created = cmdSubTrack ~= nil
    end

    if not cmdSubTrack then
        return nil, "cmd_subtrack_create_failed"
    end

    return cmdSubTrack, nil, created
end

local function prepareTrackForEventsInternal(tcNo, tgNo, trackNo, emitReply)
    local track = getTrackHandle(tcNo, tgNo, trackNo)
    if not track then
        return buildTrackError(tcNo, tgNo, trackNo, "track_not_found", nil, emitReply)
    end

    local seqNo = getAssignedSequenceNo(track)
    if not seqNo then
        return buildTrackError(tcNo, tgNo, trackNo, "no_sequence_assigned", nil, emitReply)
    end

    local timeRange, timeRangeIdx, timeRangeErr, timeRangeCreated = ensureTimeRange(track)
    if not timeRange then
        return buildTrackError(tcNo, tgNo, trackNo, timeRangeErr or "time_range_create_failed", {seq = seqNo}, emitReply)
    end

    local cmdSubTrack, cmdErr, cmdCreated = ensureCmdSubTrack(timeRange)
    if not cmdSubTrack then
        return buildTrackError(tcNo, tgNo, trackNo, cmdErr or "cmd_subtrack_create_failed", {
            seq = seqNo,
            time_range_idx = timeRangeIdx
        }, emitReply)
    end

    local payload = {
        tc = tcNo,
        tg = tgNo,
        track = trackNo,
        seq = seqNo,
        time_range_idx = timeRangeIdx,
        cmd_subtrack_ready = cmdSubTrack ~= nil
    }
    if timeRangeCreated then
        payload.time_range_created = true
    end
    if cmdCreated then
        payload.cmd_subtrack_created = true
    end

    if emitReply ~= false then
        EZ.sendMessage("track", "prepared", payload)
    end
    return true, payload
end

-- PUBLIC API: SEQUENCE MANAGEMENT
function EZ.GetSequences(startNo, endNo)
    EZ.log(string.format("GetSequences(%s, %s) called", tostring(startNo), tostring(endNo)))

    local result, err = listSequencesInRange(startNo, endNo)
    if not result then
        EZ.sendMessage("sequences", "error", {error = err})
        return nil
    end

    EZ.sendMessage("sequences", "list", {
        count = #result,
        sequences = result
    })
    return result
end

function EZ.GetCurrentSongSequenceRange()
    EZ.log("GetCurrentSongSequenceRange() called")

    local payload, err = resolveCurrentSongSequenceRange()
    if not payload then
        return buildSequenceRangeError(err)
    end

    EZ.sendMessage("sequence_range", "current_song", payload)
    return payload
end

function EZ.CreateSequenceNextAvailable(name)
    EZ.log(string.format("CreateSequenceNextAvailable('%s') called", tostring(name or "")))

    local seqNo, err = findAvailableSequenceNo(1)
    if not seqNo then
        return buildSequenceError("error", err, {mode = "next_available"})
    end

    return createSequenceAtNumber(seqNo, name, "next_available")
end

function EZ.CreateSequenceInCurrentSongRange(name)
    EZ.log(string.format("CreateSequenceInCurrentSongRange('%s') called", tostring(name or "")))

    local range, rangeErr = resolveCurrentSongSequenceRange()
    if not range then
        return buildSequenceRangeError(rangeErr)
    end

    local seqNo, err = findAvailableSequenceNo(range.start, range["end"])
    if not seqNo then
        return buildSequenceError("error", err, {
            mode = "current_song_range",
            song_label = range.song_label,
            start = range.start,
            ["end"] = range["end"]
        })
    end

    return createSequenceAtNumber(seqNo, name, "current_song_range")
end

-- Add event to a track
-- eventName/cueNo/cueLabel are optional metadata fields.
function EZ.AddEvent(tcNo, tgNo, trackNo, time, cmd, eventName, cueNo, cueLabel)
    local one_second_internally = 16777216
    local track, ma3TrackNo = getTrackHandle(tcNo, tgNo, trackNo)
    
    -- Step 1: Get the track
    if not track then
        EZ.log(string.format("FATAL: AddEvent - Track not found at %d.%d.%d (MA3 index %d)", tcNo, tgNo, trackNo, ma3TrackNo))
        EZ.sendMessage("event", "error", {tc = tcNo, tg = tgNo, track = trackNo, error = "Track not found"})
        return false
    end

    local ready, prepPayload = prepareTrackForEventsInternal(tcNo, tgNo, trackNo, false)
    if not ready then
        EZ.log(string.format("FATAL: AddEvent - Track prep failed for %d.%d.%d (%s)", tcNo, tgNo, trackNo, tostring(prepPayload and prepPayload.error or "unknown")))
        EZ.sendMessage("event", "error", {
            tc = tcNo,
            tg = tgNo,
            track = trackNo,
            error = prepPayload and prepPayload.error or "track_prepare_failed"
        })
        return false
    end

    -- Step 2: Check if time range exists
    local time_range = EZ.GetFirstTimeRange(track)
    if not time_range then
        EZ.log(string.format("FATAL: AddEvent - No TimeRange found in track %d.%d.%d after prep", tcNo, tgNo, trackNo))
        EZ.sendMessage("event", "error", {tc = tcNo, tg = tgNo, track = trackNo, error = "No TimeRange"})
        return false
    end

    -- Step 3: Check if CmdSubTrack exists
    local cmd_subtrack = EZ.GetFirstCmdSubTrack(time_range)
    if not cmd_subtrack then
        EZ.log(string.format("FATAL: AddEvent - No CmdSubTrack found in TimeRange after prep (track %d.%d.%d)", tcNo, tgNo, trackNo))
        EZ.sendMessage("event", "error", {tc = tcNo, tg = tgNo, track = trackNo, error = "No CmdSubTrack"})
        return false
    end

    -- Step 4: Create the event
    local time_units = math.floor((tonumber(time) or 0) * one_second_internally)
    local event = cmd_subtrack:Acquire()
    event:Set("Time", time_units)
    event:Set("Cmd", cmd or "")

    local resolvedName = tostring(eventName or cueLabel or ""):gsub("^%s+", ""):gsub("%s+$", "")
    if resolvedName ~= "" then
        local ok_name = pcall(function() event.name = resolvedName end)
        if not ok_name then
            pcall(function() event:Set("Name", resolvedName) end)
        end
    end

    local numericCueNo = tonumber(cueNo)
    if numericCueNo and numericCueNo > 0 then
        local ok_set_cue_no = pcall(function() event:Set("CueNo", numericCueNo) end)
        if not ok_set_cue_no then
            pcall(function() event.cueNo = numericCueNo end)
            pcall(function() event.cueno = numericCueNo end)
        end
    end

    local resolvedCueLabel = tostring(cueLabel or eventName or ""):gsub("^%s+", ""):gsub("%s+$", "")
    if resolvedCueLabel ~= "" then
        local ok_set_cue_name = pcall(function() event:Set("CueName", resolvedCueLabel) end)
        if not ok_set_cue_name then
            pcall(function() event.cueName = resolvedCueLabel end)
        end
    end

    return true
end

function EZ.AssignTrackSequence(tcNo, tgNo, trackNo, seqNo)
    local numericSeqNo = tonumber(seqNo)
    local track = getTrackHandle(tcNo, tgNo, trackNo)
    if not track then
        EZ.log(string.format("AssignTrackSequence: Track %d.%d.%d not found", tcNo, tgNo, trackNo))
        return buildTrackError(tcNo, tgNo, trackNo, "track_not_found")
    end

    if not numericSeqNo then
        return buildTrackError(tcNo, tgNo, trackNo, "invalid_sequence_number", {seq = seqNo})
    end

    local sequence = getSequenceHandle(numericSeqNo)
    if not sequence then
        EZ.log(string.format("AssignTrackSequence: Sequence %s not found", tostring(seqNo)))
        return buildTrackError(tcNo, tgNo, trackNo, "sequence_not_found", {seq = numericSeqNo})
    end

    local currentSeqNo = getAssignedSequenceNo(track)
    if currentSeqNo == numericSeqNo then
        local existingPayload = {tc = tcNo, tg = tgNo, track = trackNo, seq = numericSeqNo, changed = false}
        EZ.sendMessage("track", "assigned", existingPayload)
        return true, existingPayload
    end

    local okSet, err = pcall(function() track:Set("Target", sequence) end)
    if not okSet then
        return buildTrackError(tcNo, tgNo, trackNo, "sequence_assignment_failed", {
            seq = numericSeqNo,
            detail = tostring(err)
        })
    end

    local assignedSeqNo = getAssignedSequenceNo(track)
    if assignedSeqNo ~= numericSeqNo then
        return buildTrackError(tcNo, tgNo, trackNo, "sequence_assignment_verification_failed", {
            seq = numericSeqNo,
            assigned_seq = assignedSeqNo
        })
    end

    local payload = {tc = tcNo, tg = tgNo, track = trackNo, seq = numericSeqNo, changed = true}
    EZ.sendMessage("track", "assigned", payload)
    EZ.log(string.format("Assigned sequence %d to track %d.%d.%d", numericSeqNo, tcNo, tgNo, trackNo))
    return true, payload
end

function EZ.CreateTimeRange(tcNo, tgNo, trackNo)
    -- Creates a new TimeRange in the track
    local track = getTrackHandle(tcNo, tgNo, trackNo)
    if not track then
        EZ.log(string.format("FATAL: CreateTimeRange - Track %d.%d.%d not found", tcNo, tgNo, trackNo))
        return false
    end
    local time_range = select(1, ensureTimeRange(track))
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
    local track = getTrackHandle(tcNo, tgNo, trackNo)
    if not track then
        EZ.log(string.format("CreateCmdSubTrack: Track %d.%d.%d not found", tcNo, tgNo, trackNo))
        return false
    end

    local time_range = findChildByClass(track, "TimeRange", time_range_idx or 1)
    if not time_range then
        EZ.log(string.format("CreateCmdSubTrack: TimeRange %s not found in track %d.%d.%d", tostring(time_range_idx or 1), tcNo, tgNo, trackNo))
        return false
    end

    local cmd_subtrack, err = ensureCmdSubTrack(time_range)
    if not cmd_subtrack then
        EZ.log(string.format("CreateCmdSubTrack: %s for track %d.%d.%d", tostring(err), tcNo, tgNo, trackNo))
        return false
    end

    EZ.log(string.format("CreateCmdSubTrack: CmdSubTrack ready in track %d.%d.%d", tcNo, tgNo, trackNo))
    return true
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
    local track = getTrackHandle(tcNo, tgNo, trackNo)
    
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

function EZ.PrepareTrackForEvents(tcNo, tgNo, trackNo)
    EZ.log(string.format("PrepareTrackForEvents(%d, %d, %d) called", tcNo, tgNo, trackNo))
    return prepareTrackForEventsInternal(tcNo, tgNo, trackNo, true)
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
