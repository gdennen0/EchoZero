--[[
EchoZero Debug/Test Functions
=============================

Separate debug and testing functions for EchoZero MA3 integration.
Load this AFTER echozero.lua for debug/test functionality.

Usage: Lua "dofile('/path/to/echozero_debug.lua')"
]]--

-- Ensure EZ table exists (safe for any load order)
EZ = EZ or {}

-- =============================================================================
-- SOCKET TESTING
-- =============================================================================

function EZ.TestSocket()
    EZ.log("=== Socket Test ===")
    EZ.log(string.format("  _socketOk: %s", tostring(EZ._socketOk)))
    EZ.log(string.format("  _socket type: %s", type(EZ._socket)))
    EZ.log(string.format("  Target: %s:%d", EZ.config.ip, EZ.config.port))
    
    if not EZ._socketOk then
        EZ.log("  FAIL: Socket not available - cannot test")
        return false
    end
    
    if not EZ._socket then
        EZ.log("  FAIL: _socket is nil despite _socketOk being true")
        return false
    end
    
    -- Step 1: Check udp function
    if not EZ._socket.udp then
        EZ.log("  FAIL: _socket.udp function not found")
        return false
    end
    EZ.log("  PASS: _socket.udp function exists")
    
    -- Step 2: Create UDP socket
    local udp
    local create_ok, create_err = pcall(function()
        udp = EZ._socket.udp()
    end)
    if not create_ok then
        EZ.log(string.format("  FAIL: UDP socket creation threw error: %s", tostring(create_err)))
        return false
    end
    if not udp then
        EZ.log("  FAIL: UDP socket creation returned nil")
        return false
    end
    EZ.log("  PASS: UDP socket created")
    
    -- Step 3: Set peer name
    local peer_ok, peer_err = pcall(function()
        return udp:setpeername(EZ.config.ip, EZ.config.port)
    end)
    if not peer_ok then
        EZ.log(string.format("  FAIL: setpeername threw error: %s", tostring(peer_err)))
        udp:close()
        return false
    end
    EZ.log(string.format("  PASS: setpeername(%s, %d) succeeded", EZ.config.ip, EZ.config.port))
    
    -- Step 4: Send test data
    local test_data = "EZ_SOCKET_TEST"
    local send_result
    local send_ok, send_err = pcall(function()
        send_result = udp:send(test_data)
    end)
    if not send_ok then
        EZ.log(string.format("  FAIL: send() threw error: %s", tostring(send_err)))
        udp:close()
        return false
    end
    EZ.log(string.format("  PASS: send() returned: %s", tostring(send_result)))
    
    -- Step 5: Close socket
    udp:close()
    EZ.log("  PASS: Socket closed")
    
    EZ.log("=== Socket Test PASSED ===")
    return true
end

-- =============================================================================
-- TIMECODE DEBUGGING
-- =============================================================================

function EZ.DebugTimecode(tcNo)
    EZ.log(string.format("=== Debug Timecode %s ===", tostring(tcNo)))
    
    local dp = DataPool()
    if not dp then EZ.log("  ERROR: DataPool not accessible"); return end
    if not dp.Timecodes then EZ.log("  ERROR: No Timecodes in DataPool"); return end
    
    -- List all timecodes
    local allTCs = dp.Timecodes:Children()
    if allTCs then
        EZ.log(string.format("  Found %d timecodes:", #allTCs))
        for i, tc in ipairs(allTCs) do
            EZ.log(string.format("    [%d] no=%s, name='%s'", i, tostring(tc.no), tc.name or ""))
        end
    else
        EZ.log("  ERROR: Timecodes:Children() returned nil")
        return
    end
    
    -- Try to access specific timecode
    if tcNo then
        local tc = dp.Timecodes[tcNo]
        if not tc then
            EZ.log(string.format("  ERROR: Timecode %d not found", tcNo))
            return
        end
        EZ.log(string.format("  Timecode %d: name='%s'", tcNo, tc.name or ""))
        
        local ok, tgChildren = pcall(function() return tc:Children() end)
        if ok and tgChildren then
            EZ.log(string.format("  Found %d track groups:", #tgChildren))
            for i = 1, #tgChildren do
                local tg = tgChildren[i]
                if tg then
                    local trackCount = 0
                    local trackOk, trackChildren = pcall(function() return tg:Children() end)
                    if trackOk and trackChildren then
                        for j = 1, #trackChildren do
                            if trackChildren[j] and trackChildren[j].name ~= "Marker" then
                                trackCount = trackCount + 1
                            end
                        end
                    end
                    EZ.log(string.format("    TG[%d]: '%s' -> %d tracks", i, tg.name or "", trackCount))
                end
            end
        end
    end
    EZ.log("=== Debug Complete ===")
end

-- =============================================================================
-- TIME DIAGNOSTIC
-- =============================================================================

-- Diagnostic function to check timecode properties and time conversion
-- Usage: Lua "EZ.DiagnoseTime(101, 1, 1, 1)" to check TC101, TG1, Track1, Event1
function EZ.DiagnoseTime(tcNo, tgNo, trackNo, eventIdx)
    EZ.log("=== TIME DIAGNOSTIC ===")
    EZ.log(string.format("  Target: TC%d.TG%d.Track%d.Event%d", tcNo or 0, tgNo or 0, trackNo or 0, eventIdx or 1))
    
    local dp = DataPool()
    if not dp or not dp.Timecodes then
        EZ.log("  ERROR: DataPool not accessible")
        return
    end
    
    local tc = dp.Timecodes[tcNo]
    if not tc then
        EZ.log(string.format("  ERROR: Timecode %d not found", tcNo))
        return
    end
    
    EZ.log("  Timecode properties:")
    EZ.log(string.format("    Name: %s", tc.name or "?"))
    
    local framerate = tc.framereadout or tc.FRAMEREADOUT or tc.FrameReadout
    if framerate then
        EZ.log(string.format("    Frame rate: %s", tostring(framerate)))
    else
        EZ.log("    Frame rate: (not available)")
    end
    
    local timeformat = tc.timedisplayformat or tc.TIMEDISPLAYFORMAT
    if timeformat then
        EZ.log(string.format("    Time display: %s", tostring(timeformat)))
    end
    
    if tgNo and trackNo then
        local ma3TrackNo = trackNo + 1
        local track = dp.Timecodes[tcNo][tgNo][ma3TrackNo]
        if track then
            EZ.log(string.format("    Track: %s", track.name or "?"))
            
            -- Get events via track children
            local ok1, timeRanges = pcall(function() return track:Children() end)
            if ok1 and timeRanges and timeRanges[1] then
                local ok2, subTracks = pcall(function() return timeRanges[1]:Children() end)
                if ok2 and subTracks then
                    for _, subTrack in ipairs(subTracks) do
                        local subTrackClass = "unknown"
                        if subTrack.GetClass then
                            local ok3, cls = pcall(function() return subTrack:GetClass() end)
                            if ok3 then subTrackClass = cls or "unknown" end
                        end
                        if subTrackClass == "CmdSubTrack" then
                            local ok4, evts = pcall(function() return subTrack:Children() end)
                            if ok4 and evts then
                                EZ.log(string.format("    Events in track: %d", #evts))
                                local maxShow = math.min(5, #evts)
                                for i = 1, maxShow do
                                    local evt = evts[i]
                                    EZ.log(string.format("    Event[%d]: time=%s (type=%s), name='%s'", 
                                        i, tostring(evt.time), type(evt.time), evt.name or "?"))
                                end
                                
                                if eventIdx and eventIdx <= #evts then
                                    local evt = evts[eventIdx]
                                    EZ.log(string.format("  Detailed event %d analysis:", eventIdx))
                                    EZ.log(string.format("    time: type=%s, value=%s", type(evt.time), tostring(evt.time)))
                                    if evt.abstime then EZ.log(string.format("    abstime: %s", tostring(evt.abstime))) end
                                    if evt.rawtime then EZ.log(string.format("    rawtime: %s", tostring(evt.rawtime))) end
                                    if evt.ABSTIME then EZ.log(string.format("    ABSTIME: %s", tostring(evt.ABSTIME))) end
                                    if evt.RAWTIME then EZ.log(string.format("    RAWTIME: %s", tostring(evt.RAWTIME))) end
                                end
                            end
                            break
                        end
                    end
                end
            end
        end
    end
    EZ.log("=== END DIAGNOSTIC ===")
    return true
end

-- =============================================================================
-- HOOK TESTING
-- =============================================================================

function EZ.TestHook(tcNo, tgNo, trackNo)
    tcNo = tcNo or 101
    tgNo = tgNo or 1
    trackNo = trackNo or 4
    
    Printf("[EZ] === HOOK TEST ===")
    Printf("[EZ] Target: TC%d.TG%d.TR%d", tcNo, tgNo, trackNo)
    
    local pluginHandle = luaComponentHandle and luaComponentHandle:Parent() or nil
    Printf("[EZ] Plugin handle: %s", pluginHandle and "OK" or "MISSING")
    
    if not pluginHandle then
        Printf("[EZ] ERROR: No plugin handle - reload plugin")
        return false
    end
    
    local success = EZ.HookTrack(tcNo, tgNo, trackNo)
    
    if success then
        Printf("[EZ] SUCCESS! Modify the track to test callback")
    else
        Printf("[EZ] FAILED to create hook")
    end
    
    return success
end

function EZ.TestCmdSubTrackHook(tcNo, tgNo, trackNo, timeRangeIdx, subTrackIdx)
    tcNo = tcNo or 101
    tgNo = tgNo or 1
    trackNo = trackNo or 4
    timeRangeIdx = timeRangeIdx or 1
    
    Printf("[EZ] === CMDSUBTrack HOOK TEST ===")
    Printf("[EZ] Target: TC%d.TG%d.TR%d", tcNo, tgNo, trackNo)
    
    local success = EZ.HookCmdSubTrack(tcNo, tgNo, trackNo, timeRangeIdx, subTrackIdx)
    
    if success then
        Printf("[EZ] SUCCESS! Modify events in the CmdSubTrack to test callback")
    else
        Printf("[EZ] FAILED to create CmdSubTrack hook")
    end
    
    return success
end

function EZ.TestCreateTrack(tcNo, tgNo, trackName)
    Printf("[EZ] === CREATE TRACK TEST ===")
    Printf(string.format("[EZ] tc=%s tg=%s name=%s", tostring(tcNo), tostring(tgNo), tostring(trackName)))
    if EZ.CreateTrack then
        local result = EZ.CreateTrack(tcNo, tgNo, trackName)
        Printf(string.format("[EZ] CreateTrack result: %s", tostring(result)))
        return result
    end
    Printf("[EZ] ERROR: EZ.CreateTrack not found")
    return nil
end

function EZ.TestCreateEvent(tcNo, tgNo, trackNo, time, cmd)
    Printf("[EZ] === CREATE TRACK TEST ===")
    Printf(string.format("[EZ] tc=%s tg=%s track=%s time=%s cmd=%s", tostring(tcNo), tostring(tgNo), tostring(trackNo), tostring(time), tostring(cmd)))
    if EZ.AddEvent then
        local result = EZ.AddEvent(tcNo, tgNo, trackNo, time, cmd)
        Printf(string.format("[EZ] AddEvent result: %s", tostring(result)))
        return result
    end
    Printf("[EZ] ERROR: EZ.AddEvent not found")
    return nil
end

function EZ.TestCreateEventSeries(tcNo, tgNo, trackNo)
    tcNo = tcNo or 101
    tgNo = tgNo or 1
    trackNo = trackNo or 12

    Printf("[EZ] === CREATE EVENT SERIES TEST ===")
    Printf(string.format("[EZ] tc=%s tg=%s track=%s", tostring(tcNo), tostring(tgNo), tostring(trackNo)))
    if not EZ.AddEvent then
        Printf("[EZ] ERROR: EZ.AddEvent not found")
        return nil
    end

    local results = {}
    for i = 1, 10 do
        local time = i
        local cmd = string.format("EZ_Test_%02d", i)
        local ok = EZ.AddEvent(tcNo, tgNo, trackNo, time, cmd)
        results[i] = ok
        Printf(string.format("[EZ] AddEvent %02d at %ds => %s", i, time, tostring(ok)))
    end
    return results
end

return EZ