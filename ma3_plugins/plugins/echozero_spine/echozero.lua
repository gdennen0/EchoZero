EZ = EZ or {}

EZ.config = EZ.config or {
    ip = "127.0.0.1",
    port = 9000,
    debug = false,
    timeScaleFactor = 1,  -- Set to 2 if events appear at 2x expected time (frame rate issue)
    maxChangeEvents = 40, -- Max events to include in track.changed payload
    defaultCmdMode = "feedback"  -- "feedback" uses Cmd() (shows in console), "silent" uses CmdIndirect() (clean console)
}

EZ._hooks = EZ._hooks or {}
EZ._trackgroup_hooks = EZ._trackgroup_hooks or {}
EZ._version = "2.0"
EZ._oscInitialized = false

-- PLUGIN HANDLE CAPTURE (luaComponentHandle at load time, :Parent() for HookObjectChange)
local luaComponentHandle = select(4, ...)
local function getPluginHandle() return luaComponentHandle and luaComponentHandle:Parent() or nil end

-- LOGGING (EZ.log is the global log function, available to all EZ modules)
function EZ.log(msg) Printf("[EZ] %s", msg) end
function EZ.dbg(msg) if EZ.config.debug then Printf("[EZ DBG] %s", msg) end end

-- OSC LATE-BINDING: All OSC calls check at call time, not load time.
-- This makes load order irrelevant (echozero_osc.lua can load before or after).
local function jsonEncode(v)
    if OSC and OSC.jsonEncode then return OSC.jsonEncode(v) end
    return tostring(v)
end

-- Lazy OSC initialization: runs once on first sendMessage call
local function ensureOSC()
    if EZ._oscInitialized then return end
    if OSC and OSC.setConfig and OSC.init then
        OSC.setConfig(EZ.config)
        OSC.init()
        EZ._oscInitialized = true
        EZ.log(string.format("OSC initialized (socket: %s)", tostring(OSC._socketOk)))
    end
end

function EZ.sendMessage(msgType, changeType, data)
    ensureOSC()
    if OSC and OSC.sendMessage then
        return OSC.sendMessage(msgType, changeType, data)
    end
    return false
end

-- DATAPOOL HELPERS (global on EZ table, available to all EZ modules)
function EZ.getDP()
    local ok, dp = pcall(DataPool)
    if not ok or not dp then return nil end
    return dp
end
function EZ.getTC(tcNo)
    local dp = EZ.getDP()
    if not dp or not dp.Timecodes then return nil end
    local ok, tc = pcall(function() return dp.Timecodes[tcNo] end)
    return ok and tc or nil
end
function EZ.getTG(tcNo, tgNo)
    local tc = EZ.getTC(tcNo)
    if not tc then return nil end
    local ok, tg = pcall(function() return tc[tgNo] end)
    return (ok and tg) or nil
end
-- Get track by index (Track index 1 is Marker, user tracks start at 2)
function EZ.getTrack(tcNo, tgNo, trackNo)
    local ok, track = pcall(function() return DataPool().Timecodes[tcNo][tgNo][trackNo] end)
    return (ok and track) or nil
end
-- Get CmdSubTrack from a track (trackNo is user-visible 1-based, excluding Marker)
-- IMPORTANT: Finds the FIRST TimeRange by class, not by index
function EZ.getCmdSubTrack(tcNo, tgNo, trackNo, timeRangeIdx, subTrackIdx)
    local function getClass(obj)
        if obj and obj.GetClass then
            local ok, cls = pcall(function() return obj:GetClass() end)
            return ok and cls or nil
        end
        return nil
    end
    
    local track = EZ.getTrack(tcNo, tgNo, trackNo + 1)
    if not track then return nil end
    
    local ok1, children = pcall(function() return track:Children() end)
    if not ok1 or not children then return nil end
    
    -- Find the first TimeRange by CLASS (more reliable than index)
    local timeRange = nil
    local foundIdx = 0
    for i = 1, #children do
        if getClass(children[i]) == "TimeRange" then
            foundIdx = foundIdx + 1
            if foundIdx == (timeRangeIdx or 1) then
                timeRange = children[i]
                break
            end
        end
    end
    if not timeRange then return nil end
    
    local ok2, subTracks = pcall(function() return timeRange:Children() end)
    if not ok2 or not subTracks then return nil end
    
    if subTrackIdx then
        local st = subTracks[subTrackIdx]
        if getClass(st) == "CmdSubTrack" then return st end
    else
        for i = 1, #subTracks do
            if getClass(subTracks[i]) == "CmdSubTrack" then return subTracks[i], i end
        end
    end
    return nil
end
-- Get all events from a track
-- Hierarchy: Track -> TimeRange -> SubTrack (CmdSubTrack/FaderSubTrack) -> Event
-- All levels use :Children() for traversal!
function EZ.getTrackEvents(track)
    local events = {}
    
    if not track then return events end
    
    -- TimeRanges are children of track
    local ok1, timeRanges = pcall(function() return track:Children() end)
    if not ok1 or not timeRanges then return events end
    
    for trIdx = 1, #timeRanges do
        local timeRange = timeRanges[trIdx]
        if timeRange then
            -- SubTracks are children of TimeRange
            local ok2, subTracks = pcall(function() return timeRange:Children() end)
            if ok2 and subTracks then
                for stIdx = 1, #subTracks do
                    local subTrack = subTracks[stIdx]
                    if subTrack then
                        -- Check if this is a CmdSubTrack or FaderSubTrack
                        local subTrackClass = "unknown"
                        if subTrack.GetClass then
                            local ok3, cls = pcall(function() return subTrack:GetClass() end)
                            if ok3 then subTrackClass = cls or "unknown" end
                        end
                        
                        -- Only process CmdSubTrack and FaderSubTrack
                        if subTrackClass == "CmdSubTrack" or subTrackClass == "FaderSubTrack" then
                            -- Events are children of SubTrack
                            local ok4, evts = pcall(function() return subTrack:Children() end)
                            if ok4 and evts then
                                for evIdx = 1, #evts do
                                    local evt = evts[evIdx]
                                    if evt then
                                        local timeVal = 0
                                        if evt.time then
                                            local timeRaw = evt.time
                                            local absTimeRaw = evt.abstime or evt.ABSTIME  -- Also check abstime
                                            
                                            -- DEBUG: Always log time values when debug is on
                                            -- This helps diagnose frame rate / time conversion issues
                                            if EZ.config.debug then
                                                EZ.log(string.format("[TIME DEBUG] Event '%s':", evt.name or "?"))
                                                EZ.log(string.format("  evt.time: type=%s, value=%s", type(timeRaw), tostring(timeRaw)))
                                                if absTimeRaw then
                                                    EZ.log(string.format("  evt.abstime: type=%s, value=%s", type(absTimeRaw), tostring(absTimeRaw)))
                                                end
                                                -- Try to get rawtime property if available
                                                local rawtimeVal = evt.rawtime or evt.RAWTIME
                                                if rawtimeVal then
                                                    EZ.log(string.format("  evt.rawtime: type=%s, value=%s", type(rawtimeVal), tostring(rawtimeVal)))
                                                end
                                            end
                                            
                                            if type(timeRaw) == "number" then 
                                                -- MA3 internal time format: large values are raw ticks (divide by 16777216)
                                                -- Values > 86400 (24 hours in seconds) indicate raw format
                                                if timeRaw > 86400 then
                                                    timeVal = timeRaw / 16777216
                                                else
                                                    -- Check if time scale factor is configured (for frame rate issues)
                                                    -- Some MA3 setups return time at 2x the display value
                                                    local scaleFactor = EZ.config.timeScaleFactor or 1
                                                    timeVal = timeRaw / scaleFactor
                                                end
                                            elseif type(timeRaw) == "string" then 
                                                -- String format - check if it needs conversion
                                                local numVal = tonumber(timeRaw)
                                                if numVal and numVal > 86400 then
                                                    timeVal = numVal / 16777216
                                                else
                                                    local scaleFactor = EZ.config.timeScaleFactor or 1
                                                    timeVal = (numVal or 0) / scaleFactor
                                                end
                                            end
                                        end
                                        -- Capture all available event properties
                                        -- Core properties for sync
                                        local eventData = {
                                            no = evt.no or evIdx,
                                            time = timeVal,
                                            duration = evt.duration or 0,
                                            cmd = evt.cmd or "",
                                            name = evt.name or "",
                                            subtrack_type = subTrackClass
                                        }
                                        -- Extended properties for cue/trigger info
                                        -- These may not always be present
                                        if evt.cue then eventData.cue = evt.cue end
                                        if evt.cueNo then eventData.cue_no = evt.cueNo end
                                        if evt.cueno then eventData.cue_no = evt.cueno end
                                        if evt.cueName then eventData.cue_name = evt.cueName end
                                        if evt.trigger then eventData.trigger = evt.trigger end
                                        if evt.type then eventData.event_type = evt.type end
                                        if evt.fade then eventData.fade = evt.fade end
                                        if evt.delay then eventData.delay = evt.delay end
                                        if evt.value then eventData.value = evt.value end
                                        if evt.data then eventData.data = evt.data end
                                        table.insert(events, eventData)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    return events
end
-- =============================================================================
-- PUBLIC API: CONNECTION
-- =============================================================================
-- Retry OSC socket initialization (call from MA3 console: Lua "EZ.InitOSC()")
function EZ.InitOSC()
    if not OSC then
        EZ.log("ERROR: OSC module not loaded - echozero_osc.lua may not have loaded")
        return false
    end
    EZ._oscInitialized = false  -- Reset so ensureOSC re-runs
    OSC.setConfig(EZ.config)
    local ok = OSC.init()
    EZ._oscInitialized = true
    if ok then
        EZ.log("OSC initialized successfully")
    else
        EZ.log("OSC initialization failed - socket.core not available")
    end
    return ok
end
function EZ.Ping() EZ.log("Ping!"); EZ.sendMessage("connection", "ping", {status = "ok"}); return true end
function EZ.SetTarget(ip, port)
    EZ.config.ip = ip or EZ.config.ip; EZ.config.port = port or EZ.config.port
    -- Sync to OSC module so outbound messages use the new target
    if OSC and OSC.setConfig then OSC.setConfig(EZ.config) end
    EZ.log(string.format("Target: %s:%d", EZ.config.ip, EZ.config.port)); return true
end
function EZ.SetDebug(enabled)
    EZ.config.debug = enabled and true or false
    -- Sync to OSC module so debug logging is consistent
    if OSC and OSC.setConfig then OSC.setConfig(EZ.config) end
    EZ.log("Debug: " .. (EZ.config.debug and "ON" or "OFF")); return true
end

function EZ.SetDebounceDelay(seconds)
    EZ._debounceDelay = seconds or 0.5
    EZ.log(string.format("Debounce delay: %.3fs", EZ._debounceDelay))
    return true
end
-- Set time scale factor to correct frame rate / time conversion issues
-- If events appear at 2x the expected time, use EZ.SetTimeScale(2)
-- Default is 1 (no scaling)
function EZ.SetTimeScale(factor)
    EZ.config.timeScaleFactor = factor or 1
    EZ.log(string.format("Time scale factor: %.2f (time values will be divided by this)", EZ.config.timeScaleFactor))
    EZ.sendMessage("config", "timescale", {factor = EZ.config.timeScaleFactor})
    return true
end

-- =============================================================================
-- COMMAND EXECUTION: Cmd() vs CmdIndirect()
-- =============================================================================
-- Cmd()         -> Shows in MA3 console feedback, returns result string ('Ok', 'Syntax Error', etc.)
-- CmdIndirect() -> Keeps MA3 console clean, returns nothing
--
-- Use "feedback" mode when you need the return value or want console visibility.
-- Use "silent" mode for background/bulk operations to keep the console clean.

-- Set default command mode for EZ.RunCommand()
-- mode: "feedback" (Cmd) or "silent" (CmdIndirect)
function EZ.SetCmdMode(mode)
    if mode ~= "feedback" and mode ~= "silent" then
        EZ.log(string.format("SetCmdMode: Invalid mode '%s' (use 'feedback' or 'silent')", tostring(mode)))
        return false
    end
    EZ.config.defaultCmdMode = mode
    EZ.log(string.format("Command mode: %s (%s)", mode, mode == "silent" and "CmdIndirect" or "Cmd"))
    EZ.sendMessage("config", "cmd_mode", {mode = mode})
    return true
end

-- Central command execution wrapper
-- cmdStr: The MA3 command string to execute
-- mode: "feedback" uses Cmd(), "silent" uses CmdIndirect(), nil uses EZ.config.defaultCmdMode
-- Returns: ok (bool), result (string or nil)
--   In "feedback" mode: ok=true/false, result=Cmd() return value ('Ok', 'Syntax Error', etc.)
--   In "silent" mode: ok=true/false, result=nil (CmdIndirect returns nothing)
function EZ.RunCommand(cmdStr, mode)
    local useMode = mode or EZ.config.defaultCmdMode or "feedback"

    if useMode == "silent" then
        local ok, err = pcall(function() CmdIndirect(cmdStr) end)
        if not ok then
            EZ.log(string.format("CmdIndirect error: %s (cmd: %s)", tostring(err), cmdStr))
        end
        EZ.dbg(string.format("CmdIndirect: %s", cmdStr))
        return ok, nil
    else
        local ok, result = pcall(function() return Cmd(cmdStr) end)
        if not ok then
            EZ.log(string.format("Cmd error: %s (cmd: %s)", tostring(result), cmdStr))
            return false, tostring(result)
        end
        EZ.dbg(string.format("Cmd: %s -> %s", cmdStr, tostring(result)))
        return ok, result
    end
end

function EZ.Status()
    local hookCount = 0
    for _ in pairs(EZ._hooks) do hookCount = hookCount + 1 end
    
    local socketOk = OSC and OSC._socketOk or false
    
    local cmdMode = EZ.config.defaultCmdMode or "feedback"
    
    EZ.log("=== EchoZero Status ===")
    EZ.log(string.format("  Target: %s:%d", EZ.config.ip, EZ.config.port))
    EZ.log(string.format("  Socket: %s", socketOk and "OK" or "NOT AVAILABLE"))
    EZ.log(string.format("  Debug: %s", EZ.config.debug and "ON" or "OFF"))
    EZ.log(string.format("  Cmd Mode: %s (%s)", cmdMode, cmdMode == "silent" and "CmdIndirect" or "Cmd"))
    EZ.log(string.format("  Hooks: %d active", hookCount))
    EZ.log(string.format("  Version: %s", EZ._version))
    EZ.sendMessage("connection", "status", {
        ip = EZ.config.ip,
        port = EZ.config.port,
        socket = socketOk,
        hooks = hookCount,
        debug = EZ.config.debug,
        version = EZ._version,
        cmd_mode = cmdMode
    })
end
-- EZ.DiagnoseTime and EZ.TestSocket moved to echozero_debug.lua
-- =============================================================================
-- PUBLIC API: HOOKS (Real-time change notifications)
-- =============================================================================
-- IMPORTANT: Hook callbacks MUST be global functions (not local) to prevent garbage collection.
-- MA3's HookObjectChange requires the callback function to persist for the lifetime of the hook.
-- NOTE: All computation (diff detection, fingerprinting) now happens in EchoZero.
-- Lua plugin just sends raw events on every callback.
-- Send track change notification to EchoZero (no event payload)
local function resolveHookTrackNo(hookInfo)
    if not hookInfo or not hookInfo.track then
        return hookInfo and hookInfo.trackNo or nil
    end
    local tg = EZ.getTG(hookInfo.tc, hookInfo.tg)
    if not tg then
        return hookInfo.trackNo
    end
    local ok, children = pcall(function() return tg:Children() end)
    if not ok or not children then
        return hookInfo.trackNo
    end
    -- User-visible track number (1-based, excluding Marker)
    local trackNo = 0
    for i = 1, #children do
        local track = children[i]
        if track and track.name ~= "Marker" then
            trackNo = trackNo + 1
            if track == hookInfo.track then
                return trackNo
            end
        end
    end
    return hookInfo.trackNo
end

local function sendTrackEvents(key, hookInfo, obj)
    EZ.log(string.format(">>> sendTrackEvents for %s", key))
    local resolvedTrackNo = resolveHookTrackNo(hookInfo)
    if resolvedTrackNo and hookInfo and hookInfo.trackNo ~= resolvedTrackNo then
        EZ.log(string.format(
            ">>> Track index updated for hook %s: %s -> %s",
            tostring(key), tostring(hookInfo.trackNo), tostring(resolvedTrackNo)
        ))
        hookInfo.trackNo = resolvedTrackNo
    end
    local trackName = (hookInfo and hookInfo.track and hookInfo.track.name) or (obj and obj.name) or ""
    EZ.log(string.format("DEBUG: hook_send_enter key=%s tc=%s tg=%s track=%s obj=%s", 
        tostring(key), tostring(hookInfo.tc), tostring(hookInfo.tg), tostring(hookInfo.trackNo), tostring(obj and obj.name)))
    EZ.sendMessage("debug", "hook_send_enter", {
        key = key,
        tc = hookInfo.tc,
        tg = hookInfo.tg,
        track = hookInfo.trackNo,
        obj_name = obj and obj.name or ""
    })
    -- Only send track change notification. EchoZero requests full events via EZ.GetEvents.
    EZ.log(string.format("DEBUG: hook_send_before key=%s", tostring(key)))
    EZ.sendMessage("debug", "hook_send_before", {key = key})
    local trackNote = (hookInfo and hookInfo.track and hookInfo.track.note) or ""
    local sendOk = EZ.sendMessage("track", "changed", {
        tc = hookInfo.tc,
        tg = hookInfo.tg,
        track = hookInfo.trackNo or 0,
        name = trackName,
        note = trackNote
    })
    
    if sendOk then
        EZ.log(">>> Sent track.changed to EchoZero")
        -- #region agent log
        EZ.log(string.format("DEBUG: hook_send_after key=%s ok=true", tostring(key)))
        EZ.sendMessage("debug", "hook_send_after", {key = key, ok = true})
        -- #endregion
    else
        EZ.log(">>> ERROR: Failed to send events to EchoZero")
        -- #region agent log
        EZ.log(string.format("DEBUG: hook_send_after key=%s ok=false", tostring(key)))
        EZ.sendMessage("debug", "hook_send_after", {key = key, ok = false})
        -- #endregion
    end
    
    return sendOk
end
-- Global callback function for track changes (must be global, not local)
function EZ._onTrackChange(obj)
    -- Always log callback (even when debug is off) - this is critical for troubleshooting
    local callbackTime = os.time()
    EZ.log(string.format("=== HOOK CALLBACK [%s] ===", callbackTime))
    EZ.log(string.format("  Object: %s", obj.name or "?"))
    -- Find which hook this object belongs to
    local foundHook = false
    for key, hookInfo in pairs(EZ._hooks) do
        if hookInfo.track == obj or hookInfo.subtrack == obj then
            foundHook = true
            EZ.log(string.format("  Hook: %s - sending events to EchoZero", key))
            EZ.log(string.format("DEBUG: hook_match key=%s obj=%s", tostring(key), tostring(obj and obj.name)))
            EZ.sendMessage("debug", "hook_match", {key = key, obj_name = obj and obj.name or ""})
            -- Send raw events - EchoZero does all diff computation
            sendTrackEvents(key, hookInfo, obj)
            EZ.log("=== HOOK CALLBACK COMPLETE ===")
            return
        end
    end
    
    if not foundHook then
        EZ.log("  WARNING: No matching hook found")
        -- #region agent log
        EZ.log(string.format("DEBUG: hook_no_match obj=%s", tostring(obj and obj.name)))
        EZ.sendMessage("debug", "hook_no_match", {obj_name = obj and obj.name or ""})
        -- #endregion
    end
end
-- Global callback function for track group changes
function EZ._onTrackGroupChange(obj)
    local callbackTime = os.time()
    EZ.log(string.format("=== TRACKGROUP HOOK CALLBACK [%s] ===", callbackTime))
    EZ.log(string.format("  Object: %s", obj and obj.name or "?"))
    
    for key, hookInfo in pairs(EZ._trackgroup_hooks) do
        if hookInfo.trackgroup == obj then
            EZ.log(string.format("  TrackGroup Hook: %s - notifying EchoZero", key))
            EZ.sendMessage("trackgroup", "changed", {
                tc = hookInfo.tc,
                tg = hookInfo.tg,
                name = obj and obj.name or ""
            })
            return
        end
    end
    
    EZ.log("  WARNING: No matching trackgroup hook found")
end
-- Hook a track for real-time change notifications
-- NOTE: Unified hook path: HookTrack delegates to HookCmdSubTrack
-- NOTE: trackNo is user-visible track number (1-based, excluding Marker).
function EZ.HookTrack(tcNo, tgNo, trackNo)
    return EZ.HookCmdSubTrack(tcNo, tgNo, trackNo)
end
function EZ.UnhookTrack(tcNo, tgNo, trackNo)
    return EZ.UnhookCmdSubTrack(tcNo, tgNo, trackNo)
end
function EZ.UnhookAll()
    local count = 0
    for key, hookInfo in pairs(EZ._hooks) do
        if Unhook then
            pcall(function() Unhook(hookInfo.id) end)
        end
        count = count + 1
    end
    EZ._hooks = {}
    EZ.log(string.format("Unhooked all (%d hooks)", count))
    EZ.sendMessage("tracks", "unhooked_all", {count = count})
    return count
end
-- Hook a TrackGroup object for change notifications (track add/delete/reorder)
function EZ.HookTrackGroupChanges(tcNo, tgNo)
    EZ.log(string.format("HookTrackGroupChanges(%s, %s) called", tostring(tcNo), tostring(tgNo)))
    
    local tg = EZ.getTG(tcNo, tgNo)
    if not tg then
        EZ.log(string.format("Track group TC%d.TG%d not found", tcNo, tgNo))
        EZ.sendMessage("trackgroup", "error", {tc = tcNo, tg = tgNo, error = "Track group not found"})
        return false
    end
    
    local key = string.format("%d.%d", tcNo, tgNo)
    if EZ._trackgroup_hooks[key] then
        EZ.UnhookTrackGroupChanges(tcNo, tgNo)
    end
    
    local pluginHandle = getPluginHandle()
    if not pluginHandle then
        EZ.log("TrackGroup hook failed: plugin handle not available")
        return false
    end
    
    local function callback(obj)
        return EZ._onTrackGroupChange(obj)
    end
    
    local hookId = HookObjectChange(callback, tg, pluginHandle)
    if not hookId then
        EZ.log(string.format("TrackGroup hook failed for TC%d.TG%d", tcNo, tgNo))
        return false
    end
    
    EZ._trackgroup_hooks[key] = {
        id = hookId,
        trackgroup = tg,
        tc = tcNo,
        tg = tgNo,
        callback = callback
    }
    
    EZ.log(string.format("TrackGroup hook registered for TC%d.TG%d", tcNo, tgNo))
    EZ.sendMessage("trackgroup", "hooked", {tc = tcNo, tg = tgNo})
    return true
end
function EZ.UnhookTrackGroupChanges(tcNo, tgNo)
    local key = string.format("%d.%d", tcNo, tgNo)
    local hookInfo = EZ._trackgroup_hooks[key]
    if not hookInfo then
        return false
    end
    if Unhook then
        pcall(function() Unhook(hookInfo.id) end)
    end
    EZ._trackgroup_hooks[key] = nil
    EZ.sendMessage("trackgroup", "unhooked", {tc = tcNo, tg = tgNo})
    return true
end
-- Hook a CmdSubTrack for real-time change notifications (RECOMMENDED)
-- Hooks the CmdSubTrack instead of individual events to avoid hook limits
-- NOTE: trackNo is user-visible track number (1-based, excluding Marker).
-- timeRangeIdx: TimeRange index (default: 1)
-- subTrackIdx: SubTrack index (default: nil, finds first CmdSubTrack)
function EZ.HookCmdSubTrack(tcNo, tgNo, trackNo, timeRangeIdx, subTrackIdx)
    EZ.log("=== HookCmdSubTrack CALLED ===")
    EZ.log(string.format("  Args: tc=%s, tg=%s, track=%s, timeRange=%s, subTrack=%s", 
        tostring(tcNo), tostring(tgNo), tostring(trackNo), 
        tostring(timeRangeIdx), tostring(subTrackIdx)))
    local key = string.format("%d.%d.%d", tcNo, tgNo, trackNo)
    -- FAIL LOUD: Require timeRangeIdx - no defaults
    if not timeRangeIdx then
        EZ.log(string.format("  ERROR: timeRangeIdx is required but was nil for %s", key))
        EZ.sendMessage("hooks", "error", {
            action = "hook_failed",
            reason = "timeRangeIdx_required",
            tc = tcNo,
            tg = tgNo,
            track = trackNo
        })
        return false
    end
    local timeRange = timeRangeIdx
    -- CRITICAL FIX: Find CmdSubTrack FIRST to get actual index before creating key
    -- This prevents false "already hooked" when subTrackIdx is nil
    local cmdSubTrack, foundIdx = EZ.getCmdSubTrack(tcNo, tgNo, trackNo, timeRangeIdx, subTrackIdx)
    if not cmdSubTrack then
        EZ.log(string.format("  ERROR: CmdSubTrack not found for %s", key))
        EZ.sendMessage("hooks", "error", {
            action = "hook_failed",
            reason = "cmdsubtrack_not_found",
            tc = tcNo,
            tg = tgNo,
            track = trackNo,
            timeRange = timeRangeIdx,
            subTrackIdx = subTrackIdx
        })
        return false
    end
    -- FAIL LOUD: Resolve subTrackIdx from foundIdx if not provided, but fail if foundIdx is also nil
    if not subTrackIdx then 
        if not foundIdx then
            EZ.log(string.format("  ERROR: subTrackIdx was nil and getCmdSubTrack returned nil foundIdx for %s", key))
            EZ.sendMessage("hooks", "error", {
                action = "hook_failed",
                reason = "subtrackidx_resolution_failed",
                tc = tcNo,
                tg = tgNo,
                track = trackNo,
                timeRange = timeRangeIdx
            })
            return false
        end
        subTrackIdx = foundIdx 
        EZ.log(string.format("  Resolved subTrackIdx: nil -> %d (found)", subTrackIdx))
    end
    -- NOW create the key with the actual subTrackIdx
    local subTrackKey = string.format("%s.TR%d.ST%d", key, timeRange, subTrackIdx)
    EZ.log(string.format("  Using subTrackKey: %s", subTrackKey))
    -- Check if already hooked - but still send initial events!
    -- EchoZero may have restarted and lost its state, so we always need to send events.
    local alreadyHooked = EZ._hooks[subTrackKey] ~= nil
    if alreadyHooked then
        EZ.UnhookCmdSubTrack(tcNo, tgNo, trackNo, timeRangeIdx, subTrackIdx)
        EZ.log(string.format("  Already hooked: %s (re-sending events)", subTrackKey))
    end
    -- Get plugin handle
    local pluginHandle = getPluginHandle()
    if not pluginHandle then
        EZ.log("  ERROR: Plugin handle not available")
        return false
    end
    EZ.log("  Plugin handle: OK")
    EZ.log(string.format("  CmdSubTrack found at index: %d", subTrackIdx))
    -- Register hook (unique callback per hook)
    EZ.log("  Registering hook with HookObjectChange...")
    EZ.log(string.format("    Using resolved subTrackIdx: %d", subTrackIdx))
    local function callback(obj)
        return EZ._onTrackChange(obj)
    end
    local hookId = HookObjectChange(callback, cmdSubTrack, pluginHandle)
    if hookId then
        -- Get the parent track for reference
        local ma3TrackNo = trackNo + 1
        local track = EZ.getTrack(tcNo, tgNo, ma3TrackNo)
        local currentEvents = EZ.getTrackEvents(track)
        -- Store hook info (no state tracking - EchoZero handles all state)
        -- Key now matches the actual subTrackIdx, preventing false "already hooked"
        EZ._hooks[subTrackKey] = {
            id = hookId,
            track = track,
            subtrack = cmdSubTrack,
            tc = tcNo,
            tg = tgNo,
            trackNo = trackNo,
            timeRangeIdx = timeRange,
            subTrackIdx = subTrackIdx,
            callback = callback
        }
        EZ.log(string.format("  SUCCESS: Hook registered"))
        EZ.log(string.format("    Key: %s", subTrackKey))
        EZ.log(string.format("    Initial events: %d", #currentEvents))
        -- Send confirmation WITH initial events to EchoZero
        EZ.sendMessage("subtrack", "hooked", {
            tc = tcNo, 
            tg = tgNo, 
            track = trackNo, 
            timeRange = timeRange, 
            subtrack = subTrackIdx,
            event_count = #currentEvents,
            events = currentEvents
        })
        EZ.sendMessage("hooks", "trace", {
            action = "hook_registered",
            key = subTrackKey,
            tc = tcNo,
            tg = tgNo,
            track = trackNo,
            hook_id = hookId
        })
        EZ.log("=== HookCmdSubTrack COMPLETE ===")
        return true
    else
        EZ.log(string.format("  ERROR: HookObjectChange returned nil for %s", subTrackKey))
        EZ.log("=== HookCmdSubTrack FAILED ===")
        return false
    end
end
-- Unhook a CmdSubTrack by track coordinates
function EZ.UnhookCmdSubTrack(tcNo, tgNo, trackNo, timeRangeIdx, subTrackIdx)
    EZ.log("=== UnhookCmdSubTrack CALLED ===")
    EZ.log(string.format("  Args: tc=%s, tg=%s, track=%s, timeRange=%s, subTrack=%s",
        tostring(tcNo), tostring(tgNo), tostring(trackNo),
        tostring(timeRangeIdx), tostring(subTrackIdx)))
    local key = string.format("%d.%d.%d", tcNo, tgNo, trackNo)
    
    -- FAIL LOUD: Require timeRangeIdx - no defaults
    if not timeRangeIdx then
        EZ.log(string.format("  ERROR: timeRangeIdx is required but was nil for %s", key))
        EZ.sendMessage("hooks", "error", {
            action = "unhook_failed",
            reason = "timeRangeIdx_required",
            tc = tcNo,
            tg = tgNo,
            track = trackNo
        })
        return 0
    end
    local timeRange = timeRangeIdx
    local targetKey = nil
    local prefix = string.format("%s.TR%d.ST", key, timeRange)
    if subTrackIdx ~= nil then
        targetKey = string.format("%s.TR%d.ST%d", key, timeRange, subTrackIdx)
    end

    local unhooked = 0
    for hookKey, hookInfo in pairs(EZ._hooks) do
        local matches = false
        if targetKey then
            matches = hookKey == targetKey
        else
            matches = hookKey:sub(1, #prefix) == prefix
        end
        if matches then
            if Unhook then
                pcall(function() Unhook(hookInfo.id) end)
            end
            EZ._hooks[hookKey] = nil
            unhooked = unhooked + 1
            EZ.sendMessage("hooks", "trace", {
                action = "unhooked",
                key = hookKey,
                tc = hookInfo.tc or tcNo,
                tg = hookInfo.tg or tgNo,
                track = hookInfo.trackNo or trackNo,
                hook_id = hookInfo.id
            })
        end
    end
    EZ.log(string.format("Unhooked %d CmdSubTrack(s) for %s", unhooked, key))
    EZ.sendMessage("subtrack", "unhooked", {tc = tcNo, tg = tgNo, track = trackNo, count = unhooked})
    return unhooked
end
-- Rehook a CmdSubTrack by unhooking first, then hooking
function EZ.RehookCmdSubTrack(tcNo, tgNo, trackNo, timeRangeIdx, subTrackIdx)
    EZ.log("=== RehookCmdSubTrack CALLED ===")
    local removed = EZ.UnhookCmdSubTrack(tcNo, tgNo, trackNo, timeRangeIdx, subTrackIdx)
    local hooked = EZ.HookCmdSubTrack(tcNo, tgNo, trackNo, timeRangeIdx, subTrackIdx)
    EZ.sendMessage("hooks", "trace", {
        action = "rehook_requested",
        tc = tcNo,
        tg = tgNo,
        track = trackNo,
        removed = removed,
        hooked = hooked == true
    })
    return hooked
end
-- List currently hooked tracks
function EZ.ListHooks()
    local hooks = {}
    for key, info in pairs(EZ._hooks) do
        table.insert(hooks, {
            key = key,
            tc = info.tc,
            tg = info.tg,
            track = info.trackNo
        })
    end
    EZ.log(string.format("Active hooks: %d", #hooks))
    for _, h in ipairs(hooks) do
        EZ.log(string.format("  - %s", h.key))
    end
    -- Send to EchoZero
    EZ.sendMessage("hooks", "list", {count = #hooks, hooks = hooks})
    return hooks
end
-- Dump all MA3 hooks (for debugging)
function EZ.DumpHooks()
    if DumpAllHooks then
        EZ.log("Dumping all MA3 hooks to System Monitor...")
        DumpAllHooks()
    else
        EZ.log("DumpAllHooks not available")
    end
end
-- Hook all user-visible tracks in a track group (excludes Marker track at index 1)
-- Returns count of successfully hooked tracks
function EZ.HookTrackGroup(tcNo, tgNo)
    EZ.log(string.format("Hooking all tracks in TC%d.TG%d", tcNo, tgNo))
    local tg = EZ.getTG(tcNo, tgNo)
    if not tg then
        EZ.log(string.format("Track group TC%d.TG%d not found", tcNo, tgNo))
        EZ.sendMessage("trackgroup", "error", {tc = tcNo, tg = tgNo, error = "Track group not found"})
        return 0
    end
    local hooked = 0
    local children_ok, children = pcall(function() return tg:Children() end)
    if children_ok and children then
        -- Track index 1 is reserved for Marker, user tracks start at index 2
        -- trackNo is user-visible track number (1-based, excluding Marker)
        local trackNo = 0
        for i = 1, #children do
            local track = children[i]
            if track and track.name ~= "Marker" then
                trackNo = trackNo + 1  -- User-visible track number (1-based)
                if EZ.HookTrack(tcNo, tgNo, trackNo) then
                    hooked = hooked + 1
                end
            end
        end
    end
    
    EZ.log(string.format("Hooked %d tracks in TC%d.TG%d", hooked, tcNo, tgNo))
    EZ.sendMessage("trackgroup", "hooked", {tc = tcNo, tg = tgNo, count = hooked})
    return hooked
end
-- Unhook all tracks in a track group
function EZ.UnhookTrackGroup(tcNo, tgNo)
    EZ.log(string.format("Unhooking all tracks in TC%d.TG%d", tcNo, tgNo))
    local prefix = string.format("%d.%d.", tcNo, tgNo)
    local unhooked = 0
    for key, hookInfo in pairs(EZ._hooks) do
        if key:sub(1, #prefix) == prefix then
            if Unhook then
                pcall(function() Unhook(hookInfo.id) end)
            end
            EZ._hooks[key] = nil
            unhooked = unhooked + 1
        end
    end
    EZ.log(string.format("Unhooked %d tracks in TC%d.TG%d", unhooked, tcNo, tgNo))
    EZ.sendMessage("trackgroup", "unhooked", {tc = tcNo, tg = tgNo, count = unhooked})
    return unhooked
end
return EZ
