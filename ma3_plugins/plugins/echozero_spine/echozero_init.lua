-- EchoZero Plugin Initialization
-- This file MUST be loaded LAST. It initializes OSC, verifies all modules,
-- and reports status to the MA3 console.
-- The return init() at the end triggers execution on plugin engine load.

EZ = EZ or {}

local function init()
    Printf("==============================")
    Printf("[EZ] EchoZero v%s", EZ._version or "?")
    Printf("==============================")

    -- Initialize OSC socket
    if OSC and OSC.setConfig and OSC.init then
        OSC.setConfig(EZ.config)
        local socketOk = OSC.init()
        EZ._oscInitialized = true
        if socketOk then
            Printf("[EZ] OSC socket: OK (%s:%d)", EZ.config.ip, EZ.config.port)
        else
            Printf("[EZ] OSC socket: FAILED (socket.core not available)")
        end
    else
        Printf("[EZ] OSC module: NOT LOADED")
    end

    -- Verify core modules
    local modules = {
        {"EZ.log",           EZ.log},
        {"EZ.dbg",           EZ.dbg},
        {"EZ.sendMessage",   EZ.sendMessage},
        {"EZ.getDP",         EZ.getDP},
        {"EZ.getTC",         EZ.getTC},
        {"EZ.getTG",         EZ.getTG},
        {"EZ.getTrack",      EZ.getTrack},
        {"EZ.getCmdSubTrack",EZ.getCmdSubTrack},
        {"EZ.getTrackEvents",EZ.getTrackEvents},
    }

    local missing = 0
    for _, m in ipairs(modules) do
        if type(m[2]) ~= "function" then
            Printf("[EZ]   MISSING: %s", m[1])
            missing = missing + 1
        end
    end

    if missing == 0 then
        Printf("[EZ] Core functions: OK (%d loaded)", #modules)
    else
        Printf("[EZ] Core functions: %d/%d loaded (%d MISSING)", #modules - missing, #modules, missing)
    end

    -- Verify API modules
    local api = {
        {"EZ.GetTimecodes",  EZ.GetTimecodes},
        {"EZ.GetTrackGroups",EZ.GetTrackGroups},
        {"EZ.GetTracks",     EZ.GetTracks},
        {"EZ.GetEvents",     EZ.GetEvents},
        {"EZ.GetSequences",  EZ.GetSequences},
        {"EZ.HookTrack",     EZ.HookTrack},
        {"EZ.Ping",          EZ.Ping},
    }

    local apiMissing = 0
    for _, m in ipairs(api) do
        if type(m[2]) ~= "function" then
            Printf("[EZ]   MISSING: %s", m[1])
            apiMissing = apiMissing + 1
        end
    end

    if apiMissing == 0 then
        Printf("[EZ] API functions: OK (%d loaded)", #api)
    else
        Printf("[EZ] API functions: %d/%d loaded (%d MISSING)", #api - apiMissing, #api, apiMissing)
    end

    -- Report config
    Printf("[EZ] Target: %s:%d", EZ.config.ip, EZ.config.port)
    Printf("[EZ] Debug: %s", EZ.config.debug and "ON" or "OFF")
    Printf("[EZ] Cmd mode: %s", EZ.config.defaultCmdMode or "feedback")

    -- Send startup message to EchoZero
    EZ.sendMessage("debug", "plugin_version", {
        version = EZ._version,
        osc_ready = OSC and OSC.isReady and OSC.isReady() or false
    })

    Printf("==============================")
    Printf("[EZ] Ready")
    Printf("==============================")

    return EZ
end

return init()
