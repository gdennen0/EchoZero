--[[
EchoZero grandMA3 Autosave Showfile Plugin
=========================================

Periodically saves the current showfile under a timestamped name.

Load:
  Lua "dofile('/path/to/MA3/plugins/TC22/autosave_showfile.lua')"

Use:
  EZ_AutosaveShow.Start(10)                 -- every 10 minutes
  EZ_AutosaveShow.Start(5, "TourBackup")    -- every 5 minutes, custom prefix
  EZ_AutosaveShow.SaveNow()                 -- immediate timestamped save
  EZ_AutosaveShow.Stop()                    -- stop scheduling new saves
  EZ_AutosaveShow.Status()                  -- print current state

Notes:
- Timer cancellation in MA3 Lua is limited, so Stop() disables the active generation.
  Any already-scheduled callback will wake once and then exit without saving.
- Default command is intentionally centralized in config.commandTemplate so it can be
  adapted if a specific MA3 software version/site uses a different SaveShow syntax.
]]--

EZ_AutosaveShow = EZ_AutosaveShow or {}

EZ_AutosaveShow.config = EZ_AutosaveShow.config or {
    intervalMinutes = 10,
    prefix = "Autosave",
    timestampFormat = "%Y-%m-%d_%H-%M-%S",
    -- MA3 command template. One %s receives the escaped timestamped showfile name.
    commandTemplate = 'SaveShow "%s" /nc',
    useIndirectWait = true,
    debug = false,
}

EZ_AutosaveShow._enabled = EZ_AutosaveShow._enabled or false
EZ_AutosaveShow._generation = EZ_AutosaveShow._generation or 0
EZ_AutosaveShow._lastFilename = EZ_AutosaveShow._lastFilename or nil
EZ_AutosaveShow._lastResult = EZ_AutosaveShow._lastResult or nil
EZ_AutosaveShow._lastSavedAt = EZ_AutosaveShow._lastSavedAt or nil

local function log(message)
    if Printf then
        Printf("[EZ AutosaveShow] %s", tostring(message))
    elseif Echo then
        Echo("[EZ AutosaveShow] " .. tostring(message))
    end
end

local function debug_log(message)
    if EZ_AutosaveShow.config.debug then
        log(message)
    end
end

local function clamp_interval_minutes(value)
    local minutes = tonumber(value) or EZ_AutosaveShow.config.intervalMinutes or 10
    if minutes < 1 then minutes = 1 end
    return minutes
end

local function sanitize_filename(value)
    local name = tostring(value or "Autosave")
    -- Conservative set that works across MA3/macOS/Windows filesystems.
    name = name:gsub('[\\/:*?"<>|]', "-")
    name = name:gsub("%s+", "_")
    name = name:gsub("_+", "_")
    name = name:gsub("^-+", "")
    name = name:gsub("-+$", "")
    name = name:gsub("^_+", "")
    name = name:gsub("_+$", "")
    if name == "" then name = "Autosave" end
    return name
end

local function escape_cmd_string(value)
    return tostring(value or ""):gsub('"', '\\"')
end

function EZ_AutosaveShow.BuildFilename(prefix, timestamp)
    local resolvedPrefix = sanitize_filename(prefix or EZ_AutosaveShow.config.prefix or "Autosave")
    local resolvedTimestamp = sanitize_filename(timestamp or os.date(EZ_AutosaveShow.config.timestampFormat))
    return string.format("%s_%s", resolvedPrefix, resolvedTimestamp)
end

function EZ_AutosaveShow.BuildCommand(filename)
    local safeFilename = escape_cmd_string(sanitize_filename(filename))
    local template = EZ_AutosaveShow.config.commandTemplate or 'SaveShow "%s" /nc'
    return string.format(template, safeFilename)
end

local function run_command(command)
    debug_log("Executing: " .. tostring(command))
    if EZ_AutosaveShow.config.useIndirectWait and CmdIndirectWait then
        return pcall(function() return CmdIndirectWait(command) end)
    end
    if Cmd then
        return pcall(function() return Cmd(command) end)
    end
    return false, "Cmd/CmdIndirectWait unavailable"
end

function EZ_AutosaveShow.SaveNow(prefix)
    local filename = EZ_AutosaveShow.BuildFilename(prefix or EZ_AutosaveShow.config.prefix)
    local command = EZ_AutosaveShow.BuildCommand(filename)
    local ok, result = run_command(command)
    EZ_AutosaveShow._lastFilename = filename
    EZ_AutosaveShow._lastSavedAt = os.date("%Y-%m-%d %H:%M:%S")
    EZ_AutosaveShow._lastResult = ok and (result or "Ok") or tostring(result)
    if ok then
        log("Saved showfile as " .. filename)
        return true, filename, result
    end
    log("Save failed for " .. filename .. ": " .. tostring(result))
    return false, filename, result
end

local function schedule_next(generation)
    if not EZ_AutosaveShow._enabled or generation ~= EZ_AutosaveShow._generation then
        return
    end
    if not Timer then
        log("Timer unavailable; autosave cannot continue. Use EZ_AutosaveShow.SaveNow() manually.")
        EZ_AutosaveShow._enabled = false
        return
    end
    local seconds = math.floor(clamp_interval_minutes(EZ_AutosaveShow.config.intervalMinutes) * 60)
    Timer(function()
        if not EZ_AutosaveShow._enabled or generation ~= EZ_AutosaveShow._generation then
            debug_log("Ignoring stale timer generation " .. tostring(generation))
            return
        end
        EZ_AutosaveShow.SaveNow()
        schedule_next(generation)
    end, seconds, 1)
    debug_log("Next autosave scheduled in " .. tostring(seconds) .. " seconds")
end

function EZ_AutosaveShow.Start(intervalMinutes, prefix)
    EZ_AutosaveShow.config.intervalMinutes = clamp_interval_minutes(intervalMinutes or EZ_AutosaveShow.config.intervalMinutes)
    if prefix ~= nil and tostring(prefix) ~= "" then
        EZ_AutosaveShow.config.prefix = sanitize_filename(prefix)
    end
    EZ_AutosaveShow._generation = EZ_AutosaveShow._generation + 1
    EZ_AutosaveShow._enabled = true
    log(string.format(
        "Starting autosave every %d minute(s), prefix '%s'",
        EZ_AutosaveShow.config.intervalMinutes,
        EZ_AutosaveShow.config.prefix
    ))
    schedule_next(EZ_AutosaveShow._generation)
    return true
end

function EZ_AutosaveShow.Stop()
    EZ_AutosaveShow._enabled = false
    EZ_AutosaveShow._generation = EZ_AutosaveShow._generation + 1
    log("Stopped autosave")
    return true
end

function EZ_AutosaveShow.Status()
    log(string.format(
        "enabled=%s interval=%s minute(s) prefix='%s' last='%s' result='%s'",
        tostring(EZ_AutosaveShow._enabled),
        tostring(EZ_AutosaveShow.config.intervalMinutes),
        tostring(EZ_AutosaveShow.config.prefix),
        tostring(EZ_AutosaveShow._lastFilename or "none"),
        tostring(EZ_AutosaveShow._lastResult or "none")
    ))
    return {
        enabled = EZ_AutosaveShow._enabled,
        intervalMinutes = EZ_AutosaveShow.config.intervalMinutes,
        prefix = EZ_AutosaveShow.config.prefix,
        lastFilename = EZ_AutosaveShow._lastFilename,
        lastSavedAt = EZ_AutosaveShow._lastSavedAt,
        lastResult = EZ_AutosaveShow._lastResult,
    }
end

return EZ_AutosaveShow
