-- EchoZero OSC Communication Module
-- Handles socket initialization, JSON encoding, and OSC message sending
-- This module is loaded by echozero.lua

OSC = {}

-- Configuration (will be overridden by EZ.config)
OSC.config = {
    ip = "127.0.0.1",
    port = 9000,
    debug = false
}

OSC._socket = nil
OSC._socketOk = false

-- LOGGING
local function log(msg) Printf("[EZ-OSC] %s", msg) end
local function dbg(msg) if OSC.config.debug then Printf("[EZ-OSC DBG] %s", msg) end end

-- SOCKET INITIALIZATION
function OSC.init()
    OSC._socketWarnShown = false
    Printf("[EZ-OSC] Attempting to load socket.core...")
    
    local ok, result = pcall(function()
        return require("socket.core")
    end)
    
    if ok and result then
        OSC._socket = result
        OSC._socketOk = true
        Printf("[EZ-OSC] socket.core loaded successfully")
        if not OSC._socket.udp then 
            Printf("[EZ-OSC] WARNING: udp() not found")
            OSC._socketOk = false 
        end
    else
        OSC._socket = nil
        OSC._socketOk = false
        Printf("[EZ-OSC] ERROR: socket.core failed: %s", tostring(result))
    end
    
    if OSC._socketOk then
        local test_ok, test_err = pcall(function()
            local test_udp = OSC._socket.udp()
            if test_udp then 
                test_udp:close() 
            else 
                OSC._socketOk = false 
            end
        end)
        if not test_ok then 
            Printf("[EZ-OSC] UDP socket test failed: %s", tostring(test_err))
            OSC._socketOk = false 
        end
    end
    
    return OSC._socketOk
end

-- JSON ENCODER (minimal, no external deps)
function OSC.jsonEncode(val)
    local t = type(val)
    if t == "nil" then
        return "null"
    elseif t == "boolean" then
        return val and "true" or "false"
    elseif t == "number" then
        return tostring(val)
    elseif t == "string" then
        return '"' .. val:gsub('\\', '\\\\'):gsub('"', '\\"'):gsub('\n', '\\n'):gsub('\r', '\\r') .. '"'
    elseif t == "table" then
        -- Check if array (consecutive integer keys starting at 1)
        local isArray = true
        local n = 0
        for k, v in pairs(val) do
            n = n + 1
            if type(k) ~= "number" or k ~= n then
                isArray = false
                break
            end
        end
        
        local parts = {}
        if isArray then
            for i, v in ipairs(val) do
                table.insert(parts, OSC.jsonEncode(v))
            end
            return "[" .. table.concat(parts, ",") .. "]"
        else
            for k, v in pairs(val) do
                table.insert(parts, '"' .. tostring(k) .. '":' .. OSC.jsonEncode(v))
            end
            return "{" .. table.concat(parts, ",") .. "}"
        end
    else
        return '"[unsupported:' .. t .. ']"'
    end
end

-- OSC HELPERS
local function oscPad(s)
    local p = s .. "\0"
    return p .. string.rep("\0", (4 - #p % 4) % 4)
end

local function packInt(n)
    n = math.floor(n or 0)
    return string.char(
        math.floor(n / 16777216) % 256,
        math.floor(n / 65536) % 256,
        math.floor(n / 256) % 256,
        n % 256
    )
end

local function packFloat(f)
    f = f or 0
    if f == 0 then return "\0\0\0\0" end
    local sign = 0
    if f < 0 then sign = 1; f = -f end
    local mantissa, exponent = math.frexp(f)
    exponent = exponent + 126
    mantissa = (mantissa * 2 - 1) * 8388608
    local b3 = sign * 128 + math.floor(exponent / 2)
    local b2 = (exponent % 2) * 128 + math.floor(mantissa / 65536)
    local b1 = math.floor(mantissa / 256) % 256
    local b0 = math.floor(mantissa) % 256
    return string.char(b3, b2, b1, b0)
end

-- OSC SEND
OSC._socketWarnShown = false
function OSC.sendOSC(addr, types, ...)
    if not OSC._socketOk then
        if not OSC._socketWarnShown then
            log("WARNING: Cannot send OSC - socket.core not available (further warnings suppressed)")
            OSC._socketWarnShown = true
        end
        return false 
    end
    if not OSC._socket then 
        log("ERROR: _socket is nil despite _socketOk=true")
        return false 
    end
    
    local args = {...}
    local typeTag = "," .. (types or "")
    local data = oscPad(addr) .. oscPad(typeTag)
    
    local i = 1
    for c in types:gmatch(".") do
        local v = args[i]
        if c == "i" then
            data = data .. packInt(v or 0)
        elseif c == "f" then
            data = data .. packFloat(v or 0)
        elseif c == "s" then
            data = data .. oscPad(tostring(v or ""))
        end
        i = i + 1
    end
    
    local udp, step = nil, "create"
    local send_ok, send_err = pcall(function()
        udp = OSC._socket.udp()
        if not udp then error("udp() returned nil") end
        step = "setpeername"
        udp:setpeername(OSC.config.ip, OSC.config.port)
        step = "send"
        udp:send(data)
        OSC._last_send_bytes = #data
        step = "close"
        udp:close()
    end)
    if not send_ok then
        log(string.format(">>> ERROR: sendOSC FAILED at '%s': %s", step, tostring(send_err)))
        if udp then pcall(function() udp:close() end) end
    end
    return send_ok
end

-- Send pipe-delimited message (format EchoZero expects)
function OSC.sendMessage(msgType, changeType, data)
    local parts = {
        "type=" .. (msgType or "unknown"), 
        "change=" .. (changeType or "unknown"), 
        "timestamp=" .. os.time()
    }
    if data then
        for k, v in pairs(data) do
            table.insert(parts, k .. "=" .. (type(v) == "table" and OSC.jsonEncode(v) or tostring(v)))
        end
    end
    local msg = table.concat(parts, "|")
    OSC._last_send_len = #msg
    OSC._last_send_type = msgType
    OSC._last_send_change = changeType
    local ok = OSC.sendOSC("/ez/message", "s", msg)
    -- #region agent log
    -- Emit lightweight debug without recursion (use sendOSC directly)
    if msgType == "track" and changeType == "changed" then
        local dbg = string.format(
            "type=debug|change=osc_send|timestamp=%s|msg_type=%s|msg_change=%s|len=%d|ok=%s",
            tostring(os.time()),
            tostring(msgType),
            tostring(changeType),
            tonumber(#msg) or 0,
            tostring(ok)
        )
        OSC.sendOSC("/ez/message", "s", dbg)
    end
    -- #endregion
    return ok
end

-- Check if OSC is ready
function OSC.isReady()
    return OSC._socketOk
end

-- Update config
function OSC.setConfig(config)
    if config.ip then OSC.config.ip = config.ip end
    if config.port then OSC.config.port = config.port end
    if config.debug ~= nil then OSC.config.debug = config.debug end
end

return OSC
