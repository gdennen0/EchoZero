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
OSC._send_sequence = 0
OSC._send_ok_count = 0
OSC._send_fail_count = 0
OSC._last_send_ok_at = nil
OSC._last_send_error_at = nil
OSC._last_send_error = nil
OSC._last_send_step = nil

-- LOGGING
local function log(msg) Printf("[EZ-OSC] %s", msg) end
local function dbg(msg) if OSC.config.debug then Printf("[EZ-OSC DBG] %s", msg) end end

local function recordSendOutcome(ok, step, detail, byteCount)
    OSC._send_sequence = (OSC._send_sequence or 0) + 1
    OSC._last_send_step = tostring(step or "")
    if byteCount then OSC._last_send_bytes = tonumber(byteCount) or OSC._last_send_bytes end
    if ok then
        OSC._send_ok_count = (OSC._send_ok_count or 0) + 1
        OSC._last_send_ok_at = os.time()
        OSC._last_send_error = nil
    else
        OSC._send_fail_count = (OSC._send_fail_count or 0) + 1
        OSC._last_send_error_at = os.time()
        OSC._last_send_error = tostring(detail or "unknown send error")
    end
end

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
        local peer_ok, peer_err = udp:setpeername(OSC.config.ip, OSC.config.port)
        if not peer_ok then
            error("setpeername failed: " .. tostring(peer_err))
        end
        step = "send"
        local sent_ok, sent_err = udp:send(data)
        if not sent_ok then
            error("send failed: " .. tostring(sent_err))
        end
        OSC._last_send_bytes = tonumber(sent_ok) or #data
        step = "close"
        udp:close()
    end)
    if not send_ok then
        log(string.format(">>> ERROR: sendOSC FAILED at '%s': %s", step, tostring(send_err)))
        if udp then pcall(function() udp:close() end) end
    end
    recordSendOutcome(send_ok, step, send_err, OSC._last_send_bytes or #data)
    return send_ok
end

local function messageByteLimit()
    return math.max(512, tonumber(OSC.config.maxMessageBytes or OSC.config.max_message_bytes or 6000) or 6000)
end

local function sendRawPipeMessage(msg)
    return OSC.sendOSC("/ez/message", "s", msg)
end

function OSC.sendChunkedMessage(msgType, changeType, msg, maxBytes)
    local limit = math.max(512, tonumber(maxBytes) or messageByteLimit())
    local chunkSize = math.max(256, limit - 512)
    OSC._next_chunk_id = (OSC._next_chunk_id or 0) + 1
    local chunkId = table.concat({tostring(os.time()), tostring(OSC._next_chunk_id), tostring(msgType or "unknown"), tostring(changeType or "unknown")}, "-")
    local totalChunks = math.ceil(#msg / chunkSize)
    local allOk = true
    for chunkIndex = 1, totalChunks do
        local startIndex = ((chunkIndex - 1) * chunkSize) + 1
        local chunkText = msg:sub(startIndex, startIndex + chunkSize - 1)
        local chunkPayload = table.concat({
            "type=osc_chunk",
            "change=part",
            "timestamp=" .. os.time(),
            "chunk_id=" .. chunkId,
            "chunk_index=" .. tostring(chunkIndex),
            "total_chunks=" .. tostring(totalChunks),
            "original_type=" .. tostring(msgType or "unknown"),
            "original_change=" .. tostring(changeType or "unknown"),
            "payload=" .. OSC.jsonEncode({text = chunkText})
        }, "|")
        local ok = sendRawPipeMessage(chunkPayload)
        allOk = allOk and ok
    end
    return allOk
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
    local ok = true
    if #msg > messageByteLimit() then
        ok = OSC.sendChunkedMessage(msgType, changeType, msg, messageByteLimit())
    else
        ok = sendRawPipeMessage(msg)
    end
    -- #region agent log
    -- Emit lightweight debug without recursion (use sendOSC directly)
    if msgType == "track" and changeType == "changed" then
        local dbg = string.format(
            "type=debug|change=osc_send|timestamp=%s|msg_type=%s|msg_change=%s|len=%d|ok=%s|chunked=%s",
            tostring(os.time()),
            tostring(msgType),
            tostring(changeType),
            tonumber(#msg) or 0,
            tostring(ok),
            tostring(#msg > messageByteLimit())
        )
        sendRawPipeMessage(dbg)
    end
    -- #endregion
    return ok
end

function OSC.connectionReportFields()
    return {
        socket_ok = OSC._socketOk and true or false,
        target_ip = tostring(OSC.config.ip or ""),
        target_port = tonumber(OSC.config.port) or 0,
        send_sequence = tonumber(OSC._send_sequence or 0) or 0,
        send_ok_count = tonumber(OSC._send_ok_count or 0) or 0,
        send_fail_count = tonumber(OSC._send_fail_count or 0) or 0,
        last_send_ok_at = OSC._last_send_ok_at,
        last_send_error_at = OSC._last_send_error_at,
        last_send_error = OSC._last_send_error,
        last_send_step = OSC._last_send_step,
        last_send_type = OSC._last_send_type,
        last_send_change = OSC._last_send_change,
        last_send_len = tonumber(OSC._last_send_len or 0) or 0,
        last_send_bytes = tonumber(OSC._last_send_bytes or 0) or 0,
    }
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
    if config.maxMessageBytes then OSC.config.maxMessageBytes = config.maxMessageBytes end
    if config.max_message_bytes then OSC.config.maxMessageBytes = config.max_message_bytes end
end

return OSC
