-- Test plugin for hooking into a specific timecode object
-- Hooks into DataPool().Timecodes[101][1]

-- Extract plugin parameters (MA3 standard pattern)
local pluginName = select(1, ...)
local componentName = select(2, ...)
local signalTable = select(3, ...)
local luaComponentHandle = select(4, ...)

-- Store hook ID for cleanup
local timecode_hook_id = nil

-- Callback function when timecode changes
local function on_timecode_change(obj)
    local tc_name = obj.name or "Unknown"
    local tc_no = obj.no or "?"
    local tc_index = obj.index or "?"
    
    -- Print detailed information about the change
    Printf('Timecode changed: ' .. tc_name .. ' (No: ' .. tc_no .. ', Index: ' .. tc_index .. ')')
    Printf('Timecode object: ' .. tostring(obj))
    
    Echo('*** Timecode changed: ' .. tc_name .. ' (No: ' .. tc_no .. ', Index: ' .. tc_index .. ') ***')
    
    -- You can add more logic here to handle the change
    -- For example, send to EchoZero, update other objects, etc.
end

-- Function to start the hook
function StartTimecodeHook()
    -- Get plugin handle (required for HookObjectChange)
    local pluginHandle = luaComponentHandle:Parent()
    
    -- Get the timecode object
    local timecode = DataPool().Timecodes[101][1]
    
    if not timecode then
        Echo('*** Error: Timecode 101[1] not found ***')
        return false
    end
    
    -- Create the hook
    timecode_hook_id = HookObjectChange(on_timecode_change, timecode, pluginHandle)
    
    if timecode_hook_id then
        Echo('*** Hooked into Timecode 101[1] (Hook ID: ' .. timecode_hook_id .. ') ***')
        Printf('Timecode hook active')
        return true
    else
        Echo('*** Error: Failed to create timecode hook ***')
        return false
    end
end

-- Function to stop the hook
function StopTimecodeHook()
    if timecode_hook_id then
        Unhook(timecode_hook_id)
        Echo('*** Unhooked Timecode 101[1] (Hook ID: ' .. timecode_hook_id .. ') ***')
        timecode_hook_id = nil
        return true
    else
        Echo('*** No hook to stop ***')
        return false
    end
end

-- Main function (called when plugin loads)
return function()
    Echo('Timecode Hook Test Plugin loaded')
    Echo('Use StartTimecodeHook() to begin monitoring Timecode 101[1]')
    Echo('Use StopTimecodeHook() to stop monitoring')
    
    -- Optional: Auto-start the hook when plugin loads
    -- Uncomment the line below to automatically start monitoring
    -- StartTimecodeHook()
end

