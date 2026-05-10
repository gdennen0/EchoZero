ProgrammerMatricksPopup = ProgrammerMatricksPopup or {}
ProgrammerMatricksPopup._version = ProgrammerMatricksPopup._version or "0.1.0"
ProgrammerMatricksPopup._build = ProgrammerMatricksPopup._build or "2026-05-06.quick-popup-1"

local function _trim(value)
    return tostring(value or ""):match("^%s*(.-)%s*$")
end

local function _show_message(title, message)
    if type(MessageBox) == "function" then
        MessageBox({
            title = title or "Programmer MAtricks Popup",
            message = message or "",
            commands = {
                {value = 1, name = "OK"},
            },
        })
        return
    end
    Printf("[%s] %s", tostring(title or "Programmer MAtricks Popup"), tostring(message or ""))
end

local function _programmer_part()
    local part = nil
    local ok_part = pcall(function() part = ProgrammerPart() end)
    if ok_part and part then
        return part
    end

    local prog = nil
    local ok_prog = pcall(function() prog = Programmer() end)
    if ok_prog and prog and prog[1] then
        return prog[1]
    end
    return nil
end

local function _programmer_lines(part)
    local children = nil
    local ok_children = pcall(function() children = part:Children() end)
    if not ok_children or not children then
        return {}
    end

    local lines = {}
    for index = 1, #children do
        local child = children[index]
        if child then
            local name = _trim(child.name)
            if name == "" then
                name = "Programmer 0." .. tostring(index)
            end
            lines[#lines + 1] = {
                index = index,
                name = name,
            }
        end
    end
    return lines
end

local function _selector_values(options)
    local values = {}
    for _, option in ipairs(options or {}) do
        values[option.label] = option.id
    end
    return values
end

local function _selector_choice(return_table, selector_name, options)
    if not return_table or not return_table.selectors then
        return nil
    end
    local selected_id = return_table.selectors[selector_name]
    for _, option in ipairs(options or {}) do
        if option.id == selected_id then
            return option
        end
    end
    return nil
end

local function _state_value(return_table, state_name)
    if not return_table or not return_table.states then
        return false
    end
    return return_table.states[state_name] == true
end

local function _apply_property(line_index, property_name, value_text)
    local command = string.format(
        'Set Programmer 0.%d Property "%s" "%s"',
        tonumber(line_index) or 0,
        tostring(property_name or ""),
        tostring(value_text or "")
    )
    Printf("[ProgrammerMatricksPopup] %s", command)
    return Cmd(command)
end

local function _phase_values(option_key, flip_direction)
    local values = {
        ["0-360"] = {from_value = "0", to_value = "360"},
        ["0-180"] = {from_value = "0", to_value = "180"},
        ["0-90"] = {from_value = "0", to_value = "90"},
    }
    local chosen = values[option_key]
    if not chosen then
        return nil, nil
    end
    if flip_direction then
        return chosen.to_value, chosen.from_value
    end
    return chosen.from_value, chosen.to_value
end

function ProgrammerMatricksPopup.Run()
    local part = _programmer_part()
    if not part then
        _show_message("Programmer MAtricks Popup", "Could not access Programmer Part 0.")
        return nil
    end

    local lines = _programmer_lines(part)
    if #lines == 0 then
        _show_message(
            "Programmer MAtricks Popup",
            "Programmer Part 0 has no child lines yet.\n\nCopy or create programmer content first, then run this popup."
        )
        return nil
    end

    local target_options = {
        {id = 0, label = "All programmer lines", value = "all"},
    }
    for _, line in ipairs(lines) do
        target_options[#target_options + 1] = {
            id = line.index,
            label = string.format("Programmer 0.%d - %s", line.index, line.name),
            value = line.index,
        }
    end

    local xwings_options = {
        {id = 0, label = "Keep current XWings", value = nil},
        {id = 2, label = "XWings 2", value = "2"},
        {id = 4, label = "XWings 4", value = "4"},
        {id = 6, label = "XWings 6", value = "6"},
    }

    local speed_options = {
        {id = 0, label = "Keep current SpeedFromX", value = nil},
        {id = 15, label = "SpeedFromX 15", value = "15"},
        {id = 30, label = "SpeedFromX 30", value = "30"},
        {id = 60, label = "SpeedFromX 60", value = "60"},
        {id = 120, label = "SpeedFromX 120", value = "120"},
    }

    local phase_options = {
        {id = 0, label = "Keep current X phase", value = nil},
        {id = 1, label = "Phase 0 -> 360", value = "0-360"},
        {id = 2, label = "Phase 0 -> 180", value = "0-180"},
        {id = 3, label = "Phase 0 -> 90", value = "0-90"},
    }

    local result = MessageBox({
        title = "Programmer MAtricks Popup",
        message = "Quick presets for programmer MAtricks-style line properties.",
        selectors = {
            {
                name = "Target",
                selectedValue = 0,
                type = 1,
                values = _selector_values(target_options),
            },
            {
                name = "X Wings",
                selectedValue = 0,
                type = 1,
                values = _selector_values(xwings_options),
            },
            {
                name = "Speed From X",
                selectedValue = 0,
                type = 1,
                values = _selector_values(speed_options),
            },
            {
                name = "X Phase Pair",
                selectedValue = 0,
                type = 1,
                values = _selector_values(phase_options),
            },
        },
        states = {
            {name = "Flip Phase Direction", state = false, group = 1},
        },
        commands = {
            {value = 1, name = "Apply"},
            {value = 0, name = "Cancel"},
        },
    })

    if not result or not result.success or result.result ~= 1 then
        Printf("[ProgrammerMatricksPopup] Cancelled")
        return nil
    end

    local target_choice = _selector_choice(result, "Target", target_options)
    local xwings_choice = _selector_choice(result, "X Wings", xwings_options)
    local speed_choice = _selector_choice(result, "Speed From X", speed_options)
    local phase_choice = _selector_choice(result, "X Phase Pair", phase_options)
    local flip_phase = _state_value(result, "Flip Phase Direction")

    local target_lines = {}
    if target_choice and target_choice.value == "all" then
        for _, line in ipairs(lines) do
            target_lines[#target_lines + 1] = line.index
        end
    elseif target_choice and type(target_choice.value) == "number" then
        target_lines[#target_lines + 1] = target_choice.value
    end

    if #target_lines == 0 then
        _show_message("Programmer MAtricks Popup", "No programmer line target was resolved.")
        return nil
    end

    if (not xwings_choice or xwings_choice.value == nil)
        and (not speed_choice or speed_choice.value == nil)
        and (not phase_choice or phase_choice.value == nil) then
        _show_message("Programmer MAtricks Popup", "Nothing selected to apply.")
        return nil
    end

    local applied_changes = {}
    for _, line_index in ipairs(target_lines) do
        if xwings_choice and xwings_choice.value ~= nil then
            _apply_property(line_index, "XWings", xwings_choice.value)
            applied_changes[#applied_changes + 1] = string.format("0.%d XWings=%s", line_index, xwings_choice.value)
        end
        if speed_choice and speed_choice.value ~= nil then
            _apply_property(line_index, "SpeedFromX", speed_choice.value)
            applied_changes[#applied_changes + 1] = string.format("0.%d SpeedFromX=%s", line_index, speed_choice.value)
        end
        if phase_choice and phase_choice.value ~= nil then
            local from_value, to_value = _phase_values(phase_choice.value, flip_phase)
            if from_value ~= nil and to_value ~= nil then
                _apply_property(line_index, "PhaseFromX", from_value)
                _apply_property(line_index, "PhaseToX", to_value)
                applied_changes[#applied_changes + 1] = string.format(
                    "0.%d PhaseFromX=%s PhaseToX=%s",
                    line_index,
                    from_value,
                    to_value
                )
            end
        end
    end

    _show_message(
        "Programmer MAtricks Popup",
        "Applied:\n- " .. table.concat(applied_changes, "\n- ")
    )

    return {
        target_lines = target_lines,
        xwings = xwings_choice and xwings_choice.value or nil,
        speed_from_x = speed_choice and speed_choice.value or nil,
        phase_pair = phase_choice and phase_choice.value or nil,
        flip_phase = flip_phase,
        applied_changes = applied_changes,
    }
end

return ProgrammerMatricksPopup.Run
