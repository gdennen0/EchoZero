HitMaker = {}
HitMaker._version = HitMaker._version or "1.1.0"
HitMaker._build = HitMaker._build or "2026-04-30.hitmaker-x-hit-release-5"

function HitMaker.get_version_info()
    return {
        version = tostring(HitMaker._version or ""),
        build = tostring(HitMaker._build or ""),
        supports_event_type_create = type(HitMaker.create_sequence_for_event_type) == "function",
        supports_event_type_batch = type(HitMaker.create_event_type_sequences) == "function",
        supports_go_hit = type(HitMaker.go_hit) == "function",
        supports_registry = type(HitMaker.register_sequence_type) == "function",
        supports_x_hit_release = type(HitMaker.x_hit_release) == "function",
    }
end

-- Keyword argument helper functions
function HitMaker.merge_settings_with_defaults(defaults, provided)
    local merged = {}
    
    -- Start with defaults
    for key, value in pairs(defaults) do
        merged[key] = value
    end
    
    -- Override with provided values
    if provided then
        for key, value in pairs(provided) do
            if value ~= nil then
                merged[key] = value
            end
        end
    end
    
    return merged
end

function HitMaker.validate_required_args(args, required_fields)
    for _, field in ipairs(required_fields) do
        if not args[field] or args[field] == "" then
            HitMakerUI.ShowError("Missing Required Field", "Required field '" .. field .. "' is missing or empty")
            return false
        end
    end
    return true
end

function HitMaker.get_default_sequence_settings()
    return {
        name = "",
        autoAssign = true,
        tapOnExec = false,
        doNotAssign = false,
        clearFirst = true
    }
end

function HitMaker.get_default_go_hit_settings()
    return {
        name = "",
        offtime = "1",
        autoAssign = true,
        tapOnExec = false,
        doNotAssign = false,
        clearFirst = true,
        followTrigger = true,
        timingType = 1  -- BPM
    }
end

function HitMaker.get_default_four_hit_settings()
    return {
        name = "",
        baseFadeTime = "1",
        autoAssign = true,
        tapOnExec = false,
        doNotAssign = false,
        clearFirst = true,
        createIndividual = true,
        setRestartMode = true,
        matrixType = 1,  -- 4 Hit Standard
        fadeTimeType = 1  -- BPM
    }
end

function HitMaker.get_default_x_hit_release_settings()
    return {
        name = "",
        cueCount = 8,
        releaseFade = "1",
        orderMode = "ascending",
        xgroups = nil,
        groups = nil,
        hitPreset = nil,
        releasePreset = nil,
        autoAssign = true,
        tapOnExec = false,
        doNotAssign = false,
        clearFirst = true
    }
end

-- Enhanced UI functions (backwards compatible)

function HitMaker.StrInput(title, message)
    return HitMakerUI.StrInput(title, message)
end

function HitMaker.place_in_commandline_helper(command)
    local macro_number = 9999
    Cmd("Delete Macro "..macro_number.. ".1 thru /nc")
    Cmd("Store Macro "..macro_number)
    Cmd("Set Macro "..macro_number..".1 Property 'Command' \""..command.."\"")
    Cmd("Set Macro "..macro_number.." Property 'EXECUTE' \"No\"")
    Cmd("Set Macro "..macro_number.." Property 'ADDTOCMDLINE' \"Yes\"")
    Cmd("Call Macro "..macro_number)

    DataPool().Macros[macro_number] = nil
end

function HitMaker.get_available_exec()
    local available_exec = nil
    local hit_buttons = {101, 115}
    local current_song = GetVar(GlobalVars(), "song")
    local page = DataPool().Pages[tostring(current_song)]
    local executors = page:Children()
    
    -- Create a table to track which executor numbers are used
    local used_executors = {}
    for i = 1, #executors do
        local executor = executors[i]
        if executor and executor.no then
            if executor.no >= hit_buttons[1] and executor.no <= hit_buttons[2] then
                used_executors[executor.no] = true
                --Printf("Found used executor: " .. executor.no)
            end
        end
    end
    
    -- Find the first unused executor number in the range
    for exec_num = hit_buttons[1], hit_buttons[2] do
        if not used_executors[exec_num] then
            available_exec = exec_num
            Printf("First available executor: " .. available_exec)
            break
        end
    end
    
    if available_exec == nil then
        Printf("No available executors in range " .. hit_buttons[1] .. " to " .. hit_buttons[2])
    end
    
    return available_exec
end

local function trim_text(value)
    return tostring(value or ""):gsub("^%s+", ""):gsub("%s+$", "")
end

local function normalize_text(value)
    return trim_text(value):lower()
end

local function escape_for_cmd(value)
    return tostring(value or ""):gsub('"', '\\"')
end

HitMaker._integration_hooks = HitMaker._integration_hooks or {}

function HitMaker.set_integration_hooks(hooks)
    if type(hooks) ~= "table" then
        return false
    end
    HitMaker._integration_hooks.resolve_song_sequence_range = hooks.resolve_song_sequence_range
    HitMaker._integration_hooks.create_sequence_in_range = hooks.create_sequence_in_range
    return true
end

function HitMaker.resolve_song_sequence_range()
    local current_song = trim_text(GetVar(GlobalVars(), "song"))
    if current_song == "" then
        return nil, nil, nil
    end

    local hook = HitMaker._integration_hooks.resolve_song_sequence_range
    if type(hook) == "function" then
        local ok, start_no, end_no, song_label = pcall(hook, current_song)
        if ok and tonumber(start_no) and tonumber(end_no) then
            start_no = tonumber(start_no)
            end_no = tonumber(end_no)
            if start_no > 0 and end_no >= start_no then
                return start_no, end_no, trim_text(song_label or current_song)
            end
        end
    end

    local sequences = DataPool() and DataPool().Sequences or nil
    if not sequences then
        return nil, nil, nil
    end
    for i = 1, sequences.count do
        local sequence = sequences[i]
        if sequence and normalize_text(sequence.name) == normalize_text(current_song) then
            local anchor = tonumber(sequence.no) or i
            return anchor, anchor + 99, current_song
        end
    end
    return nil, nil, current_song
end

local function find_first_free_sequence_in_range(start_no, end_no)
    local sequences = DataPool() and DataPool().Sequences or nil
    if not sequences then
        return nil
    end
    for i = tonumber(start_no) or 1, tonumber(end_no) or 0 do
        if not sequences[i] or sequences[i] == "" then
            return i
        end
    end
    return nil
end

function HitMaker.get_available_sequence()
    local song_start, song_end, song_label = HitMaker.resolve_song_sequence_range()
    if not song_start or not song_end then
        Printf("HitMaker: could not resolve song range for '" .. tostring(song_label or "?") .. "'")
        return nil
    end
    local available_sequence = find_first_free_sequence_in_range(song_start, song_end)
    Printf("HitMaker: song '" .. tostring(song_label) .. "' range " .. tostring(song_start) .. "-" .. tostring(song_end) .. ", next available sequence " .. tostring(available_sequence))
    return available_sequence
end

function HitMaker.create_auto_sequence(sequence_name)
    local desired_name = trim_text(sequence_name)
    local song_start, song_end, song_label = HitMaker.resolve_song_sequence_range()
    if not song_start or not song_end then
        HitMakerUI.ShowError(
            "Sequence Range Missing",
            "Could not resolve the current song range for '" .. tostring(song_label or "?") .. "'."
        )
        return nil
    end

    local planned_sequence = find_first_free_sequence_in_range(song_start, song_end)
    if not planned_sequence then
        HitMakerUI.ShowError(
            "No Free Sequence",
            "No free sequence numbers were found in range " .. tostring(song_start) .. "-" .. tostring(song_end) .. "."
        )
        return nil
    end

    local create_hook = HitMaker._integration_hooks.create_sequence_in_range
    if type(create_hook) == "function" then
        local ok, created = pcall(create_hook, {
            sequence_name = desired_name,
            sequence_no = planned_sequence,
            song_start = song_start,
            song_end = song_end,
            song_label = song_label
        })
        if ok then
            if type(created) == "table" then
                local seq_no = tonumber(created.no or created.sequence_no)
                if seq_no then
                    return seq_no
                end
            elseif tonumber(created) then
                return tonumber(created)
            end
        end
    end

    local cmd = "Store Sequence " .. tostring(planned_sequence)
    if desired_name ~= "" then
        cmd = cmd .. string.format(' /name="%s"', escape_for_cmd(desired_name))
    end
    cmd = cmd .. " /nc"
    Cmd(cmd)

    local created_sequence = DataPool() and DataPool().Sequences and DataPool().Sequences[planned_sequence] or nil
    if not created_sequence then
        HitMakerUI.ShowError(
            "Sequence Create Failed",
            "Failed to create sequence " .. tostring(planned_sequence) .. " in range " .. tostring(song_start) .. "-" .. tostring(song_end) .. "."
        )
        return nil
    end

    return planned_sequence
end

HitMaker._sequence_type_registry = HitMaker._sequence_type_registry or {}

function HitMaker.register_sequence_type(sequence_type, builder)
    local type_key = normalize_text(sequence_type)
    if type_key == "" then
        return false
    end
    if type(builder) ~= "function" and type(builder) ~= "string" then
        return false
    end
    HitMaker._sequence_type_registry[type_key] = builder
    return true
end

function HitMaker.register_default_sequence_types()
    local defaults = {
        go_hit = "go_hit",
        temp_hit = "temp_hit",
        go_on_off = "go_on_go_off_hit",
        four_hit = "four_hit_matrix",
        x_hit_release = "x_hit_release"
    }
    for sequence_type, builder in pairs(defaults) do
        if HitMaker._sequence_type_registry[sequence_type] == nil then
            HitMaker.register_sequence_type(sequence_type, builder)
        end
    end
end

local function resolve_sequence_builder(sequence_type)
    HitMaker.register_default_sequence_types()
    local type_key = normalize_text(sequence_type)
    local registered = HitMaker._sequence_type_registry[type_key]
    if type(registered) == "function" then
        return registered
    end
    if type(registered) == "string" and type(HitMaker[registered]) == "function" then
        return HitMaker[registered]
    end
    return nil
end

local function normalize_event_type_label(event_type)
    local label = trim_text(event_type):gsub("%s+", " ")
    return label
end

local function normalize_event_type_key(event_type)
    return normalize_event_type_label(event_type):lower()
end

local function event_type_name_segment(event_type)
    local normalized = normalize_event_type_label(event_type)
    if normalized == "" then
        return "Event"
    end
    normalized = normalized:gsub("[%s/]+", "_")
    normalized = normalized:gsub("[^%w_%-]+", "")
    if normalized == "" then
        return "Event"
    end
    return normalized
end

local function normalize_x_order_mode(raw_mode)
    local normalized = normalize_text(raw_mode)
    if normalized == "asc" or normalized == "ascending" then
        return "ascending"
    end
    if normalized == "desc" or normalized == "descending" or normalized == "decending" then
        return "descending"
    end
    if normalized == "rand" or normalized == "random" then
        return "random"
    end
    return nil
end

local function build_x_order_values(pair_count, order_mode)
    local order = {}
    for i = 1, pair_count do
        order[i] = i
    end

    if order_mode == "descending" then
        for i = 1, pair_count do
            order[i] = pair_count - i + 1
        end
        return order
    end

    if order_mode == "random" then
        if not HitMaker._x_hit_random_seeded then
            math.randomseed(os.time())
            -- Warm-up calls so immediate invocations are less predictable.
            math.random()
            math.random()
            math.random()
            HitMaker._x_hit_random_seeded = true
        end
        for i = pair_count, 2, -1 do
            local j = math.random(i)
            order[i], order[j] = order[j], order[i]
        end
    end

    return order
end

function HitMaker.normalize_event_types(event_types)
    local result = {}
    local seen = {}

    local function push_one(raw_value)
        local label = normalize_event_type_label(raw_value)
        if label == "" then
            return
        end
        local key = normalize_event_type_key(label)
        if seen[key] then
            return
        end
        seen[key] = true
        table.insert(result, label)
    end

    if type(event_types) == "string" then
        for token in string.gmatch(event_types, "([^,;]+)") do
            push_one(token)
        end
        return result
    end

    if type(event_types) == "table" then
        for _, value in ipairs(event_types) do
            push_one(value)
        end
        for _, value in pairs(event_types) do
            if type(value) == "string" then
                push_one(value)
            end
        end
    end

    return result
end

function HitMaker.create_event_type_sequences(keyword_args)
    local settings = keyword_args or {}
    local event_types = HitMaker.normalize_event_types(settings.event_types or settings.types)
    if #event_types == 0 then
        HitMakerUI.ShowError("Missing Event Types", "Provide at least one event type.")
        return nil
    end

    local sequence_type = normalize_text(settings.sequence_type or "go_hit")
    local builder = resolve_sequence_builder(sequence_type)
    if not builder then
        HitMakerUI.ShowError(
            "Unknown Sequence Type",
            "No builder is registered for sequence type '" .. tostring(sequence_type) .. "'."
        )
        return nil
    end

    local current_song = trim_text(settings.song_name or GetVar(GlobalVars(), "song"))
    local name_prefix = settings.name_prefix
    if name_prefix == nil then
        name_prefix = current_song ~= "" and (current_song .. "_") or ""
    else
        name_prefix = trim_text(name_prefix)
    end
    local assign_to_timecode = settings.assign_to_timecode ~= false
    local target_timecode = trim_text(settings.timecode_name or current_song)
    local target_track_group = tonumber(settings.timecode_track_group_no) or 1
    local clear_first = settings.clearFirst == true
    local clear_first_once = settings.clear_first_once ~= false

    local created = {}
    local first = true
    for _, event_type in ipairs(event_types) do
        local event_segment = event_type_name_segment(event_type)
        local build_args = HitMaker.merge_settings_with_defaults({}, settings)
        build_args.name = name_prefix .. event_segment
        build_args.skip_dialog = true
        if clear_first and clear_first_once then
            build_args.clearFirst = first
        elseif build_args.clearFirst == nil then
            build_args.clearFirst = clear_first
        end

        local sequence_no = builder(build_args)
        if not sequence_no then
            return nil
        end

        local target_track_name = normalize_event_type_key(event_type)
        if assign_to_timecode and target_timecode ~= "" then
            Cmd(
                "Assign Sequence "
                    .. tostring(sequence_no)
                    .. " At Timecode "
                    .. target_timecode
                    .. "."
                    .. tostring(target_track_group)
                    .. ".\""
                    .. escape_for_cmd(target_track_name)
                    .. "\" Property \"TARGET\""
            )
        end

        created[target_track_name] = sequence_no
        table.insert(created, {
            event_type = event_type,
            event_key = target_track_name,
            sequence_no = sequence_no
        })
        first = false
    end

    return created
end

function HitMaker.create_sequence_for_event_type(keyword_args)
    local settings = keyword_args or {}
    local event_type = normalize_event_type_label(settings.event_type or settings.type)
    if event_type == "" then
        HitMakerUI.ShowError("Missing Event Type", "Provide event_type for sequence creation.")
        return nil
    end

    local sequence_type = normalize_text(settings.sequence_type or "go_hit")
    local builder = resolve_sequence_builder(sequence_type)
    if not builder then
        HitMakerUI.ShowError(
            "Unknown Sequence Type",
            "No builder is registered for sequence type '" .. tostring(sequence_type) .. "'."
        )
        return nil
    end

    local build_args = HitMaker.merge_settings_with_defaults({}, settings)
    build_args.skip_dialog = true

    local explicit_name = trim_text(settings.sequence_name or settings.name)
    if explicit_name ~= "" then
        build_args.name = explicit_name
    else
        local current_song = trim_text(settings.song_name or GetVar(GlobalVars(), "song"))
        local prefix = trim_text(settings.name_prefix or "")
        if prefix == "" and current_song ~= "" then
            prefix = current_song .. "_"
        end
        build_args.name = prefix .. event_type_name_segment(event_type)
    end

    if build_args.clearFirst == nil then
        build_args.clearFirst = settings.clearFirst == true
    end
    if build_args.doNotAssign == nil then
        build_args.doNotAssign = false
    end
    if build_args.autoAssign == nil then
        build_args.autoAssign = true
    end
    if build_args.tapOnExec == nil then
        build_args.tapOnExec = false
    end

    local sequence_no = builder(build_args)
    if not sequence_no then
        return nil
    end

    if settings.assign_to_timecode == true then
        local current_song = trim_text(settings.song_name or GetVar(GlobalVars(), "song"))
        local target_timecode = trim_text(settings.timecode_name or current_song)
        local target_track_group = tonumber(settings.timecode_track_group_no) or 1
        if target_timecode ~= "" then
            Cmd(
                "Assign Sequence "
                    .. tostring(sequence_no)
                    .. " At Timecode "
                    .. target_timecode
                    .. "."
                    .. tostring(target_track_group)
                    .. ".\""
                    .. escape_for_cmd(normalize_event_type_key(event_type))
                    .. "\" Property \"TARGET\""
            )
        end
    end

    return {
        sequence_no = sequence_no,
        event_type = event_type,
        sequence_type = sequence_type,
        name = build_args.name
    }
end

function HitMaker.get_beat_times(beat_count)
    -- calculates how many seconds are in the given number of beats
    local beat_time = nil
    local bpm = GetVar(GlobalVars(), "BPM")
    beat_time = math.floor((60/tonumber(bpm)) * 100 + 0.5) / 100
    Printf("Beat time for bpm "..bpm.." is "..beat_time)
    beat_count_time = math.floor((beat_time * beat_count) * 100 + 0.5) / 100
    Printf("Beat time for "..beat_count.." beats is "..beat_time)
    return beat_time
end

function HitMaker.generate_default_name(hit_type)
    -- Generate default names based on hit type
    if hit_type == "temp" then
        return "Temp"
    elseif hit_type == "go_hit" then
        return "Go Hit"
    elseif hit_type == "go_on_off" then
        return "Go On/Off"
    elseif hit_type == "four_hit" then
        return "Four Hit"
    elseif hit_type == "x_hit_release" then
        return "X Hit Release"
    else
        return "Hit"
    end
end

function HitMaker.process_timing_command(command_text)
    -- Check if command contains timing parameters (case insensitive, more flexible patterns)
    local has_timing = command_text:lower():find("delayfrom[xyz]") or command_text:lower():find("delayto[xyz]") or 
                      command_text:lower():find("fadefrom[xyz]") or command_text:lower():find("fadeto[xyz]")
    
    if not has_timing then
        return command_text, false -- No timing found, return original
    end
    
    Printf("Timing command detected: " .. command_text)
    
    -- Get timing from user using enhanced UI
    local timing_settings = HitMakerUI.TimingInputDialog("Timing Input", "Enter timing value:")
    if not timing_settings then
        return command_text, false -- User cancelled
    end
    
    local timing_input = tonumber(timing_settings.value)
    if not timing_input then
        HitMakerUI.ShowError("Error", "Invalid timing value entered")
        return command_text, false
    end
    
    local timing_value = timing_input
    
    -- Convert based on selected unit
    if timing_settings.unit == 1 then -- Beats
        local timing_value = HitMaker.get_beat_times(timing_input)
        Printf("Converting " .. timing_input .. " beats to " .. timing_value .. " seconds")
    elseif timing_settings.unit == 2 then -- Seconds
        Printf("Using " .. timing_input .. " seconds")
    elseif timing_settings.unit == 3 then -- Milliseconds
        timing_value = timing_input / 1000
        Printf("Converting " .. timing_input .. " milliseconds to " .. timing_value .. " seconds")
    end
    
    -- Replace timing parameters with more specific patterns
    local processed_command = command_text
    
    -- Replace any parentheses containing DelayFrom or FadeFrom (case insensitive)
    processed_command = processed_command:gsub("%(([^%)]*[Dd]elay[Ff]rom[XYZxyz][^%)]*)%)", "\"" .. timing_value .. "\"")
    processed_command = processed_command:gsub("%(([^%)]*[Ff]ade[Ff]rom[XYZxyz][^%)]*)%)", "\"" .. timing_value .. "\"")
    
    -- Replace any parentheses containing DelayTo or FadeTo (case insensitive)
    processed_command = processed_command:gsub("%(([^%)]*[Dd]elay[Tt]o[XYZxyz][^%)]*)%)", "\"0\"")
    processed_command = processed_command:gsub("%(([^%)]*[Ff]ade[Tt]o[XYZxyz][^%)]*)%)", "\"0\"")
    
    Printf("Processed command: " .. processed_command)
    
    return processed_command, true
end

function HitMaker.macro_to_table(macro_int)
    local macro_table = {}

    local macro_integer = tonumber(macro_int)

    macro_object = DataPool().Macros[macro_integer]

    macro_table = macro_object:Children()

    return macro_table
end

function HitMaker.temp_hit(keyword_args)
    local settings
    
    -- Handle keyword arguments
    if keyword_args and keyword_args.skip_dialog then
        -- Use provided arguments directly
        settings = HitMaker.merge_settings_with_defaults(HitMaker.get_default_sequence_settings(), keyword_args)
        
        -- Validate required fields if any
        if not HitMaker.validate_required_args(settings, {}) then
            return
        end
    else
        -- Use enhanced UI dialog (with pre-filled values if provided)
        if keyword_args then
            -- Pre-fill dialog with provided values
            local dialog_settings = HitMakerUI.SequenceCreationDialog()
            if not dialog_settings then
                Printf("Temp hit creation cancelled by user")
                return
            end
            -- Merge dialog results with provided keyword args
            settings = HitMaker.merge_settings_with_defaults(dialog_settings, keyword_args)
        else
            -- Standard dialog
            settings = HitMakerUI.SequenceCreationDialog()
            if not settings then
                Printf("Temp hit creation cancelled by user")
                return
            end
        end
    end
    
    local sequence_label = settings.name
    if not sequence_label or sequence_label == "" then
        sequence_label = HitMaker.generate_default_name("temp")
        Printf("No name provided, using default name: " .. sequence_label)
    end
    
    local available_sequence = HitMaker.create_auto_sequence(sequence_label)
    if not available_sequence then
        return nil
    end
    local available_exec = HitMaker.get_available_exec()
    local current_song = GetVar(GlobalVars(), "song")
    local page = DataPool().Pages[tostring(current_song)]

    if settings.clearFirst then
    Cmd("ClearAll")
    end
    
    if not settings.doNotAssign then
        if settings.autoAssign then
    Cmd("Assign Sequence "..available_sequence.." At Page "..page.no.."."..available_exec)
    Cmd("Set Page "..page.no.."."..available_exec.." Property \"KEYCOMMAND\" \"Temp\"")
        elseif settings.tapOnExec then
            -- Use command line helper to place assign command
            HitMaker.place_in_commandline_helper("Assign Sequence "..available_sequence)
        end
    end
    
    Cmd("Set Sequence "..available_sequence.." Property \"Name\" \""..sequence_label.."\"")
    return available_sequence
end

function HitMaker.go_hit(keyword_args)
    local settings
    
    -- Handle keyword arguments
    if keyword_args and keyword_args.skip_dialog then
        -- Use provided arguments directly
        settings = HitMaker.merge_settings_with_defaults(HitMaker.get_default_go_hit_settings(), keyword_args)
        
        -- Validate required fields
        if not HitMaker.validate_required_args(settings, {"offtime"}) then
            return
        end
    else
        -- Use enhanced UI dialog (with pre-filled values if provided)
        if keyword_args then
            -- Pre-fill dialog with provided values
            local dialog_settings = HitMakerUI.GoHitCreationDialog()
            if not dialog_settings then
                Printf("Go hit creation cancelled by user")
                return
            end
            -- Merge dialog results with provided keyword args
            settings = HitMaker.merge_settings_with_defaults(dialog_settings, keyword_args)
        else
            -- Standard dialog
            settings = HitMakerUI.GoHitCreationDialog()
            if not settings then
                Printf("Go hit creation cancelled by user")
                return
            end
        end
    end
    
    local sequence_label = settings.name
    if not sequence_label or sequence_label == "" then
        sequence_label = HitMaker.generate_default_name("go_hit")
        Printf("No name provided, using default name: " .. sequence_label)
    end
    
    local offtime = tonumber(settings.offtime)
    if not offtime then
        HitMakerUI.ShowError("Error", "Invalid offtime value")
        return
    end
    
    -- Convert BPM to seconds if needed
    if settings.timingType == 1 then -- BPM
        offtime = tonumber(HitMaker.get_beat_times(offtime))
    end
    
    local available_sequence = HitMaker.create_auto_sequence(sequence_label)
    if not available_sequence then
        return nil
    end
    local available_exec = HitMaker.get_available_exec()
    local current_song = GetVar(GlobalVars(), "song")
    local page = DataPool().Pages[tostring(current_song)]

    if settings.clearFirst then
    Cmd("ClearAll")
    end
    
    if not settings.doNotAssign then
        if settings.autoAssign then
    Cmd("Assign Sequence "..available_sequence.." At Page "..page.no.."."..available_exec)
        elseif settings.tapOnExec then
            -- Use command line helper to place assign command
            HitMaker.place_in_commandline_helper("Assign Sequence "..available_sequence)
        end
    end
    
    Cmd("Set Sequence "..available_sequence.." Property \"Name\" \""..sequence_label.."\"")
    
    if settings.followTrigger then
    Cmd("Set Sequence "..available_sequence.." Cue OffCue Property \"TrigType\" \"Follow\"")
    end
    
    Cmd("Set Sequence "..available_sequence.." Cue OffCue Property \"CueFade\" \""..offtime.."\"")
    return available_sequence
end

function HitMaker.go_on_go_off_hit(keyword_args)
    local settings
    
    -- Handle keyword arguments
    if keyword_args and keyword_args.skip_dialog then
        -- Use provided arguments directly
        settings = HitMaker.merge_settings_with_defaults(HitMaker.get_default_sequence_settings(), keyword_args)
        
        -- Validate required fields if any
        if not HitMaker.validate_required_args(settings, {}) then
            return
        end
    else
        -- Use enhanced UI dialog (with pre-filled values if provided)
        if keyword_args then
            -- Pre-fill dialog with provided values
            local dialog_settings = HitMakerUI.SequenceCreationDialog()
            if not dialog_settings then
                Printf("Go on/off hit creation cancelled by user")
                return
            end
            -- Merge dialog results with provided keyword args
            settings = HitMaker.merge_settings_with_defaults(dialog_settings, keyword_args)
        else
            -- Standard dialog
            settings = HitMakerUI.SequenceCreationDialog()
            if not settings then
                Printf("Go on/off hit creation cancelled by user")
                return
            end
        end
    end
    
    local sequence_label = settings.name
    if not sequence_label or sequence_label == "" then
        sequence_label = HitMaker.generate_default_name("go_on_off")
        Printf("No name provided, using default name: " .. sequence_label)
    end
    
    local available_sequence = HitMaker.create_auto_sequence(sequence_label)
    if not available_sequence then
        return nil
    end
    local available_exec = HitMaker.get_available_exec()
    local current_song = GetVar(GlobalVars(), "song")
    local page = DataPool().Pages[tostring(current_song)]

    if settings.clearFirst then
    Cmd("ClearAll")
    end
    
    Cmd("Store Sequence "..available_sequence.. " Cue 1 Thru 2")
    
    if not settings.doNotAssign then
        if settings.autoAssign then
    Cmd("Assign Sequence "..available_sequence.." At Page "..page.no.."."..available_exec)
        elseif settings.tapOnExec then
            -- Use command line helper to place assign command
            HitMaker.place_in_commandline_helper("Assign Sequence "..available_sequence)
        end
    end
    
    Cmd("Set Sequence "..available_sequence.." Property \"Name\" \""..sequence_label.."\"")
    Cmd("Set Sequence "..available_sequence.." Cue OffCue Property \"TrigType\" \"Follow\"")
    return available_sequence
end

function HitMaker.x_hit_release(keyword_args)
    local defaults = HitMaker.get_default_x_hit_release_settings()
    local settings
    local preset_types = {
        {name = "Dimmer", id = 1},
        {name = "Position", id = 2},
        {name = "Color", id = 4},
        {name = "Beam", id = 5},
        {name = "Focus", id = 6},
        {name = "Phaser", id = 21},
        {name = "All", id = 22}
    }

    local function normalize_preset_ref(raw_ref)
        local text = trim_text(raw_ref)
        if text == "" then
            return nil
        end

        local function normalize_numeric_ref(type_no_text, preset_no_text)
            local type_no = tonumber(type_no_text)
            local preset_no = tonumber(preset_no_text)
            if not type_no or not preset_no then
                return nil
            end
            return "Preset " .. tostring(math.floor(type_no)) .. "." .. tostring(math.floor(preset_no))
        end

        local numeric_with_prefix_type, numeric_with_prefix_preset = text:match("^Preset%s+(%d+)%.(%d+)$")
        if numeric_with_prefix_type and numeric_with_prefix_preset then
            return normalize_numeric_ref(numeric_with_prefix_type, numeric_with_prefix_preset)
        end

        local numeric_type, numeric_preset = text:match("^(%d+)%.(%d+)$")
        if numeric_type and numeric_preset then
            return normalize_numeric_ref(numeric_type, numeric_preset)
        end

        local named_type, named_name = text:match("^Preset%s+(%d+)%.\"(.-)\"$")
        if not named_type then
            named_type, named_name = text:match("^(%d+)%.\"(.-)\"$")
        end
        if named_type and named_name then
            local type_no = tonumber(named_type)
            local presets = DataPool().PresetPools[type_no]
            if presets then
                local wanted = normalize_text(named_name)
                for i = 1, presets.count do
                    local preset = presets[i]
                    if preset and normalize_text(preset.name or "") == wanted then
                        local preset_no = tonumber(preset.no) or i
                        return "Preset " .. tostring(type_no) .. "." .. tostring(math.floor(preset_no))
                    end
                end
            end
            return nil
        end

        return nil
    end

    local function wait_for_ma()
        local ok = pcall(function()
            coroutine.yield(0.02)
        end)
        if not ok then
            pcall(function()
                coroutine.yield()
            end)
        end
    end

    local function set_property_with_wait(command_text)
        wait_for_ma()
        Cmd(command_text)
    end

    local function normalize_group_names(raw_groups)
        local groups = {}
        local seen = {}

        local function push_one(raw_group)
            local group_name = trim_text(raw_group)
            if group_name == "" then
                return
            end
            local dedupe_key = normalize_text(group_name)
            if seen[dedupe_key] then
                return
            end
            seen[dedupe_key] = true
            table.insert(groups, group_name)
        end

        if type(raw_groups) == "string" then
            for token in string.gmatch(raw_groups, "([^,;]+)") do
                push_one(token)
            end
            return groups
        end

        if type(raw_groups) == "table" then
            for _, value in ipairs(raw_groups) do
                push_one(value)
            end
            for _, value in pairs(raw_groups) do
                if type(value) == "string" then
                    push_one(value)
                end
            end
        end

        return groups
    end

    local function select_preset_reference(dialog_title)
        local type_options = {}
        for _, preset_type in ipairs(preset_types) do
            table.insert(type_options, preset_type.name)
        end

        local selected_type = HitMaker.SwipeSelector(
            dialog_title,
            "Select preset type:",
            "Preset Type",
            type_options,
            1
        )
        if not selected_type then
            return nil
        end

        local selected_type_id = nil
        for _, preset_type in ipairs(preset_types) do
            if preset_type.name == selected_type then
                selected_type_id = preset_type.id
                break
            end
        end
        if not selected_type_id then
            return nil
        end

        local presets = DataPool().PresetPools[selected_type_id]
        local preset_labels = {}
        local preset_ref_by_label = {}
        if presets then
            for i = 1, presets.count do
                local preset = presets[i]
                if preset then
                    local preset_no = tonumber(preset.no) or i
                    local preset_name = trim_text(preset.name or "")
                    local label
                    if preset_name ~= "" then
                        label = string.format("%03d - %s", preset_no, preset_name)
                        preset_ref_by_label[label] = "Preset " .. selected_type_id .. "." .. tostring(preset_no)
                    else
                        label = string.format("%03d", preset_no)
                        preset_ref_by_label[label] = "Preset " .. selected_type_id .. "." .. tostring(preset_no)
                    end
                    table.insert(preset_labels, label)
                end
            end
        end

        if #preset_labels == 0 then
            HitMakerUI.ShowError("No Presets", "No presets found in pool " .. tostring(selected_type_id) .. ".")
            return nil
        end

        local selected_preset_label = HitMaker.SwipeSelector(
            dialog_title,
            "Select preset:",
            "Preset",
            preset_labels,
            1
        )
        if not selected_preset_label then
            return nil
        end
        return preset_ref_by_label[selected_preset_label]
    end

    local function set_object_property(target, property_name, property_value)
        if not target then
            return false
        end
        wait_for_ma()
        local ok = pcall(function()
            if target.Set then
                target:Set(property_name, tostring(property_value))
            else
                target[property_name] = property_value
            end
        end)
        return ok
    end

    local function get_cue_handle(sequence_no, cue_no)
        local datapool = DataPool()
        if not datapool then
            return nil
        end
        local sequence_pool = datapool[6]
        if not sequence_pool then
            return nil
        end
        local sequence_handle = sequence_pool[tonumber(sequence_no)]
        if not sequence_handle then
            return nil
        end
        local cue_index = tonumber(cue_no) + 2
        return sequence_handle[cue_index]
    end

    local function set_release_cue_properties(sequence_no, release_cue_no, release_fade_seconds)
        local release_cue_handle = get_cue_handle(sequence_no, release_cue_no)
        local trig_set_ok = false
        local fade_set_ok = false

        if release_cue_handle then
            trig_set_ok = set_object_property(release_cue_handle, "TrigType", "Follow")
            local cue_part_zero = release_cue_handle[1]
            if cue_part_zero then
                fade_set_ok = set_object_property(cue_part_zero, "CueInFade", release_fade_seconds)
                if not fade_set_ok then
                    fade_set_ok = set_object_property(cue_part_zero, "cueinfade", release_fade_seconds)
                end
            end
        end

        if not trig_set_ok then
            set_property_with_wait("Set Sequence " .. sequence_no .. " Cue " .. release_cue_no .. " Property \"TrigType\" \"Follow\"")
        end
        if not fade_set_ok then
            set_property_with_wait("Set Sequence " .. sequence_no .. " Cue " .. release_cue_no .. " Part 0 Property \"CueInFade\" \"" .. tostring(release_fade_seconds) .. "\"")
        end
    end

    local function store_recipe_line(sequence_no, cue_no, recipe_line_no, group_name, preset_ref)
        Cmd("ClearAll")
        Cmd("SelectFixtures Group \"" .. escape_for_cmd(group_name) .. "\"")
        Cmd("At " .. preset_ref)
        Cmd("Store Sequence " .. sequence_no .. " Cue " .. cue_no .. " Part 0." .. recipe_line_no .. " /Merge")
        Cmd("Assign Group \"" .. escape_for_cmd(group_name) .. "\" At Sequence " .. sequence_no .. " Cue " .. cue_no .. " Part 0." .. recipe_line_no)
        Cmd("Assign " .. preset_ref .. " At Sequence " .. sequence_no .. " Cue " .. cue_no .. " Part 0." .. recipe_line_no)
    end

    local function set_recipe_line_x(sequence_no, cue_no, recipe_line_no, x_value, xgroup_value)
        local recipe_handle = nil
        local cue_handle = get_cue_handle(sequence_no, cue_no)
        if cue_handle and cue_handle[1] then
            recipe_handle = cue_handle[1][recipe_line_no]
        end

        local set_x_ok = false
        local set_xgroup_ok = false
        if recipe_handle then
            set_x_ok = set_object_property(recipe_handle, "X", x_value)
            set_xgroup_ok = set_object_property(recipe_handle, "XGroup", xgroup_value)
        end
        if not set_x_ok then
            set_property_with_wait("Set Sequence " .. sequence_no .. " Cue " .. cue_no .. " Part 0." .. recipe_line_no .. " Property \"X\" \"" .. tostring(x_value) .. "\"")
        end
        if not set_xgroup_ok then
            set_property_with_wait("Set Sequence " .. sequence_no .. " Cue " .. cue_no .. " Part 0." .. recipe_line_no .. " Property \"XGroup\" \"" .. tostring(xgroup_value) .. "\"")
        end
    end

    if keyword_args and keyword_args.skip_dialog then
        settings = HitMaker.merge_settings_with_defaults(defaults, keyword_args)
    else
        local initial_settings = HitMaker.merge_settings_with_defaults(defaults, keyword_args or {})

        local sequence_name = HitMakerUI.StrInput(
            "X Hit Release",
            "Sequence name:",
            tostring(initial_settings.name or ""),
            30
        )
        if sequence_name == nil then
            Printf("X hit/release creation cancelled by user")
            return
        end

        local cue_count_input = HitMakerUI.StrInput(
            "X Hit Release",
            "Total cue count (even number):",
            tostring(initial_settings.cueCount or 8),
            8
        )
        if cue_count_input == nil then
            Printf("X hit/release creation cancelled by user")
            return
        end

        local release_fade_input = HitMakerUI.StrInput(
            "X Hit Release",
            "Release cue fade time (seconds):",
            tostring(initial_settings.releaseFade or "1"),
            8
        )
        if release_fade_input == nil then
            Printf("X hit/release creation cancelled by user")
            return
        end

        local xgroups_input = HitMakerUI.StrInput(
            "X Hit Release",
            "XGroup value (blank uses pair count):",
            tostring(initial_settings.xgroups or ""),
            8
        )
        if xgroups_input == nil then
            Printf("X hit/release creation cancelled by user")
            return
        end

        local selected_order = HitMaker.SwipeSelector(
            "X Hit Release",
            "X order mode:",
            "Order Mode",
            {"ascending", "descending", "random"},
            1
        )
        if selected_order == nil then
            Printf("X hit/release creation cancelled by user")
            return
        end

        local selected_groups = HitMaker.group_selection_popup(
            "X Hit Release Groups",
            "Select groups to store as recipe lines:"
        )
        if not selected_groups or #selected_groups == 0 then
            HitMakerUI.ShowError("Missing Groups", "Select at least one group for recipe line creation.")
            return nil
        end

        local hit_preset_ref = select_preset_reference("Hit Preset")
        if not hit_preset_ref then
            Printf("X hit/release creation cancelled while selecting hit preset")
            return nil
        end

        local release_preset_ref = select_preset_reference("Release Preset")
        if not release_preset_ref then
            Printf("X hit/release creation cancelled while selecting release preset")
            return nil
        end

        settings = HitMaker.merge_settings_with_defaults(initial_settings, {
            name = sequence_name,
            cueCount = cue_count_input,
            releaseFade = release_fade_input,
            xgroups = xgroups_input,
            orderMode = selected_order,
            groups = selected_groups,
            hitPreset = hit_preset_ref,
            releasePreset = release_preset_ref
        })
    end

    local cue_count = tonumber(settings.cueCount or settings.cue_count or settings.x_cues or settings.count)
    if not cue_count or cue_count <= 1 or cue_count % 1 ~= 0 then
        HitMakerUI.ShowError("Invalid Cue Count", "Cue count must be a whole number greater than 1.")
        return nil
    end
    cue_count = math.floor(cue_count)
    if cue_count % 2 ~= 0 then
        HitMakerUI.ShowError("Invalid Cue Count", "Cue count must be even so hits/releases can be paired.")
        return nil
    end

    local release_fade = tonumber(settings.releaseFade or settings.release_fade or settings.release_fade_time or settings.fade)
    if not release_fade or release_fade < 0 then
        HitMakerUI.ShowError("Invalid Fade", "Release fade must be a number greater than or equal to 0.")
        return nil
    end

    local pair_count = cue_count / 2
    local xgroups_raw = settings.xgroups
    if xgroups_raw == nil then
        xgroups_raw = settings.x_groups
    end
    local xgroups_value
    if trim_text(xgroups_raw) == "" then
        xgroups_value = pair_count
    else
        xgroups_value = tonumber(xgroups_raw)
        if not xgroups_value or xgroups_value <= 0 or xgroups_value % 1 ~= 0 then
            HitMakerUI.ShowError("Invalid XGroup", "XGroup must be a whole number greater than 0.")
            return nil
        end
        xgroups_value = math.floor(xgroups_value)
    end

    local order_mode = normalize_x_order_mode(settings.orderMode or settings.order_mode or settings.order or settings.x_order)
    if not order_mode then
        HitMakerUI.ShowError("Invalid Order", "Order mode must be ascending, descending, or random.")
        return nil
    end

    local groups = normalize_group_names(settings.groups or settings.group_names or settings.groupNames)
    if #groups == 0 then
        HitMakerUI.ShowError("Missing Groups", "Provide one or more groups for recipe lines.")
        return nil
    end

    local hit_preset_ref = normalize_preset_ref(settings.hitPreset or settings.hit_preset)
    local release_preset_ref = normalize_preset_ref(settings.releasePreset or settings.release_preset)
    if not hit_preset_ref or not release_preset_ref then
        HitMakerUI.ShowError("Missing Presets", "Both hit and release presets are required.")
        return nil
    end

    local sequence_label = trim_text(settings.name)
    if sequence_label == "" then
        sequence_label = HitMaker.generate_default_name("x_hit_release")
        Printf("No name provided, using default name: " .. sequence_label)
    end

    local available_sequence = HitMaker.create_auto_sequence(sequence_label)
    if not available_sequence then
        return nil
    end
    local available_exec = HitMaker.get_available_exec()
    local current_song = GetVar(GlobalVars(), "song")
    local page = DataPool().Pages[tostring(current_song)]

    if settings.clearFirst then
        Cmd("ClearAll")
    end

    -- 1) Build sequence and cue skeleton.
    Cmd("Store Sequence " .. available_sequence .. " Cue 1 Thru " .. cue_count)

    if not settings.doNotAssign then
        if settings.autoAssign then
            Cmd("Assign Sequence " .. available_sequence .. " At Page " .. page.no .. "." .. available_exec)
        elseif settings.tapOnExec then
            HitMaker.place_in_commandline_helper("Assign Sequence " .. available_sequence)
        end
    end

    set_property_with_wait("Set Sequence " .. available_sequence .. " Property \"Name\" \"" .. escape_for_cmd(sequence_label) .. "\"")

    local x_order_values = build_x_order_values(pair_count, order_mode)
    local release_cues_to_finalize = {}
    for pair_index = 1, pair_count do
        local hit_cue = (pair_index * 2) - 1
        local release_cue = hit_cue + 1
        local x_value = x_order_values[pair_index]

        table.insert(release_cues_to_finalize, release_cue)

        -- 3) Create recipe lines for each selected group on hit/release cues.
        for recipe_line_no, group_name in ipairs(groups) do
            store_recipe_line(available_sequence, hit_cue, recipe_line_no, group_name, hit_preset_ref)
            store_recipe_line(available_sequence, release_cue, recipe_line_no, group_name, release_preset_ref)

            -- 4) Set X / XGroup on recipe lines.
            set_recipe_line_x(available_sequence, hit_cue, recipe_line_no, x_value, xgroups_value)
            set_recipe_line_x(available_sequence, release_cue, recipe_line_no, x_value, xgroups_value)
        end
    end

    -- Final pass: assign release cue trigger/fade after all recipe-line storage.
    for _, release_cue in ipairs(release_cues_to_finalize) do
        set_release_cue_properties(available_sequence, release_cue, release_fade)
    end

    Cmd("ClearAll")
    return available_sequence
end


function HitMaker.four_hit_matrix(keyword_args)
    local settings
    
    -- Handle keyword arguments
    if keyword_args and keyword_args.skip_dialog then
        -- Use provided arguments directly
        settings = HitMaker.merge_settings_with_defaults(HitMaker.get_default_four_hit_settings(), keyword_args)
        
        -- Validate required fields
        if not HitMaker.validate_required_args(settings, {"baseFadeTime"}) then
            return
        end
    else
        -- Use enhanced UI dialog (with pre-filled values if provided)
        if keyword_args then
            -- Pre-fill dialog with provided values
            local dialog_settings = HitMakerUI.FourHitMatrixDialog()
            if not dialog_settings then
                Printf("Four hit matrix creation cancelled by user")
                return
            end
            -- Merge dialog results with provided keyword args
            settings = HitMaker.merge_settings_with_defaults(dialog_settings, keyword_args)
        else
            -- Standard dialog
            settings = HitMakerUI.FourHitMatrixDialog()
            if not settings then
                Printf("Four hit matrix creation cancelled by user")
                return
            end
        end
    end
    
    local sequence_label = settings.name
    if not sequence_label or sequence_label == "" then
        sequence_label = HitMaker.generate_default_name("four_hit")
        Printf("No name provided, using default name: " .. sequence_label)
    end
    
    local available_sequence = HitMaker.create_auto_sequence(sequence_label)
    if not available_sequence then
        return nil
    end
    local available_exec = HitMaker.get_available_exec()
    local current_song = GetVar(GlobalVars(), "song")
    local page = DataPool().Pages[tostring(current_song)]
    local base_command = "Off Sequence "..available_sequence

    if settings.clearFirst then
        Cmd("ClearAll")
    end

    Cmd("Store Sequence "..available_sequence.." Cue 1 + 1.1 + 2 + 2.1 + 3 + 3.1 + 4 + 4.1")
    Cmd("Set Sequence "..available_sequence.."Cue 1.1 + 2.1 + 3.1 + 4.1 Property \"TrigType\" \"Follow\"")
    
    if not settings.doNotAssign then
        if settings.autoAssign then
    Cmd("Assign Sequence "..available_sequence.." At Page "..page.no.."."..available_exec)
        elseif settings.tapOnExec then
            -- Use command line helper to place assign command
            HitMaker.place_in_commandline_helper("Assign Sequence "..available_sequence)
        end
    end
    
    Cmd("Set Sequence "..available_sequence.." Property \"Name\" \""..sequence_label.."\"")
    
    if settings.setRestartMode then
    Cmd("Set Sequence "..available_sequence.." Property \"RESTARTMODE\" \"Next Cue\"")
    end

    if settings.createIndividual then
    for i = 1, 4, 1 do
        local avail_sequence = HitMaker.create_auto_sequence(sequence_label.."_"..i)
        if not avail_sequence then
            return nil
        end
        local avail_exec = HitMaker.get_available_exec()
        local available_seq_command = base_command.."; Go Sequence "..avail_sequence.." Cue 1\""
            
        Cmd("Store Sequence "..avail_sequence)
            
            if not settings.doNotAssign then
                if settings.autoAssign then
        Cmd("Assign Sequence "..avail_sequence.." At Page "..page.no.."."..avail_exec)
                elseif settings.tapOnExec then
                    -- Use command line helper to place assign command for individual hits
                    HitMaker.place_in_commandline_helper("Assign Sequence "..avail_sequence)
                end
            end
            
        Cmd("Set Sequence "..avail_sequence.." Property \"Name\" \""..sequence_label.."_"..i.." \"")    
        Cmd("Set Sequence "..available_sequence.."Cue "..i..".1 Property \"Name\" \""..sequence_label.."_"..i.." \"")
        Cmd("Set Sequence "..avail_sequence.."Cue 1 Property \"Command\" \"Go Sequence "..avail_sequence.." Cue OffCue\"")
        Cmd("Set Sequence "..avail_sequence.."Cue OffCue Property \"TrigType\" \"Follow\"")
        Cmd("Set Sequence "..available_sequence.."Cue "..i..".1 Property \"Command\" \""..available_seq_command.."\"")
            
            -- Calculate fade time based on settings
            local fadeTime = tonumber(settings.baseFadeTime) or 1
            if settings.fadeTimeType == 1 then -- BPM
                fadeTime = HitMaker.get_beat_times(fadeTime)
            end
            
            Cmd("Set Sequence "..avail_sequence.."Cue OffCue Property \"CueFade\" \""..fadeTime.."\"")
        end
    end

    return available_sequence
end

function HitMaker.store_kick_snare(keyword_args)
    local settings
    
    -- Handle keyword arguments
    if keyword_args and keyword_args.skip_dialog then
        -- Use provided arguments directly
        settings = HitMaker.merge_settings_with_defaults(HitMaker.get_default_go_hit_settings(), keyword_args)
        
        -- Validate required fields
        if not HitMaker.validate_required_args(settings, {"offtime"}) then
            return
        end
    else
        -- Use enhanced UI dialog (with pre-filled values if provided)
        if keyword_args then
            -- Pre-fill dialog with provided values
            local dialog_settings = HitMakerUI.GoHitCreationDialog()
            if not dialog_settings then
                Printf("Store kick snare creation cancelled by user")
                return
            end
            -- Merge dialog results with provided keyword args
            settings = HitMaker.merge_settings_with_defaults(dialog_settings, keyword_args)
        else
            -- Standard dialog
            settings = HitMakerUI.GoHitCreationDialog()
            if not settings then
                Printf("Store kick snare creation cancelled by user")
                return
            end
        end
    end



    settings.doNotAssign = false
    settings.autoAssign = true
    settings.clearFirst = true
    settings.clear_first_once = true
    settings.sequence_type = "go_hit"
    settings.event_types = {"kick", "snare"}
    settings.assign_to_timecode = true
    settings.timecode_track_group_no = 1

    local created = HitMaker.create_event_type_sequences(settings)
    if not created then
        return nil
    end

    return {
        kick_sequence = created.kick,
        snare_sequence = created.snare,
        created = created
    }
end


function HitMaker.replace_macroline_range(keyword_args)
    -- args: {
    --   macro_range_int_start = int,
    --   macro_range_int_end = int,
    --   line_num_to_replace = int,
    --   replace_this = string,
    --   replace_with_this = string,
    --   skip_dialog = bool (optional)
    -- }

    local settings

    -- Handle keyword arguments
    if keyword_args and keyword_args.skip_dialog then
        -- Use provided arguments directly
        settings = keyword_args
        
        -- Validate required fields
        if not HitMaker.validate_required_args(settings, {"macro_range_int_start", "macro_range_int_end", "line_num_to_replace", "replace_this", "replace_with_this"}) then
            return nil
        end
    else
        -- Loop to handle dialog and preview interactions
        local dialog_completed = false
        local current_settings = keyword_args or {}
        
        while not dialog_completed do
            -- Show dialog with current settings (either from keyword args or previous dialog input)
            local dialog_settings = HitMakerUI.ReplaceMacroLineRangeDialogWithValues(current_settings)
            if not dialog_settings then
                Printf("Replace macro line range cancelled by user")
                return nil
            end
            
            -- Update current settings with dialog results
            settings = dialog_settings
            
            -- Handle preview action
            if settings.action == "preview" then
                HitMaker.preview_macroline_replacements(settings)
                -- Store the current settings for the next dialog iteration
                current_settings = {
                    macro_range_int_start = settings.macro_range_int_start,
                    macro_range_int_end = settings.macro_range_int_end,
                    line_num_to_replace = settings.line_num_to_replace,
                    replace_this = settings.replace_this,
                    replace_with_this = settings.replace_with_this
                }
                -- Don't set dialog_completed to true, so we loop back to the dialog
            elseif settings.action == "cancel" then
                return nil
            else
                -- User chose to replace, exit the loop
                dialog_completed = true
            end
        end
    end

    local macro_range_int_start = settings.macro_range_int_start
    local macro_range_int_end = settings.macro_range_int_end
    local line_num_to_replace = settings.line_num_to_replace
    local replace_this = settings.replace_this
    local replace_with_this = settings.replace_with_this

    -- Validate required fields
    if not (macro_range_int_start and macro_range_int_end and line_num_to_replace and replace_this and replace_with_this) then
        HitMakerUI.ShowError("Error", "All fields are required and must be valid.")
        return nil
    end

    local macro_table = nil
    for i = macro_range_int_start, macro_range_int_end do
        macro_table = HitMaker.macro_to_table(i)
        for line_number, macro_line in ipairs(macro_table) do
            if line_number == line_num_to_replace then
                local command_string = macro_line.command
                command_string = command_string:gsub(replace_this, replace_with_this)
                macro_line.command = command_string
            end
        end
    end
    return macro_table
end

-- Preview function for macro line replacements
function HitMaker.preview_macroline_replacements(settings)
    local macro_range_int_start = settings.macro_range_int_start
    local macro_range_int_end = settings.macro_range_int_end
    local line_num_to_replace = settings.line_num_to_replace
    local replace_this = settings.replace_this
    local replace_with_this = settings.replace_with_this

    local preview_text = "Preview of macro line replacements:\n\n"
    local changes_found = false

    for i = macro_range_int_start, macro_range_int_end do
        local macro_table = HitMaker.macro_to_table(i)
        for line_number, macro_line in ipairs(macro_table) do
            if line_number == line_num_to_replace then
                local original_command = macro_line.command
                local modified_command = original_command:gsub(replace_this, replace_with_this)
                
                if original_command ~= modified_command then
                    changes_found = true
                    preview_text = preview_text .. "Macro " .. i .. ", Line " .. line_number .. ":\n"
                    preview_text = preview_text .. "  Original: " .. original_command .. "\n"
                    preview_text = preview_text .. "  Modified: " .. modified_command .. "\n\n"
                end
            end
        end
    end

    if not changes_found then
        preview_text = preview_text .. "No changes would be made with the current settings."
    end

    HitMakerUI.ShowInfo("Preview", preview_text)
end

-- CODAS OLDER FUNCTIONS 
-- THESE DO NOT USE MESSAGEBOXMODULE BUT THEY COULD/SHOULD

function HitMaker.SwipeSelector(title, message, selectorName, options, defaultSelection)
    local selected_value = nil
    
    -- Validate inputs
    if not options or #options == 0 then
        Printf("Error: No options provided for swipe selector")
        return nil
    end
    
    -- Set default selection to first option if not specified
    if not defaultSelection then
        defaultSelection = 1
    end
    
    -- Define command buttons for the message box
    local commandButtons = {
        {value = 1, name = "OK"},
        {value = 2, name = "Cancel"}
    }
    
    -- Build the values table for the swipe selector
    -- Format: ["DisplayName"] = returnValue
    local selectorValues = {}
    for i, option in ipairs(options) do
        selectorValues[option] = i
    end
    
    -- Define the swipe selector
    local selectorButtons = {
        {
            name = selectorName or "Select Option",
            selectedValue = defaultSelection,
            type = 0, -- 0 = swipe selector, 1 = radio buttons
            values = selectorValues
        }
    }
    
    -- Create the message box table
    local messageTable = {
        title = title,
        message = message,
        commands = commandButtons,
        selectors = selectorButtons
    }
    
    -- Display the message box and get the result
    local returnTable = MessageBox(messageTable)
    
    -- Check if user clicked OK and extract the selection
    if returnTable.success and returnTable.result == 1 then
        local selectorResult = returnTable.selectors[selectorName or "Select Option"]
        selected_value = options[selectorResult] -- Return the actual option string
        Printf("User selected: " .. selected_value)
    else
        Printf("Selection cancelled by user")
    end
    
    return selected_value
end

function HitMaker.group_selection_popup(title, message)
    -- Set default values if not provided
    title = title or "Group Selector"
    message = message or "Select one or more groups:"
    
    local groups = DataPool().Groups
    local groupData = {}
    local groupStateButtons = {}
    
    -- Extract groups with their numbers for sorting
    for i = 1, groups.count do
        local group = groups[i]
        if group and group.name then
            table.insert(groupData, {
                number = group.no or i,
                name = group.name
            })
        end
    end
    
    -- Sort groups by number
    table.sort(groupData, function(a, b)
        return a.number < b.number
    end)
    
    -- Create state buttons with prefixed names for proper alphabetical sorting
    for _, groupInfo in ipairs(groupData) do
        -- Add zero-padded number prefix to ensure alphabetical order matches numerical order
        local displayName = string.format("%03d - %s", groupInfo.number, groupInfo.name)
        table.insert(groupStateButtons, {name = displayName, state = false})
    end
    
    -- Check if any groups were found
    if #groupStateButtons == 0 then
        Printf("No groups found in DataPool")
        return nil
    end
    
    -- Define command buttons for the message box
    local commandButtons = {
        {value = 1, name = "OK"},
        {value = 2, name = "Cancel"}
    }
    
    -- Create the message box table with state selectors
    local messageTable = {
        title = title,
        message = message,
        commands = commandButtons,
        states = groupStateButtons
    }
    
    -- Display the message box and get the result
    local returnTable = MessageBox(messageTable)
    
    -- Check if user clicked OK and extract selected groups
    if returnTable.success and returnTable.result == 1 then
        local selectedGroups = {}
        
        -- Collect all selected groups and extract original names
        for displayName, isSelected in pairs(returnTable.states) do
            if isSelected then
                -- Extract original group name by removing the number prefix
                local originalName = displayName:match("^%d+%s*-%s*(.+)$") or displayName
                table.insert(selectedGroups, originalName)
            end
        end
        
        return selectedGroups
    else
        Printf("Group selection cancelled by user")
        return nil
    end
end

function HitMaker.execute_group_selection(selectedGroups)
    if not selectedGroups or #selectedGroups == 0 then
        Printf("No groups provided for selection")
        return
    end
    
    -- Execute command for selected groups
    local groupCommand = "SelectFixtures Group"
    for i, groupName in ipairs(selectedGroups) do
        groupCommand = groupCommand .. " \"" .. groupName .. "\""
        if i < #selectedGroups then
            groupCommand = groupCommand .. " +"
        end
    end
    Printf("Selecting groups: " .. table.concat(selectedGroups, ", "))
    Cmd(groupCommand)
end

function HitMaker.group_popup()
    local selectedGroups = HitMaker.group_selection_popup()
    if selectedGroups then
        HitMaker.execute_group_selection(selectedGroups)
    end
end

function HitMaker.sequence_selection_popup(title, message, songName)
    -- Set default values if not provided
    title = title or "Sequence Selector"
    message = message or "Select one or more sequences:"
    
    local sequences = DataPool().Sequences
    local sequenceData = {}
    local sequenceStateButtons = {}
    local song_start = nil
    local song_end = nil
    
    -- If songName is provided, find the song range
    if songName then
        for i = 1, sequences.count do
            local sequence = sequences[i]
            if sequence and sequence.name == songName then
                song_start = sequence.no or i
                song_end = song_start + 99
                Printf("Song range found for '" .. songName .. "': " .. song_start .. " to " .. song_end)
                break
            end
        end
        
        if not song_start then
            Printf("Song '" .. songName .. "' not found in sequences")
            return nil
        end
    end
    
    -- Extract sequences with their numbers for sorting
    for i = 1, sequences.count do
        local sequence = sequences[i]
        if sequence and sequence.name and sequence.name ~= '' then
            local sequenceNumber = sequence.no or i
            
            -- If song filtering is enabled, only include sequences in the song range
            local includeSequence = true
            if songName and song_start and song_end then
                includeSequence = (sequenceNumber >= song_start and sequenceNumber <= song_end)
            end
            
            if includeSequence then
                table.insert(sequenceData, {
                    number = sequenceNumber,
                    name = sequence.name
                })
            end
        end
    end
    
    -- Sort sequences by number
    table.sort(sequenceData, function(a, b)
        return a.number < b.number
    end)
    
    -- Create state buttons with prefixed names for proper alphabetical sorting
    for _, sequenceInfo in ipairs(sequenceData) do
        -- Add zero-padded number prefix to ensure alphabetical order matches numerical order
        local displayName = string.format("%03d - %s", sequenceInfo.number, sequenceInfo.name)
        table.insert(sequenceStateButtons, {name = displayName, state = false})
    end
    
    -- Check if any sequences were found
    if #sequenceStateButtons == 0 then
        Printf("No sequences found in DataPool")
        return nil
    end
    
    -- Define command buttons for the message box
    local commandButtons = {
        {value = 1, name = "OK"},
        {value = 2, name = "Cancel"}
    }
    
    -- Create the message box table with state selectors
    local messageTable = {
        title = title,
        message = message,
        commands = commandButtons,
        states = sequenceStateButtons
    }
    
    -- Display the message box and get the result
    local returnTable = MessageBox(messageTable)
    
    -- Check if user clicked OK and extract selected sequences
    if returnTable.success and returnTable.result == 1 then
        local selectedSequences = {}
        
        -- Collect all selected sequences and extract original names and numbers
        for displayName, isSelected in pairs(returnTable.states) do
            if isSelected then
                -- Extract original sequence name by removing the number prefix
                local originalName = displayName:match("^%d+%s*-%s*(.+)$") or displayName
                -- Extract sequence number from the prefix
                local sequenceNumber = tonumber(displayName:match("^(%d+)%s*-"))
                table.insert(selectedSequences, {
                    name = originalName,
                    number = sequenceNumber
                })
            end
        end
        
        return selectedSequences
    else
        Printf("Sequence selection cancelled by user")
        return nil
    end
end

function HitMaker.preset_popup()
    -- Define preset types and their corresponding integers in display order
    local presetTypes = {
        {name = "Dimmer", id = 1},
        {name = "Position", id = 2},
        {name = "Color", id = 4},
        {name = "Focus", id = 6},
        {name = "Beam", id = 5},
        {name = "Optical", id = 22}
    }
    
    local selectorButtons = {}
    local presetOptionsMap = {}
    
    -- Build selectors for each preset type
    for _, presetType in ipairs(presetTypes) do
        local presets = DataPool().PresetPools[presetType.id]
        local presetOptions = {"NONE"}  -- Start with NONE option
        
        if presets then
            -- Extract preset names from the preset collection
            for i = 1, presets.count do
                local preset = presets[i]
                if preset and preset.name then
                    table.insert(presetOptions, preset.name)
                end
            end
        end
        
        -- Store options for later use
        presetOptionsMap[presetType.name] = {
            options = presetOptions,
            id = presetType.id
        }
        
        -- Build the values table for the swipe selector
        local selectorValues = {}
        for i, option in ipairs(presetOptions) do
            selectorValues[option] = i
        end
        
        -- Create swipe selector for this preset type with order prefix
        -- Add number prefix to force alphabetical ordering to match display order
        local orderPrefix = string.format("%d. ", _)
        table.insert(selectorButtons, {
            name = orderPrefix .. presetType.name,
            selectedValue = 1, -- Default to "NONE" (first option)
            type = 0, -- 0 = swipe selector
            values = selectorValues
        })
    end
    
    -- Define command buttons for the message box
    local commandButtons = {
        {value = 1, name = "OK"},
        {value = 2, name = "Cancel"}
    }
    
    -- Create the message box table with multiple swipe selectors
    local messageTable = {
        title = "Multi-Preset Selector",
        message = "Select presets for each type (swipe to choose, default is NONE):",
        commands = commandButtons,
        selectors = selectorButtons
    }
    
    -- Display the message box and get the result
    local returnTable = MessageBox(messageTable)
    
    -- Check if user clicked OK and process selections
    if returnTable.success and returnTable.result == 1 then
        local selectedPresets = {}
        
        -- Extract selections from each selector
        for i, presetType in ipairs(presetTypes) do
            local orderPrefix = string.format("%d. ", i)
            local prefixedName = orderPrefix .. presetType.name
            local selectorResult = returnTable.selectors[prefixedName]
            local selectedOption = presetOptionsMap[presetType.name].options[selectorResult]
            
            if selectedOption and selectedOption ~= "NONE" then
                selectedPresets[presetType.name] = {
                    name = selectedOption,
                    id = presetOptionsMap[presetType.name].id
                }
                Printf("Selected " .. presetType.name .. ": " .. selectedOption)
            else
                Printf("No " .. presetType.name .. " preset selected")
            end
        end
        
        -- Execute commands for all selected presets
        for presetTypeName, presetInfo in pairs(selectedPresets) do
            local cmd = "At Preset " .. presetInfo.id .. ".\"" .. presetInfo.name .. "\""
            Printf("Executing: " .. cmd)
            Cmd(cmd)
        end
        
        if next(selectedPresets) == nil then
            Printf("No presets were selected")
        end
    else
        Printf("Preset selection cancelled by user")
    end
end


function HitMaker.store_phaser()
    local sequence_label = HitMaker.StrInput("Phaser Name", "Enter Phaser Name:")
    local available_preset = HitMaker.get_available_presets(21)  -- Pass preset type parameter
    local current_song = GetVar(GlobalVars(), "song")
    local page = DataPool().Pages[tostring(current_song)]
    local bpm = GetVar(GlobalVars(), "BPM")
    bpm = bpm / 2

    Cmd("Store Preset 21."..available_preset)

    local available_recipe = HitMaker.get_available_presets(21)

    Cmd("ClearAll")
    Cmd("Store Preset 21."..available_recipe)
    Cmd("Move Preset 21."..available_recipe.." At Preset 21."..available_recipe..".1")
    Cmd("Assign Preset 21."..available_preset.." at Preset 21."..available_recipe..".1")
    Cmd("Set Preset 21."..available_recipe..".1 Property \"SpeedFromX\" \""..bpm.."\"")
    Cmd("Set Preset 21."..available_recipe..".1 Property \"PhaseFromX\" \"0\"")
    Cmd("Set Preset 21."..available_recipe..".1 Property \"PhaseToX\" \"360\"")
    Cmd("Set Preset 21."..available_recipe..".1 Property \"SelectionMode\" \"Strict\"")

    Cmd("Set Preset 21."..available_preset.." Property \"Name\" \""..sequence_label.."_recipe_"..page.name.."\"")
    Cmd("Set Preset 21."..available_recipe.." Property \"Name\" \""..sequence_label.."_phaser_"..page.name.."\"")
end

function HitMaker.get_available_presets(type_int)
    local available_presets = {}
    local available_preset = nil
    local song_start = nil
    local song_end = nil
    local presets = DataPool().PresetPools[type_int]
    local sequences = DataPool().Sequences
    local song_sequence = nil
    local current_song = GetVar(GlobalVars(), "song")

    Printf("Looking for song: '"..current_song.."' in "..sequences.count.." sequences")
    
    -- Checks if song sequence is in the sequence datapool and returns the integer
    for i = 1, sequences.count do
        local sequence = sequences[i]
        if sequence then
            Printf("Checking sequence "..i..": '"..tostring(sequence.name).."'")
            if sequence.name == current_song then
                song_start = sequence.no
                song_end = sequence.no + 99
                Printf("Song Int Found "..song_start.." (range: "..song_start.." to "..song_end..")")
                break
            end
        end
    end
    
    if not song_start then
        Printf("Error: Song '"..current_song.."' not found in sequences")
        return nil
    end

    if song_start then
        for i = song_start, song_end do
            if (presets[i] == nil or presets[i] == '') then
                available_presets[i] = ""
                Printf("Adding available preset: "..i)
                if available_preset == nil then
                    available_preset = i
                    Printf("Found first available preset: "..i)
                    break
                end
            end
        end
    end
    if available_preset then
        Printf("Available Preset: "..type_int.."."..available_preset)
        return available_preset
    else
        Printf("Error: No available preset found in range "..song_start.." to "..song_end.." for preset type "..type_int)
        return nil
    end
end

function HitMaker.get_used_sequences()
    local used_sequences = {}
    local song_start = nil
    local song_end = nil
    local sequences = DataPool().Sequences
    local current_song = GetVar(GlobalVars(), "song")

    Printf("current_song "..current_song)

    -- Checks if song sequence is in datapool and returns integer
    for i = 1, sequences.count do
        local sequence = sequences[i]
        if sequence then
            if sequence.name == current_song then
                song_start = sequence.no
                song_end = song_start + 99
                Printf("Song range found: "..song_start.." to "..song_end)
                break
            end
        end
    end

    -- Only proceed if we found the song sequence
    if song_start then
        -- Collect all used sequences in the song range
        for i = song_start, song_end do
            local sequence = DataPool().Sequences[i]
            if sequence and sequence ~= '' then
                table.insert(used_sequences, {
                    number = i,
                    name = sequence.name or ("Sequence " .. i),
                    sequence = sequence
                })
                Printf("Found used sequence: " .. i .. " - " .. (sequence.name or "Unnamed"))
            end
        end

        -- Sort sequences by number
        table.sort(used_sequences, function(a, b)
            return a.number < b.number
        end)

        Printf("Total used sequences found: "..#used_sequences)
    end

    return used_sequences
end


function HitMaker.process_timing_command(command_text)
    -- Check if command contains timing parameters (case insensitive, more flexible patterns)
    local has_timing = command_text:lower():find("delayfrom[xyz]") or command_text:lower():find("delayto[xyz]") or 
                      command_text:lower():find("fadefrom[xyz]") or command_text:lower():find("fadeto[xyz]")
    
    if not has_timing then
        return command_text, false -- No timing found, return original
    end
    
    Printf("Timing command detected: " .. command_text)
    
    -- Get beats from user
    local beats_input = HitMaker.StrInput("Timing Input", "Enter number of beats:")
    if not beats_input then
        return command_text, false -- User cancelled
    end
    
    local beats = tonumber(beats_input)
    if not beats then
        Printf("Invalid number entered for beats")
        return command_text, false
    end
    
    local timing_value = HitMaker.get_beat_times(beats)
    local timing_value = beats * beat_time
    
    Printf("Converting " .. beats .. " beats to " .. timing_value .. " seconds")
    
    -- Replace timing parameters with more specific patterns
    local processed_command = command_text
    
    -- Replace any parentheses containing DelayFrom or FadeFrom (case insensitive)
    processed_command = processed_command:gsub("%(([^%)]*[Dd]elay[Ff]rom[XYZxyz][^%)]*)%)", "\"" .. timing_value .. "\"")
    processed_command = processed_command:gsub("%(([^%)]*[Ff]ade[Ff]rom[XYZxyz][^%)]*)%)", "\"" .. timing_value .. "\"")
    
    -- Replace any parentheses containing DelayTo or FadeTo (case insensitive)
    processed_command = processed_command:gsub("%(([^%)]*[Dd]elay[Tt]o[XYZxyz][^%)]*)%)", "\"0\"")
    processed_command = processed_command:gsub("%(([^%)]*[Ff]ade[Tt]o[XYZxyz][^%)]*)%)", "\"0\"")
    
    Printf("Processed command: " .. processed_command)
    
    return processed_command, true
end

function HitMaker.set_all_programmer_matricks()
    Printf("set_all_programmer_matricks")
    -- Get the programmer object
    local prog = Programmer()
    if not prog then
        Printf("Error: Could not access programmer")
        return
    else
        Printf("Programmer is Valid")
    end

    -- Get Part 0
    local part0 = prog[1]
    if not part0 then
        Printf("Error: Could not access Part 0")
        return
    else
        Printf("Part 0 is Valid")
    end

    local partchildren = part0:Children()

    if partchildren then
        Printf("Part Children are valid")
        -- Dump all children
        for i, child in pairs(partchildren) do
            if child then
                Printf(string.format("Child %d: %s", i, tostring(child)))
            end
        end
    else
        Printf("No children for cuepart")
    end
    
    -- Properties to get
    local properties = {
        -- X Properties
        "FadeFromX", "FadeToX",
        "DelayFromX", "DelayToX",
        "SpeedFromX", "SpeedToX",
        "PhaseFromX", "PhaseToX",
        "X", "XGroup", "XBlock", "XWings",
        
        -- Y Properties
        "FadeFromY", "FadeToY",
        "DelayFromY", "DelayToY",
        "SpeedFromY", "SpeedToY",
        "PhaseFromY", "PhaseToY",
        "Y", "YGroup", "YBlock", "YWings",
        
        -- Z Properties
        "FadeFromZ", "FadeToZ",
        "DelayFromZ", "DelayToZ",
        "SpeedFromZ", "SpeedToZ",
        "PhaseFromZ", "PhaseToZ",
        "Z", "ZGroup", "ZBlock", "ZWings"
    }

    local lastline = nil
    last_index = nil
    if partchildren then
        -- Find the last valid index in the children table
        local lastIndex = 0
        for i, _ in pairs(partchildren) do
            if type(i) == "number" and i > lastIndex then
                lastIndex = i
            end
        end
        if lastIndex > 0 then
            lastline = partchildren[lastIndex]
            last_index = lastIndex
        end
    end

    local children_to_update = {}
    for i = 1, last_index - 1 do
        if partchildren[i] then
            table.insert(children_to_update, partchildren[i])
        end
    end
    

    target_properties = {}

    if lastline then
        Printf("Last Line is valid: " .. tostring(lastline))
    else
        Printf("Last Line not valid")
    end

    if #children_to_update <= 1 then
        Printf("There is 1 or fewer lines, exiting method")
        return
    else
        Printf("There is more than 1 child, running matrick copy")
        for i = 1, #properties, 1 do
            local value = lastline[properties[i]]
            if value == nil or value == "None" then
                Printf("Skipping Nil/None property: "..properties[i])
            else
                target_properties[properties[i]] = value
                Printf("Adding to target_properties: "..properties[i].." : "..tostring(value))
            end
        end
        Printf("target_properties created -> Updating lines")
        for i = 1, #children_to_update, 1 do
            Printf("Updating Recipe: "..children_to_update[i].name)
            local recipeline = tonumber(children_to_update[i].name:match("(%d+)$")) 
            for property, value in pairs(target_properties) do
                Printf("Updating Property: "..property.." for recipe: "..children_to_update[i].name)
                Cmd("Set Programmer 0."..recipeline.." Property ".."\""..property.."\" \""..value.."\"")
            end
        end
    end
end

function HitMaker.get_cues_in_sequence(sequence_int)
    local cue_table = DataPool().Sequences[sequence_int]
    return cue_table
end

function HitMaker.get_sequence_list(song_name)
    local sequence_range = song_name or ""
    local sequences = DataPool().Sequences

end

function HitMaker.CopyAndReplaceRecipes()
    local current_song = GetVar(GlobalVars(), "song")
    local source_group = HitMaker.group_selection_popup("Copy + Replace Source Group", "Set the source group")
    local dest_group = HitMaker.group_selection_popup("Copy + Replace Destination Group", "Set the destination group")
    local target_sequences = HitMaker.sequence_selection_popup("Target Sequence", "Select Sequence to Copy + Replace Groups in: ", current_song)
    
    source_group = source_group[1]
    dest_group = dest_group[1]

    Printf("Source: "..source_group.." Dest Group: "..dest_group)

    if not target_sequences then
        Printf("No sequences selected")
        return
    end

    -- Iterate through selected sequences
    for _, seq in ipairs(target_sequences) do
        Printf("Processing sequence: " .. seq.name .. " (Number: " .. seq.number .. ")")
        
        local sequence = DataPool().Sequences[seq.number]
        if not sequence then
            Printf("Could not find sequence number: " .. seq.number)
            goto continue
        end

        -- Get all cues in the sequence
        local cues = sequence:Children()
        if not cues then
            Printf("No cues found in sequence: " .. seq.number)
            goto continue
        end

        -- Iterate through actual cues
        for cueIndex = 1, #cues do
            local cue = cues[cueIndex]
            if not cue then
                goto continue_cue
            end
            
            -- Get the actual cue number (might be different from index)
            local actualCueNumber = cue.number or cue.no
            if actualCueNumber then
                actualCueNumber = actualCueNumber/1000
            end
        
            if not actualCueNumber then
                Printf("Could not get cue number for cue at index: " .. cueIndex)
                goto continue_cue
            end

            --Printf("Processing cue: " .. (cue.name or "Unnamed") .. " at index " .. cueIndex)
            
            -- Get cue parts
            local cueparts = cue:Children()
            if not cueparts then
                Printf("No cue parts found for cue: " .. cueIndex)
                goto continue_cue
            end

            -- Iterate through cue parts using count or length
            local numParts = cueparts.count or #cueparts
            for partIndex = 1, numParts do
                local part = cueparts[partIndex]
                if not part then
                    Printf("Skipping nil part: " .. partIndex)
                    goto continue_part
                end

                --Printf("Processing part: " .. (part.name or "Unnamed") .. " at index " .. partIndex)
                
                -- Get recipes
                local recipes = part:Children()
                if not recipes then
                    Printf("No recipes found for part: " .. partIndex)
                    goto continue_part
                end

                -- Iterate through recipes
                local numRecipes = recipes.count or #recipes
                --Printf("Found " .. numRecipes .. " recipes in part " .. partIndex)
                
                for recipeIndex = 1, numRecipes do
                    local recipe = recipes[recipeIndex]
                    if recipe then
                        -- Add your recipe processing logic here
                        if recipe.selection == DataPool().Groups[source_group] then
                            -- Verify the recipe exists and has valid data
                            if not recipe.selection then
                                Printf("Warning: Recipe at index " .. recipeIndex .. " has no selection data")
                                goto continue_recipe
                            end

                            Cmd("ClearAll")
                            -- Construct the part number in y.z format where y starts at 0 ((partIndex-1).recipeIndex)
                            local partNumber = string.format("%d.%d", partIndex - 1, recipeIndex)
                            Printf("Processing Part Number: " .. partNumber)

                            -- Build and execute commands with proper part number
                            local copyCmd = string.format("Copy Sequence %d Cue %g Part %s At Programmer 0.1", 
                                seq.number, actualCueNumber, partNumber)
                            Printf("Executing copy command: " .. copyCmd)
                            local copyResult = Cmd(copyCmd)
                            
                            -- Verify the copy was successful
                            if not copyResult then
                                Printf("Warning: Failed to copy part " .. partNumber)
                                goto continue_recipe
                            end

                            local assignCmd = string.format("Assign Group \"%s\" At Programmer 0.1", dest_group)
                            Printf("Executing assign command: " .. assignCmd)
                            local assignResult = Cmd(assignCmd)
                            
                            -- Verify the assign was successful
                            if not assignResult then
                                Printf("Warning: Failed to assign group " .. dest_group)
                                goto continue_recipe
                            end

                            -- Store command should merge changes to the specific part
                            -- Extract just the part number (before the dot) for storing
                            local storePart = partIndex - 1  -- Convert to 0-based part number
                            local storeCmd = string.format("Store Sequence %d Cue %g Part %d /merge", 
                                seq.number, actualCueNumber, storePart)
                            Printf("Executing store command: " .. storeCmd)
                            local storeResult = Cmd(storeCmd)
                            
                            -- Verify the store was successful
                            if not storeResult then
                                Printf("Warning: Failed to store changes to part " .. partNumber)
                            end
                            
                            ::continue_recipe::
                        end
                    end
                end

                ::continue_part::
            end

            ::continue_cue::
        end

        ::continue::
    end
end

function HitMaker.macro_to_table(macro_int)
    local macro_table = {}

    local macro_integer = tonumber(macro_int)

    macro_object = DataPool().Macros[macro_integer]

    macro_table = macro_object:Children()

    return macro_table
end

-- Enhanced UI utility functions (expose HitMakerUI functionality)
function HitMaker.ShowError(title, message)
    return HitMakerUI.ShowError(title, message)
end

function HitMaker.ShowWarning(title, message)
    return HitMakerUI.ShowWarning(title, message)
end

function HitMaker.ShowInfo(title, message)
    return HitMakerUI.ShowInfo(title, message)
end

function HitMaker.ShowConfirmation(title, message)
    return HitMakerUI.ShowConfirmation(title, message)
end

-- Configuration dialogs
function HitMaker.SongPageDialog()
    return HitMakerUI.SongPageDialog()
end

function HitMaker.ExecutorRangeDialog()
    return HitMakerUI.ExecutorRangeDialog()
end

function HitMaker.SequenceRangeDialog()
    return HitMakerUI.SequenceRangeDialog()
end

return HitMaker
