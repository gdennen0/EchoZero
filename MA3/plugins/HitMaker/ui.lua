-- HitMaker UI Module
-- Enhanced UI functionality for HitMaker plugin using MessageBoxModule
-- Version: 1.0.0


HitMakerUI = {}

-- Basic text input with validation (backwards compatible with StrInput)
function HitMakerUI.StrInput(title, message, defaultValue, maxLength)
    local result = MessageBoxModule.TextInput(
        title or "Text Input",
        message or "Enter text:",
        "Text Input",
        defaultValue or "",
        maxLength or 50
    )
    
    if result.success and MessageBoxModule.WasCommandClicked(result, 1) then
        return MessageBoxModule.GetInputValue(result, "Text Input")
    end
    
    return nil
end

-- Enhanced sequence creation dialog
function HitMakerUI.SequenceCreationDialog()
    local inputs = {
        {name = "Sequence Name", value = "", maxTextLength = 30}
    }
    
    local states = {
        {name = "Clear All First", state = false, group = 2}
    }
    
    local selectors = {
        {
            name = "Assignment Method",
            selectedValue = 3,
            values = {["Auto-assign Executor"] = 1, ["Tap on Exec After Close"] = 2, ["Do Not Assign"] = 3},
            type = 1  -- Radio selector for mutually exclusive options
        }
    }
    
    local commands = {
        {value = 1, name = "Create"},
        {value = 0, name = "Cancel"}
    }
    
    local result = MessageBoxModule.Complex(
        "Create Sequence",
        "Configure new sequence settings:",
        inputs,
        states,
        selectors,
        commands
    )
    
    if result.success and MessageBoxModule.WasCommandClicked(result, 1) then
        local assignmentMethod = MessageBoxModule.GetSelectorValue(result, "Assignment Method")
        
        return {
            name = MessageBoxModule.GetInputValue(result, "Sequence Name"),
            autoAssign = (assignmentMethod == 1),
            tapOnExec = (assignmentMethod == 2),
            doNotAssign = (assignmentMethod == 3),
            clearFirst = MessageBoxModule.GetStateValue(result, "Clear All First")
        }
    end
    
    return nil
end

-- Enhanced go hit creation dialog with timing options
function HitMakerUI.GoHitCreationDialog()
    local inputs = {
        {name = "Sequence Name", value = "", maxTextLength = 30},
        {name = "Off Time (Beats/Seconds)", value = "1", whiteFilter = "0123456789.", maxTextLength = 8}
    }
    
    local states = {
        {name = "Clear All First", state = false, group = 2},
        {name = "Follow Trigger", state = true, group = 3}
    }
    
    local selectors = {
        {
            name = "Assignment Method",
            selectedValue = 1,
            values = {["Auto-assign Executor"] = 1, ["Tap on Exec After Close"] = 2, ["Do Not Assign"] = 3},
            type = 1  -- Radio selector for mutually exclusive options
        },
        {
            name = "Timing Type",
            selectedValue = 1,
            values = {["BPM"] = 1, ["Seconds"] = 2},
            type = 1
        }
    }
    
    local commands = {
        {value = 1, name = "Create"},
        {value = 0, name = "Cancel"}
    }
    
    local result = MessageBoxModule.Complex(
        "Create Go Hit",
        "Configure go hit sequence with timing:",
        inputs,
        states,
        selectors,
        commands
    )
    
    if result.success and MessageBoxModule.WasCommandClicked(result, 1) then
        local assignmentMethod = MessageBoxModule.GetSelectorValue(result, "Assignment Method")
        
        return {
            name = MessageBoxModule.GetInputValue(result, "Sequence Name"),
            offtime = MessageBoxModule.GetInputValue(result, "Off Time (Beats/Seconds)"),
            autoAssign = (assignmentMethod == 1),
            tapOnExec = (assignmentMethod == 2),
            doNotAssign = (assignmentMethod == 3),
            clearFirst = MessageBoxModule.GetStateValue(result, "Clear All First"),
            followTrigger = MessageBoxModule.GetStateValue(result, "Follow Trigger"),
            timingType = MessageBoxModule.GetSelectorValue(result, "Timing Type")
        }
    end
    
    return nil
end

-- Enhanced four hit matrix creation dialog
function HitMakerUI.FourHitMatrixDialog()
    local inputs = {
        {name = "Hit Name", value = "", maxTextLength = 25},
        {name = "Base Fade Time", value = "1", whiteFilter = "0123456789.", maxTextLength = 8}
    }
    
    local states = {
        {name = "Clear All First", state = false, group = 2},
        {name = "Create Individual Hits", state = true, group = 3},
        {name = "Set Restart Mode", state = true, group = 3}
    }
    
    local selectors = {
        {
            name = "Assignment Method",
            selectedValue = 1,
            values = {["Auto-assign Executors"] = 1, ["Tap on Exec After Close"] = 2, ["Do Not Assign"] = 3},
            type = 1  -- Radio selector for mutually exclusive options
        },
        {
            name = "Matrix Type",
            selectedValue = 1,
            values = {["4 Hit Standard"] = 1, ["4 Hit Advanced"] = 2, ["Custom"] = 3},
            type = 0
        },
        {
            name = "Fade Time Type",
            selectedValue = 1,
            values = {["BPM"] = 1, ["Seconds"] = 2},
            type = 1
        }
    }
    
    local commands = {
        {value = 1, name = "Create Matrix"},
        {value = 2, name = "Preview"},
        {value = 0, name = "Cancel"}
    }
    
    local result = MessageBoxModule.Complex(
        "Create Four Hit Matrix",
        "Configure four hit matrix settings:",
        inputs,
        states,
        selectors,
        commands
    )
    
    if result.success and MessageBoxModule.WasCommandClicked(result, 1) then
        local assignmentMethod = MessageBoxModule.GetSelectorValue(result, "Assignment Method")
        
        return {
            name = MessageBoxModule.GetInputValue(result, "Hit Name"),
            baseFadeTime = MessageBoxModule.GetInputValue(result, "Base Fade Time"),
            autoAssign = (assignmentMethod == 1),
            tapOnExec = (assignmentMethod == 2),
            doNotAssign = (assignmentMethod == 3),
            clearFirst = MessageBoxModule.GetStateValue(result, "Clear All First"),
            createIndividual = MessageBoxModule.GetStateValue(result, "Create Individual Hits"),
            setRestartMode = MessageBoxModule.GetStateValue(result, "Set Restart Mode"),
            matrixType = MessageBoxModule.GetSelectorValue(result, "Matrix Type"),
            fadeTimeType = MessageBoxModule.GetSelectorValue(result, "Fade Time Type")
        }
    end
    
    return nil
end

-- Enhanced timing input dialog for beat/time conversion
function HitMakerUI.TimingInputDialog(title, message)
    local inputs = {
        {name = "Beats", value = "1", whiteFilter = "0123456789.", maxTextLength = 8}
    }
    
    local selectors = {
        {
            name = "Time Unit",
            selectedValue = 1,
            values = {["Beats"] = 1, ["Seconds"] = 2, ["Milliseconds"] = 3},
            type = 1
        }
    }
    
    local states = {
        {name = "Apply to All", state = false, group = 1}
    }
    
    local commands = {
        {value = 1, name = "Apply"},
        {value = 0, name = "Cancel"}
    }
    
    local result = MessageBoxModule.Complex(
        title or "Timing Input",
        message or "Enter timing value:",
        inputs,
        states,
        selectors,
        commands
    )
    
    if result.success and MessageBoxModule.WasCommandClicked(result, 1) then
        return {
            value = MessageBoxModule.GetInputValue(result, "Beats"),
            unit = MessageBoxModule.GetSelectorValue(result, "Time Unit"),
            applyToAll = MessageBoxModule.GetStateValue(result, "Apply to All")
        }
    end
    
    return nil
end

-- BPM Settings dialog
function HitMakerUI.BPMSettingsDialog()
    local inputs = {
        {name = "Master BPM", value = tostring(GetVar(GlobalVars(), "BPM") or "120"), whiteFilter = "0123456789.", maxTextLength = 6}
    }
    
    local states = {
        {name = "Set Master 3.1 (BPM/4)", state = true, group = 1},
        {name = "Set Master 3.2 (BPM/2)", state = true, group = 1},
        {name = "Set Master 3.3 (BPM)", state = true, group = 1},
        {name = "Set Master 3.4 (BPM*2)", state = true, group = 1}
    }
    
    local commands = {
        {value = 1, name = "Apply"},
        {value = 2, name = "Test"},
        {value = 0, name = "Cancel"}
    }
    
    local result = MessageBoxModule.Complex(
        "BPM Settings",
        "Configure BPM settings for masters:",
        inputs,
        states,
        {},
        commands
    )
    
    if result.success and MessageBoxModule.WasCommandClicked(result, 1) then
        return {
            bpm = MessageBoxModule.GetInputValue(result, "Master BPM"),
            setMaster1 = MessageBoxModule.GetStateValue(result, "Set Master 3.1 (BPM/4)"),
            setMaster2 = MessageBoxModule.GetStateValue(result, "Set Master 3.2 (BPM/2)"),
            setMaster3 = MessageBoxModule.GetStateValue(result, "Set Master 3.3 (BPM)"),
            setMaster4 = MessageBoxModule.GetStateValue(result, "Set Master 3.4 (BPM*2)")
        }
    end
    
    return nil
end

-- Error dialogs for better user feedback
function HitMakerUI.ShowError(title, message)
    return MessageBoxModule.Error(title, message)
end

-- Warning dialogs
function HitMakerUI.ShowWarning(title, message)
    return MessageBoxModule.Warning(title, message)
end

-- Info dialogs
function HitMakerUI.ShowInfo(title, message)
    return MessageBoxModule.Info(title, message)
end

-- Confirmation dialogs
function HitMakerUI.ShowConfirmation(title, message)
    return MessageBoxModule.YesNo(title, message)
end

-- Song/Page selection dialog
function HitMakerUI.SongPageDialog()
    local currentSong = GetVar(GlobalVars(), "song") or "1"
    
    local inputs = {
        {name = "Song Number", value = currentSong, whiteFilter = "0123456789", maxTextLength = 3}
    }
    
    local states = {
        {name = "Auto-create if missing", state = true, group = 1},
        {name = "Set as current song", state = true, group = 1}
    }
    
    local commands = {
        {value = 1, name = "Select"},
        {value = 0, name = "Cancel"}
    }
    
    local result = MessageBoxModule.Complex(
        "Select Song/Page",
        "Choose song/page for hit creation:",
        inputs,
        states,
        {},
        commands
    )
    
    if result.success and MessageBoxModule.WasCommandClicked(result, 1) then
        return {
            songNumber = MessageBoxModule.GetInputValue(result, "Song Number"),
            autoCreate = MessageBoxModule.GetStateValue(result, "Auto-create if missing"),
            setAsCurrent = MessageBoxModule.GetStateValue(result, "Set as current song")
        }
    end
    
    return nil
end

-- Available executor range dialog
function HitMakerUI.ExecutorRangeDialog()
    local inputs = {
        {name = "Start Executor", value = "101", whiteFilter = "0123456789", maxTextLength = 3},
        {name = "End Executor", value = "115", whiteFilter = "0123456789", maxTextLength = 3}
    }
    
    local states = {
        {name = "Show used executors", state = true, group = 1},
        {name = "Auto-find next available", state = true, group = 1}
    }
    
    local commands = {
        {value = 1, name = "Set Range"},
        {value = 2, name = "Scan Available"},
        {value = 0, name = "Cancel"}
    }
    
    local result = MessageBoxModule.Complex(
        "Executor Range",
        "Configure executor range for hits:",
        inputs,
        states,
        {},
        commands
    )
    
    if result.success then
        return {
            startExecutor = MessageBoxModule.GetInputValue(result, "Start Executor"),
            endExecutor = MessageBoxModule.GetInputValue(result, "End Executor"),
            showUsed = MessageBoxModule.GetStateValue(result, "Show used executors"),
            autoFind = MessageBoxModule.GetStateValue(result, "Auto-find next available"),
            action = MessageBoxModule.WasCommandClicked(result, 1) and "set" or 
                    MessageBoxModule.WasCommandClicked(result, 2) and "scan" or "cancel"
        }
    end
    
    return nil
end

-- Sequence range dialog
function HitMakerUI.SequenceRangeDialog()
    local currentSong = GetVar(GlobalVars(), "song") or "1"
    local songStart = tonumber(currentSong) * 100 or 100
    
    local inputs = {
        {name = "Start Sequence", value = tostring(songStart), whiteFilter = "0123456789", maxTextLength = 4},
        {name = "End Sequence", value = tostring(songStart + 99), whiteFilter = "0123456789", maxTextLength = 4}
    }
    
    local states = {
        {name = "Show used sequences", state = true, group = 1},
        {name = "Auto-find next available", state = true, group = 1}
    }
    
    local commands = {
        {value = 1, name = "Set Range"},
        {value = 2, name = "Scan Available"},
        {value = 0, name = "Cancel"}
    }
    
    local result = MessageBoxModule.Complex(
        "Sequence Range",
        "Configure sequence range for hits:",
        inputs,
        states,
        {},
        commands
    )
    
    if result.success then
        return {
            startSequence = MessageBoxModule.GetInputValue(result, "Start Sequence"),
            endSequence = MessageBoxModule.GetInputValue(result, "End Sequence"),
            showUsed = MessageBoxModule.GetStateValue(result, "Show used sequences"),
            autoFind = MessageBoxModule.GetStateValue(result, "Auto-find next available"),
            action = MessageBoxModule.WasCommandClicked(result, 1) and "set" or 
                    MessageBoxModule.WasCommandClicked(result, 2) and "scan" or "cancel"
        }
    end
    
    return nil
end

-- Replace macro line range dialog
function HitMakerUI.ReplaceMacroLineRangeDialog()
    return HitMakerUI.ReplaceMacroLineRangeDialogWithValues({})
end

-- Replace macro line range dialog with pre-filled values
function HitMakerUI.ReplaceMacroLineRangeDialogWithValues(prefill_values)
    local inputs = {
        {name = "Macro Range Start", value = tostring(prefill_values.macro_range_int_start or "1"), whiteFilter = "0123456789", maxTextLength = 5},
        {name = "Macro Range End", value = tostring(prefill_values.macro_range_int_end or "5"), whiteFilter = "0123456789", maxTextLength = 5},
        {name = "Line Number To Replace", value = tostring(prefill_values.line_num_to_replace or "1"), whiteFilter = "0123456789", maxTextLength = 5},
        {name = "Replace This", value = prefill_values.replace_this or "", maxTextLength = 50},
        {name = "Replace With This", value = prefill_values.replace_with_this or "", maxTextLength = 50}
    }
    
    local states = {
        {name = "Show Preview", state = true, group = 1},
        {name = "Confirm Each Macro", state = false, group = 1}
    }
    
    local commands = {
        {value = 1, name = "Replace"},
        {value = 2, name = "Preview"},
        {value = 0, name = "Cancel"}
    }
    
    local result = MessageBoxModule.Complex(
        "Replace Macro Line Range",
        "Configure macro line replacement settings:",
        inputs,
        states,
        {},
        commands
    )
    
    if result.success then
        local action = MessageBoxModule.WasCommandClicked(result, 1) and "replace" or 
                      MessageBoxModule.WasCommandClicked(result, 2) and "preview" or "cancel"
        
        return {
            macro_range_int_start = tonumber(MessageBoxModule.GetInputValue(result, "Macro Range Start")),
            macro_range_int_end = tonumber(MessageBoxModule.GetInputValue(result, "Macro Range End")),
            line_num_to_replace = tonumber(MessageBoxModule.GetInputValue(result, "Line Number To Replace")),
            replace_this = MessageBoxModule.GetInputValue(result, "Replace This"),
            replace_with_this = MessageBoxModule.GetInputValue(result, "Replace With This"),
            showPreview = MessageBoxModule.GetStateValue(result, "Show Preview"),
            confirmEach = MessageBoxModule.GetStateValue(result, "Confirm Each Macro"),
            action = action
        }
    end
    
    return nil
end

return HitMakerUI
