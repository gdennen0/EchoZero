EZ = EZ or {}

function EZ.PrintTimeRanges(tcNo, tgNo, trackNo)
    local ma3TrackNo = (trackNo or 0) + 1
    local track = DataPool().Timecodes[tcNo][tgNo][ma3TrackNo]
    local children = track:Children() or {}
    EZ.log("Printing TimeRanges for track: " .. tostring(track.index).. " " .. tostring(track.name or "nil"))

    for i = 1, #children do
        local child = children[i]
        EZ.log("index: " .. tostring(i))
        EZ.log("name: " .. tostring(child and child.name or "nil"))
        EZ.log("class: " .. tostring(child and child.GetClass and child:GetClass() or "nil"))
    end
end

function EZ.PrintCmdSubTracks(tcNo, tgNo, trackNo)
    local ma3TrackNo = (trackNo or 0) + 1
    local track = DataPool().Timecodes[tcNo][tgNo][ma3TrackNo]
    local time_ranges = track:Children() or {}
    EZ.log("Printing CmdSubTracks for track: " .. tostring(track.index).. " " .. tostring(track.name or "nil"))

    for i = 1, #time_ranges do
        local time_range = time_ranges[i]
        if time_range and time_range.GetClass and time_range:GetClass() == "TimeRange" then
            EZ.log("TimeRange: " .. tostring(i) .. " name: " .. tostring(time_range and time_range.name or "nil"))
            local cmd_subtracks = time_range:Children() or {}
            for j = 1, #cmd_subtracks do
                local cmd_subtrack = cmd_subtracks[j]
                EZ.log("cmd_subtrack: " .. tostring(j) .. " name: " .. tostring(cmd_subtrack and cmd_subtrack.name or "nil"))
                EZ.log("cmd_subtrack class: " .. tostring(cmd_subtrack and cmd_subtrack:GetClass() or "nil"))
            end
        end
    end
end

