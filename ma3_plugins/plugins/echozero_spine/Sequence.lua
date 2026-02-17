EZ = EZ or {}

-- PUBLIC API: SEQUENCE MANAGEMENT
-- These functions support Editor->MA3 sync by managing sequences

-- Get all sequences from the DataPool
function EZ.GetSequences()
    EZ.log("GetSequences() called")
    local dp = EZ.getDP()
    if not dp then
        EZ.log("ERROR: GetSequences() - DataPool not accessible")
        EZ.sendMessage("sequences", "error", {error = "DataPool not accessible"})
        return nil
    end
    
    if not dp.Sequences then
        EZ.log("ERROR: GetSequences() - No Sequences in DataPool")
        EZ.sendMessage("sequences", "list", {count = 0, sequences = {}})
        return {}
    end
    
    local result = {}
    local ok, children = pcall(function() return dp.Sequences:Children() end)
    if ok and children then
        for _, seq in ipairs(children) do
            if seq then
                table.insert(result, {
                    no = seq.no,
                    name = seq.name or "",
                })
            end
        end
        EZ.log(string.format("GetSequences() -> Found %d sequences", #result))
    else
        EZ.log("ERROR: GetSequences() - Failed to enumerate sequences")
    end
    
    EZ.sendMessage("sequences", "list", {count = #result, sequences = result})
    return result
end

-- Check if a sequence exists by number
function EZ.SequenceExists(seqNo)
    EZ.log(string.format("SequenceExists(%s) called", tostring(seqNo)))
    local dp = EZ.getDP()
    if not dp or not dp.Sequences then
        EZ.sendMessage("sequence", "exists", {no = seqNo, exists = false})
        return false
    end
    
    local exists = dp.Sequences[seqNo] ~= nil
    EZ.log(string.format("  Sequence %d exists: %s", seqNo, tostring(exists)))
    EZ.sendMessage("sequence", "exists", {no = seqNo, exists = exists})
    return exists
end

-- Find the next available (unused) sequence number
function EZ.GetNextAvailableSequence(startFrom)
    startFrom = startFrom or 1
    EZ.log(string.format("GetNextAvailableSequence(%d) called", startFrom))
    local dp = EZ.getDP()
    if not dp then
        EZ.sendMessage("sequence", "next_available", {no = startFrom, error = "DataPool not accessible"})
        return startFrom
    end
    
    -- Find first unused sequence number
    local seqNo = startFrom
    local maxCheck = 9999  -- Don't loop forever
    while seqNo < maxCheck do
        if not dp.Sequences or not dp.Sequences[seqNo] then
            EZ.log(string.format("  Next available sequence: %d", seqNo))
            EZ.sendMessage("sequence", "next_available", {no = seqNo})
            return seqNo
        end
        seqNo = seqNo + 1
    end
    
    EZ.log("ERROR: Could not find available sequence number")
    EZ.sendMessage("sequence", "error", {error = "No available sequence numbers"})
    return nil
end

-- Create a new sequence at a specific number
-- Returns true if created, false if already exists or failed
function EZ.CreateSequence(seqNo, name)
    EZ.log(string.format("CreateSequence(%s, '%s') called", tostring(seqNo), tostring(name or "")))
    local dp = EZ.getDP()
    if not dp then
        EZ.log("ERROR: CreateSequence() - DataPool not accessible")
        EZ.sendMessage("sequence", "error", {no = seqNo, error = "DataPool not accessible"})
        return false
    end
    
    -- Check if sequence already exists
    if dp.Sequences and dp.Sequences[seqNo] then
        EZ.log(string.format("  Sequence %d already exists", seqNo))
        EZ.sendMessage("sequence", "exists", {no = seqNo, exists = true, created = false})
        return false
    end
    
    -- Create sequence using MA3 command
    local cmdStr = string.format('Store Sequence %d', seqNo)
    if name and name ~= "" then
        cmdStr = cmdStr .. string.format(' /name="%s"', name)
    end
    cmdStr = cmdStr .. ' /nc'  -- No confirmation
    
    EZ.log(string.format("  Executing: %s", cmdStr))
    local ok, result = EZ.RunCommand(cmdStr, "feedback")
    
    if ok then
        -- Verify it was created
        if dp.Sequences and dp.Sequences[seqNo] then
            EZ.log(string.format("  Created sequence %d", seqNo))
            EZ.sendMessage("sequence", "created", {no = seqNo, name = name or ""})
            return true
        else
            EZ.log(string.format("  WARNING: Command succeeded but sequence not found"))
        end
    end
    
    EZ.log(string.format("ERROR: Failed to create sequence %d", seqNo))
    EZ.sendMessage("sequence", "error", {no = seqNo, error = "Failed to create sequence"})
    return false
end
