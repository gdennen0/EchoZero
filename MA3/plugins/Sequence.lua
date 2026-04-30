EZ = EZ or {}

-- PUBLIC API: SEQUENCE MANAGEMENT
-- These functions support Editor->MA3 sync by managing sequences

-- Get all sequences from the DataPool (optionally filtered and correlated).
function EZ.GetSequences(startNo, endNo, request_id)
    EZ.log(string.format("GetSequences(%s, %s, %s) called", tostring(startNo), tostring(endNo), tostring(request_id)))
    local dp = EZ.getDP()
    if not dp then
        EZ.log("ERROR: GetSequences() - DataPool not accessible")
        local errorPayload = {error = "DataPool not accessible"}
        if request_id ~= nil then
            errorPayload.request_id = request_id
        end
        EZ.sendMessage("sequences", "error", errorPayload)
        return nil
    end
    
    if not dp.Sequences then
        EZ.log("ERROR: GetSequences() - No Sequences in DataPool")
        local emptyPayload = {
            count = 0,
            offset = 1,
            chunk_index = 1,
            total_chunks = 1,
            sequences = {}
        }
        if request_id ~= nil then
            emptyPayload.request_id = request_id
        end
        EZ.sendMessage("sequences", "list", emptyPayload)
        return {}
    end
    
    local result = {}
    local ok, children = pcall(function() return dp.Sequences:Children() end)
    if ok and children then
        for _, seq in ipairs(children) do
            if seq then
                local seqNo = tonumber(seq.no)
                if seqNo ~= nil and (startNo == nil or seqNo >= tonumber(startNo)) and (endNo == nil or seqNo <= tonumber(endNo)) then
                    table.insert(result, {
                        no = seqNo,
                        name = seq.name or "",
                    })
                end
            end
        end
        EZ.log(string.format("GetSequences() -> Found %d sequences", #result))
    else
        EZ.log("ERROR: GetSequences() - Failed to enumerate sequences")
    end
    
    local count = #result
    local maxPer = 40
    local totalChunks = math.max(1, math.ceil(count / maxPer))
    for chunkIdx = 1, totalChunks do
        local startIdx = (chunkIdx - 1) * maxPer + 1
        local endIdx = math.min(startIdx + maxPer - 1, count)
        local chunk = {}
        for index = startIdx, endIdx do
            table.insert(chunk, result[index])
        end
        local payload = {
            count = count,
            offset = startIdx,
            chunk_index = chunkIdx,
            total_chunks = totalChunks,
            sequences = chunk
        }
        if request_id ~= nil then
            payload.request_id = request_id
        end
        EZ.sendMessage("sequences", "list", payload)
    end
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
