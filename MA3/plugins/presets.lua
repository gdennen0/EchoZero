EZ = EZ or {}

local function trimPresetText(value)
    return tostring(value or ""):gsub("^%s+", ""):gsub("%s+$", "")
end

local function escapePresetCmdString(value)
    return tostring(value or ""):gsub('"', '\\"')
end

local function normalizeStoreMode(storeMode)
    local normalized = trimPresetText(storeMode):lower()
    if normalized == "" then
        return "Auto", "/Auto"
    end
    if normalized == "forceglobal" or normalized == "force global" then
        return "ForceGlobal", "/ForceGlobal"
    end
    if normalized == "selective" then
        return "Selective", "/Selective"
    end
    if normalized == "global" then
        return "Global", "/Global"
    end
    if normalized == "universal" then
        return "Universal", "/Universal"
    end
    if normalized == "auto" then
        return "Auto", "/Auto"
    end
    return nil, nil
end

local function sendPresetError(presetTypeNo, presetNo, errorText)
    EZ.sendMessage("preset", "error", {
        preset_type = tonumber(presetTypeNo) or 0,
        number = tonumber(presetNo) or 0,
        error = tostring(errorText or "Preset authoring failed"),
    })
end

local function sendPresetSnapshot(change, payload)
    EZ.sendMessage("preset", change, payload)
end

local function executeAuthoringCommand(commandText)
    local normalized = trimPresetText(commandText)
    if normalized == "" then
        return true
    end
    local ok, err = pcall(function() Cmd(normalized) end)
    if not ok then
        return false, tostring(err or "Command failed")
    end
    return true
end

local function resolvePresetHandle(presetTypeNo, presetNo)
    local dp = EZ.getDP()
    if not dp or not dp.PresetPools then
        return nil
    end
    local pool = nil
    local okPool = pcall(function() pool = dp.PresetPools[tonumber(presetTypeNo)] end)
    if not okPool or not pool then
        return nil
    end
    local preset = nil
    local okPreset = pcall(function() preset = pool[tonumber(presetNo)] end)
    if okPreset and preset then
        return preset
    end
    return nil
end

local function resolvePresetPoolHandle(presetTypeNo)
    local dp = EZ.getDP()
    if not dp or not dp.PresetPools then
        return nil
    end
    local pool = nil
    local okPool = pcall(function() pool = dp.PresetPools[tonumber(presetTypeNo)] end)
    if not okPool or not pool then
        return nil
    end
    return pool
end

local function copyPresetPropertyItems(handle)
    if type(EZ.describeHandlePropertyItems) ~= "function" then
        return {}, 0, false
    end
    return EZ.describeHandlePropertyItems(handle)
end

local function propertyValueByName(propertyItems)
    local values = {}
    for _, item in ipairs(propertyItems or {}) do
        local key = tostring(item.name or ""):upper()
        if key ~= "" then
            values[key] = item.value
        end
    end
    return values
end

local function inferPresetStoreMode(propertyValues)
    local candidates = {
        propertyValues.PRESETMODE,
        propertyValues.STOREDMODE,
        propertyValues.STOREDDATA,
        propertyValues.PRESETMODEINTERNAL,
    }
    for _, candidate in ipairs(candidates) do
        local text = trimPresetText(candidate)
        if text ~= "" then
            return text
        end
    end
    return ""
end

local function inferPresetKind(presetTypeNo, childDescriptions)
    local child = childDescriptions and childDescriptions[1] or nil
    if not child then
        return "static"
    end
    local values = propertyValueByName(child.property_items or {})
    local phaserKeys = {
        "SPEEDFROMX",
        "SPEEDTOX",
        "PHASEFROMX",
        "PHASETOX",
        "FADEFROMX",
        "FADETOX",
        "DELAYFROMX",
        "DELAYTOX",
        "PHASERTRANSFORM",
    }
    for _, key in ipairs(phaserKeys) do
        if values[key] ~= nil and tostring(values[key]) ~= "" then
            return "phaser"
        end
    end
    if tonumber(presetTypeNo) == 21 then
        return "phaser"
    end
    return "recipe"
end

local function buildPresetChildDescription(presetTypeNo, presetNo, childHandle, childIndex)
    local browseToken = EZ.getDataPoolBrowseToken(childHandle, childIndex)
    local childPath = string.format("PresetPools/%d/%d/%s", presetTypeNo, presetNo, browseToken)
    local description = EZ.describeHandle(childHandle, {
        path = childPath,
        browse_token = browseToken,
        child_index = childIndex,
    }) or {}
    local propertyItems, propertyCount, propertiesTruncated = copyPresetPropertyItems(childHandle)
    description.property_items = propertyItems
    description.property_count = propertyCount
    description.properties_truncated = propertiesTruncated
    return description
end

local function buildPresetDescription(presetTypeNo, presetNo, presetHandle)
    local path = string.format("PresetPools/%d/%d", presetTypeNo, presetNo)
    local description = EZ.describeHandle(presetHandle, {
        path = path,
        browse_token = tostring(presetNo),
    }) or {}
    local propertyItems, propertyCount, propertiesTruncated = copyPresetPropertyItems(presetHandle)
    local children = EZ.safeChildren(presetHandle)
    local childDescriptions = {}
    for childIndex = 1, #children do
        table.insert(
            childDescriptions,
            buildPresetChildDescription(presetTypeNo, presetNo, children[childIndex], childIndex)
        )
    end
    local propertyValues = propertyValueByName(propertyItems)
    description.preset_type = tonumber(presetTypeNo) or 0
    description.number = tonumber(presetNo) or 0
    description.store_mode = inferPresetStoreMode(propertyValues)
    description.kind = inferPresetKind(presetTypeNo, childDescriptions)
    description.step_count = #childDescriptions > 0 and #childDescriptions or 1
    description.property_items = propertyItems
    description.property_count = propertyCount
    description.properties_truncated = propertiesTruncated
    description.children = childDescriptions
    return description
end

local function ensurePresetName(presetTypeNo, presetNo, presetName)
    local normalizedName = trimPresetText(presetName)
    if normalizedName == "" then
        return true
    end
    return executeAuthoringCommand(
        string.format(
            'Set Preset %d.%d Name "%s"',
            tonumber(presetTypeNo) or 0,
            tonumber(presetNo) or 0,
            escapePresetCmdString(normalizedName)
        )
    )
end

local function deletePresetIfPresent(presetTypeNo, presetNo)
    local existingPreset = resolvePresetHandle(presetTypeNo, presetNo)
    if not existingPreset then
        return true
    end
    return executeAuthoringCommand(
        string.format(
            "Delete Preset %d.%d /nc",
            tonumber(presetTypeNo) or 0,
            tonumber(presetNo) or 0
        )
    )
end

local function parseStepPresetSpec(stepSpec)
    local normalizedSpec = trimPresetText(stepSpec)
    local parsed = {}
    if normalizedSpec == "" then
        return parsed
    end
    for rawStep in string.gmatch(normalizedSpec, "([^;]+)") do
        local refs = {}
        for rawRef in string.gmatch(rawStep, "([^+]+)") do
            local normalizedRef = trimPresetText(rawRef)
            if normalizedRef ~= "" then
                table.insert(refs, normalizedRef)
            end
        end
        if #refs > 0 then
            table.insert(parsed, refs)
        end
    end
    return parsed
end

local function splitCsv(rawText)
    local values = {}
    local text = trimPresetText(rawText)
    if text == "" then
        return values
    end
    for part in string.gmatch(text, "([^,]+)") do
        local normalized = trimPresetText(part)
        if normalized ~= "" then
            table.insert(values, normalized)
        end
    end
    return values
end

local function normalizePresetRef(rawRef)
    local text = trimPresetText(rawRef)
    if text == "" then
        return nil
    end

    local function normalizeNumericRef(typeNoText, presetNoText)
        local typeNo = tonumber(typeNoText)
        local presetNo = tonumber(presetNoText)
        if not typeNo or not presetNo then
            return nil
        end
        return "Preset " .. tostring(math.floor(typeNo)) .. "." .. tostring(math.floor(presetNo))
    end

    local numericWithPrefixType, numericWithPrefixPreset = text:match("^Preset%s+(%d+)%.(%d+)$")
    if numericWithPrefixType and numericWithPrefixPreset then
        return normalizeNumericRef(numericWithPrefixType, numericWithPrefixPreset)
    end

    local numericType, numericPreset = text:match("^(%d+)%.(%d+)$")
    if numericType and numericPreset then
        return normalizeNumericRef(numericType, numericPreset)
    end

    local namedType, namedName = text:match("^Preset%s+(%d+)%.\"(.-)\"$")
    if not namedType then
        namedType, namedName = text:match("^(%d+)%.\"(.-)\"$")
    end
    if namedType and namedName then
        local typeNo = tonumber(namedType)
        local pool = resolvePresetPoolHandle(typeNo)
        if pool then
            local wanted = trimPresetText(namedName):lower()
            local presets = EZ.safeChildren(pool)
            for index = 1, #presets do
                local preset = presets[index]
                if preset and trimPresetText(EZ.safeStringProperty(preset, "name")):lower() == wanted then
                    local presetNo = tonumber(EZ.safeNumberProperty(preset, "no")) or index
                    return "Preset " .. tostring(typeNo) .. "." .. tostring(math.floor(presetNo))
                end
            end
        end
        return nil
    end

    return nil
end

local function resolveGroupFilterLookup(rawGroupFilter)
    local groups = DataPool() and DataPool().Groups or nil
    local lookup = {}
    local labels = {}
    if not groups then
        return lookup, labels
    end
    for _, rawGroupName in ipairs(splitCsv(rawGroupFilter)) do
        local groupHandle = groups[rawGroupName]
        if groupHandle then
            lookup[groupHandle] = rawGroupName
            table.insert(labels, rawGroupName)
        end
    end
    return lookup, labels
end

local function resolveTargetSequences(rawSequenceNumbers)
    local sequences = {}
    local pool = DataPool() and DataPool().Sequences or nil
    if not pool then
        return sequences
    end
    for _, rawSequenceNo in ipairs(splitCsv(rawSequenceNumbers)) do
        local sequenceNo = tonumber(rawSequenceNo)
        if sequenceNo and sequenceNo > 0 then
            local sequenceHandle = pool[sequenceNo]
            if sequenceHandle then
                table.insert(sequences, {
                    number = math.floor(sequenceNo),
                    name = EZ.safeStringProperty(sequenceHandle, "name"),
                    handle = sequenceHandle,
                })
            end
        end
    end
    return sequences
end

local function actualCueNumberFromHandle(cueHandle)
    local rawCueNumber = EZ.safeNumberProperty(cueHandle, "number") or EZ.safeNumberProperty(cueHandle, "no")
    if not rawCueNumber then
        return nil
    end
    return tonumber(rawCueNumber) / 1000
end

local function recipeLineReferencesPreset(recipeHandle, normalizedSourcePresetRef)
    local candidates = {
        EZ.safeProperty(recipeHandle, "preset"),
        EZ.safeProperty(recipeHandle, "PRESET"),
        EZ.safeProperty(recipeHandle, "values"),
        EZ.safeProperty(recipeHandle, "VALUES"),
    }
    local wantedLower = trimPresetText(normalizedSourcePresetRef):lower()
    local wantedBare = wantedLower:gsub("^preset%s+", "")
    for _, candidate in ipairs(candidates) do
        local text = trimPresetText(candidate)
        if text ~= "" then
            local lowered = text:lower()
            if lowered == wantedLower or lowered == wantedBare then
                return true
            end
            if lowered:find(wantedLower, 1, true) or lowered:find(wantedBare, 1, true) then
                return true
            end
        end
    end
    return false
end

local function collectPresetReplacementFindings(targetSequences, filterGroupLookup, normalizedSourcePresetRef, sourcePresetName, destPresetName)
    local findings = {}
    for _, seq in ipairs(targetSequences) do
        local cues = EZ.safeChildren(seq.handle)
        for cueIndex = 1, #cues do
            local cue = cues[cueIndex]
            if cue then
                local actualCueNumber = actualCueNumberFromHandle(cue)
                if actualCueNumber then
                    local parts = EZ.safeChildren(cue)
                    for partIndex = 1, #parts do
                        local part = parts[partIndex]
                        if part then
                            local recipes = EZ.safeChildren(part)
                            for recipeIndex = 1, #recipes do
                                local recipe = recipes[recipeIndex]
                                if recipe then
                                    local matchedGroup = filterGroupLookup[recipe.selection]
                                    if matchedGroup and recipeLineReferencesPreset(recipe, normalizedSourcePresetRef) then
                                        local partNumber = string.format("%d.%d", partIndex - 1, recipeIndex)
                                        table.insert(findings, {
                                            description = string.format(
                                                "Seq %d \"%s\" Cue %g Part %s [Group: %s]: %s -> %s",
                                                seq.number,
                                                seq.name,
                                                actualCueNumber,
                                                partNumber,
                                                matchedGroup,
                                                sourcePresetName,
                                                destPresetName
                                            ),
                                            seqNumber = seq.number,
                                            actualCueNumber = actualCueNumber,
                                            partNumber = partNumber,
                                            matched_group = matchedGroup,
                                        })
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return findings
end

local function sendRecipeCuePayload(change, payload)
    EZ.sendMessage("recipe_cue", change, payload)
end

local function normalizeCueNumberText(value)
    local cueNumber = tonumber(value)
    if not cueNumber or cueNumber <= 0 then
        return nil
    end
    if math.floor(cueNumber) == cueNumber then
        return tostring(math.floor(cueNumber))
    end
    local text = tostring(cueNumber)
    text = text:gsub("(%..-)0+$", "%1")
    text = text:gsub("%.$", "")
    return text
end

local function cueNumberValue(value)
    local normalized = normalizeCueNumberText(value)
    if not normalized then
        return nil
    end
    return tonumber(normalized)
end

local function resolveSequenceHandle(sequenceNo)
    local sequences = DataPool() and DataPool().Sequences or nil
    if not sequences then
        return nil
    end
    return sequences[tonumber(sequenceNo)]
end

local function resolveCueHandle(sequenceHandle, cueNo)
    if not sequenceHandle then
        return nil, nil
    end
    local wantedText = normalizeCueNumberText(cueNo)
    if not wantedText then
        return nil, nil
    end
    local cues = EZ.safeChildren(sequenceHandle)
    for cueIndex = 1, #cues do
        local cueHandle = cues[cueIndex]
        local actualCueNumber = actualCueNumberFromHandle(cueHandle)
        if normalizeCueNumberText(actualCueNumber) == wantedText then
            return cueHandle, actualCueNumber
        end
    end
    return nil, nil
end

local function recipeTextProperty(recipeHandle, propertyNames)
    for _, propertyName in ipairs(propertyNames or {}) do
        local value = trimPresetText(EZ.safeProperty(recipeHandle, propertyName))
        if value ~= "" then
            return value
        end
    end
    return ""
end

local function inferRecipeFeatureGroup(recipeHandle)
    local candidates = {
        recipeTextProperty(recipeHandle, {"values", "VALUES"}),
        recipeTextProperty(recipeHandle, {"preset", "PRESET"}),
    }
    for _, candidate in ipairs(candidates) do
        local featureGroupName = candidate:match("FeatureGroup%s+%d+%s+'([^']+)'")
        if featureGroupName and trimPresetText(featureGroupName) ~= "" then
            return trimPresetText(featureGroupName)
        end
        local presetFeatureName = candidate:match("^(%d+)%s+'([^']+)'")
        if presetFeatureName then
            local matched = candidate:match("^%d+%s+'([^']+)'")
            if matched and trimPresetText(matched) ~= "" then
                return trimPresetText(matched)
            end
        end
    end
    return ""
end

local function inferRecipeMode(recipeHandle)
    local candidates = {
        recipeTextProperty(recipeHandle, {"mode", "MODE", "valuemode", "VALUEMODE"}),
        recipeTextProperty(recipeHandle, {"values", "VALUES"}),
        recipeTextProperty(recipeHandle, {"preset", "PRESET"}),
    }
    for _, candidate in ipairs(candidates) do
        local lowered = trimPresetText(candidate):lower()
        if lowered ~= "" then
            if lowered:find("relative", 1, true) then
                return "relative"
            end
            if lowered:find("absolute", 1, true) then
                return "absolute"
            end
        end
    end
    return ""
end

local function buildCueRecipeRows(sequenceNo, cueNo)
    local warnings = {}
    local unsupportedReasons = {}
    local sequenceHandle = resolveSequenceHandle(sequenceNo)
    if not sequenceHandle then
        table.insert(unsupportedReasons, string.format("Sequence %d not found.", tonumber(sequenceNo) or 0))
        return {}, warnings, unsupportedReasons, nil
    end

    local cueHandle, actualCueNumber = resolveCueHandle(sequenceHandle, cueNo)
    if not cueHandle then
        table.insert(
            unsupportedReasons,
            string.format(
                "Cue %s was not found in sequence %d.",
                tostring(normalizeCueNumberText(cueNo) or cueNo or ""),
                tonumber(sequenceNo) or 0
            )
        )
        return {}, warnings, unsupportedReasons, nil
    end

    local rows = {}
    local seqName = EZ.safeStringProperty(sequenceHandle, "name")
    local parts = EZ.safeChildren(cueHandle)
    for partIndex = 1, #parts do
        local partHandle = parts[partIndex]
        local recipes = EZ.safeChildren(partHandle)
        if #recipes == 0 then
            table.insert(
                warnings,
                string.format(
                    "Cue %s part %d has no detectable recipe children; direct stored values are not modeled.",
                    tostring(normalizeCueNumberText(actualCueNumber) or ""),
                    partIndex - 1
                )
            )
        end
        for recipeIndex = 1, #recipes do
            local recipeHandle = recipes[recipeIndex]
            local matchedGroup = trimPresetText(EZ.safeStringProperty(recipeHandle.selection, "name"))
            local featureGroup = inferRecipeFeatureGroup(recipeHandle)
            local recipeMode = inferRecipeMode(recipeHandle)
            local selectionKey = ""
            if matchedGroup ~= "" and featureGroup ~= "" then
                selectionKey = matchedGroup .. ":" .. featureGroup
            end
            if selectionKey == "" then
                table.insert(unsupportedReasons, "One or more recipe lines are missing a stable selection key.")
            end
            if recipeMode == "" then
                table.insert(unsupportedReasons, "One or more recipe lines are missing relative/absolute mode metadata.")
            end
            local partNumber = string.format("%d.%d", partIndex - 1, recipeIndex)
            table.insert(rows, {
                seq_number = tonumber(sequenceNo) or 0,
                seq_name = tostring(seqName or ""),
                actual_cue_number = tonumber(actualCueNumber) or 0,
                part_number = partNumber,
                feature_group = featureGroup,
                recipe_mode = recipeMode ~= "" and recipeMode or "unknown",
                matched_group = matchedGroup,
                line_index = recipeIndex,
                selection_key = selectionKey,
                source_cue_number = tonumber(actualCueNumber) or 0,
                source_part_number = partNumber,
                preset_ref = recipeTextProperty(recipeHandle, {"preset", "PRESET"}),
                values_ref = recipeTextProperty(recipeHandle, {"values", "VALUES"}),
                selection_mode = recipeTextProperty(recipeHandle, {"selectionmode", "SELECTIONMODE"}),
                preset_mode = recipeTextProperty(recipeHandle, {"presetmode", "PRESETMODE", "presetmodeinternal", "PRESETMODEINTERNAL"}),
            })
        end
    end

    table.sort(rows, function(left, right)
        local leftCue = tonumber(left.actual_cue_number) or 0
        local rightCue = tonumber(right.actual_cue_number) or 0
        if leftCue ~= rightCue then
            return leftCue < rightCue
        end
        local leftIndex = tonumber(left.line_index) or 0
        local rightIndex = tonumber(right.line_index) or 0
        if leftIndex ~= rightIndex then
            return leftIndex < rightIndex
        end
        return tostring(left.part_number or "") < tostring(right.part_number or "")
    end)

    return rows, warnings, unsupportedReasons, tonumber(actualCueNumber)
end

local function recipeStateKey(row)
    local explicit = trimPresetText(row.selection_key)
    if explicit ~= "" then
        return explicit
    end
    return trimPresetText(row.matched_group) .. ":" .. trimPresetText(row.feature_group)
end

local function collectSequenceRecipeRowsUpTo(sequenceNo, maxCueNumber)
    local sequenceHandle = resolveSequenceHandle(sequenceNo)
    if not sequenceHandle then
        return {}, {"Sequence not found."}, {"Sequence not found."}
    end
    local rows = {}
    local warnings = {}
    local unsupportedReasons = {}
    local cues = EZ.safeChildren(sequenceHandle)
    for cueIndex = 1, #cues do
        local cueHandle = cues[cueIndex]
        local actualCueNumber = actualCueNumberFromHandle(cueHandle)
        if actualCueNumber and tonumber(actualCueNumber) <= tonumber(maxCueNumber) then
            local cueRows, cueWarnings, cueUnsupported = buildCueRecipeRows(sequenceNo, actualCueNumber)
            for _, row in ipairs(cueRows) do
                table.insert(rows, row)
            end
            for _, warning in ipairs(cueWarnings) do
                table.insert(warnings, warning)
            end
            for _, reason in ipairs(cueUnsupported) do
                table.insert(unsupportedReasons, reason)
            end
        end
    end
    return rows, warnings, unsupportedReasons
end

local function dedupeTextList(values)
    local seen = {}
    local ordered = {}
    for _, value in ipairs(values or {}) do
        local text = trimPresetText(value)
        if text ~= "" and not seen[text] then
            seen[text] = true
            table.insert(ordered, text)
        end
    end
    return ordered
end

local function cloneRows(rows)
    local cloned = {}
    for _, row in ipairs(rows or {}) do
        local item = {}
        for key, value in pairs(row) do
            item[key] = value
        end
        table.insert(cloned, item)
    end
    return cloned
end

local function effectiveRecipeContributorsFromRows(rows)
    local relevantRows = cloneRows(rows)
    table.sort(relevantRows, function(left, right)
        local leftCue = tonumber(left.actual_cue_number) or 0
        local rightCue = tonumber(right.actual_cue_number) or 0
        if leftCue ~= rightCue then
            return leftCue < rightCue
        end
        local leftIndex = tonumber(left.line_index) or 0
        local rightIndex = tonumber(right.line_index) or 0
        if leftIndex ~= rightIndex then
            return leftIndex < rightIndex
        end
        return tostring(left.part_number or "") < tostring(right.part_number or "")
    end)

    local contributorsByKey = {}
    for _, row in ipairs(relevantRows) do
        local key = recipeStateKey(row)
        local recipeMode = trimPresetText(row.recipe_mode):lower()
        if key ~= "" and key ~= ":" and (recipeMode == "absolute" or recipeMode == "relative") then
            if recipeMode == "relative" then
                local bucket = contributorsByKey[key] or {}
                table.insert(bucket, row)
                contributorsByKey[key] = bucket
            else
                contributorsByKey[key] = {row}
            end
        end
    end

    local flattened = {}
    for _, bucket in pairs(contributorsByKey) do
        for _, row in ipairs(bucket) do
            table.insert(flattened, row)
        end
    end
    table.sort(flattened, function(left, right)
        local leftCue = tonumber(left.actual_cue_number) or 0
        local rightCue = tonumber(right.actual_cue_number) or 0
        if leftCue ~= rightCue then
            return leftCue < rightCue
        end
        local leftIndex = tonumber(left.line_index) or 0
        local rightIndex = tonumber(right.line_index) or 0
        if leftIndex ~= rightIndex then
            return leftIndex < rightIndex
        end
        return tostring(left.part_number or "") < tostring(right.part_number or "")
    end)
    return flattened
end

local function contributorSignature(rows)
    local signatures = {}
    for _, row in ipairs(rows or {}) do
        local signature = table.concat({
            trimPresetText(recipeStateKey(row)),
            trimPresetText(row.recipe_mode),
            trimPresetText(row.preset_ref),
            trimPresetText(row.matched_group),
        }, "|")
        signatures[signature] = true
    end
    return signatures
end

local function signatureSetsMatch(leftRows, rightRows)
    local left = contributorSignature(leftRows)
    local right = contributorSignature(rightRows)
    for signature, _ in pairs(left) do
        if not right[signature] then
            return false
        end
    end
    for signature, _ in pairs(right) do
        if not left[signature] then
            return false
        end
    end
    return true
end

local function sendPresetReplacePayload(change, payload)
    EZ.sendMessage("preset_replace", change, payload)
end

local function sendRecipeCuePayload(change, payload)
    EZ.sendMessage("recipe_cue", change, payload)
end

local function cueNumberText(rawCueNo)
    local cueNo = tonumber(rawCueNo)
    if not cueNo then
        return nil
    end
    if cueNo == math.floor(cueNo) then
        return tostring(math.floor(cueNo))
    end
    return tostring(cueNo)
end

local function resolveSequenceHandle(sequenceNo)
    local pool = DataPool() and DataPool().Sequences or nil
    if not pool then
        return nil
    end
    return pool[tonumber(sequenceNo)]
end

local function resolveCueHandle(sequenceHandle, cueNo)
    local cues = EZ.safeChildren(sequenceHandle)
    for cueIndex = 1, #cues do
        local cue = cues[cueIndex]
        if cue and actualCueNumberFromHandle(cue) == tonumber(cueNo) then
            return cue
        end
    end
    return nil
end

local function recipeRowPresetRef(recipeHandle)
    local candidates = {
        EZ.safeProperty(recipeHandle, "preset"),
        EZ.safeProperty(recipeHandle, "PRESET"),
        EZ.safeProperty(recipeHandle, "values"),
        EZ.safeProperty(recipeHandle, "VALUES"),
    }
    for _, candidate in ipairs(candidates) do
        local normalized = normalizePresetRef(candidate)
        if normalized then
            return normalized
        end
    end
    local fallback = trimPresetText(EZ.safeProperty(recipeHandle, "values") or EZ.safeProperty(recipeHandle, "VALUES"))
    return fallback ~= "" and fallback or nil
end

local function inferRecipeFeatureGroup(recipeHandle, presetRef)
    local normalizedPresetRef = trimPresetText(presetRef)
    local presetTypeNo = normalizedPresetRef:match("^Preset%s+(%d+)%.%d+$")
    if presetTypeNo then
        return tostring(presetTypeNo)
    end
    local valuesText = trimPresetText(EZ.safeProperty(recipeHandle, "values") or EZ.safeProperty(recipeHandle, "VALUES"))
    local valuesFeatureGroup = valuesText:match("FeatureGroup%s+(%d+)")
    if valuesFeatureGroup then
        return tostring(valuesFeatureGroup)
    end
    return "unknown"
end

local function inferRecipeMode(recipeHandle)
    local candidates = {
        EZ.safeProperty(recipeHandle, "valuemode"),
        EZ.safeProperty(recipeHandle, "VALUEMODE"),
        EZ.safeProperty(recipeHandle, "presetmode"),
        EZ.safeProperty(recipeHandle, "PRESETMODE"),
    }
    for _, candidate in ipairs(candidates) do
        local lowered = trimPresetText(candidate):lower()
        if lowered:find("relative", 1, true) then
            return "relative"
        end
    end
    return "absolute"
end

local function recipeSelectionLabel(recipeHandle)
    local selection = recipeHandle and recipeHandle.selection or nil
    local label = trimPresetText(EZ.safeStringProperty(selection, "name"))
    if label ~= "" then
        return label
    end
    local direct = trimPresetText(EZ.safeProperty(recipeHandle, "selection") or EZ.safeProperty(recipeHandle, "SELECTION"))
    if direct ~= "" then
        return direct
    end
    return "selection"
end

local function buildCueRecipeRow(sequenceNo, sequenceName, cueNo, partIndex, recipeIndex, recipeHandle)
    local presetRef = recipeRowPresetRef(recipeHandle)
    local matchedGroup = recipeSelectionLabel(recipeHandle)
    local featureGroup = inferRecipeFeatureGroup(recipeHandle, presetRef)
    local recipeMode = inferRecipeMode(recipeHandle)
    local partNumber = string.format("%d.%d", partIndex - 1, recipeIndex)
    return {
        seq_number = tonumber(sequenceNo) or 0,
        seq_name = tostring(sequenceName or ""),
        actual_cue_number = tonumber(cueNo) or 0,
        part_number = partNumber,
        matched_group = matchedGroup,
        feature_group = featureGroup,
        recipe_mode = recipeMode,
        line_index = recipeIndex,
        selection_key = matchedGroup .. ":" .. featureGroup,
        source_cue_number = tonumber(cueNo) or 0,
        source_part_number = partNumber,
        preset_ref = presetRef or "",
    }
end

local function cueRecipeRows(sequenceNo, cueNo)
    local rows = {}
    local sequenceHandle = resolveSequenceHandle(sequenceNo)
    if not sequenceHandle then
        return rows
    end
    local cueHandle = resolveCueHandle(sequenceHandle, cueNo)
    if not cueHandle then
        return rows
    end
    local parts = EZ.safeChildren(cueHandle)
    for partIndex = 1, #parts do
        local part = parts[partIndex]
        if part then
            local recipes = EZ.safeChildren(part)
            for recipeIndex = 1, #recipes do
                local recipe = recipes[recipeIndex]
                if recipe then
                    table.insert(
                        rows,
                        buildCueRecipeRow(
                            sequenceNo,
                            EZ.safeStringProperty(sequenceHandle, "name"),
                            cueNo,
                            partIndex,
                            recipeIndex,
                            recipe
                        )
                    )
                end
            end
        end
    end
    table.sort(rows, function(left, right)
        if tonumber(left.actual_cue_number) ~= tonumber(right.actual_cue_number) then
            return tonumber(left.actual_cue_number) < tonumber(right.actual_cue_number)
        end
        return tonumber(left.line_index or 0) < tonumber(right.line_index or 0)
    end)
    return rows
end

local function effectiveRecipeContributorsFromRows(rows, sequenceNo, cueNo)
    local contributorsByKey = {}
    for _, row in ipairs(rows) do
        if tonumber(row.seq_number) == tonumber(sequenceNo) and tonumber(row.actual_cue_number) <= tonumber(cueNo) then
            local key = tostring(row.selection_key or "")
            local mode = trimPresetText(row.recipe_mode):lower()
            if mode == "relative" then
                if not contributorsByKey[key] then
                    contributorsByKey[key] = {}
                end
                table.insert(contributorsByKey[key], row)
            else
                contributorsByKey[key] = {row}
            end
        end
    end
    local flattened = {}
    for _, bucket in pairs(contributorsByKey) do
        for _, row in ipairs(bucket) do
            table.insert(flattened, row)
        end
    end
    table.sort(flattened, function(left, right)
        if tonumber(left.actual_cue_number) ~= tonumber(right.actual_cue_number) then
            return tonumber(left.actual_cue_number) < tonumber(right.actual_cue_number)
        end
        return tonumber(left.line_index or 0) < tonumber(right.line_index or 0)
    end)
    return flattened
end

local function collectSequenceRecipeRows(sequenceNo)
    local sequenceHandle = resolveSequenceHandle(sequenceNo)
    if not sequenceHandle then
        return {}
    end
    local rows = {}
    local cues = EZ.safeChildren(sequenceHandle)
    for cueIndex = 1, #cues do
        local cue = cues[cueIndex]
        if cue then
            local cueNo = actualCueNumberFromHandle(cue)
            if cueNo then
                local cueRows = cueRecipeRows(sequenceNo, cueNo)
                for _, row in ipairs(cueRows) do
                    table.insert(rows, row)
                end
            end
        end
    end
    return rows
end

local function replaceCueRowsInSnapshot(rows, sequenceNo, cueNo, replacementRows)
    local survivors = {}
    for _, row in ipairs(rows) do
        if not (tonumber(row.seq_number) == tonumber(sequenceNo) and tonumber(row.actual_cue_number) == tonumber(cueNo)) then
            table.insert(survivors, row)
        end
    end
    for index, row in ipairs(replacementRows or {}) do
        local cloned = {}
        for key, value in pairs(row) do
            cloned[key] = value
        end
        cloned.seq_number = tonumber(sequenceNo)
        cloned.actual_cue_number = tonumber(cueNo)
        cloned.line_index = index
        cloned.part_number = string.format("0.%d", index)
        table.insert(survivors, cloned)
    end
    return survivors
end

local function contributorSignature(rows, selectionKey)
    local signature = {}
    for _, row in ipairs(rows or {}) do
        if tostring(row.selection_key or "") == tostring(selectionKey or "") then
            table.insert(
                signature,
                table.concat(
                    {
                        tostring(row.selection_key or ""),
                        tostring(row.recipe_mode or ""),
                        tostring(row.preset_ref or ""),
                        tostring(row.matched_group or ""),
                    },
                    "|"
                )
            )
        end
    end
    table.sort(signature)
    return table.concat(signature, ";")
end

local function clearCueRecipeRows(sequenceNo, cueNo)
    local localRows = cueRecipeRows(sequenceNo, cueNo)
    for index = #localRows, 1, -1 do
        local row = localRows[index]
        executeAuthoringCommand(
            string.format(
                "Delete Sequence %d Cue %s Part %s /nc",
                tonumber(sequenceNo) or 0,
                cueNumberText(cueNo) or tostring(cueNo),
                tostring(row.part_number or "")
            )
        )
    end
end

local function copyRecipeRowsToCue(sequenceNo, cueNo, rows)
    clearCueRecipeRows(sequenceNo, cueNo)
    for index, row in ipairs(rows or {}) do
        executeAuthoringCommand(
            string.format(
                "Copy Sequence %d Cue %s Part %s At Sequence %d Cue %s Part 0.%d /Merge",
                tonumber(row.seq_number) or tonumber(sequenceNo) or 0,
                cueNumberText(row.source_cue_number or row.actual_cue_number) or tostring(row.source_cue_number or row.actual_cue_number),
                tostring(row.source_part_number or row.part_number or ""),
                tonumber(sequenceNo) or 0,
                cueNumberText(cueNo) or tostring(cueNo),
                index
            )
        )
    end
end

function EZ.CreateStaticPreset(
    presetTypeNo,
    presetNo,
    storeMode,
    presetName,
    selectionCommand,
    valueCommand
)
    local resolvedPresetTypeNo = tonumber(presetTypeNo)
    local resolvedPresetNo = tonumber(presetNo)
    local normalizedName = trimPresetText(presetName)
    local normalizedSelectionCommand = trimPresetText(selectionCommand)
    local normalizedValueCommand = trimPresetText(valueCommand)
    local normalizedMode, storeOption = normalizeStoreMode(storeMode)

    if not resolvedPresetTypeNo or resolvedPresetTypeNo < 1 then
        sendPresetError(presetTypeNo, presetNo, "Preset type is required")
        return nil
    end
    if not resolvedPresetNo or resolvedPresetNo < 1 then
        sendPresetError(presetTypeNo, presetNo, "Preset number is required")
        return nil
    end
    if normalizedName == "" then
        sendPresetError(presetTypeNo, presetNo, "Preset name is required")
        return nil
    end
    if normalizedSelectionCommand == "" then
        sendPresetError(presetTypeNo, presetNo, "Selection command is required")
        return nil
    end
    if normalizedValueCommand == "" then
        sendPresetError(presetTypeNo, presetNo, "Value command is required")
        return nil
    end
    if not normalizedMode or not storeOption then
        sendPresetError(
            presetTypeNo,
            presetNo,
            "Store mode must be Auto, Selective, Global, Universal, or ForceGlobal"
        )
        return nil
    end

    local existingPreset = resolvePresetHandle(resolvedPresetTypeNo, resolvedPresetNo)
    if existingPreset then
        sendPresetSnapshot("exists", {
            preset_type = resolvedPresetTypeNo,
            number = resolvedPresetNo,
            name = normalizedName,
            store_mode = normalizedMode,
            kind = "static",
            step_count = 1,
        })
        return existingPreset
    end

    executeAuthoringCommand("ClearAll")
    local okSelection, selectionErr = executeAuthoringCommand(normalizedSelectionCommand)
    if not okSelection then
        sendPresetError(presetTypeNo, presetNo, "Selection command failed: " .. tostring(selectionErr))
        return nil
    end
    local okValue, valueErr = executeAuthoringCommand(normalizedValueCommand)
    if not okValue then
        sendPresetError(presetTypeNo, presetNo, "Value command failed: " .. tostring(valueErr))
        return nil
    end
    local okStore, storeErr = executeAuthoringCommand(
        string.format("Store Preset %d.%d %s", resolvedPresetTypeNo, resolvedPresetNo, storeOption)
    )
    if not okStore then
        sendPresetError(presetTypeNo, presetNo, "Store command failed: " .. tostring(storeErr))
        return nil
    end
    local okName, nameErr = ensurePresetName(resolvedPresetTypeNo, resolvedPresetNo, normalizedName)
    if not okName then
        sendPresetError(presetTypeNo, presetNo, "Preset rename failed: " .. tostring(nameErr))
        return nil
    end
    executeAuthoringCommand("ClearAll")

    sendPresetSnapshot("created", {
        preset_type = resolvedPresetTypeNo,
        number = resolvedPresetNo,
        name = normalizedName,
        store_mode = normalizedMode,
        kind = "static",
        step_count = 1,
    })
    return resolvedPresetNo
end

function EZ.CreatePhaserPreset(
    presetTypeNo,
    presetNo,
    storeMode,
    presetName,
    selectionCommand,
    stepSpec,
    speedBpm
)
    local resolvedPresetTypeNo = tonumber(presetTypeNo)
    local resolvedPresetNo = tonumber(presetNo)
    local normalizedName = trimPresetText(presetName)
    local normalizedSelectionCommand = trimPresetText(selectionCommand)
    local normalizedMode, storeOption = normalizeStoreMode(storeMode)
    local parsedSteps = parseStepPresetSpec(stepSpec)
    local resolvedSpeedBpm = tonumber(speedBpm)

    if not resolvedPresetTypeNo or resolvedPresetTypeNo < 1 then
        sendPresetError(presetTypeNo, presetNo, "Preset type is required")
        return nil
    end
    if not resolvedPresetNo or resolvedPresetNo < 1 then
        sendPresetError(presetTypeNo, presetNo, "Preset number is required")
        return nil
    end
    if normalizedName == "" then
        sendPresetError(presetTypeNo, presetNo, "Preset name is required")
        return nil
    end
    if normalizedSelectionCommand == "" then
        sendPresetError(presetTypeNo, presetNo, "Selection command is required")
        return nil
    end
    if not normalizedMode or not storeOption then
        sendPresetError(
            presetTypeNo,
            presetNo,
            "Store mode must be Auto, Selective, Global, Universal, or ForceGlobal"
        )
        return nil
    end
    if #parsedSteps < 2 then
        sendPresetError(presetTypeNo, presetNo, "Phaser presets require at least two steps")
        return nil
    end
    if resolvedSpeedBpm and resolvedSpeedBpm <= 0 then
        sendPresetError(presetTypeNo, presetNo, "Speed BPM must be positive")
        return nil
    end

    local existingPreset = resolvePresetHandle(resolvedPresetTypeNo, resolvedPresetNo)
    if existingPreset then
        sendPresetSnapshot("exists", {
            preset_type = resolvedPresetTypeNo,
            number = resolvedPresetNo,
            name = normalizedName,
            store_mode = normalizedMode,
            kind = "phaser",
            step_count = #parsedSteps,
        })
        return existingPreset
    end

    executeAuthoringCommand("ClearAll")
    local okSelection, selectionErr = executeAuthoringCommand(normalizedSelectionCommand)
    if not okSelection then
        sendPresetError(presetTypeNo, presetNo, "Selection command failed: " .. tostring(selectionErr))
        return nil
    end

    for stepIndex = 1, #parsedSteps do
        if stepIndex > 1 then
            local okStep, stepErr = executeAuthoringCommand(string.format("Step %d", stepIndex))
            if not okStep then
                sendPresetError(
                    presetTypeNo,
                    presetNo,
                    string.format("Failed to select step %d: %s", stepIndex, tostring(stepErr))
                )
                return nil
            end
        end
        local refs = parsedSteps[stepIndex]
        for refIndex = 1, #refs do
            local okRef, refErr = executeAuthoringCommand("At Preset " .. refs[refIndex])
            if not okRef then
                sendPresetError(
                    presetTypeNo,
                    presetNo,
                    string.format("Failed to apply preset ref '%s': %s", refs[refIndex], tostring(refErr))
                )
                return nil
            end
        end
    end

    if resolvedSpeedBpm then
        local okSpeed, speedErr = executeAuthoringCommand(
            string.format("At Speed BPM %s", tostring(resolvedSpeedBpm))
        )
        if not okSpeed then
            sendPresetError(presetTypeNo, presetNo, "Speed command failed: " .. tostring(speedErr))
            return nil
        end
    end

    local okStore, storeErr = executeAuthoringCommand(
        string.format("Store Preset %d.%d %s", resolvedPresetTypeNo, resolvedPresetNo, storeOption)
    )
    if not okStore then
        sendPresetError(presetTypeNo, presetNo, "Store command failed: " .. tostring(storeErr))
        return nil
    end
    local okName, nameErr = ensurePresetName(resolvedPresetTypeNo, resolvedPresetNo, normalizedName)
    if not okName then
        sendPresetError(presetTypeNo, presetNo, "Preset rename failed: " .. tostring(nameErr))
        return nil
    end
    executeAuthoringCommand("ClearAll")

    sendPresetSnapshot("created", {
        preset_type = resolvedPresetTypeNo,
        number = resolvedPresetNo,
        name = normalizedName,
        store_mode = normalizedMode,
        kind = "phaser",
        step_count = #parsedSteps,
    })
    return resolvedPresetNo
end

function EZ.CreateRecipePreset(
    presetTypeNo,
    presetNo,
    storeMode,
    presetName,
    selectionCommand,
    sourcePresetRef,
    selectionMode
)
    local resolvedPresetTypeNo = tonumber(presetTypeNo)
    local resolvedPresetNo = tonumber(presetNo)
    local normalizedName = trimPresetText(presetName)
    local normalizedSelectionCommand = trimPresetText(selectionCommand)
    local normalizedSourcePresetRef = trimPresetText(sourcePresetRef)
    local normalizedSelectionMode = trimPresetText(selectionMode)
    local normalizedMode, storeOption = normalizeStoreMode(storeMode)

    if not resolvedPresetTypeNo or resolvedPresetTypeNo < 1 then
        sendPresetError(presetTypeNo, presetNo, "Preset type is required")
        return nil
    end
    if not resolvedPresetNo or resolvedPresetNo < 1 then
        sendPresetError(presetTypeNo, presetNo, "Preset number is required")
        return nil
    end
    if normalizedName == "" then
        sendPresetError(presetTypeNo, presetNo, "Preset name is required")
        return nil
    end
    if normalizedSelectionCommand == "" then
        sendPresetError(presetTypeNo, presetNo, "Selection command is required")
        return nil
    end
    if normalizedSourcePresetRef == "" then
        sendPresetError(presetTypeNo, presetNo, "Source preset ref is required")
        return nil
    end
    if normalizedSelectionMode == "" then
        sendPresetError(presetTypeNo, presetNo, "Selection mode is required")
        return nil
    end
    if not normalizedMode or not storeOption then
        sendPresetError(
            presetTypeNo,
            presetNo,
            "Store mode must be Auto, Selective, Global, Universal, or ForceGlobal"
        )
        return nil
    end

    local existingPreset = resolvePresetHandle(resolvedPresetTypeNo, resolvedPresetNo)
    if existingPreset then
        sendPresetSnapshot("exists", {
            preset_type = resolvedPresetTypeNo,
            number = resolvedPresetNo,
            name = normalizedName,
            store_mode = normalizedMode,
            kind = "recipe",
            step_count = 1,
        })
        return existingPreset
    end

    executeAuthoringCommand("ClearAll")
    local okSelection, selectionErr = executeAuthoringCommand(normalizedSelectionCommand)
    if not okSelection then
        sendPresetError(presetTypeNo, presetNo, "Selection command failed: " .. tostring(selectionErr))
        return nil
    end
    local okStore, storeErr = executeAuthoringCommand(
        string.format("Store Preset %d.%d %s", resolvedPresetTypeNo, resolvedPresetNo, storeOption)
    )
    if not okStore then
        sendPresetError(presetTypeNo, presetNo, "Store command failed: " .. tostring(storeErr))
        return nil
    end
    local okMove, moveErr = executeAuthoringCommand(
        string.format(
            "Move Preset %d.%d At Preset %d.%d.1",
            resolvedPresetTypeNo,
            resolvedPresetNo,
            resolvedPresetTypeNo,
            resolvedPresetNo
        )
    )
    if not okMove then
        sendPresetError(presetTypeNo, presetNo, "Recipe line creation failed: " .. tostring(moveErr))
        return nil
    end
    local okAssign, assignErr = executeAuthoringCommand(
        string.format(
            "Assign Preset %s At Preset %d.%d.1",
            normalizedSourcePresetRef,
            resolvedPresetTypeNo,
            resolvedPresetNo
        )
    )
    if not okAssign then
        sendPresetError(presetTypeNo, presetNo, "Recipe line assign failed: " .. tostring(assignErr))
        return nil
    end
    local okSelectionMode, selectionModeErr = executeAuthoringCommand(
        string.format(
            'Set Preset %d.%d.1 Property "SelectionMode" "%s"',
            resolvedPresetTypeNo,
            resolvedPresetNo,
            escapePresetCmdString(normalizedSelectionMode)
        )
    )
    if not okSelectionMode then
        sendPresetError(
            presetTypeNo,
            presetNo,
            "Recipe selection mode failed: " .. tostring(selectionModeErr)
        )
        return nil
    end
    local okName, nameErr = ensurePresetName(resolvedPresetTypeNo, resolvedPresetNo, normalizedName)
    if not okName then
        sendPresetError(presetTypeNo, presetNo, "Preset rename failed: " .. tostring(nameErr))
        return nil
    end
    executeAuthoringCommand("ClearAll")

    sendPresetSnapshot("created", {
        preset_type = resolvedPresetTypeNo,
        number = resolvedPresetNo,
        name = normalizedName,
        store_mode = normalizedMode,
        kind = "recipe",
        step_count = 1,
    })
    return resolvedPresetNo
end

function EZ.EditStaticPreset(
    presetTypeNo,
    presetNo,
    storeMode,
    presetName,
    selectionCommand,
    valueCommand
)
    local okDelete, deleteErr = deletePresetIfPresent(presetTypeNo, presetNo)
    if not okDelete then
        sendPresetError(presetTypeNo, presetNo, "Preset delete failed: " .. tostring(deleteErr))
        return nil
    end
    local created = EZ.CreateStaticPreset(
        presetTypeNo,
        presetNo,
        storeMode,
        presetName,
        selectionCommand,
        valueCommand
    )
    if created == nil then
        return nil
    end
    sendPresetSnapshot("updated", {
        preset_type = tonumber(presetTypeNo) or 0,
        number = tonumber(presetNo) or 0,
        name = trimPresetText(presetName),
        store_mode = normalizeStoreMode(storeMode),
        kind = "static",
        step_count = 1,
    })
    return created
end

function EZ.EditPhaserPreset(
    presetTypeNo,
    presetNo,
    storeMode,
    presetName,
    selectionCommand,
    stepSpec,
    speedBpm
)
    local okDelete, deleteErr = deletePresetIfPresent(presetTypeNo, presetNo)
    if not okDelete then
        sendPresetError(presetTypeNo, presetNo, "Preset delete failed: " .. tostring(deleteErr))
        return nil
    end
    local created = EZ.CreatePhaserPreset(
        presetTypeNo,
        presetNo,
        storeMode,
        presetName,
        selectionCommand,
        stepSpec,
        speedBpm
    )
    if created == nil then
        return nil
    end
    sendPresetSnapshot("updated", {
        preset_type = tonumber(presetTypeNo) or 0,
        number = tonumber(presetNo) or 0,
        name = trimPresetText(presetName),
        store_mode = normalizeStoreMode(storeMode),
        kind = "phaser",
        step_count = #parseStepPresetSpec(stepSpec),
    })
    return created
end

function EZ.EditRecipePreset(
    presetTypeNo,
    presetNo,
    storeMode,
    presetName,
    selectionCommand,
    sourcePresetRef,
    selectionMode
)
    local okDelete, deleteErr = deletePresetIfPresent(presetTypeNo, presetNo)
    if not okDelete then
        sendPresetError(presetTypeNo, presetNo, "Preset delete failed: " .. tostring(deleteErr))
        return nil
    end
    local created = EZ.CreateRecipePreset(
        presetTypeNo,
        presetNo,
        storeMode,
        presetName,
        selectionCommand,
        sourcePresetRef,
        selectionMode
    )
    if created == nil then
        return nil
    end
    sendPresetSnapshot("updated", {
        preset_type = tonumber(presetTypeNo) or 0,
        number = tonumber(presetNo) or 0,
        name = trimPresetText(presetName),
        store_mode = normalizeStoreMode(storeMode),
        kind = "recipe",
        step_count = 1,
    })
    return created
end

function EZ.ListPresets(presetTypeNo, requestId)
    local resolvedPresetTypeNo = tonumber(presetTypeNo)
    if not resolvedPresetTypeNo or resolvedPresetTypeNo < 1 then
        sendPresetError(presetTypeNo, 0, "Preset type is required")
        return nil
    end

    local pool = resolvePresetPoolHandle(resolvedPresetTypeNo)
    if not pool then
        sendPresetError(presetTypeNo, 0, "Preset pool not found")
        return nil
    end

    local presets = {}
    local children = EZ.safeChildren(pool)
    for _, presetHandle in ipairs(children) do
        if presetHandle then
            local presetNo = tonumber(EZ.safeNumberProperty(presetHandle, "no"))
            if presetNo and presetNo > 0 then
                local description = buildPresetDescription(resolvedPresetTypeNo, presetNo, presetHandle)
                table.insert(presets, {
                    preset_type = resolvedPresetTypeNo,
                    number = presetNo,
                    name = tostring(description.name or ""),
                    store_mode = tostring(description.store_mode or ""),
                    kind = tostring(description.kind or ""),
                    step_count = tonumber(description.step_count) or 1,
                    path = tostring(description.path or ""),
                })
            end
        end
    end

    table.sort(presets, function(left, right)
        return (tonumber(left.number) or 0) < (tonumber(right.number) or 0)
    end)

    local payload = {
        preset_type = resolvedPresetTypeNo,
        presets = presets,
    }
    if requestId ~= nil then
        payload.request_id = requestId
    end
    EZ.sendMessage("presets", "list", payload)
    return presets
end

function EZ.DescribePreset(presetTypeNo, presetNo, requestId)
    local resolvedPresetTypeNo = tonumber(presetTypeNo)
    local resolvedPresetNo = tonumber(presetNo)
    if not resolvedPresetTypeNo or resolvedPresetTypeNo < 1 then
        sendPresetError(presetTypeNo, presetNo, "Preset type is required")
        return nil
    end
    if not resolvedPresetNo or resolvedPresetNo < 1 then
        sendPresetError(presetTypeNo, presetNo, "Preset number is required")
        return nil
    end

    local presetHandle = resolvePresetHandle(resolvedPresetTypeNo, resolvedPresetNo)
    if not presetHandle then
        sendPresetError(presetTypeNo, presetNo, "Preset not found")
        return nil
    end

    local description = buildPresetDescription(resolvedPresetTypeNo, resolvedPresetNo, presetHandle)
    local payload = {
        preset_type = resolvedPresetTypeNo,
        number = resolvedPresetNo,
        object = description,
    }
    if requestId ~= nil then
        payload.request_id = requestId
    end
    EZ.sendMessage("preset", "described", payload)
    return description
end

function EZ.PreviewReplacePresetWhenGroup(
    presetTypeNo,
    sourcePresetRef,
    destPresetRef,
    groupFilterCsv,
    sequenceNumbersCsv,
    requestId
)
    local resolvedPresetTypeNo = tonumber(presetTypeNo)
    local normalizedSourcePresetRef = normalizePresetRef(sourcePresetRef)
    local normalizedDestPresetRef = normalizePresetRef(destPresetRef)
    if not resolvedPresetTypeNo or resolvedPresetTypeNo < 1 then
        sendPresetError(presetTypeNo, 0, "Preset type is required")
        return nil
    end
    if not normalizedSourcePresetRef then
        sendPresetError(presetTypeNo, 0, "Source preset ref is required")
        return nil
    end
    if not normalizedDestPresetRef then
        sendPresetError(presetTypeNo, 0, "Destination preset ref is required")
        return nil
    end

    local filterGroupLookup, filterGroupLabels = resolveGroupFilterLookup(groupFilterCsv)
    local targetSequences = resolveTargetSequences(sequenceNumbersCsv)
    local findings = collectPresetReplacementFindings(
        targetSequences,
        filterGroupLookup,
        normalizedSourcePresetRef,
        normalizedSourcePresetRef,
        normalizedDestPresetRef
    )
    local payload = {
        preset_type = resolvedPresetTypeNo,
        source_preset_ref = normalizedSourcePresetRef,
        dest_preset_ref = normalizedDestPresetRef,
        group_filter = filterGroupLabels,
        sequence_numbers = splitCsv(sequenceNumbersCsv),
        count = #findings,
        findings = findings,
    }
    if requestId ~= nil then
        payload.request_id = requestId
    end
    sendPresetReplacePayload("preview", payload)
    return findings
end

function EZ.ReplacePresetWhenGroup(
    presetTypeNo,
    sourcePresetRef,
    destPresetRef,
    groupFilterCsv,
    sequenceNumbersCsv,
    requestId
)
    local resolvedPresetTypeNo = tonumber(presetTypeNo)
    local normalizedSourcePresetRef = normalizePresetRef(sourcePresetRef)
    local normalizedDestPresetRef = normalizePresetRef(destPresetRef)
    if not resolvedPresetTypeNo or resolvedPresetTypeNo < 1 then
        sendPresetError(presetTypeNo, 0, "Preset type is required")
        return nil
    end
    if not normalizedSourcePresetRef then
        sendPresetError(presetTypeNo, 0, "Source preset ref is required")
        return nil
    end
    if not normalizedDestPresetRef then
        sendPresetError(presetTypeNo, 0, "Destination preset ref is required")
        return nil
    end

    local filterGroupLookup, filterGroupLabels = resolveGroupFilterLookup(groupFilterCsv)
    local targetSequences = resolveTargetSequences(sequenceNumbersCsv)
    local findings = collectPresetReplacementFindings(
        targetSequences,
        filterGroupLookup,
        normalizedSourcePresetRef,
        normalizedSourcePresetRef,
        normalizedDestPresetRef
    )

    local replaceCount = 0
    for _, finding in ipairs(findings) do
        local assignCmd = string.format(
            "Assign %s At Sequence %d Cue %g Part %s",
            normalizedDestPresetRef,
            finding.seqNumber,
            finding.actualCueNumber,
            finding.partNumber
        )
        local assignResult = Cmd(assignCmd)
        if assignResult then
            replaceCount = replaceCount + 1
        end
    end

    local payload = {
        preset_type = resolvedPresetTypeNo,
        source_preset_ref = normalizedSourcePresetRef,
        dest_preset_ref = normalizedDestPresetRef,
        group_filter = filterGroupLabels,
        sequence_numbers = splitCsv(sequenceNumbersCsv),
        findings = findings,
        count = #findings,
        replaced_count = replaceCount,
    }
    if requestId ~= nil then
        payload.request_id = requestId
    end
    sendPresetReplacePayload("applied", payload)
    return replaceCount
end

function EZ.AnalyzeCueRecipeState(sequenceNo, cueNo, requestId)
    local resolvedSequenceNo = tonumber(sequenceNo)
    local normalizedCueNo = normalizeCueNumberText(cueNo)
    if not resolvedSequenceNo or resolvedSequenceNo < 1 then
        sendPresetError(0, 0, "Sequence number is required")
        return nil
    end
    if not normalizedCueNo then
        sendPresetError(0, 0, "Cue number is required")
        return nil
    end

    local localLines, warnings, unsupportedReasons, actualCueNumber = buildCueRecipeRows(
        resolvedSequenceNo,
        normalizedCueNo
    )
    if not actualCueNumber then
        sendPresetError(0, 0, table.concat(dedupeTextList(unsupportedReasons), " "))
        return nil
    end
    local sequenceRows, sequenceWarnings, sequenceUnsupported = collectSequenceRecipeRowsUpTo(
        resolvedSequenceNo,
        actualCueNumber
    )
    for _, warning in ipairs(sequenceWarnings) do
        table.insert(warnings, warning)
    end
    for _, reason in ipairs(sequenceUnsupported) do
        table.insert(unsupportedReasons, reason)
    end
    local contributors = effectiveRecipeContributorsFromRows(sequenceRows)
    local stateKeyLookup = {}
    for _, row in ipairs(contributors) do
        local key = recipeStateKey(row)
        if key ~= "" and key ~= ":" then
            stateKeyLookup[key] = true
        end
    end
    local stateKeys = {}
    for key, _ in pairs(stateKeyLookup) do
        table.insert(stateKeys, key)
    end
    table.sort(stateKeys)

    local payload = {
        sequence_no = resolvedSequenceNo,
        cue_no = tonumber(actualCueNumber),
        supported = (#dedupeTextList(unsupportedReasons) == 0),
        status = (#dedupeTextList(unsupportedReasons) == 0) and "ready" or "unsupported",
        warnings = dedupeTextList(warnings),
        unsupported_reasons = dedupeTextList(unsupportedReasons),
        local_line_count = #localLines,
        contributor_count = #contributors,
        state_keys = stateKeys,
        local_lines = localLines,
        contributors = contributors,
    }
    if requestId ~= nil then
        payload.request_id = requestId
    end
    sendRecipeCuePayload("analysis", payload)
    return payload
end

function EZ.PreviewRecipeCueOnly(sequenceNo, sourceCueNo, targetCueNo, requestId)
    local resolvedSequenceNo = tonumber(sequenceNo)
    local normalizedSourceCueNo = normalizeCueNumberText(sourceCueNo)
    local normalizedTargetCueNo = normalizeCueNumberText(targetCueNo)
    if not resolvedSequenceNo or resolvedSequenceNo < 1 then
        sendPresetError(0, 0, "Sequence number is required")
        return nil
    end
    if not normalizedSourceCueNo then
        sendPresetError(0, 0, "Source cue number is required")
        return nil
    end
    if not normalizedTargetCueNo then
        sendPresetError(0, 0, "Target cue number is required")
        return nil
    end

    local sourceAnalysis = EZ.AnalyzeCueRecipeState(resolvedSequenceNo, normalizedSourceCueNo, nil)
    if not sourceAnalysis then
        return nil
    end

    local sourceCueValue = cueNumberValue(normalizedSourceCueNo)
    local targetCueValue = cueNumberValue(normalizedTargetCueNo)
    local nextCueValue = targetCueValue and (targetCueValue + 1.0) or nil
    if not sourceCueValue or not targetCueValue or not nextCueValue then
        sendPresetError(0, 0, "Cue numbers must be positive numeric data")
        return nil
    end

    local sequenceRows, sequenceWarnings, sequenceUnsupported = collectSequenceRecipeRowsUpTo(
        resolvedSequenceNo,
        nextCueValue
    )
    local beforeNext = effectiveRecipeContributorsFromRows(sequenceRows)
    local simulatedRows = {}
    for _, row in ipairs(sequenceRows) do
        if not (
            tonumber(row.seq_number or 0) == resolvedSequenceNo
            and tonumber(row.actual_cue_number or 0) == targetCueValue
        ) then
            table.insert(simulatedRows, row)
        end
    end
    local storedLines = cloneRows(sourceAnalysis.local_lines or {})
    for _, row in ipairs(storedLines) do
        row.actual_cue_number = targetCueValue
        row.seq_number = resolvedSequenceNo
    end
    for lineIndex, row in ipairs(storedLines) do
        row.line_index = lineIndex
        row.part_number = string.format("0.%d", lineIndex)
        row.source_cue_number = tonumber(row.source_cue_number or sourceCueValue)
        row.source_part_number = tostring(row.source_part_number or row.part_number)
        table.insert(simulatedRows, row)
    end
    local afterNext = effectiveRecipeContributorsFromRows(simulatedRows)

    local affectedLookup = {}
    for _, row in ipairs(sourceAnalysis.local_lines or {}) do
        affectedLookup[recipeStateKey(row)] = true
    end
    local restoreRows = {}
    for _, row in ipairs(beforeNext) do
        local key = recipeStateKey(row)
        if affectedLookup[key] then
            table.insert(restoreRows, row)
        end
    end
    local changedKeys = {}
    for key, _ in pairs(affectedLookup) do
        local beforeBucket = {}
        local afterBucket = {}
        for _, row in ipairs(beforeNext) do
            if recipeStateKey(row) == key then
                table.insert(beforeBucket, row)
            end
        end
        for _, row in ipairs(afterNext) do
            if recipeStateKey(row) == key then
                table.insert(afterBucket, row)
            end
        end
        if not signatureSetsMatch(beforeBucket, afterBucket) then
            table.insert(changedKeys, key)
        end
    end
    table.sort(changedKeys)

    local filteredRestoreRows = {}
    local changedLookup = {}
    for _, key in ipairs(changedKeys) do
        changedLookup[key] = true
    end
    for _, row in ipairs(restoreRows) do
        if changedLookup[recipeStateKey(row)] then
            table.insert(filteredRestoreRows, row)
        end
    end

    local warnings = {}
    local unsupportedReasons = {}
    for _, warning in ipairs(sourceAnalysis.warnings or {}) do
        table.insert(warnings, warning)
    end
    for _, warning in ipairs(sequenceWarnings or {}) do
        table.insert(warnings, warning)
    end
    for _, reason in ipairs(sourceAnalysis.unsupported_reasons or {}) do
        table.insert(unsupportedReasons, reason)
    end
    for _, reason in ipairs(sequenceUnsupported or {}) do
        table.insert(unsupportedReasons, reason)
    end
    if sourceCueValue == targetCueValue then
        table.insert(unsupportedReasons, "Source cue and target cue must be different.")
    end
    if #(sourceAnalysis.local_lines or {}) == 0 then
        table.insert(unsupportedReasons, "Source cue does not expose local recipe lines.")
    end
    if #beforeNext > 0 then
        table.insert(
            warnings,
            "Cue-only preview only restores detected recipe contributors in the following cue; direct stored values are not modeled."
        )
    end

    local payload = {
        sequence_no = resolvedSequenceNo,
        source_cue_no = sourceCueValue,
        target_cue_no = targetCueValue,
        next_cue_no = nextCueValue,
        supported = (#dedupeTextList(unsupportedReasons) == 0),
        status = (#dedupeTextList(unsupportedReasons) == 0) and "ready" or "unsupported",
        warnings = dedupeTextList(warnings),
        unsupported_reasons = dedupeTextList(unsupportedReasons),
        stored_lines = storedLines,
        restore_lines = filteredRestoreRows,
        changed_keys = changedKeys,
    }
    if requestId ~= nil then
        payload.request_id = requestId
    end
    sendRecipeCuePayload("cue_only_preview", payload)
    return payload
end

function EZ.PreviewCopyCueWithStatus(sequenceNo, sourceCueNo, destCueNo, requestId)
    local resolvedSequenceNo = tonumber(sequenceNo)
    local normalizedSourceCueNo = normalizeCueNumberText(sourceCueNo)
    local normalizedDestCueNo = normalizeCueNumberText(destCueNo)
    if not resolvedSequenceNo or resolvedSequenceNo < 1 then
        sendPresetError(0, 0, "Sequence number is required")
        return nil
    end
    if not normalizedSourceCueNo then
        sendPresetError(0, 0, "Source cue number is required")
        return nil
    end
    if not normalizedDestCueNo then
        sendPresetError(0, 0, "Destination cue number is required")
        return nil
    end

    local analysis = EZ.AnalyzeCueRecipeState(resolvedSequenceNo, normalizedSourceCueNo, nil)
    if not analysis then
        return nil
    end
    local warnings = {}
    local unsupportedReasons = {}
    for _, warning in ipairs(analysis.warnings or {}) do
        table.insert(warnings, warning)
    end
    for _, reason in ipairs(analysis.unsupported_reasons or {}) do
        table.insert(unsupportedReasons, reason)
    end
    if normalizedSourceCueNo == normalizedDestCueNo then
        table.insert(unsupportedReasons, "Source cue and destination cue must be different.")
    end
    if tonumber(analysis.contributor_count or 0) > tonumber(analysis.local_line_count or 0) then
        table.insert(
            warnings,
            "Status preview includes tracked contributors from earlier cues, not only local recipe lines."
        )
    end

    local payload = {
        sequence_no = resolvedSequenceNo,
        source_cue_no = cueNumberValue(normalizedSourceCueNo),
        dest_cue_no = cueNumberValue(normalizedDestCueNo),
        supported = (#dedupeTextList(unsupportedReasons) == 0),
        status = (#dedupeTextList(unsupportedReasons) == 0) and "ready" or "unsupported",
        warnings = dedupeTextList(warnings),
        unsupported_reasons = dedupeTextList(unsupportedReasons),
        copied_lines = cloneRows(analysis.contributors or {}),
        copied_line_count = tonumber(analysis.contributor_count or 0),
        local_line_count = tonumber(analysis.local_line_count or 0),
        contributor_count = tonumber(analysis.contributor_count or 0),
    }
    if requestId ~= nil then
        payload.request_id = requestId
    end
    sendRecipeCuePayload("copy_with_status_preview", payload)
    return payload
end

function EZ.ApplyRecipeCueOnly(sequenceNo, sourceCueNo, targetCueNo, requestId)
    local payload = EZ.PreviewRecipeCueOnly(sequenceNo, sourceCueNo, targetCueNo, nil)
    if payload == nil then
        return nil
    end
    if requestId ~= nil then
        payload.request_id = requestId
    end
    payload.supported = false
    payload.status = "unsupported"
    payload.unsupported_reasons = dedupeTextList({
        "ApplyRecipeCueOnly is intentionally blocked in this preview-only pass.",
    })
    sendRecipeCuePayload("cue_only_applied", payload)
    return nil
end

function EZ.CopyCueWithStatus(sequenceNo, sourceCueNo, destCueNo, requestId)
    local payload = EZ.PreviewCopyCueWithStatus(sequenceNo, sourceCueNo, destCueNo, nil)
    if payload == nil then
        return nil
    end
    if requestId ~= nil then
        payload.request_id = requestId
    end
    payload.supported = false
    payload.status = "unsupported"
    payload.unsupported_reasons = dedupeTextList({
        "CopyCueWithStatus is intentionally blocked in this preview-only pass. Use PreviewCopyCueWithStatus for planning output.",
    })
    sendRecipeCuePayload("copied_with_status", payload)
    return nil
end
