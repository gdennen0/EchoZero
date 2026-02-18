# Timecode Structure Exploration Guide

Step-by-step guide for using the exploration tools to discover MA3 timecode structure.

## Setup

1. **Copy the plugin** to your MA3 show:
   ```
   [MA3 Show]/datapools/plugins/timecode_explorer.lua
   ```

2. **Load the plugin** in grandMA3:
   - Go to **Setup > Plugins**
   - Load `timecode_explorer`

3. **Verify it loaded**:
   - Check command line for: "Timecode Explorer Plugin loaded"

## Quick Start

### Step 0: Show Full Hierarchy

**NEW**: Show the complete hierarchy at once:

```lua
Lua "PrintHierarchy(101)"
```

Or with event properties:

```lua
Lua "PrintHierarchyWithProperties(101)"
```

This shows:
- Complete timecode structure
- All track groups
- All layers (with Marker noted)
- All events (with properties if requested)

### Step 1: Quick Test

Run a quick test to verify basic access:

```lua
Lua "QuickTest()"
```

This will:
- Check if Timecode 101 exists
- Try to access its first child
- Print basic information

### Step 2: Explore Specific Timecode

Explore a specific timecode track:

```lua
Lua "ExploreTimecodeByNumber(101)"
```

Replace `101` with your timecode number. This will:
- Show timecode properties
- Explore its children (tracks/layers)
- Try direct index access
- Show nested structure

### Step 3: Test Access Patterns

Compare different ways to access objects:

```lua
Lua "TestAccessPatterns(101)"
```

This compares:
- Direct index access: `tc[1]`
- Children() method: `tc:Children()[1]`
- Shows which works and when

### Step 4: Explore Specific Path

Test a specific nested path:

```lua
Lua "ExplorePath(101, 1, 1, 1)"
```

This tests:
- `DataPool().Timecodes[101][1][1][1]`
- Each level of nesting
- Properties at each level

### Step 5: Full Exploration

Run complete exploration:

```lua
Lua "FullExploration()"
```

This runs all tests and provides comprehensive output.

## Understanding the Output

### Object Information

Each object shows:
- `type` - Lua type (table, userdata, etc.)
- `name` - Object name (if available)
- `no` - Object number (if available)
- `index` - Array index (if available)
- `children_count` - Number of children (if available)

### Access Patterns

The explorer tests:
1. **Direct Index**: `obj[1]`, `obj[2]`, etc.
2. **Children() Method**: `obj:Children()` returns array
3. **Comparison**: Shows which method works

### Structure Discovery

The output shows:
- What exists at each level
- How many children each object has
- Properties available at each level
- Nested structure depth

## Documenting Findings

After running exploration:

1. **Update TIMECODE_STRUCTURE.md** with discovered properties
2. **Update MA3_NESTED_ACCESS.md** with working patterns
3. **Add examples** to documentation
4. **Create helper functions** for common operations

## Example Workflow

```lua
-- 1. Show full hierarchy (RECOMMENDED FIRST STEP)
Lua "PrintHierarchy(101)"

-- 2. Show hierarchy with event properties
Lua "PrintHierarchyWithProperties(101)"

-- 3. Quick test
Lua "QuickTest()"

-- 4. Explore your timecode
Lua "ExploreTimecodeByNumber(101)"

-- 5. Test specific path
Lua "ExplorePath(101, 1, 1, 1)"

-- 6. Compare access methods
Lua "TestAccessPatterns(101)"

-- 7. Full exploration
Lua "FullExploration()"
```

## Troubleshooting

### "Timecode not found"

- Verify the timecode number exists in your show
- Try a different timecode number
- Check MA3 command line for errors

### "Could not get children"

- Object may not have children
- Try direct index access instead
- Check if object is nil

### No output

- Check MA3 command line (not just Echo)
- Verify plugin loaded correctly
- Try `QuickTest()` first

## Next Steps

After exploration:

1. **Document structure** in TIMECODE_STRUCTURE.md
2. **Create helper functions** for common access patterns
3. **Build integration** using discovered patterns
4. **Test with real data** from your show

## Advanced Usage

### Custom Exploration

Modify `timecode_explorer.lua` to add custom tests:

```lua
function CustomTest()
    -- Your custom exploration code
end
```

### Hook Testing

Combine with `test.lua` to hook into discovered objects:

```lua
-- After discovering structure, hook into specific level
local event = DataPool().Timecodes[101][1][1][1]
-- Hook into event changes
```

## Related Documentation

- `TIMECODE_STRUCTURE.md` - Document your findings here
- `MA3_NESTED_ACCESS.md` - Access pattern reference
- `MA3_DATA_STRUCTURES.md` - Complete structure docs

