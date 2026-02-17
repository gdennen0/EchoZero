# Set Target Timecode Command

## Overview

The `SetTargetTimecodeCommand` allows users to configure which timecode pool number in MA3 the ShowManager block should use to fetch structure (tracks) and events.

## Purpose

In MA3 (grandMA3 lighting console), timecode pools are numbered (e.g., 101, 102, 103). The ShowManager needs to know which timecode pool to query for tracks and events. This command provides an undoable way to change that setting.

## Implementation

### Command Details

**Location:** `src/application/commands/layer_sync/set_target_timecode_command.py`

**Class:** `SetTargetTimecodeCommand`

**Command Type:** `"layer_sync.set_target_timecode"`

### Parameters

- `facade: ApplicationFacade` - Application facade instance
- `show_manager_block_id: str` - ID of the ShowManager block
- `timecode_no: int` - Target timecode number (must be >= 1)

### Behavior

**Redo:**
- Validates timecode number (must be >= 1)
- Updates `target_timecode` setting via `ShowManagerSettingsManager`
- Logs the change with INFO level

**Undo:**
- Restores the previous `target_timecode` value
- Logs the restoration with INFO level

### Usage Example

```python
from src.application.commands import SetTargetTimecodeCommand

# Set target timecode to 101
cmd = SetTargetTimecodeCommand(
    facade=facade,
    show_manager_block_id="block-123",
    timecode_no=101
)
facade.command_bus.execute(cmd)

# Undo the change
facade.command_bus.undo()
```

## Integration Points

### Settings

The command updates `ShowManagerSettings.target_timecode`, which is already defined in:
- `src/application/settings/show_manager_settings.py`

The setting defaults to `1` if not explicitly set.

### UI Integration

This command should be integrated into the ShowManagerPanel UI, allowing users to:
1. View the current target timecode
2. Change it via a spinbox or input field
3. Have the change be undoable

Example UI integration:
```python
# In ShowManagerPanel
def _on_timecode_changed(self, value: int):
    from src.application.commands import SetTargetTimecodeCommand
    cmd = SetTargetTimecodeCommand(
        facade=self.facade,
        show_manager_block_id=self.block_id,
        timecode_no=value
    )
    self.facade.command_bus.execute(cmd)
```

## Related Components

- **ShowManagerSettingsManager**: Manages the `target_timecode` setting
- **ShowManagerPanel**: UI that should expose this setting
- **Auto-fetch logic**: Uses `target_timecode` when fetching MA3 structure/events

## Testing Considerations

1. **Validation**: Test that timecode < 1 raises ValueError
2. **Undo/Redo**: Verify setting is correctly restored on undo
3. **Persistence**: Confirm setting is saved to block metadata
4. **UI Update**: Ensure UI reflects change immediately

## Future Enhancements

- Could add validation for maximum timecode number (if MA3 has limits)
- Could add warning if changing timecode while synced layers exist
- Could auto-refresh MA3 data after timecode change
