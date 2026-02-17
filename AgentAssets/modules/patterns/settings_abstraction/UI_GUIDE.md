# UI Settings Guide

> **Note**: This document is specific to UI/widget settings. For the complete settings standard, see [SETTINGS_STANDARD.md](./SETTINGS_STANDARD.md).

## Overview

All user-configurable UI options, widget settings, and visual preferences **MUST** be persisted using the standardized settings system. This ensures:
- User preferences are remembered across sessions
- Settings are easily extensible
- Single source of truth for UI state
- Consistent patterns across the application

## Timeline Widget Settings

The Timeline Widget uses `TimelineSettingsManager` (inheriting from `BaseSettingsManager`) for all UI settings.

### Adding a New Setting

When adding any new UI option or setting to the timeline widget (or any block containing it), follow these steps:

#### 1. Add to Settings Schema

In `ui/qt_gui/widgets/timeline/settings_storage.py`, add the field to `TimelineSettings` dataclass:

```python
@dataclass
class TimelineSettings:
    # ... existing settings ...
    
    # Add new setting with default value
    my_new_setting: bool = True  # Description of what this controls
```

#### 2. Add Property Accessor (Optional but Recommended)

For type-safe access and validation, add property getter/setter in `TimelineSettingsManager`:

```python
@property
def my_new_setting(self) -> bool:
    return self._settings.my_new_setting

@my_new_setting.setter
def my_new_setting(self, value: bool):
    if value != self._settings.my_new_setting:
        self._settings.my_new_setting = value
        self._save_setting('my_new_setting')
```

#### 3. Apply Setting in Widget

In `TimelineWidget` or relevant component, apply the setting:

```python
def _apply_my_setting(self):
    """Apply my_new_setting from settings manager."""
    if self._settings_manager.my_new_setting:
        # Apply when True
        pass
    else:
        # Apply when False
        pass
```

#### 4. Provide Public API (If Needed)

If the setting should be changeable at runtime:

```python
def set_my_new_setting(self, value: bool):
    """Set my_new_setting (persists to settings)."""
    self._settings_manager.my_new_setting = value
    self._apply_my_setting()
```

### Current Settings Categories

| Category | Settings |
|----------|----------|
| Layer Panel | `layer_column_width`, `layer_column_min_width`, `layer_column_max_width`, `default_layer_height` |
| Grid/Snap | `snap_enabled`, `snap_to_grid`, `snap_threshold` |
| Zoom | `default_pixels_per_second`, `min_pixels_per_second`, `max_pixels_per_second` |
| Playback | `playhead_follow_mode` |
| Visual | `show_grid_lines`, `show_waveform`, `waveform_opacity` |
| Scrollbar | `vertical_scrollbar_always_visible`, `horizontal_scrollbar_always_visible` |
| Inspector | `inspector_visible`, `inspector_width` |
| Colors | `recent_layer_colors` |

## Block Panel Settings

For block-specific settings that should persist:
- Use the existing `preferences_repo` from the `ApplicationFacade`
- Pass `preferences_repo` to widgets that need settings persistence

Example:
```python
preferences_repo = getattr(self.facade, 'preferences_repo', None)
self.timeline_widget = TimelineWidget(preferences_repo=preferences_repo)
```

## Testing Settings

When adding new settings, ensure:
1. Default value works correctly
2. Setting persists across app restarts
3. Setting loads correctly when missing (backwards compatibility)
4. UI updates immediately when setting changes

## Key Principles

1. **Single Source of Truth**: All UI state goes through settings manager
2. **Auto-Save**: Settings save automatically when changed (debounced where appropriate)
3. **Backwards Compatible**: New settings must have defaults that work with old saved data
4. **Type-Safe**: Use dataclass for schema, properties for validation
5. **Signals**: Emit `settings_changed` signal so UI can react to changes







