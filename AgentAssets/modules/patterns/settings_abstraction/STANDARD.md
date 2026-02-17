# EchoZero Settings Standard

## Overview

All user-configurable settings, preferences, and UI state in EchoZero **MUST** follow this standardized pattern. This ensures:

- Consistent save/load behavior across the application
- Easy addition/removal of settings
- Type safety and validation
- Automatic persistence
- UI reactivity via signals

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Settings Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ AppSettings      │  │ TimelineSettings │  │ BlockSettings │  │
│  │ (Global)         │  │ (Widget)         │  │ (Per-block)   │  │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬───────┘  │
│           │                     │                     │          │
│           └─────────────────────┼─────────────────────┘          │
│                                 │                                │
│                                 ▼                                │
│           ┌─────────────────────────────────────────┐            │
│           │         BaseSettingsManager             │            │
│           │  - Dataclass schema (BaseSettings)      │            │
│           │  - Property accessors                   │            │
│           │  - Auto-save with debouncing           │            │
│           │  - Signal emission                     │            │
│           │  - Backwards-compatible loading        │            │
│           └─────────────────────┬───────────────────┘            │
│                                 │                                │
│                                 ▼                                │
│           ┌─────────────────────────────────────────┐            │
│           │         PreferencesRepository           │            │
│           │  (SQLite preferences table)             │            │
│           └─────────────────────────────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Settings Categories

| Category | Namespace | Location | Scope |
|----------|-----------|----------|-------|
| Application | `app` | `src/application/settings/app_settings.py` | Global |
| Timeline Widget | `timeline` | `ui/qt_gui/widgets/timeline/settings/storage.py` | Per-widget |
| Block-specific | `block.<type>` | `src/application/settings/` or block panel files | Per-block type |
| Show Manager | `block.show_manager` | `src/application/settings/show_manager_settings.py` | Per-block |

## Creating a New Settings Manager

### Step 1: Define the Settings Schema

Create a dataclass inheriting from `BaseSettings`:

```python
from dataclasses import dataclass, field
from typing import List
from src.application.settings import BaseSettings

@dataclass
class MyComponentSettings(BaseSettings):
    """
    Settings schema for MyComponent.
    
    All fields MUST have default values for backwards compatibility.
    """
    
    # Group related settings with comments
    # Visual settings
    theme: str = "dark"
    font_size: int = 12
    
    # Behavior settings
    auto_refresh: bool = True
    refresh_interval: int = 5
    
    # Collections use field(default_factory=...)
    recent_items: List[str] = field(default_factory=list)
```

### Step 2: Create the Settings Manager

Inherit from `BaseSettingsManager`:

```python
from src.application.settings import BaseSettingsManager
from typing import Optional

class MyComponentSettingsManager(BaseSettingsManager[MyComponentSettings]):
    """Settings manager for MyComponent."""
    
    # Required: Unique namespace for storage key
    NAMESPACE = "my_component"
    SETTINGS_CLASS = MyComponentSettings
    
    # Optional: Override debounce delay (default 300ms)
    SAVE_DEBOUNCE_MS = 500
    
    def __init__(self, preferences_repo=None, parent=None):
        super().__init__(preferences_repo, parent)
    
    # Add type-safe property accessors
    @property
    def theme(self) -> str:
        return self._settings.theme
    
    @theme.setter
    def theme(self, value: str):
        valid_themes = {"dark", "light"}
        if value in valid_themes and value != self._settings.theme:
            self._settings.theme = value
            self._save_setting('theme')  # Triggers debounced save + signal
    
    @property
    def font_size(self) -> int:
        return self._settings.font_size
    
    @font_size.setter
    def font_size(self, value: int):
        if value != self._settings.font_size:
            # Apply validation
            self._settings.font_size = max(8, min(value, 72))
            self._save_setting('font_size')
```

### Step 3: Initialize with PreferencesRepository

```python
# In your component initialization
preferences_repo = facade.preferences_repo  # or get from bootstrap
settings_manager = MyComponentSettingsManager(preferences_repo)

# Use settings
current_theme = settings_manager.theme
settings_manager.font_size = 14  # Auto-saves after debounce
```

### Step 4: React to Setting Changes

```python
# Connect to signals for UI updates
settings_manager.settings_changed.connect(self._on_setting_changed)
settings_manager.settings_loaded.connect(self._on_settings_loaded)

def _on_setting_changed(self, setting_name: str):
    if setting_name == 'theme':
        self._apply_theme()
    elif setting_name == 'font_size':
        self._update_font()
```

## Adding a New Setting to Existing Manager

### Quick Checklist

1. [ ] Add field to dataclass with default value
2. [ ] Add property getter in manager
3. [ ] Add property setter in manager (with validation)
4. [ ] Apply setting where needed in code
5. [ ] Update tests

### Example: Adding `show_tooltips` to AppSettings

```python
# 1. In AppSettings dataclass
@dataclass
class AppSettings(BaseSettings):
    # ... existing fields ...
    show_tooltips: bool = True  # New setting with default

# 2. & 3. In AppSettingsManager
@property
def show_tooltips(self) -> bool:
    return self._settings.show_tooltips

@show_tooltips.setter
def show_tooltips(self, value: bool):
    if value != self._settings.show_tooltips:
        self._settings.show_tooltips = value
        self._save_setting('show_tooltips')

# 4. Apply in UI
def _update_tooltips(self):
    for widget in self._widgets:
        widget.setToolTipsVisible(self._settings_manager.show_tooltips)
```

## Best Practices

### DO:
- Always provide default values for new settings
- Use descriptive field names
- Group related settings with comments
- Validate values in setters
- Use `_save_setting(key)` to trigger save + signal
- Test that settings persist across restarts

### DON'T:
- Store sensitive data in settings (use secure storage)
- Use complex objects (keep it JSON-serializable)
- Forget to emit signals when settings change
- Create circular dependencies between settings managers

## Namespace Conventions

| Pattern | Example | Use For |
|---------|---------|---------|
| `app` | `app.settings` | Global application settings |
| `timeline` | `timeline.settings` | Timeline widget settings |
| `block.<type>` | `block.editor.settings` | Block-specific settings |
| `panel.<name>` | `panel.inspector.settings` | Panel-specific settings |

## Storage Format

Settings are stored in the SQLite `preferences` table as JSON:

```
Key: "timeline.settings"
Value: {
    "layer_column_width": 150,
    "snap_enabled": true,
    "vertical_scrollbar_always_visible": true,
    ...
}
```

## Migration / Versioning

When adding new settings:
- New fields with defaults are automatically handled
- Removed fields are ignored when loading
- No migration code needed for simple additions

For breaking changes (rare):
- Add schema version field
- Implement migration in `from_dict()`

## Testing Settings

```python
import pytest
from unittest.mock import MagicMock

class TestMyComponentSettings:
    
    def test_default_values(self):
        settings = MyComponentSettings()
        assert settings.theme == "dark"
        assert settings.font_size == 12
    
    def test_persistence(self):
        mock_repo = MagicMock()
        mock_repo.get.return_value = {'theme': 'light'}
        
        manager = MyComponentSettingsManager(mock_repo)
        assert manager.theme == 'light'
    
    def test_auto_save(self):
        mock_repo = MagicMock()
        mock_repo.get.return_value = {}
        
        manager = MyComponentSettingsManager(mock_repo)
        manager.theme = 'light'
        
        # Force save (bypass debounce for testing)
        manager.force_save()
        
        mock_repo.set.assert_called()
```

## File Locations

| File | Purpose |
|------|---------|
| `src/application/settings/base_settings.py` | BaseSettings + BaseSettingsManager base classes |
| `src/application/settings/block_settings.py` | BlockSettingsManager (metadata-backed, per-block) |
| `src/application/settings/app_settings.py` | AppSettings + AppSettingsManager (global) |
| `src/application/settings/show_manager_settings.py` | ShowManagerSettings + ShowManagerSettingsManager |
| `src/application/settings/__init__.py` | Public exports |
| `ui/qt_gui/widgets/timeline/settings/storage.py` | Timeline settings |

## Agent Instructions

When asked to add a new setting or preference:

1. **Identify the scope**: Is it global (app), widget-specific (timeline), or block-specific?
2. **Follow the pattern**: Add to appropriate dataclass + manager
3. **Include validation**: Setters should validate and clamp values
4. **Apply the setting**: Make sure it's actually used somewhere
5. **Test persistence**: Verify it saves and loads correctly
6. **Document**: Update this guide if adding a new settings category







