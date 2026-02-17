# Settings Abstraction Preset for AI Agents

## Purpose

This preset defines the standardized pattern for implementing configurable settings in EchoZero. Use this pattern whenever you need to add or modify settings that:
- Persist across sessions
- Are accessed from multiple UI locations (panels, quick actions)
- Need validation
- Should be undoable

## Core Principle

**Single Source of Truth:** `block.metadata` in the database is the single source of truth for all settings. All UI components (panels, quick actions) read from and write to this same source.

**Single Pathway Rule:** All settings changes MUST go through the settings manager. Multiple update pathways create inconsistencies and bugs.

## Pattern Template

### 1. Define Settings Schema

Create a dataclass with all settings and default values:

```python
from dataclasses import dataclass
from typing import Optional
from src.application.settings.base_settings import BaseSettings

@dataclass
class MyBlockSettings(BaseSettings):
    """
    Settings for MyBlock.
    
    All fields MUST have default values for backwards compatibility.
    """
    # Group related settings with comments
    # Model settings
    model: str = "default_model"
    
    # Processing settings
    device: str = "auto"
    threshold: float = 0.5
    
    # Optional settings
    optional_mode: Optional[str] = None
```

### 2. Create Settings Manager

Inherit from `BlockSettingsManager` (for block settings) or `BaseSettingsManager` (for app/widget settings):

```python
from src.application.settings.block_settings import BlockSettingsManager

class MyBlockSettingsManager(BlockSettingsManager):
    """
    Settings manager for MyBlock.
    Provides type-safe property accessors with validation.
    """
    NAMESPACE = "block.myblock"  # Unique identifier
    SETTINGS_CLASS = MyBlockSettings
    
    def __init__(self, facade, block_id, preferences_repo=None):
        super().__init__(facade, block_id, preferences_repo)
    
    # Property accessors with validation
    @property
    def model(self) -> str:
        return self._settings.model
    
    @model.setter
    def model(self, value: str):
        # Validate
        valid_models = {"default_model", "model_a", "model_b"}
        if value not in valid_models:
            raise ValueError(
                f"Invalid model: '{value}'. "
                f"Valid options: {', '.join(valid_models)}"
            )
        
        # Update if changed
        if value != self._settings.model:
            self._settings.model = value
            self._save_setting('model')  # Auto-saves and emits event
    
    @property
    def threshold(self) -> float:
        return self._settings.threshold
    
    @threshold.setter
    def threshold(self, value: float):
        # Validate and clamp
        if not 0.0 <= value <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        # Update if changed
        if abs(value - self._settings.threshold) > 0.001:
            self._settings.threshold = value
            self._save_setting('threshold')
```

### 3. Use in UI Components

**In Block Panels:**

```python
class MyBlockPanel(BlockPanelBase):
    def __init__(self, block_id: str, facade, parent=None):
        # Call parent init first
        super().__init__(block_id, facade, parent)
        
        # Initialize settings manager AFTER parent init
        self._settings_manager = MyBlockSettingsManager(facade, block_id, parent=self)
        
        # Connect to settings changes for UI updates (local changes)
        self._settings_manager.settings_changed.connect(self._on_setting_changed)
        
        # Refresh UI now that settings manager is ready
        if self.block:
            self.refresh()
    
    def create_content_widget(self):
        # Create UI controls
        self.model_combo = QComboBox()
        self.model_combo.addItem("Default Model", "default_model")
        self.model_combo.addItem("Model A", "model_a")
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        
        return widget
    
    def refresh(self):
        """Update UI with current settings"""
        if not self.block:
            return
        
        # Block signals while updating
        self.model_combo.blockSignals(True)
        
        # Load settings from manager
        current_model = self._settings_manager.model
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == current_model:
                self.model_combo.setCurrentIndex(i)
                break
        
        # Unblock signals
        self.model_combo.blockSignals(False)
    
    def _on_model_changed(self, index: int):
        """Handle user change - update via settings manager"""
        new_model = self.model_combo.itemData(index)
        if new_model:
            try:
                # Single pathway: settings manager handles everything
                self._settings_manager.model = new_model
                self.set_status_message(f"Model set to {new_model}")
            except ValueError as e:
                self.set_status_message(str(e), error=True)
    
    def refresh_for_undo(self):
        """
        Refresh panel after undo/redo operation.
        
        Reloads settings from database (single source of truth) and refreshes UI.
        """
        if hasattr(self, '_settings_manager') and self._settings_manager:
            self._settings_manager.reload_from_storage()
        self.refresh()
    
    def _on_block_updated(self, event):
        """
        Handle block update event - reload settings and refresh UI.
        
        This ensures panel stays in sync when settings change via quick actions
        or other sources. Single source of truth: block.metadata in database.
        """
        updated_block_id = event.data.get('id')
        if updated_block_id == self.block_id:
            # Skip if we triggered this update (prevents refresh loop)
            if self._is_saving:
                return
            
            # Reload settings from database (single source of truth)
            if hasattr(self, '_settings_manager') and self._settings_manager:
                self._settings_manager.reload_from_storage()
            
            # Refresh UI to reflect changes
            self.refresh()
    
    def _on_setting_changed(self, setting_name: str):
        """React to settings changes from this panel's settings manager"""
        if setting_name == 'model':
            self.refresh()  # Update UI to reflect change
```

**In Quick Actions:**

```python
@quick_action(
    "MyBlock",
    "Set Model",
    description="Choose processing model",
    category=ActionCategory.CONFIGURE,
    icon="model",
    primary=True
)
def action_set_model(facade, block_id: str, model: str = None, **kwargs):
    """
    Set model via settings manager.
    
    CRITICAL: Always read current value from settings manager before showing
    input dialog. Never use hardcoded defaults - use current value from database
    (single source of truth).
    """
    from src.application.settings.my_block_settings import MyBlockSettingsManager
    
    if model:
        # Write path: set the model
        try:
            settings_manager = MyBlockSettingsManager(facade, block_id)
            settings_manager.model = model
            # CRITICAL: Force immediate save for quick actions
            # This bypasses debounce (300ms) to ensure BlockUpdated event fires immediately
            # Panel changes use debounced save (better for rapid changes), but quick actions
            # need immediate save for responsive UI updates
            settings_manager.force_save()
            return {"success": True, "message": f"Model set to {model}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    # Read path: get current value from single source of truth for dialog default
    try:
        settings_manager = MyBlockSettingsManager(facade, block_id)
        current_model = settings_manager.model  # Read from database
    except Exception:
        # Fallback if settings manager fails to load
        current_model = "default_model"
    
    # Return input requirements with CURRENT value as default
    return {
        "needs_input": True,
        "input_type": "choice",
        "choices": ["default_model", "model_a", "model_b"],
        "default": current_model,  #  Current value from database, not hardcoded
        "title": "Select Model"
    }
```

**For Number Inputs (with increment/decrement arrows):**

```python
@quick_action("MyBlock", "Set Threshold", ...)
def action_set_threshold(facade, block_id: str, value: float = None, **kwargs):
    """Set threshold via settings manager"""
    from src.application.settings.my_block_settings import MyBlockSettingsManager
    
    if value is not None:
        # Write path
        try:
            settings_manager = MyBlockSettingsManager(facade, block_id)
            settings_manager.threshold = float(value)
            settings_manager.force_save()
            return {"success": True, "message": f"Threshold set to {value:.2f}"}
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    # Read path: get current value
    try:
        settings_manager = MyBlockSettingsManager(facade, block_id)
        current_value = settings_manager.threshold
    except Exception:
        current_value = 0.5
    
    return {
        "needs_input": True,
        "input_type": "number",
        "min": 0.0,                    #  REQUIRED: Minimum value
        "max": 1.0,                    #  REQUIRED: Maximum value
        "default": current_value,      #  Current value from database
        "decimals": 2,                 #  REQUIRED: Decimal places
        "increment_jump": 0.05,        #  REQUIRED: Step size for arrows (preferred name)
        # "step": 0.05,                # Alternative name (fallback if increment_jump not provided)
        "title": "Set Threshold (0.0-1.0)"
    }
```

**CRITICAL: Number Input Parameters**
- **min/max**: Always specify - prevents invalid input
- **decimals**: Always specify - controls precision
- **increment_jump**: Always specify - step size for spinbox arrows (preferred name)
- **step**: Alternative name for increment_jump (fallback support)

**See:** `AgentAssets/QUICK_ACTIONS_INPUT_DIALOGS.md` for complete reference guide

## Key Rules

### DO:

1. **Always use settings manager** - Never directly mutate `block.metadata`
2. **Single source of truth** - `block.metadata` in database is the source of truth
3. **Read current value before showing dialogs** - Quick actions must read current value from settings manager and use as default
4. **Force immediate save in quick actions** - Quick actions must call `force_save()` after setting values to ensure immediate UI refresh (bypasses 300ms debounce)
4. **Reload after external changes** - Call `reload_from_storage()` after undo/redo or when BlockUpdated event fires
5. **Provide default values** - All settings must have defaults for backwards compatibility
6. **Validate in setters** - Catch invalid values before saving
7. **Use property accessors** - Type-safe access, not `get()`/`set()` methods
8. **Let manager handle persistence** - Don't manually save to database
9. **Listen to events** - Subscribe to `BlockUpdated` for external changes, `settings_changed` for local changes
10. **Block signals during refresh** - Prevent recursive updates
11. **Override `refresh_for_undo()`** - Reload settings from database after undo/redo

### DON'T:

1. **Don't use multiple update pathways** - No `execute_block_command`, no direct metadata mutation
2. **Don't use hardcoded defaults in dialogs** - Always read current value from settings manager
3. **Don't use hardcoded mappings** - No command-name-to-metadata-key mappings
4. **Don't use flag-based refresh guards** - Use event-driven refresh instead
5. **Don't store settings inconsistently** - All settings in schema, no ad-hoc metadata keys
6. **Don't skip validation** - Always validate in setters
7. **Don't forget defaults** - New settings must have defaults for backwards compatibility

## Migration Checklist

When migrating existing settings to this pattern:

- [ ] Identify all current update pathways (commands, direct mutations, etc.)
- [ ] Create settings schema dataclass with all current settings
- [ ] Add default values matching current behavior
- [ ] Create settings manager with property accessors
- [ ] Add validation to setters
- [ ] Update UI panel to use settings manager
- [ ] Update quick actions to use settings manager
- [ ] Test persistence (close/open project)
- [ ] Test refresh (change in panel, verify quick action reflects it)
- [ ] Test undo/redo
- [ ] Remove old update pathways
- [ ] Update tests

## Common Patterns

### Pattern: Boolean Setting

```python
@property
def enabled(self) -> bool:
    return self._settings.enabled

@enabled.setter
def enabled(self, value: bool):
    if value != self._settings.enabled:
        self._settings.enabled = value
        self._save_setting('enabled')
```

### Pattern: Enum/Choice Setting

```python
@property
def mode(self) -> str:
    return self._settings.mode

@mode.setter
def mode(self, value: str):
    valid_modes = {"fast", "balanced", "quality"}
    if value not in valid_modes:
        raise ValueError(f"Invalid mode: {value}. Valid: {valid_modes}")
    if value != self._settings.mode:
        self._settings.mode = value
        self._save_setting('mode')
```

### Pattern: Numeric Setting with Range

```python
@property
def threshold(self) -> float:
    return self._settings.threshold

@threshold.setter
def threshold(self, value: float):
    if not 0.0 <= value <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")
    if abs(value - self._settings.threshold) > 0.001:  # Float comparison
        self._settings.threshold = value
        self._save_setting('threshold')
```

### Pattern: Optional Setting

```python
@property
def optional_param(self) -> Optional[str]:
    return self._settings.optional_param

@optional_param.setter
def optional_param(self, value: Optional[str]):
    if value != self._settings.optional_param:
        self._settings.optional_param = value
        self._save_setting('optional_param')
```

## Error Handling

Always provide clear error messages:

```python
@model.setter
def model(self, value: str):
    valid_models = {"model_a", "model_b", "model_c"}
    if value not in valid_models:
        raise ValueError(
            f"Invalid model '{value}'. "
            f"Valid options: {', '.join(sorted(valid_models))}"
        )
    # ... rest of setter
```

Catch and display errors in UI:

```python
def _on_model_changed(self, index: int):
    new_model = self.model_combo.itemData(index)
    try:
        self._settings_manager.model = new_model
        self.set_status_message(f"Model set to {new_model}")
    except ValueError as e:
        # Show error to user
        self.set_status_message(str(e), error=True)
        # Optionally: revert combo box to previous value
        self.refresh()
```

## Testing Checklist

When adding or modifying settings:

- [ ] Settings persist when project is saved and reloaded
- [ ] UI refreshes when settings change from another source
- [ ] Invalid values show clear error messages
- [ ] Settings changes are undoable
- [ ] Settings work from both panel and quick actions
- [ ] Default values work for existing projects (backwards compatibility)

## Red Flags (Anti-Patterns)

 **Multiple Update Pathways**
```python
# BAD: Three different ways to update same setting
block.metadata["model"] = "new_model"  # Direct mutation
facade.execute_block_command(block_id, "set_model", ["new_model"])  # Command
settings_manager.model = "new_model"  # Settings manager

# GOOD: One way only
settings_manager.model = "new_model"
```

 **Hardcoded Mappings**
```python
# BAD: Mapping command names to metadata keys
mappings = {"set_model": "model", "set_device": "device"}

# GOOD: Settings schema defines structure
@dataclass
class Settings(BaseSettings):
    model: str = "default"
    device: str = "auto"
```

 **Flag-Based Refresh Guards**
```python
# BAD: Flag-based guards (unreliable timing)
self._is_saving = True
# ... save ...
self._is_saving = False
# ... refresh checks flag ...

# GOOD: Event-driven refresh
settings_manager.settings_changed.connect(self._on_setting_changed)
```

 **Inconsistent Storage**
```python
# BAD: Some settings nested, some flat
metadata["model"] = "x"  # Flat
metadata["separator_settings"]["device"] = "y"  # Nested

# GOOD: All settings in schema (manager handles storage)
settings_manager.model = "x"
settings_manager.device = "y"
```

## References

- **Full Council Decision:** `AgentAssets/commands/COUNCIL_SETTINGS_SYSTEM.md`
- **Settings Standard:** `AgentAssets/SETTINGS_STANDARD.md`
- **Base Settings:** `src/application/settings/base_settings.py`
- **Example Implementation:** `src/application/settings/app_settings.py` (for app-level settings pattern)

## Quick Reference

**To add a new setting:**
1. Add field to settings schema dataclass (with default)
2. Add property getter/setter to settings manager (with validation)
3. Update UI to use property accessor
4. Test persistence, refresh, undo

**To migrate existing setting:**
1. Add to settings schema (match current metadata key)
2. Create property accessor
3. Update all update pathways to use settings manager
4. Test thoroughly
5. Remove old pathways

**The best part is no part. In this case, the best settings system is the one with a single, clear pathway.**

---

*This preset was created based on the Council Decision on Settings System Fire. Refer to `COUNCIL_SETTINGS_SYSTEM.md` for full reasoning and context.*
