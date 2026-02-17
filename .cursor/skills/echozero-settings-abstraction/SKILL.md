---
name: echozero-settings-abstraction
description: Implement configurable settings in EchoZero blocks using BlockSettingsManager. Use when adding settings to blocks, implementing block configuration, quick action dialogs, or when the user asks about block settings, metadata, or settings persistence.
---

# Settings Abstraction

## Core Principle

**Single Source of Truth:** `block.metadata` is the only storage. All UI (panels, quick actions) read/write through the settings manager.

**Single Pathway:** All changes MUST go through the settings manager.

## Pattern

### 1. Define Schema

```python
from dataclasses import dataclass
from src.application.settings.base_settings import BaseSettings

@dataclass
class MyBlockSettings(BaseSettings):
    model: str = "default"
    threshold: float = 0.5
```

### 2. Create Manager

```python
from src.application.settings.block_settings import BlockSettingsManager

class MyBlockSettingsManager(BlockSettingsManager):
    NAMESPACE = "block.myblock"
    SETTINGS_CLASS = MyBlockSettings

    @property
    def model(self) -> str:
        return self._settings.model

    @model.setter
    def model(self, value: str):
        if value not in {"default", "a", "b"}:
            raise ValueError(f"Invalid model: {value}")
        if value != self._settings.model:
            self._settings.model = value
            self._save_setting('model')
```

### 3. Use in Panel

```python
self._settings_manager = MyBlockSettingsManager(facade, block_id)
self._settings_manager.settings_changed.connect(self._on_setting_changed)

def refresh(self):
    current_model = self._settings_manager.model  # Read from manager
    # ... update UI ...

def _on_model_changed(self, index):
    self._settings_manager.model = new_model  # Write through manager
```

### 4. Use in Quick Actions

```python
if value is not None:
    settings_manager = MyBlockSettingsManager(facade, block_id)
    settings_manager.model = value
    settings_manager.force_save()  # CRITICAL: immediate save for quick actions
    return {"success": True}
# Read path
settings_manager = MyBlockSettingsManager(facade, block_id)
current = settings_manager.model
return {"needs_input": True, "input_type": "choice", "default": current, ...}
```

## Critical Rules

**DO:**
- Always read current value from settings manager before dialogs
- Call `force_save()` in quick actions (bypasses debounce)
- Call `reload_from_storage()` after undo/redo or BlockUpdated event
- Validate in setters; provide defaults for backwards compatibility

**DON'T:**
- Mutate `block.metadata` directly
- Use hardcoded defaults in dialogs
- Multiple update pathways (commands, direct mutation, manager)

## Number Input Parameters (Quick Actions)

Always include: `min`, `max`, `default` (from manager), `decimals`, `increment_jump` (or `step`)

```python
return {
    "needs_input": True,
    "input_type": "number",
    "min": 0.0,
    "max": 1.0,
    "default": current_value,  # From settings manager
    "decimals": 2,
    "increment_jump": 0.05,
    "title": "Set Threshold"
}
```

## Two Tiers

- **BlockSettingsManager:** Stores in `block.metadata`, undo support
- **BaseSettingsManager:** Global/widget settings, PreferencesRepository (SQLite)

## Key Files

- Base: `src/application/settings/base_settings.py`
- Block: `src/application/settings/block_settings.py`
- App example: `src/application/settings/app_settings.py`

## Reference

Full preset: `AgentAssets/modules/patterns/settings_abstraction/PRESET.md`
