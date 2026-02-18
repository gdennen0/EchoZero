# Settings System

Type-safe settings with persistence, validation, and UI integration.

## Overview

- **Block settings** – Per-block configuration stored in `block.metadata`, undo via command bus. Use `BaseSettings` (schema) + `BlockSettingsManager` (manager) + `@register_block_settings` for auto-discovery.
- **App settings** – Global preferences stored via `PreferencesRepository`. Use `AppSettings` (schema) + `AppSettingsManager`; create with `init_app_settings_manager()` (see bootstrap).

Features: dataclass schemas, validation (`FieldValidator`, `ValidationResult`), debounced save, signals for UI, backwards-compatible loading.

## Architecture

```
settings/
├── base_settings.py       # BaseSettings, BaseSettingsManager, ValidationResult, FieldValidator
├── block_settings.py      # BlockSettingsManager (block.metadata, undo)
├── app_settings.py       # AppSettings, AppSettingsManager, init_app_settings_manager
├── [block]_settings.py   # Per-block schema + manager (e.g. load_audio_settings.py)
└── show_manager_settings.py  # ShowManagerSettingsManager (synced layers, MA3)
```

Block settings are registered with `@register_block_settings(block_type_id, ...)` from `src.shared.application.settings`.

## Creating block settings

1. Define a dataclass schema extending `BaseSettings` (fields with defaults).
2. Decorate with `@register_block_settings("BlockTypeId", description=..., tags=[...])`.
3. Add a manager class extending `BlockSettingsManager` with `SETTINGS_CLASS` and property accessors that call `_save_setting(name)` on change.

```python
from dataclasses import dataclass
from src.application.settings.base_settings import BaseSettings
from src.application.settings.block_settings import BlockSettingsManager
from src.shared.application.settings import register_block_settings

@register_block_settings("LoadAudio", description="Audio file loading", tags=["audio", "input"])
@dataclass
class LoadAudioBlockSettings(BaseSettings):
    audio_path: Optional[str] = None

class LoadAudioSettingsManager(BlockSettingsManager):
    SETTINGS_CLASS = LoadAudioBlockSettings

    @property
    def audio_path(self) -> Optional[str]:
        return self._settings.audio_path

    @audio_path.setter
    def audio_path(self, value: Optional[str]):
        if value != self._settings.audio_path:
            self._settings.audio_path = value
            self._save_setting("audio_path")
```

## Using block settings

Managers are constructed with `(facade, block_id)`. Panels and services typically create or obtain the appropriate manager for the block type (e.g. `LoadAudioSettingsManager(facade, block_id)`). Block metadata is read/written through the manager; changes are persisted to the block and participate in undo when using the facade’s command bus.

## Using app settings

```python
from src.application.settings import AppSettingsManager, init_app_settings_manager

# Typically created in bootstrap and provided via service container
app_settings = AppSettingsManager(preferences_repo)
# or
app_settings = init_app_settings_manager(preferences_repo)

app_settings.theme_preset = "default dark"
preset = app_settings.theme_preset  # Auto-saved, type-safe
```

## References

- **Block settings guide:** `AgentAssets/modules/patterns/settings_abstraction/` (PRESET, STANDARD, UI_GUIDE)
- **App/block settings standard:** `AgentAssets/modules/patterns/settings_abstraction/STANDARD.md`
- **Module API:** `src/application/settings/__init__.py` exports `BaseSettings`, `BaseSettingsManager`, `BlockSettingsManager`, `AppSettings`, `AppSettingsManager`, `init_app_settings_manager`
