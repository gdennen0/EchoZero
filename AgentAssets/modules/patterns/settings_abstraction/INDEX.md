# Settings Abstraction Module

## Purpose

Provides the standardized pattern for implementing configurable settings in EchoZero blocks, widgets, and application components.

## When to Use

- When adding settings to a block that need to persist
- When implementing UI settings for widgets
- When creating settings that need validation
- When settings are accessed from multiple UI locations (panels, quick actions)
- When settings should be undoable

## Quick Start

1. Read **PRESET.md** for the complete pattern template
2. Define your settings schema (dataclass inheriting from `BaseSettings`)
3. Create a settings manager:
   - For blocks: inherit from `BlockSettingsManager` (`src/application/settings/block_settings.py`)
   - For global/widget: inherit from `BaseSettingsManager` (`src/application/settings/base_settings.py`)
4. Use the manager in UI components (panels, quick actions)

## Contents

- **PRESET.md** - Complete pattern template with code examples for settings schema, manager, and UI integration
- **STANDARD.md** - Settings architecture, categories, standards, and file locations
- **UI_GUIDE.md** - Specific guidance for UI/widget settings implementation

## Key Files

| File | Purpose |
|------|---------|
| `src/application/settings/base_settings.py` | BaseSettings dataclass + BaseSettingsManager (global/widget settings) |
| `src/application/settings/block_settings.py` | BlockSettingsManager (stores in block.metadata, undo support) |
| `src/application/settings/app_settings.py` | AppSettings + AppSettingsManager (global app settings) |
| `src/application/settings/show_manager_settings.py` | ShowManagerSettings + ShowManagerSettingsManager |
| `src/application/settings/__init__.py` | Public exports |

## Two Settings Tiers

1. **BaseSettingsManager** - For global/widget settings
   - Stores in PreferencesRepository (SQLite)
   - Auto-save with debouncing
   - Signal emission for UI updates
   - Used by: AppSettingsManager, timeline settings

2. **BlockSettingsManager** - For per-block settings
   - Stores in `block.metadata` (not preferences)
   - Undo support via `facade.command_bus`
   - Signal emission for UI updates
   - Used by: ShowManagerSettingsManager, block panel settings

## Related Modules

- [`modules/patterns/block_implementation/`](../block_implementation/) - When adding settings to new blocks
- [`modules/patterns/ui_components/`](../ui_components/) - When settings affect UI components
- [`modules/commands/feature/`](../../commands/feature/) - When settings are part of feature development

## Core Values Alignment

This module embodies "the best part is no part" by:
- **Single source of truth** - `block.metadata` (blocks) or preferences table (global) is the only storage location
- **Single pathway** - All settings changes go through the manager, preventing inconsistencies
- **Simple contract** - Clear dataclass schema, explicit property accessors
- **No magic** - Explicit save calls, clear validation, obvious data flow

The settings abstraction eliminates the complexity of multiple update pathways and inconsistent storage, providing a simple, unified approach.

