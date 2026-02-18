---
name: echozero-project-overview
description: EchoZero project structure, key interfaces, and architecture reference. Use when navigating the codebase, finding where to add code, understanding architecture, or when the user asks about EchoZero structure, layers, or how components connect.
---

# EchoZero Project Overview

## Project Structure

```
EchoZero/
├── src/
│   ├── features/               # Vertical feature modules
│   │   ├── blocks/             # Block entities, editor API, expected outputs
│   │   ├── connections/        # Connection management
│   │   ├── execution/          # Graph execution, progress tracker
│   │   ├── projects/           # Project persistence, snapshots
│   │   ├── setlists/           # Setlist processing, song switching
│   │   ├── show_manager/       # MA3 sync (layer sync, divergence)
│   │   └── ma3/                # GrandMA3 OSC communication
│   ├── application/
│   │   ├── api/                # ApplicationFacade, feature APIs
│   │   ├── blocks/             # Block processors, quick actions
│   │   ├── commands/           # QUndoCommand implementations
│   │   ├── processing/         # BlockProcessor base
│   │   ├── settings/           # App, block, show manager settings
│   │   ├── events/             # Domain events
│   │   └── services/           # Application services
│   ├── shared/
│   │   ├── application/        # EventBus, progress, registry
│   │   ├── domain/             # Entities, repository interfaces
│   │   └── infrastructure/     # Repository implementations
│   └── utils/                  # paths, message
├── ui/qt_gui/                  # PyQt6 interface
│   ├── block_panels/           # Block configuration panels
│   ├── core/                   # Base components, progress bar
│   ├── node_editor/            # Visual graph editor
│   ├── dialogs/                # Dialogs (filters, setlist, tracks)
│   ├── views/                  # Setlist views, action editor
│   └── widgets/               # Timeline, settings, logs
├── tests/                      # unit, integration, application
├── AgentAssets/                # AI agent context (transitioning to .cursor/skills)
└── ma3_plugins/                # GrandMA3 Lua plugins
```

## Key Interfaces

| Interface | Location | Purpose |
|-----------|----------|---------|
| ApplicationFacade | `src/application/api/application_facade.py` | Unified API |
| BlockProcessor | `src/application/processing/block_processor.py` | Block execution |
| EchoZeroCommand | `src/application/commands/base_command.py` | Undoable operations |
| BlockSettingsManager | `src/application/settings/block_settings.py` | Block settings |
| ProgressContext | `src/shared/application/services/` | Hierarchical progress |
| EditorAPI | `src/features/blocks/application/editor_api.py` | Editor layer/event ops |
| SyncSystemManager | `src/features/show_manager/` | MA3 sync orchestration |
| SetlistService | `src/features/setlists/application/` | Setlist coordinator |

## Key Files

- Block registration: `src/application/block_registry.py`
- Processor registration: `src/application/blocks/__init__.py`
- Quick actions: `src/application/blocks/quick_actions.py`
- Block panel base: `ui/qt_gui/block_panels/block_panel_base.py`
- Design system: `ui/qt_gui/design_system.py`

## Scripts

- Quality checks: `AgentAssets/scripts/quality_checks.py`
- Context CLI: `AgentAssets/scripts/context_cli.py`
- Auto-sync: `AgentAssets/scripts/auto_sync.py`

## Documentation

- Architecture: `docs/architecture/ARCHITECTURE.md`
- Show manager sync: `docs/ma3/show_manager_sync_system.md`
- Progress tracking: `docs/progress_tracking.md`
