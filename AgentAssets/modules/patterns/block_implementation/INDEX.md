# Block Implementation Module

## Purpose

Provides patterns and guidance for implementing new block types in EchoZero, ensuring consistency and proper integration.

## When to Use

- When creating a new block processor
- When adding a new block type to the system
- When understanding block architecture and lifecycle
- When implementing block cleanup or resource management
- When adding quick actions to blocks
- When integrating progress tracking into block execution

## Quick Start

1. Read **PRESET.md** for the block implementation template (processor + command)
2. Follow **GUIDE.md** for step-by-step implementation (10 steps)
3. Review existing block processors in `src/application/blocks/` for examples
4. Register block in BlockRegistry (`src/application/block_registry.py`)
5. Register quick actions in `src/application/blocks/quick_actions.py`
6. Create UI panel if needed (inherit from `BlockPanelBase`)

## Contents

- **PRESET.md** - Block processor template, command template, and registration patterns
- **GUIDE.md** - Step-by-step block implementation guide (10 steps)

## Key Files

| File | Purpose |
|------|---------|
| `src/application/processing/block_processor.py` | BlockProcessor base class (5-step flow) |
| `src/application/block_registry.py` | Block type registration |
| `src/application/blocks/__init__.py` | Processor auto-registration |
| `src/application/blocks/quick_actions.py` | Quick action registration |
| `src/application/commands/base_command.py` | EchoZeroCommand base class |
| `src/application/settings/block_settings.py` | BlockSettingsManager base class |
| `ui/qt_gui/block_panels/block_panel_base.py` | BlockPanelBase for UI panels |

## Related Modules

- [`modules/patterns/settings_abstraction/`](../settings_abstraction/) - When block needs configurable settings
- [`modules/patterns/progress_tracking/`](../progress_tracking/) - When block has long-running operations
- [`modules/commands/cleanup/`](../../commands/cleanup/) - For resource cleanup patterns
- [`modules/patterns/ui_components/`](../ui_components/) - When creating block UI panels
- [`modules/commands/feature/`](../../commands/feature/) - When block is part of feature development

## Documentation Links

- `docs/architecture/ARCHITECTURE.md` - Overall architecture
- `src/application/processing/README.md` - Processing layer details
- `src/application/blocks/README.md` - Block processors overview
- `src/application/commands/README.md` - Command pattern reference

## Core Values Alignment

This module embodies "the best part is no part" by:
- **Simple contract** - BlockProcessor interface is minimal and clear (5-step flow)
- **Auto-registration** - Call `register_processor_class()`, import in `__init__.py`
- **Explicit cleanup** - cleanup() method is explicit, no hidden lifecycle
- **No magic** - Clear data flow, obvious execution order
- **Quick actions** - Simple decorator-based action registration

Block implementation should be straightforward: implement the interface, register automatically, handle resources explicitly.

