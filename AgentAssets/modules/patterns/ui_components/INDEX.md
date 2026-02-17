# UI Components Module

## Purpose

Provides standards and patterns for implementing UI components in EchoZero's Qt GUI, ensuring consistency and preventing common errors.

## When to Use

- When creating input dialogs for quick actions
- When implementing UI components that need user input
- When working with Qt dialogs and widgets
- When creating block panels
- When ensuring consistency across UI components

## Quick Start

1. Read **QUICK_ACTIONS.md** for input dialog patterns
2. Follow the parameter order for QInputDialog.getDouble()
3. Always read current values from settings manager (single source of truth)
4. Test dialogs immediately after creation

## Contents

- **QUICK_ACTIONS.md** - Reference guide for creating input dialogs in quick actions, including PyQt6 API details and common pitfalls

## Key UI Files

| File | Purpose |
|------|---------|
| `ui/qt_gui/design_system.py` | Colors, Spacing, Typography, Sizes, Effects, global stylesheet |
| `ui/qt_gui/core/base_components.py` | Panel, Section, Button, BaseGraphicsNode base classes |
| `ui/qt_gui/block_panels/block_panel_base.py` | BlockPanelBase for block configuration panels |
| `ui/qt_gui/core/progress_bar.py` | StatusBarProgress for execution progress |
| `ui/qt_gui/core/actions_panel.py` | Actions panel (quick actions UI) |
| `ui/qt_gui/core/properties_panel.py` | Properties panel |
| `ui/qt_gui/block_panels/components/data_filter_widget.py` | Reusable data filter component |
| `ui/qt_gui/block_panels/components/expected_outputs_display.py` | Expected outputs display component |

## Design System

The design system (`ui/qt_gui/design_system.py`) provides:
- `Colors` - Color palette with theme support (`Colors.apply_theme()`)
- `Colors.get_port_color()` / `Colors.get_block_color()` - Semantic colors
- `Spacing` - Consistent spacing constants
- `Typography` - Font definitions
- `Sizes` - Size constants
- `Effects` - Visual effect constants
- `set_sharp_corners()` / `is_sharp_corners()` - Global corner preference
- `border_radius()` - CSS helper
- `get_stylesheet()` - Global stylesheet generator

## Block Panel Pattern

Block panels inherit from `BlockPanelBase` (`ui/qt_gui/block_panels/block_panel_base.py`):

```python
from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase

class MyBlockPanel(BlockPanelBase):
    def __init__(self, block_id, facade, parent=None):
        super().__init__(block_id, facade, parent)
    
    def create_content_widget(self):
        # Build block-specific UI here
        widget = QWidget()
        # ... create controls ...
        return widget
    
    def refresh(self):
        # Reload UI state from block metadata
        pass
```

Key `BlockPanelBase` methods:
- `create_content_widget()` - Override for block-specific UI
- `refresh()` / `refresh_for_undo()` - Refresh UI state
- `set_block_metadata_key(key, value)` - Undoable metadata update
- `set_multiple_metadata(updates_dict)` - Batch undoable updates
- `execute_block_setting(key, value)` - Undoable block configuration
- `create_filter_widget()` - Add data filter component
- `create_expected_outputs_display()` - Add expected outputs display
- `add_port_filter_sections()` - Auto-add filter sections for ports

## Related Modules

- [`modules/patterns/settings_abstraction/`](../settings_abstraction/) - Settings system used by UI components
- [`modules/patterns/block_implementation/`](../block_implementation/) - When creating block UI panels
- [`modules/commands/feature/`](../../commands/feature/) - When UI components are part of feature development

## Core Values Alignment

This module embodies "the best part is no part" by:
- **Single source of truth** - Always read from settings manager, never hardcode
- **Simple patterns** - Clear, consistent dialog creation
- **Explicit parameters** - No magic, clear parameter order
- **Prevent errors** - Document common pitfalls to avoid them
- **Reusable components** - BlockPanelBase, data_filter_widget, expected_outputs_display

UI components should be simple, consistent, and follow clear patterns to prevent errors and maintainability issues.

