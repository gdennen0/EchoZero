---
name: echozero-ui-components
description: Create UI components in EchoZero's Qt GUI. Use when creating block panels, input dialogs, quick action UI, or when the user asks about BlockPanelBase, design system, or Qt dialogs in EchoZero.
---

# UI Components

## Block Panel Pattern

Inherit from `BlockPanelBase` (`ui/qt_gui/block_panels/block_panel_base.py`):

```python
from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase

class MyBlockPanel(BlockPanelBase):
    def create_content_widget(self):
        # Build block-specific UI
        return widget

    def refresh(self):
        # Reload UI from block metadata
        pass
```

## Key BlockPanelBase Methods

- `create_content_widget()` - Override for block UI
- `refresh()` / `refresh_for_undo()` - Reload UI state
- `set_block_metadata_key(key, value)` - Undoable metadata update
- `set_multiple_metadata(updates_dict)` - Batch undoable updates
- `create_filter_widget()` - Data filter component
- `create_expected_outputs_display()` - Output info
- `add_port_filter_sections()` - Auto filter sections for ports

## Design System

`ui/qt_gui/design_system.py`:
- `Colors` - Palette, theme, `Colors.apply_theme()`
- `Colors.get_port_color()`, `Colors.get_block_color()`
- `Spacing`, `Typography`, `Sizes`, `Effects`
- `border_radius()`, `get_stylesheet()`

## Key Files

- Base: `ui/qt_gui/block_panels/block_panel_base.py`
- Components: `block_panels/components/data_filter_widget.py`, `expected_outputs_display.py`
- Design: `ui/qt_gui/design_system.py`
- Progress: `ui/qt_gui/core/progress_bar.py`
- Actions panel: `ui/qt_gui/core/actions_panel.py`

## Quick Actions Input Dialogs

**QInputDialog.getDouble() parameter order (PyQt6):**
```
parent, title, label, value, min, max, decimals, flags, step
```

**Common error:** Omitting `flags` or wrong order causes "argument 8 has unexpected type 'float'"

**Number input return format:**
```python
{
    "needs_input": True,
    "input_type": "number",
    "min": 0.0,
    "max": 1.0,
    "default": current_value,  # From settings manager
    "decimals": 2,
    "increment_jump": 0.05,
    "title": "Dialog Title"
}
```

Always read `default` from settings manager. Always include `min`, `max`, `decimals`, `increment_jump`.

## Single Source of Truth

Always read current values from settings manager before showing dialogs. Never hardcode defaults.

## Reference

- QUICK_ACTIONS: `AgentAssets/modules/patterns/ui_components/QUICK_ACTIONS.md`
- Settings: echozero-settings-abstraction skill
