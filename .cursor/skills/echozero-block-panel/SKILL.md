---
name: echozero-block-panel
description: Create block panels (UI configuration) in EchoZero. Use when creating block panels, BlockPanelBase subclasses, block configuration UI, or when the user asks about block panels, block UI, or panel refresh.
---

# Block Panel Creation

## Base Class

`BlockPanelBase` in `ui/qt_gui/block_panels/block_panel_base.py`

Inherit for block-specific configuration UI.

## Override Points

```python
class MyBlockPanel(BlockPanelBase):
    def create_content_widget(self) -> QWidget:
        """Return block-specific UI. Required."""
        widget = QWidget()
        # ... add controls ...
        return widget

    def refresh(self):
        """Update UI with current block data. Required."""
        if not self.block:
            return
        # Load from block.metadata or settings manager
        # Update control values
```

## Key Methods

| Method | Purpose |
|--------|---------|
| `create_content_widget()` | Build block UI - return root widget |
| `refresh()` | Sync UI from block state |
| `refresh_for_undo()` | Reload after undo/redo (calls reload_from_storage on settings) |
| `set_block_metadata_key(key, value)` | Undoable single metadata update |
| `set_multiple_metadata(updates)` | Undoable batch metadata update |
| `create_filter_widget()` | Add data filter component |
| `create_expected_outputs_display()` | Add expected outputs display |
| `add_port_filter_sections()` | Auto-generate filter sections for ports |
| `set_status_message(msg, error=False)` | Show status in footer |

## Panel Structure

- Header: block name, type, status dot
- Content: block-specific (create_content_widget)
- Footer: status message, actions

## Settings Integration

Initialize settings manager after parent init:

```python
def __init__(self, block_id, facade, parent=None):
    super().__init__(block_id, facade, parent)
    self._settings_manager = MyBlockSettingsManager(facade, block_id, parent=self)
    self._settings_manager.settings_changed.connect(self._on_setting_changed)
    if self.block:
        self.refresh()
```

## Event Handling

- Subscribe to BlockUpdated for external changes
- Call `reload_from_storage()` when BlockUpdated fires (if settings manager)
- Block signals during refresh: `combo.blockSignals(True)` ... `blockSignals(False)`

## Registration

Panels are registered in `ui/qt_gui/block_panels/__init__.py` - mapping block type to panel class.

## Key Files

- Base: `ui/qt_gui/block_panels/block_panel_base.py`
- Components: `block_panels/components/data_filter_widget.py`, `expected_outputs_display.py`
- Design system: `ui/qt_gui/design_system.py`
