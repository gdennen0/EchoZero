# EchoZero UI Design System

**Philosophy: Best Part is No Part**

This design system provides a clean, minimal foundation for building UI components. Everything is standardized, reusable, and follows consistent patterns.

---

##  Visual Design

### Colors

All colors are defined in `ui/qt_gui/design_system.py`:

```python
from ui.qt_gui.design_system import Colors

# Usage
background = Colors.BG_DARK
text = Colors.TEXT_PRIMARY
accent = Colors.ACCENT_BLUE
block_color = Colors.get_block_color('LoadAudio')
```

**Color Categories:**
- **Backgrounds**: `BG_DARK`, `BG_MEDIUM`, `BG_LIGHT`
- **UI Elements**: `BORDER`, `HOVER`, `SELECTED`
- **Text**: `TEXT_PRIMARY`, `TEXT_SECONDARY`, `TEXT_DISABLED`
- **Accents**: `ACCENT_BLUE`, `ACCENT_GREEN`, `ACCENT_RED`, `ACCENT_YELLOW`
- **Blocks**: `BLOCK_LOAD`, `BLOCK_ANALYZE`, `BLOCK_TRANSFORM`, etc.
- **Connections**: `CONNECTION_NORMAL`, `CONNECTION_HOVER`, `CONNECTION_SELECTED`
- **Ports**: `PORT_INPUT`, `PORT_OUTPUT`

### Spacing

Consistent spacing scale:

```python
from ui.qt_gui.design_system import Spacing

# XS = 4px, SM = 8px, MD = 16px, LG = 24px, XL = 32px, XXL = 48px
padding = Spacing.MD
margin = Spacing.LG
```

### Typography

Three font styles:

```python
from ui.qt_gui.design_system import Typography

default_font = Typography.default_font()  # Body text
heading_font = Typography.heading_font()  # Headings
mono_font = Typography.mono_font()        # Code/IDs
```

### Sizes

Standard sizes for UI elements:

```python
from ui.qt_gui.design_system import Sizes

# Block dimensions
width = Sizes.BLOCK_WIDTH        # 180px
height = Sizes.BLOCK_HEIGHT      # 80px
corner = Sizes.BLOCK_CORNER_RADIUS  # 6px

# Port dimensions
port_radius = Sizes.PORT_RADIUS  # 6px

# Connections
line_width = Sizes.CONNECTION_WIDTH  # 2px

# Grid
grid_size = Sizes.GRID_SIZE      # 20px
```

---

##  Component Patterns

### Base Components

Use these standardized building blocks for all UI elements:

```python
from ui.qt_gui.core.base_components import Panel, Section, Button, create_vertical_layout

# Panel - container widget
panel = Panel()

# Section - titled group within panel
section = Section("Block Properties")

# Button - standardized button
primary_btn = Button("Execute", variant="primary")
secondary_btn = Button("Cancel", variant="secondary")

# Layouts
v_layout = create_vertical_layout(spacing=Spacing.MD)
h_layout = create_horizontal_layout(spacing=Spacing.SM)
```

### Graphics Items

For node editor components:

```python
from ui.qt_gui.node_editor.block_item import BlockItem
from ui.qt_gui.node_editor.connection_item import ConnectionItem

# Block visualization
block_item = BlockItem(block, facade)

# Connection visualization
connection = ConnectionItem(
    source_block, source_port,
    target_block, target_port,
    connection_data
)
```

---

## ️ Architecture Patterns

### 1. Single Responsibility

Each component does ONE thing well:
- `BlockItem` → visualize a block
- `ConnectionItem` → visualize a connection  
- `PropertiesPanel` → display properties
- `NodeScene` → manage the graph

### 2. Facade Pattern

All backend interaction goes through `ApplicationFacade`:

```python
#  Good
result = self.facade.describe_block(block_id)
if result.success:
    block = result.data.get('block')
    # Use block data

#  Bad - Don't access services directly
block = self.facade.block_service.get_block(...)
```

### 3. Event-Driven Updates

Use the event bus for UI updates:

```python
# Subscribe to events
self.facade.event_bus.subscribe('project.loaded', self._on_project_loaded)

# Emit events happen automatically in facade
# UI just reacts to them
```

### 4. Signals for Internal Communication

Use Qt signals for component-to-component communication:

```python
class MyWidget(QWidget):
    item_selected = pyqtSignal(str)  # Emit when something happens
    
    def _on_click(self):
        self.item_selected.emit(self.item_id)

# Connect in parent
widget.item_selected.connect(self._handle_selection)
```

---

##  File Organization

```
ui/
├── qt_gui/
│   ├── design_system.py        # ⭐ All visual constants
│   ├── qt_application.py       # App initialization
│   ├── main_window.py          # Main window
│   │
│   ├── core/                   # Reusable components
│   │   ├── base_components.py  # Base widgets & patterns
│   │   └── properties_panel.py # Properties display
│   │
│   └── node_editor/            # Node editor specific
│       ├── node_editor_widget.py   # Main editor widget
│       ├── node_scene.py           # Scene management
│       ├── node_graphics_view.py   # View (zoom, pan)
│       ├── block_item.py           # Block visualization
│       └── connection_item.py      # Connection visualization
│
├── DESIGN_SYSTEM.md            # This file
└── README.md                   # UI overview
```

---

##  Creating New Components

### Step 1: Use Design System

Always import from design system:

```python
from ui.qt_gui.design_system import Colors, Spacing, Sizes, Typography
```

### Step 2: Extend Base Components

Start from base components when possible:

```python
from ui.qt_gui.core.base_components import Panel, BaseGraphicsNode

class MyCustomPanel(Panel):
    def __init__(self):
        super().__init__()
        # Custom logic
```

### Step 3: Follow Patterns

- Use signals for events
- Keep components focused (single responsibility)
- Use facade for all backend calls
- Apply consistent styling

### Step 4: Keep It Simple

**Before adding code, ask:**
- Is this absolutely necessary?
- Can existing components handle this?
- Am I duplicating functionality?

**Best part is no part.**

---

## ️ Standardized Item Deletion Pattern

All visual elements should follow this deletion pattern:

### Three Ways to Delete

1. **Toolbar Button** - "Remove [Item]" button shows selection dialog
2. **Keyboard Shortcut** - Delete/Backspace on selected items
3. **Context Menu** - Right-click → Delete (future)

### Implementation Pattern

```python
from ui.qt_gui.core.base_components import ItemSelectionDialog

# 1. Toolbar button handler
def _on_remove_item(self):
    """Show dialog to select and remove items"""
    items_result = self.facade.list_items()  # Get items from backend
    if not items_result.success or not items_result.data:
        return
    
    selected_items = ItemSelectionDialog.show_delete_dialog(
        parent=self,
        title="Remove Items",
        items=items_result.data,
        item_name_getter=lambda item: f"{item.name} ({item.type})"
    )
    
    if selected_items:
        for item in selected_items:
            self.facade.remove_item(item.id)
        self.refresh()

# 2. Scene deletion (keyboard shortcut)
def delete_selected_items(self):
    """Delete currently selected items"""
    selected = self.selectedItems()
    # Filter to your item type
    # Confirm with user
    # Call facade.remove_item()
    # Refresh

# 3. Graphics view keyboard handler
def keyPressEvent(self, event):
    """Handle keyboard shortcuts"""
    if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
        if self.scene():
            self.scene().delete_selected_items()
        event.accept()
```

### Usage Example

**Block deletion is fully implemented:**
-  "Remove Block" button in toolbar
-  Delete/Backspace on selected blocks
-  Confirmation dialog
-  Multi-select support (Ctrl/Cmd+Click)

**For new visual element types:**
Just follow the same pattern with your item type!

---

##  Styling Guidelines

### Use Inline Styles Sparingly

Prefer design system constants over magic numbers:

```python
#  Good
padding = Spacing.MD
color = Colors.TEXT_PRIMARY

#  Bad
padding = 16
color = QColor(240, 240, 245)
```

### Apply Global Stylesheet

The main window applies a global stylesheet from `get_stylesheet()`. This handles:
- Menu bars
- Toolbars
- Dock widgets
- Status bars
- Basic widgets

Only override when you need component-specific styling.

---

##  Refactor Checklist

When refactoring or adding new UI:

- [ ] Uses design system constants (no magic numbers)
- [ ] Extends base components where applicable
- [ ] Follows single responsibility principle
- [ ] Uses facade for backend calls
- [ ] Uses signals for communication
- [ ] Has clear, focused purpose
- [ ] Removes unnecessary code
- [ ] Documented if non-obvious

---

##  Future Enhancements

As the UI grows, consider:

1. **Theme Switching** - Light/dark mode toggle
2. **Component Library** - More reusable widgets
3. **Animation System** - Smooth transitions
4. **Keyboard Shortcuts** - Comprehensive hotkey system
5. **Layout Persistence** - Save window states

**But only add what's needed. Best part is no part.**

---

##  Key Files

| File | Purpose |
|------|---------|
| `design_system.py` | All visual constants and global styles |
| `base_components.py` | Reusable UI building blocks |
| `block_item.py` | Block visualization |
| `connection_item.py` | Connection visualization |
| `main_window.py` | Main application window |

---

**Remember: Simplicity, consistency, and focus. Every line of code should earn its place.**

