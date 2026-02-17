# UI Improvement Guide

## Step-by-Step Workflow

### Step 1: Audit the Current State

Before changing anything, answer these questions:

1. **What does the user look at first?** (Ask them -- don't assume)
2. **What context is this displayed in?** (Narrow side panel? Dialog? Node editor?)
3. **How many fields are typically shown?** (Determines if cards vs. flat list)
4. **Is anything editable, or all read-only?**
5. **Is there a custom node editor item, or just the default BlockItem?**

### Step 2: Establish Information Hierarchy

Divide all displayed information into three tiers:

| Tier | Description | Treatment |
|------|-------------|-----------|
| **Hero** | User checks every time (timestamps, classification, confidence) | Top of panel, monospace, full-size, accent color |
| **Interactive** | User interacts with (audio player, knobs, mode selector) | Middle section, compact but functional controls |
| **Debug** | Rarely checked (IDs, display mode, raw metadata) | Bottom, smaller font, dimmer color, de-emphasized |

### Step 3: Apply Layout Patterns

#### For Panels (narrow side-panel context, ~200-300px wide)

```python
# Always wrap labels above fields in narrow panels
layout = QFormLayout(group)
layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
layout.setSpacing(Spacing.XS)

# Set small minimum widths so widgets can compress
spin = QDoubleSpinBox()
spin.setMinimumWidth(60)

# Shorten labels
layout.addRow("Fade:", self.fade_spin)  # NOT "Fade Duration:"
```

#### For Dialogs (resizable, 400px+ wide)

```python
# Reduce default dialog size from oversized to compact
self.setMinimumSize(400, 500)
self.resize(560, 640)

# Use shared group box style helper
def _group_box_style(self) -> str:
    return f"""
        QGroupBox {{
            color: {Colors.TEXT_PRIMARY.name()};
            border: 1px solid {Colors.BORDER.name()};
            border-radius: {border_radius(4)};
            margin-top: 6px;
            padding: 8px 6px 6px 6px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 6px;
            padding: 0 3px;
        }}
    """
```

#### For Inspector Panels (event details, properties)

```python
# Single scroll area for everything -- NO nested scrolls
scroll = QScrollArea()
scroll.setWidgetResizable(True)
scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

# Content widget with cards
content = QWidget()
content_layout = QVBoxLayout(content)
content_layout.setContentsMargins(8, 6, 8, 8)
content_layout.setSpacing(6)

# Card helper for visual grouping
def _make_card(parent_layout):
    card = QFrame()
    card.setStyleSheet(f"""
        QFrame {{
            background-color: {Colors.BG_MEDIUM.name()};
            border: 1px solid {Colors.BORDER.name()};
            border-radius: {border_radius(6)};
        }}
    """)
    inner = QVBoxLayout(card)
    inner.setContentsMargins(10, 8, 10, 8)
    inner.setSpacing(4)
    parent_layout.addWidget(card)
    return inner
```

### Step 4: Node Editor Embedding (if applicable)

When a block has 2+ configurable parameters, consider a custom BlockItem:

```python
# 1. Create widget class (embeds inside the node)
class MyBlockWidget(QWidget):
    def __init__(self, block_id, facade, parent=None):
        super().__init__(parent)
        self.setFixedWidth(Sizes.MY_BLOCK_WIDTH - 12)
        self._build_ui()
        self._load_from_metadata()

# 2. Create BlockItem subclass
class MyBlockItem(BlockItem):
    def _calculate_dimensions(self):
        super()._calculate_dimensions()
        self._width = Sizes.MY_BLOCK_WIDTH
        self._height += Sizes.MY_CONTROL_HEIGHT

# 3. Register in node_scene.py
elif block.type == "MyBlock":
    block_item = MyBlockItem(block, self.facade, undo_stack=self.undo_stack)

# 4. Add size constants to design_system.py
MY_BLOCK_WIDTH = 210
MY_CONTROL_HEIGHT = 260
```

#### RotaryKnob Usage

```python
knob = RotaryKnob(
    label="FREQ",          # 8px bold label above knob
    min_val=20.0,
    max_val=20000.0,
    default=1000.0,
    suffix="Hz",           # shown in value readout
    decimals=1,
    accent_color=Colors.ACCENT_BLUE,  # arc fill color
)
knob.valueChanged.connect(lambda v: self._save_debounced("key", v))
```

### Step 5: Validate

- [ ] Panel renders correctly at minimum width (200px for side panels, 400px for dialogs)
- [ ] Hero info is immediately visible without scrolling
- [ ] Interactive controls are reachable and responsive
- [ ] Debug info is present but not competing for attention
- [ ] Zero linter errors
- [ ] Uses design system tokens throughout (no magic hex colors)
