# UI Improvement Patterns

Proven patterns extracted from real EchoZero UI improvements.

---

## Pattern 1: Compact Form Layout (Narrow Panels)

**Problem:** QFormLayout with side-by-side label + widget requires ~250px+ width.

**Solution:** Wrap labels above fields so the widget gets full width.

```python
layout = QFormLayout(group)
layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
layout.setSpacing(Spacing.XS)

spin = QDoubleSpinBox()
spin.setMinimumWidth(60)  # Allow compression to 60px
layout.addRow("Fade:", spin)  # Short label
```

**Applied in:** `audio_negate_panel.py`, `event_filter_dialog.py`

---

## Pattern 2: Single Scroll Container (Inspector Panels)

**Problem:** Nested scroll areas (scroll inside scroll) create confusing interaction.

**Solution:** One QScrollArea wrapping all content. Individual sections use cards, not sub-scrolls.

```python
# Root layout
root = QVBoxLayout(self)
root.setContentsMargins(0, 0, 0, 0)

# Fixed header (outside scroll)
header = QWidget()
header.setFixedHeight(28)
root.addWidget(header)

# Single scroll for everything
scroll = QScrollArea()
scroll.setWidgetResizable(True)
scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

content = QWidget()
content_layout = QVBoxLayout(content)
content_layout.setContentsMargins(8, 6, 8, 8)
content_layout.setSpacing(6)
scroll.setWidget(content)
root.addWidget(scroll, 1)
```

**Applied in:** `inspector.py` (Event Inspector)

---

## Pattern 3: Card Grouping (Visual Sections)

**Problem:** HLine separators are weak visual cues; flat lists lack hierarchy.

**Solution:** Subtle card containers with BG_MEDIUM background and thin border.

```python
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

def _add_section_header(layout, text):
    lbl = QLabel(text.upper())
    font = QFont("SF Pro Text, Segoe UI, sans-serif")
    font.setPixelSize(10)
    font.setBold(True)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
    lbl.setFont(font)
    lbl.setStyleSheet(f"color: {Colors.TEXT_DISABLED.name()};")
    layout.addWidget(lbl)
```

**Applied in:** `inspector.py` (Hero, Player, Details cards)

---

## Pattern 4: Hero Row (Primary Information)

**Problem:** All values look the same -- user can't quickly scan for what matters.

**Solution:** Fixed-width label column (64px) + monospace value. Important values get accent color.

```python
def _add_hero_row(grid, row, label, value):
    lbl = QLabel(label)
    lbl.setFont(label_font(10))
    lbl.setStyleSheet(f"color: {Colors.TEXT_DISABLED.name()};")
    lbl.setFixedWidth(64)

    val = QLabel(value)
    val.setFont(mono_font(12))
    val.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()};")

    grid.addWidget(lbl, row, 0)
    grid.addWidget(val, row, 1)
    return row + 1
```

**Applied in:** `inspector.py` (timestamps, classification, confidence)

---

## Pattern 5: Compact Waveform (Audio Preview)

**Problem:** Waveform at 120-150px dominates the inspector panel.

**Solution:** 48-64px waveform with 2px padding. Still interactive (click-to-seek works).

```python
waveform = WaveformWidget()
waveform.setMinimumHeight(48)
waveform.setMaximumHeight(64)
```

**Applied in:** `inspector.py`, `waveform_widget.py`

---

## Pattern 6: Embedded Node Knobs (Node Editor)

**Problem:** Block has 3+ parameters but user must open a panel to adjust them.

**Solution:** Custom BlockItem with embedded RotaryKnob widgets.

Key dimensions:
- **Default node:** 150px wide
- **Knob node:** 210px wide (fits label + knob per row)
- **Player node:** 350px wide (fits waveform + transport)
- **Knob row height:** 76px (generous touch target)

Knob row layout:
```python
def _make_knob_row(label_text, knob):
    row = QWidget()
    row.setFixedHeight(76)
    h = QHBoxLayout(row)
    h.setContentsMargins(4, 0, 4, 0)

    lbl = QLabel(label_text)
    lbl.setFixedWidth(52)
    h.addWidget(lbl)
    h.addStretch()
    h.addWidget(knob)
    return row
```

Metadata persistence:
```python
def _save_metadata_debounced(self, key, value):
    if self._save_timer:
        self._save_timer.stop()
        self._save_timer.deleteLater()
    self._save_timer = QTimer(self)
    self._save_timer.setSingleShot(True)
    self._save_timer.timeout.connect(lambda: self._save_metadata(key, value))
    self._save_timer.start(150)  # 150ms debounce
```

**Applied in:** `audio_negate_block_item.py`, `audio_filter_block_item.py`

---

## Pattern 7: Shared Group Box Style (DRY Dialogs)

**Problem:** Each group box repeats the same 10-line stylesheet.

**Solution:** Extract a helper method.

```python
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

**Applied in:** `event_filter_dialog.py`

---

## Design System Quick Reference

| Token | Value | Use |
|-------|-------|-----|
| `Colors.BG_DARK` | #1c1c20 | Panel backgrounds |
| `Colors.BG_MEDIUM` | #2a2a2f | Card backgrounds |
| `Colors.BG_LIGHT` | #38383e | Input backgrounds, hover |
| `Colors.BORDER` | #4b4b50 | Card borders, separators |
| `Colors.TEXT_PRIMARY` | #f0f0f5 | Values, important text |
| `Colors.TEXT_SECONDARY` | #b4b4b9 | Labels, descriptions |
| `Colors.TEXT_DISABLED` | #78787d | Section headers, dim labels |
| `Colors.ACCENT_BLUE` | #4682dc | Primary accent, links, layer names |
| `Spacing.XS` | 4px | Tight row spacing |
| `Spacing.SM` | 8px | Standard group margins |
| `Spacing.MD` | 16px | Section spacing |
| `border_radius(N)` | Npx or 0px | Respects global sharp corners setting |
