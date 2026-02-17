# UI Improvement Preset

## Evaluation Template

**Target Component:** [Panel / Dialog / Node Item name]

**Current Problem:**
- [ ] Hard to read (poor hierarchy, everything same visual weight)
- [ ] Too wide for narrow panels (fixed widths, side-by-side labels)
- [ ] Nested scrolling (scroll area inside scroll area)
- [ ] Important info buried (timestamps, classification, confidence hidden in metadata dump)
- [ ] Inconsistent with design system (ad-hoc colors, fonts, spacing)
- [ ] Missing inline controls (node editor item has no knobs/sliders)

**User Priority:**
1. [What does the user look at first?]
2. [What does the user interact with most?]
3. [What is debug/rarely-checked info?]

**Context:**
- Panel width: [Fixed / Narrow side-panel / Dialog / Node embed]
- Typical data density: [Few fields / Many fields / Variable]
- Editable: [Yes / No / Partially]

---

## Improvement Checklist

### Layout Phase
- [ ] Switch QFormLayout to `WrapAllRows` for narrow panels
- [ ] Set `setMinimumWidth(60)` on QDoubleSpinBox / QComboBox
- [ ] Reduce group box margins (SM/XS, not MD/LG)
- [ ] Remove nested QScrollAreas (one scroll container for everything)
- [ ] Shorten labels ("Fade Duration:" -> "Fade:", "Minimum time:" -> "Min:")

### Hierarchy Phase
- [ ] Identify hero info (user looks at first) and place at top
- [ ] Use monospace font for numeric values (timestamps, IDs)
- [ ] Use accent color for primary classification/category
- [ ] De-emphasize debug info (smaller font, dimmer color, bottom of panel)
- [ ] Add section headers (uppercase, 10px, letter-spaced)

### Visual Grouping Phase
- [ ] Use subtle card containers (BG_MEDIUM + thin border) for related info
- [ ] Replace HLine separators with card grouping
- [ ] Consistent label column width (e.g. 64px fixed for hero rows)

### Node Editor Item Phase (if applicable)
- [ ] Determine if block needs custom BlockItem subclass
- [ ] Choose node width (150 default, 210 for knobs, 350 for players)
- [ ] Add RotaryKnob widgets for numeric parameters
- [ ] Add mode selector button (QPushButton + QMenu popup)
- [ ] Wire knob changes to metadata with debounced saves (150ms)
- [ ] Subscribe to BlockUpdated events for external sync

### Completion Phase
- [ ] Zero linter errors
- [ ] Design system tokens used (Colors, Spacing, Typography)
- [ ] Panel works at 200px width (minimum viable narrow)
- [ ] Dialog works at 400px width (minimum viable dialog)
- [ ] Node item registered in node_scene.py

---

## Red Flags (Reconsider)

- Adding more information instead of removing noise
- Creating new color constants instead of using design system
- Fixed pixel widths that break in narrow panels
- Nested scroll areas
- Labels longer than 8 characters in narrow panels
- "While we're at it, let's also add..."

---

## Green Flags (Proceed)

- Removing visual elements that users never check
- Surfacing frequently-used values to the top
- Replacing verbose labels with concise ones
- Switching from side-by-side to stacked layout for narrow contexts
- Adding inline controls that reduce need to open separate panels
