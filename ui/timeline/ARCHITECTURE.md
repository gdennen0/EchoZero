# EchoZero Timeline Prototype — Architecture

## Overview

A standalone visual test harness for exploring timeline UI feel/interaction in EchoZero 2.

**Goal:** Griff runs this, drags events around, zooms/pans, tweaks FEEL.py constants, and evaluates whether the interaction model and visual style feel right.

## Design Principles

1. **Model ≠ View** — Pure Python data model in `model.py` (no Qt imports). Reusable anywhere.
2. **All Feel Constants in One Place** — `FEEL.py` has every visual constant. No magic numbers elsewhere.
3. **Fast Culling** — Spatial indexing with bisect + viewport. Only render visible events.
4. **Simple But Correct** — Prototype-grade code: clear, straightforward, no over-engineering.
5. **Just Enough Interaction** — Click, drag, scroll, zoom. No animation, no real audio, no persistence.

## File Layout

```
C:\Users\griff\EchoZero\ui\timeline\
├── prototype.py          # Entry point: opens window, creates fake data, runs timer
├── canvas.py             # Core canvas: QWidget + paintEvent, rendering pipeline
├── model.py              # Data model: pure Python, no Qt
├── input.py              # Input handling: mouse/keyboard → model updates
├── ruler.py              # Time ruler: ticks, labels, playhead control
├── layers_panel.py       # Layer list: names, colors, selection
├── FEEL.py               # Visual constants (edit these to adjust feel)
├── README.md             # Quick start guide
├── ARCHITECTURE.md       # This file
└── verify.py             # Syntax check + basic model validation
```

## Data Model (model.py)

Pure Python—no Qt. Can be tested, debugged, or reused independently.

### Classes

```python
class TimelineEvent:
    id: str
    time: float           # seconds
    duration: float       # seconds
    layer_id: str
    label: str
    color: Tuple[int, int, int]  # RGB
    classifications: Dict[str, str]  # arbitrary metadata

class TimelineLayer:
    id: str
    name: str
    color: Tuple[int, int, int]
    order: int            # sort order
    collapsed: bool
    height: float         # row height in pixels

class TimelineState:
    events: List[TimelineEvent]
    layers: List[TimelineLayer]
    zoom_level: float           # px/sec
    scroll_x: float             # seconds
    scroll_y: float             # pixels (vertical)
    playhead_time: float
    selection: Set[str]         # selected event ids
    selected_layer_id: Optional[str]
    
    # Sorted index (rebuilt after mutations)
    _sorted_times: List[float]
    _sorted_ids: List[str]

class ViewportRect:
    time_start: float
    time_end: float
    layer_start: int
    layer_end: int

def visible_events(state, viewport) -> List[TimelineEvent]:
    """Fast culling: return events visible in viewport using bisect."""
```

### Model Flow

1. **Load/create** state with events and layers
2. **Call rebuild_index()** after adding/removing events
3. **Query visible_events(state, viewport)** for rendering
4. **Mutate events** (drag, resize, delete)
5. **Call rebuild_index()** again
6. **Repeat**

## Canvas (canvas.py)

QWidget subclass with custom `paintEvent()`. NOT QGraphicsView—we use QPainter for total control.

### Rendering Pipeline (7 passes)

```
1. Grid Background
   - Adaptive major/minor ticks based on zoom
   - Pixel-snapped for clean look

2. Section Regions
   - Colored background bands (alternating) per layer
   - Very low alpha for subtlety

3. Event Backgrounds (batched)
   - Rounded rectangles with fill
   - Use event.color + EVENT_ALPHA

4. Event Borders (batched)
   - Gray for normal, bright yellow for selected
   - EVENT_SELECTED_BORDER_WIDTH = 2px

5. Event Labels (batched)
   - Elided text, only if event wide enough
   - Clipped to visible area

6. Playhead
   - Red vertical line
   - Ruler has triangle indicator at top

7. Snap Indicator (if dragging)
   - Yellow vertical line shows snap target
```

### Coordinate System

- **X axis:** Time in seconds
- **Y axis:** Layer indices (top to bottom)
- **Origin:** Top-left corner

### Viewport Culling

Only render events visible in the current scroll/zoom window:

```python
viewport = _build_viewport()  # from scroll_x, scroll_y, zoom_level
visible = visible_events(state, viewport)  # fast bisect-based lookup
# render only visible events
```

## Input (input.py)

`InputHandler` class processes all mouse/keyboard events and updates state.

### Controls

| Input | Action |
|-------|--------|
| Mouse wheel | Zoom (cursor-anchored: zoom point stays under cursor) |
| Middle drag / Alt+Left drag | Pan (horizontal + vertical) |
| Left click | Select event / deselect on empty |
| Shift+Click | Add to selection |
| Drag selected event | Move (with snap indicator) |
| Drag event edge | Resize (left/right edges extend/shrink + shift time) |
| Escape | Deselect all |
| Delete | Remove selected events |
| Space | Play/pause playhead |

### Snap

When dragging, if event snaps within `SNAP_THRESHOLD_PX`, show snap indicator:

```python
SNAP_GRID_SECONDS = 0.25  # snap to 0.25s grid
SNAP_THRESHOLD_PX = 8     # 8px from target = snap
```

## Ruler & Layers Panel

### TimeRuler (ruler.py)

- Horizontal ruler at top
- Adaptive tick marks (major/minor)
- Time labels (MM:SS.ms format)
- Playhead triangle indicator
- Click to move playhead

### LayersPanel (layers_panel.py)

- Vertical panel on left
- Layer names + color swatches
- Highlight on hover / selection
- Click to select layer
- Scrolls vertically with canvas

## Feel (FEEL.py)

**Every visual constant lives here.** No hardcoded values elsewhere.

### Categories

```python
# Backgrounds & grid
BG_COLOR, GRID_MINOR_COLOR, GRID_MAJOR_COLOR, ...

# Ruler
RULER_HEIGHT, RULER_BG_COLOR, RULER_FONT_SIZE, ...

# Playhead
PLAYHEAD_COLOR, PLAYHEAD_WIDTH, ...

# Events
EVENT_HEIGHT, EVENT_RADIUS, EVENT_ALPHA, EVENT_BORDER_WIDTH, ...

# Selection
SELECTION_RECT_COLOR, SELECTION_RECT_BORDER, ...

# Snap
SNAP_INDICATOR_COLOR, SNAP_THRESHOLD_PX, SNAP_GRID_SECONDS, ...

# Zoom
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP, ...
```

### How to Tweak

1. Edit FEEL.py constants
2. Run prototype.py
3. Observe changes immediately
4. Adjust colors, sizes, thresholds, etc.
5. Repeat until feel is right

## Fake Data Generation (prototype.py)

```python
def generate_fake_state(
    num_events: int = 500,
    num_layers: int = 10,
    total_duration: float = 300.0  # 5 minutes
) -> TimelineState:
```

Creates:
- 10 layers with names, colors, fixed heights
- 500 events with random:
  - Start times (0–300s)
  - Durations (0.2–60s, biased toward short)
  - Layer assignments
  - Colors (from palette)
  - Labels (from pool: "Speech", "Music", "Ambient", etc.)
  - Classifications (arbitrary metadata)

Seed is fixed (42) for reproducibility.

## Performance

### Current (Prototype)

- 500 events + 10 layers
- 60fps target (16ms per frame via QTimer)
- Viewport culling with bisect: ~O(log n) lookup
- Batch rendering in 7 passes

### Bottlenecks (if needed)

- Very large datasets (1000+ events): replace bisect + sorted list with interval tree
- Complex rendering: GPU acceleration via OpenGL context
- Audio waveform: sample at display resolution, cache tiles

For this prototype: **more than sufficient**.

## Testing & Validation

### Pure Python (no Qt)

Run `python verify.py` to check:
- Syntax of all files
- Model creation, mutation, culling
- Basic correctness

### With Qt

Run `python prototype.py` to:
- Open window
- Scroll, zoom, interact
- Verify rendering pipeline
- Check feel/responsiveness

## Future Directions

### Short Term (for prototyping)
- Undo/redo (history in state)
- Keyboard shortcuts (J/K for next/prev, L for playhead)
- Copy/paste events
- Multiple selection modes (rectangle, layer)

### Medium Term (for real app)
- Persistence (save/load to JSON or SQLite)
- Real audio integration (waveform rendering, playback sync)
- Scripting/automation (Python API for event manipulation)
- Multi-track editing (copy events across layers, etc.)

### Long Term
- Collaborative editing (WebSocket sync)
- Plugin system (custom event types, visualization)
- Bidirectional sync with DAW (Ableton, Logic, etc.)

---

**Current Status:** ✓ Syntax verified, ✓ Model logic tested, ✓ Ready to run.

When PyQt5 DLLs are available, `python prototype.py` opens the window.

Until then: `python verify.py` confirms code structure is sound.
