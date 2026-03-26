# EchoZero 2 — Timeline Editor: Incremental Build Tasks

**Document version:** 1.0  
**Date:** 2026-03-17  
**Architecture reference:** [PANEL.md](../../.openclaw/workspace/sass/seminars/echozero-timeline-feasibility/PANEL.md)

---

## Overview

This document defines 10 incremental agent tasks to build EchoZero 2's timeline editor from scratch. Each task is designed to be handed to an AI coding agent with clear input/output and reviewed by Griff before the next task begins.

### Architectural Principles (Mandatory — Every Agent Reads This)

1. **Three-layer architecture — non-negotiable:**
   - `data/` — Pure Python. No Qt imports. Unit-testable. (`TimelineModel`, `EventModel`, `SpatialIndex`, `SnapCalculator`)
   - `layout/` — Pure Python. No Qt imports. Computes screen rects from model state. (`LayoutEngine`, `CoordTransform`)
   - `ui/` — Qt only. Reads pre-computed layout. Never computes positions. (`TimelineCanvas`, renderers, input handlers)

2. **FEEL.py is the only place for tunable constants.** No magic numbers anywhere else. All pixel values, durations, colors, easing curves → imported from `FEEL.py`.

3. **Agents never edit `FEEL.py` defaults.** They create new entries with documented names and sensible starting values, then Griff tunes them manually.

4. **All rendering is in batched passes.** Paint order: grid → event backgrounds → waveforms → event foregrounds → labels → selection → playhead. Never draw one event completely before moving to the next.

5. **QPainter on QWidget. No QGraphicsView anywhere.**

### EZ1 Reference Location

```
C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\
```

The EZ1 codebase has real, working logic for: grid intervals, snap calculation, coordinate transforms, waveform display, event drag state machines. Feed these to agents as reference — not as code to copy, but as logic to re-express in the new architecture.

---

## Task 1: Canvas + Adaptive Grid

### 1.1 Description

Build the base `TimelineCanvas` widget: a `QWidget` that renders via `QPainter`. Implements the adaptive time ruler with major/minor tick marks, background grid lines, and the coordinate transform between screen pixels and timeline seconds.

This is the foundation everything else builds on. It must be correct, fast, and fully exercised before any events are added.

### 1.2 Input

Provide the agent with:
- This document (all 10 tasks for architectural context)
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\timing\grid_calculator.py` — EZ1 grid interval math
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\timing\grid_renderer.py` — EZ1 grid rendering reference
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\constants.py` — EZ1 constants (reference only, do not copy — port to FEEL.py)
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\timing\time_converter.py` — time format utilities

### 1.3 Output

Create these files in `C:\Users\griff\EchoZero\ui\timeline_v2\`:

```
ui/timeline_v2/
├── FEEL.py                        # Tunable constants (agent creates stubs, Griff fills)
├── data/
│   └── coord_transform.py         # TimeToScreen / ScreenToTime (pure Python)
├── layout/
│   └── grid_layout.py             # GridLayout: computes tick positions (pure Python)
├── ui/
│   ├── canvas.py                  # TimelineCanvas(QWidget) — main widget
│   └── renderers/
│       ├── __init__.py
│       └── grid_renderer.py       # GridRenderer: draws ticks + grid lines
└── tests/
    └── test_coord_transform.py    # Unit tests for coord math
```

**`FEEL.py` entries for this task (agent creates these):**

```python
# Grid
RULER_HEIGHT_PX = 30
GRID_MIN_MINOR_SPACING_PX = 12      # Skip minor ticks below this density
GRID_MIN_MAJOR_SPACING_PX = 80      # Skip major ticks below this density
GRID_MAX_LINES = 500                 # Hard cap on lines drawn per frame
GRID_MINOR_LINE_COLOR = "#1E2A38"
GRID_MAJOR_LINE_COLOR = "#2A3848"
GRID_RULER_BG_COLOR = "#141E28"
GRID_RULER_TEXT_COLOR = "#8899AA"
GRID_RULER_FONT_SIZE_PX = 11

# Zoom defaults
ZOOM_DEFAULT_PPS = 100.0            # pixels per second at default zoom
ZOOM_MIN_PPS = 10.0
ZOOM_MAX_PPS = 2000.0
```

### 1.4 Acceptance Criteria

**Visual:**
- [ ] Ruler is a horizontal strip at top of widget, correct height, correct background color
- [ ] Major tick marks visible with time labels (seconds or timecode depending on zoom)
- [ ] Minor tick marks visible between major ticks, no visual crowding
- [ ] Tick density adapts: zoom in → more ticks; zoom out → fewer. No ticks ever overlap.
- [ ] Grid lines extend vertically through the track area, aligned with ruler ticks
- [ ] Background is dark, distinct from ruler strip

**Functional:**
- [ ] `coord_transform.time_to_screen(t, pps, scroll_x)` and `screen_to_time(x, pps, scroll_x)` return correct values
- [ ] Round-trip test: `screen_to_time(time_to_screen(t))` == `t` within floating point tolerance
- [ ] Widget can be resized; grid redraws correctly at new size
- [ ] Changing `pps` (pixels per second) causes grid to update immediately on next paint

**Performance:**
- [ ] At max zoom (2000 pps), no more than `GRID_MAX_LINES` lines drawn
- [ ] `paintEvent` completes in < 2ms measured with `QElapsedTimer`

**Tests:**
- [ ] All `test_coord_transform.py` tests pass
- [ ] Tests cover: zero scroll, non-zero scroll, edge cases (t=0, large t), sub-pixel positions

### 1.5 Agent Instructions

> You are building the base canvas widget for EchoZero 2's timeline editor.
>
> **Architecture rules (mandatory):**
> 1. `data/coord_transform.py` — pure Python, zero Qt imports. Exports `CoordTransform` class with `time_to_screen(t, pps, scroll_x) -> float` and `screen_to_time(x, pps, scroll_x) -> float`. Both accept and return floats. Sub-pixel precision — never round.
> 2. `layout/grid_layout.py` — pure Python, zero Qt imports. `GridLayout` takes `(pps, scroll_x, viewport_width, ruler_height)` and returns a list of `GridTick` named tuples: `(x_screen: float, label: str | None, is_major: bool)`. Apply `GRID_MIN_MINOR_SPACING_PX` and `GRID_MIN_MAJOR_SPACING_PX` density culling here, not in the renderer.
> 3. `ui/canvas.py` — `TimelineCanvas(QWidget)`. In `paintEvent`, instantiate a `QPainter`, call `GridLayout.compute(...)` once to get all tick data, pass the result to `GridRenderer.draw(painter, ticks, ...)`. The canvas does not compute grid positions itself.
> 4. `ui/renderers/grid_renderer.py` — `GridRenderer`. Receives pre-computed `GridTick` list. Draws grid lines first (all in one pass with a single QPainter pen), then ruler background, then tick marks, then labels. NO state switches inside the line drawing loop.
>
> **Grid interval logic:** Port EZ1's `GridCalculator.get_intervals()` logic into `GridLayout`. The key algorithm: start with a base interval (e.g. 1.0 second), multiply/divide by 2 or 5 until the pixel spacing is within `[GRID_MIN_MAJOR_SPACING_PX, GRID_MIN_MAJOR_SPACING_PX * 5]`. Minor ticks are 1/4 of major interval. Reference: `timing/grid_calculator.py` in EZ1.
>
> **Time label formatting:** At < 1s intervals, show milliseconds ("0.5s", "100ms"). At >= 1s intervals, show whole seconds ("4s", "1:30"). At >= 60s, show minutes:seconds ("1:30"). Reference: `timing/time_converter.py` in EZ1.
>
> **All feel constants** go in `FEEL.py`. Import them at the top of each file that uses them. No inline numbers.
>
> **Anti-aliasing:** Grid lines should be pixel-snapped (`round(x) + 0.5`) for crisp rendering. Do NOT enable `Antialiasing` for grid lines. Enable `TextAntialiasing` only.
>
> **Write `tests/test_coord_transform.py`** using pytest. Test round-trips, edge cases, and that `time_to_screen` increases monotonically with t.

### 1.6 FEEL.py Parameters Used

`RULER_HEIGHT_PX`, `GRID_MIN_MINOR_SPACING_PX`, `GRID_MIN_MAJOR_SPACING_PX`, `GRID_MAX_LINES`, `GRID_MINOR_LINE_COLOR`, `GRID_MAJOR_LINE_COLOR`, `GRID_RULER_BG_COLOR`, `GRID_RULER_TEXT_COLOR`, `GRID_RULER_FONT_SIZE_PX`, `ZOOM_DEFAULT_PPS`, `ZOOM_MIN_PPS`, `ZOOM_MAX_PPS`

### 1.7 Dependencies

None. This is the foundation.

### 1.8 Estimated Agent Time

45–75 minutes (straightforward math + Qt widget boilerplate)

### 1.9 Risk Areas

- **Pixel snapping on grid lines:** Agent may forget to snap grid line X to integer + 0.5. Result is blurry 2px-wide lines. Check: zoom in on grid lines in screenshot — each should be exactly 1px crisp.
- **Sub-pixel coord transform:** Agent may use `int()` instead of preserving float precision. Check: round-trip test will catch this.
- **Qt import in data layer:** Agent may accidentally `from PyQt6.QtCore import QRectF` in `coord_transform.py`. Enforce: run `grep -r "PyQt" data/` — should return nothing.
- **Adaptive grid density:** Agent may implement a fixed tick count instead of the proper adaptive algorithm. Check: zoom smoothly from min to max — tick count should change continuously, not at random jumps.
- **Label collision:** At certain zoom levels, labels may overlap. `GridLayout` should cull major-tick labels when `x_spacing < GRID_MIN_MAJOR_SPACING_PX * 2`.

### 1.10 EZ1 Reference Files

- `timing/grid_calculator.py` — interval math (primary reference)
- `timing/grid_renderer.py` — rendering pattern (secondary reference)
- `timing/time_converter.py` — label formatting
- `constants.py` — zoom range, default values to port to FEEL.py

---

## Task 2: Event Rendering + Spatial Culling

### 2.1 Description

Add event data models and rendering. Events are colored rectangles with text labels, organized into layers (tracks). An interval tree provides O(log n) viewport queries. Rendering uses batched passes: all backgrounds, then all waveform placeholder areas, then all labels.

### 2.2 Input

Provide the agent with:
- Output from Task 1 (full codebase)
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\types.py` — EZ1 event/layer types
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\events\items.py` — EZ1 event item reference (skim top 200 lines for data model; ignore QGraphicsItem parts)
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\events\layer_manager.py` — layer organization

### 2.3 Output

New/modified files in `ui/timeline_v2/`:

```
data/
├── models.py          # EventModel, LayerModel, TimelineModel (pure Python dataclasses)
└── spatial_index.py   # SpatialIndex (interval tree for viewport queries)
layout/
└── event_layout.py    # EventLayout: computes EventRect list from model + viewport
ui/renderers/
└── event_renderer.py  # EventRenderer: draws backgrounds, borders, labels
```

**`FEEL.py` entries to add:**

```python
# Track / Layer
TRACK_HEIGHT_PX = 48
TRACK_SPACING_PX = 2
TRACK_LABEL_WIDTH_PX = 0            # 0 = not yet implemented, Task 9 adds this

# Events
EVENT_HEIGHT_PX = 40                # Height of event rect (< track height)
EVENT_CORNER_RADIUS_PX = 4
EVENT_BORDER_WIDTH_PX = 1
EVENT_LABEL_FONT_SIZE_PX = 11
EVENT_LABEL_PADDING_PX = 6          # Left padding for label text
EVENT_LABEL_MIN_WIDTH_PX = 30       # Don't draw label if event narrower than this
EVENT_DEFAULT_COLOR = "#3A6EA5"
EVENT_SELECTED_COLOR = "#0066FF"
EVENT_SELECTED_BORDER_COLOR = "#66AAFF"
EVENT_BORDER_COLOR = "#2A5080"
EVENT_TEXT_COLOR = "#DDEEFF"
EVENT_TEXT_SELECTED_COLOR = "#FFFFFF"
```

### 2.4 Acceptance Criteria

**Visual:**
- [ ] Events render as rounded rectangles with correct color and border
- [ ] Labels visible when event is wide enough (`EVENT_LABEL_MIN_WIDTH_PX`); absent when too narrow
- [ ] Selected events render with `EVENT_SELECTED_COLOR` fill and `EVENT_SELECTED_BORDER_COLOR` border
- [ ] Multiple layers stack vertically with `TRACK_SPACING_PX` gap between them
- [ ] Events outside the viewport are not rendered (confirm via log or profiling, not visual)

**Functional:**
- [ ] `SpatialIndex.query(start, end)` returns correct events for any time range
- [ ] `SpatialIndex` handles: empty state, single event, 5000 events, overlapping events
- [ ] `EventLayout.compute(model, viewport)` returns a list of `EventRect(event, x, y, w, h)` — only visible events
- [ ] Events with negative x (partially off left edge) render correctly (clipped)
- [ ] Events wider than viewport (partially off right edge) render correctly

**Performance:**
- [ ] With 5,000 events in model, `SpatialIndex.query()` completes in < 0.5ms (Python timeit)
- [ ] With 200 visible events, full `paintEvent` completes in < 5ms

**Tests:**
- [ ] `test_spatial_index.py`: correctness at 1, 100, 5000 events; overlapping; touching edges; empty
- [ ] `test_event_layout.py`: correct screen rects for various pps/scroll values

### 2.5 Agent Instructions

> **Data layer (`data/models.py`):**
>
> ```python
> @dataclass
> class EventModel:
>     id: str
>     layer_id: str
>     start_time: float       # seconds
>     duration: float         # seconds
>     label: str
>     color: str | None = None   # If None, use layer color or EVENT_DEFAULT_COLOR
>     metadata: dict = field(default_factory=dict)
>
> @dataclass
> class LayerModel:
>     id: str
>     name: str
>     color: str
>     is_visible: bool = True
>     is_collapsed: bool = False
>     y_offset: float = 0.0   # computed by layout engine, not stored manually
>
> @dataclass
> class TimelineModel:
>     layers: list[LayerModel]
>     events: list[EventModel]
>     duration: float = 0.0
>     selection: set[str] = field(default_factory=set)  # set of event IDs
> ```
>
> **Spatial index (`data/spatial_index.py`):**
> Implement a simple interval tree or augmented BST. A sorted list with `bisect` is acceptable for v1 if query is O(log n + k) where k = results. The public API must be:
> ```python
> class SpatialIndex:
>     def build(self, events: list[EventModel]) -> None: ...
>     def query(self, start: float, end: float) -> list[EventModel]: ...
>     def invalidate(self) -> None: ...   # forces rebuild on next query
> ```
> Do not use any Qt types here.
>
> **Layout engine (`layout/event_layout.py`):**
> ```python
> @dataclass
> class EventRect:
>     event: EventModel
>     x: float       # screen x (may be negative for partially visible)
>     y: float       # screen y
>     w: float       # screen width (may extend beyond viewport)
>     h: float       # screen height
>     is_selected: bool
>
> class EventLayout:
>     def compute(
>         self,
>         model: TimelineModel,
>         pps: float,
>         scroll_x: float,
>         scroll_y: float,
>         viewport_width: float,
>         viewport_height: float,
>         coord: CoordTransform,
>         spatial_index: SpatialIndex
>     ) -> list[EventRect]: ...
> ```
> Y position for each event: `layer.y_offset + (TRACK_HEIGHT_PX - EVENT_HEIGHT_PX) / 2`. Y offsets for layers are computed top-to-bottom in layer order: `y_offset[i] = sum(TRACK_HEIGHT_PX + TRACK_SPACING_PX for layer in layers[:i])`. Collapsed layers use a reduced height (defined in FEEL.py as `TRACK_COLLAPSED_HEIGHT_PX = 8`).
>
> **Renderer (`ui/renderers/event_renderer.py`):**
> Three methods, called in this order from `TimelineCanvas.paintEvent`:
> ```python
> class EventRenderer:
>     def draw_backgrounds(self, painter: QPainter, rects: list[EventRect]) -> None: ...
>     def draw_labels(self, painter: QPainter, rects: list[EventRect]) -> None: ...
> ```
> In `draw_backgrounds`: set pen+brush once, loop drawing all rounded rects. Check if selected for color, but don't change pen per-rect — selected rects go in a second sub-pass with selected colors. So the full order is: normal backgrounds → selected backgrounds → labels.
>
> Use `painter.drawRoundedRect(QRectF(x, y, w, h), EVENT_CORNER_RADIUS_PX, EVENT_CORNER_RADIUS_PX)`. Clip labels: `painter.setClipRect(QRectF(x + EVENT_LABEL_PADDING_PX, y, w - EVENT_LABEL_PADDING_PX, h))`.

### 2.6 FEEL.py Parameters Used

`TRACK_HEIGHT_PX`, `TRACK_SPACING_PX`, `TRACK_COLLAPSED_HEIGHT_PX`, `TRACK_LABEL_WIDTH_PX`, `EVENT_HEIGHT_PX`, `EVENT_CORNER_RADIUS_PX`, `EVENT_BORDER_WIDTH_PX`, `EVENT_LABEL_FONT_SIZE_PX`, `EVENT_LABEL_PADDING_PX`, `EVENT_LABEL_MIN_WIDTH_PX`, `EVENT_DEFAULT_COLOR`, `EVENT_SELECTED_COLOR`, `EVENT_SELECTED_BORDER_COLOR`, `EVENT_BORDER_COLOR`, `EVENT_TEXT_COLOR`, `EVENT_TEXT_SELECTED_COLOR`

### 2.7 Dependencies

Task 1 complete.

### 2.8 Estimated Agent Time

60–90 minutes (interval tree + batched renderer are the complexity)

### 2.9 Risk Areas

- **Qt in data layer:** Agent may use `QRectF` in `EventRect`. Must be plain floats. Enforce: `grep -r "PyQt\|Qt" data/` → empty.
- **Spatial index correctness:** Bisect-based implementations often have off-by-one errors on interval boundaries. Watch for: events that start exactly at viewport edge being excluded. Test with events at `start == viewport_start` and `end == viewport_end`.
- **Batching order:** Agent may draw each event's background+label before moving to next. This causes many pen/brush switches. Check that agent's renderer is structured with separate loop passes.
- **Y position math:** Agent may put events at wrong Y within their track. Verify visually that event is vertically centered in track area.
- **Label clipping:** Without `setClipRect`, labels bleed into adjacent events. Check on narrow events.

### 2.10 EZ1 Reference Files

- `types.py` — data model types to understand (port the dataclass structure, not the Qt items)
- `events/items.py` — first 200 lines for event model fields; ignore all `QGraphicsItem` code
- `events/layer_manager.py` — layer ordering and Y offset calculation logic

---

## Task 3: Scroll + Zoom

### 3.1 Description

Implement horizontal and vertical scroll, cursor-anchored zoom (pinch + mouse wheel), momentum scrolling via trackpad `pixelDelta`, and zoom preset keyboard shortcuts. The viewport state (`pps`, `scroll_x`, `scroll_y`) lives in the model and triggers a repaint when changed.

### 3.2 Input

- Full codebase from Tasks 1–2
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\constants.py` — zoom sensitivity constants
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\core\view.py` — EZ1 scroll/zoom implementation (reference the `wheelEvent` and zoom-anchor math)

### 3.3 Output

```
data/
└── viewport_state.py      # ViewportState dataclass (pps, scroll_x, scroll_y)
ui/
├── input/
│   ├── __init__.py
│   ├── scroll_controller.py   # Handles wheel + trackpad scroll + inertia
│   └── zoom_controller.py     # Handles wheel+modifier zoom, pinch, presets
└── canvas.py                  # Modified: wires up new controllers
```

**`FEEL.py` entries to add:**

```python
# Scroll inertia (momentum scrolling)
SCROLL_INERTIA_DECAY = 0.88          # Per-frame velocity multiplier (lower = faster stop)
SCROLL_INERTIA_TIMER_MS = 16         # Inertia update interval (~60fps)
SCROLL_MIN_VELOCITY_PX = 0.8         # Stop inertia below this px/frame
SCROLL_PIXEL_SENSITIVITY = 1.0       # Multiplier for trackpad pixelDelta scroll
SCROLL_ANGLE_SENSITIVITY = 0.5       # Multiplier for mouse angleDelta scroll (notch)

# Zoom
ZOOM_PIXEL_SENSITIVITY = 0.003       # Ctrl+trackpad zoom sensitivity
ZOOM_ANGLE_SENSITIVITY = 0.12        # Ctrl+wheel zoom per notch (fraction of pps)
ZOOM_ACCUMULATOR_THRESHOLD = 0.005   # Min accumulated delta before applying zoom
ZOOM_ANCHOR_TO_CURSOR = True         # If False, anchor to viewport center
```

### 3.4 Acceptance Criteria

**Visual:**
- [ ] Mouse wheel scrolls horizontally (content moves left/right)
- [ ] Shift+wheel scrolls vertically (tracks move up/down)
- [ ] Ctrl+wheel zooms; the time position under cursor stays fixed during zoom
- [ ] Trackpad two-finger swipe scrolls smoothly with momentum (content coasts after finger lifts)
- [ ] Trackpad pinch-to-zoom works (macOS) — anchored to pinch center point
- [ ] Momentum animation looks natural: fast coast, smooth deceleration to stop

**Functional:**
- [ ] `ViewportState.pps` is clamped to `[ZOOM_MIN_PPS, ZOOM_MAX_PPS]`
- [ ] `ViewportState.scroll_x` is clamped to `[0, max_scroll_x]` where `max_scroll_x = model.duration * pps - viewport_width`
- [ ] After zooming, `screen_to_time(cursor_x_before_zoom, pps_after, scroll_x_after)` == the same time as before zoom
- [ ] Inertia timer is stopped when `velocity < SCROLL_MIN_VELOCITY_PX`
- [ ] Inertia is killed immediately on new touch/click

**Performance:**
- [ ] Repaint triggered only once per scroll event (not multiple times)
- [ ] Inertia timer fires QTimer at `SCROLL_INERTIA_TIMER_MS`, calls `update()` — not `repaint()`

### 3.5 Agent Instructions

> **Cursor-anchored zoom formula (critical — implement exactly):**
> ```python
> def zoom_anchored(viewport: ViewportState, cursor_x: float, zoom_factor: float) -> ViewportState:
>     # The time under the cursor must not change after zoom
>     time_at_cursor = (cursor_x + viewport.scroll_x) / viewport.pps
>     new_pps = clamp(viewport.pps * zoom_factor, ZOOM_MIN_PPS, ZOOM_MAX_PPS)
>     new_scroll_x = time_at_cursor * new_pps - cursor_x
>     return ViewportState(pps=new_pps, scroll_x=max(0, new_scroll_x), scroll_y=viewport.scroll_y)
> ```
> This is the exact 3-line formula. Do not modify it.
>
> **Trackpad vs mouse wheel handling (critical):**
> ```python
> def wheelEvent(self, event: QWheelEvent):
>     pixel_delta = event.pixelDelta()
>     angle_delta = event.angleDelta()
>
>     if not pixel_delta.isNull():
>         # Trackpad — high resolution, use pixelDelta
>         dx = pixel_delta.x() * SCROLL_PIXEL_SENSITIVITY
>         dy = pixel_delta.y() * SCROLL_PIXEL_SENSITIVITY
>         self._scroll_controller.add_velocity(dx, dy)
>     else:
>         # Mouse wheel — use angleDelta (unit: 1/8 degree, standard notch = 120)
>         notches = angle_delta.y() / 120.0
>         dy = notches * TRACK_HEIGHT_PX * SCROLL_ANGLE_SENSITIVITY
>         self._scroll_controller.scroll_by(0, dy)  # No inertia for mouse wheel
>
>     event.accept()
> ```
> If `Ctrl` modifier is held, route to `ZoomController.on_wheel(event)` instead of scroll.
>
> **ScrollController inertia:** Use a `QTimer` (not a thread). On `add_velocity(dx, dy)`, set `self._vx += dx, self._vy += dy`. Timer fires every `SCROLL_INERTIA_TIMER_MS` ms and calls `_tick()`:
> ```python
> def _tick(self):
>     self._vx *= SCROLL_INERTIA_DECAY
>     self._vy *= SCROLL_INERTIA_DECAY
>     if abs(self._vx) < SCROLL_MIN_VELOCITY_PX and abs(self._vy) < SCROLL_MIN_VELOCITY_PX:
>         self._timer.stop()
>         return
>     self.viewport.scroll_x = clamp(self.viewport.scroll_x - self._vx, 0, self._max_scroll_x())
>     self.viewport.scroll_y = clamp(self.viewport.scroll_y - self._vy, 0, self._max_scroll_y())
>     self.canvas.update()  # schedule repaint, do NOT call repaint() directly
> ```
>
> **Zoom presets (keyboard shortcuts — implement in ZoomController):**
> - `Ctrl+0` → reset to `ZOOM_DEFAULT_PPS`, center scroll
> - `Ctrl+=` or `Ctrl++` → zoom in by `ZOOM_ANGLE_SENSITIVITY * 3`
> - `Ctrl+-` → zoom out by `ZOOM_ANGLE_SENSITIVITY * 3`
> - `Ctrl+Shift+=` → zoom to fit all events (pps = viewport_width / model.duration)
>
> **`ViewportState` is a plain Python dataclass. No Qt.** The canvas reads it in `paintEvent`.

### 3.6 FEEL.py Parameters Used

`SCROLL_INERTIA_DECAY`, `SCROLL_INERTIA_TIMER_MS`, `SCROLL_MIN_VELOCITY_PX`, `SCROLL_PIXEL_SENSITIVITY`, `SCROLL_ANGLE_SENSITIVITY`, `ZOOM_PIXEL_SENSITIVITY`, `ZOOM_ANGLE_SENSITIVITY`, `ZOOM_ACCUMULATOR_THRESHOLD`, `ZOOM_ANCHOR_TO_CURSOR`, `ZOOM_MIN_PPS`, `ZOOM_MAX_PPS`, `ZOOM_DEFAULT_PPS`, `TRACK_HEIGHT_PX`

### 3.7 Dependencies

Tasks 1–2 complete.

### 3.8 Estimated Agent Time

60–90 minutes (inertia timer + zoom formula are the tricky parts)

### 3.9 Risk Areas

- **Zoom anchor drift:** The most common bug. Agent implements zoom without the anchor formula, resulting in zoom that "drifts" to the right. Test: zoom in 10 times at cursor position X; the content under the cursor should not have moved. Verify manually.
- **pixelDelta ignored:** Agent uses only `angleDelta`. On macOS trackpad, scrolling will feel like a 2010 app. Test on macOS with two-finger swipe — should be glass-smooth, not notchy.
- **Inertia never stops:** Missing the `< SCROLL_MIN_VELOCITY_PX` stop condition. Timer runs forever. Check: after scroll, content should come to a clean stop within ~0.5 seconds.
- **`repaint()` vs `update()`:** Agent calls `repaint()` in the inertia timer, which forces a synchronous paint in the timer callback. Must be `update()` to schedule a deferred paint.
- **Scroll clamping:** Agent may allow negative `scroll_x` (events fly off left edge). Clamp at `[0, max_scroll_x]` always.

### 3.10 EZ1 Reference Files

- `core/view.py` — zoom and scroll implementation (lines 1–200 are most relevant)
- `constants.py` — `PIXEL_ZOOM_SENSITIVITY`, `ANGLE_ZOOM_SENSITIVITY`, `ZOOM_ACCUMULATOR_THRESHOLD`

---

## Task 4: Selection System

### 4.1 Description

Implement click-select, shift-click multi-select, rubber-band lasso (drag on empty space), and cross-layer selection. Selection state persists in `TimelineModel.selection`. Render: selected events use the selected color scheme (already defined). Lasso renders as a semi-transparent rect during drag.

### 4.2 Input

- Full codebase from Tasks 1–3
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\events\items.py` — EZ1 selection logic (search for "selected" in file)
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\types.py` — `DragState` enum

### 4.3 Output

```
ui/input/
└── selection_controller.py    # Handles all selection mouse logic
ui/renderers/
└── selection_renderer.py      # Draws lasso rect + selection highlight overlay
data/
└── hit_test.py                # HitTester: which event/layer is under a screen point?
```

**`FEEL.py` entries to add:**

```python
# Selection
SELECTION_LASSO_FILL_COLOR = "#0066FF"
SELECTION_LASSO_FILL_ALPHA = 30           # 0-255
SELECTION_LASSO_BORDER_COLOR = "#4499FF"
SELECTION_LASSO_BORDER_WIDTH_PX = 1
SELECTION_CLICK_RADIUS_PX = 4             # Tolerance for click-hit on events
SELECTION_DRAG_THRESHOLD_PX = 4          # Min drag distance before lasso starts
```

### 4.4 Acceptance Criteria

**Visual:**
- [ ] Click on an event selects it (highlighted with selected colors)
- [ ] Click on empty space deselects all
- [ ] Shift+click adds/removes event from selection
- [ ] Drag on empty space shows lasso rect (semi-transparent blue fill, 1px border)
- [ ] Events fully or partially inside lasso are selected on release
- [ ] Lasso works across multiple layers simultaneously

**Functional:**
- [ ] `TimelineModel.selection` is a `set[str]` of event IDs; always the source of truth
- [ ] `HitTester.event_at(x, y, model, layout)` returns the topmost event under a screen point, or `None`
- [ ] `HitTester.layer_at(y, model)` returns the layer under a Y coordinate
- [ ] Lasso selection replaces current selection (no shift = replace); Shift+lasso adds to selection
- [ ] `EventLayout` re-renders with `is_selected=True` for events in `model.selection` — no extra state

**Edge cases:**
- [ ] Click on overlapping events selects the topmost (by layer order, top layer wins)
- [ ] Zero-size lasso (click without drag) does NOT create a selection — falls through to click logic
- [ ] Selection persists through scroll and zoom (it's in the model, not the view)

### 4.5 Agent Instructions

> **State machine in `SelectionController`:**
> ```python
> class SelectionState(Enum):
>     IDLE = "idle"
>     PENDING = "pending"        # Mouse down, not yet moved SELECTION_DRAG_THRESHOLD_PX
>     LASSO = "lasso"            # Dragging lasso
>
> class SelectionController:
>     def on_mouse_press(self, event: QMouseEvent, model, layout): ...
>     def on_mouse_move(self, event: QMouseEvent, model, layout): ...
>     def on_mouse_release(self, event: QMouseEvent, model, layout): ...
>     def get_lasso_rect(self) -> tuple[float,float,float,float] | None: ...
>     # Returns (x, y, w, h) in screen coords, or None if not in LASSO state
> ```
>
> **`on_mouse_press` logic:**
> 1. Call `HitTester.event_at(x, y)`. If event found: enter `PENDING` state, record hit event.
> 2. If no event: enter `PENDING` state, record press position for potential lasso start.
> 3. Do NOT modify selection yet — wait for release or drag threshold.
>
> **`on_mouse_move` logic:**
> 1. If in `PENDING` and distance > `SELECTION_DRAG_THRESHOLD_PX` and no event was hit: transition to `LASSO`.
> 2. If in `PENDING` and distance > threshold and event was hit: transition to drag (handled by `DragController` — emit `drag_started` signal).
> 3. If in `LASSO`: update lasso rect, call `canvas.update()`.
>
> **`on_mouse_release` logic:**
> 1. If in `PENDING` (no drag occurred): commit click-select (hit event or deselect all).
> 2. If in `LASSO`: compute which events intersect lasso rect using `SpatialIndex.query(lasso_start_time, lasso_end_time)`, then filter by Y. Update `model.selection`.
> 3. Return to `IDLE`.
>
> **Shift behavior:** If `Qt.KeyboardModifier.ShiftModifier` is held on release, add to selection instead of replacing.
>
> **`HitTester` is pure Python (no Qt).** Takes screen coordinates + the current `EventRect` list from layout. Returns `EventModel | None`. For point-in-rect: add `SELECTION_CLICK_RADIUS_PX` tolerance on all sides.
>
> **`SelectionRenderer.draw_lasso(painter, lasso_rect)`:** Use `QRectF`, set brush to `QColor(SELECTION_LASSO_FILL_COLOR)` with alpha `SELECTION_LASSO_FILL_ALPHA`, set pen to `SELECTION_LASSO_BORDER_COLOR` with width `SELECTION_LASSO_BORDER_WIDTH_PX`. Draw after events, before playhead.

### 4.6 FEEL.py Parameters Used

`SELECTION_LASSO_FILL_COLOR`, `SELECTION_LASSO_FILL_ALPHA`, `SELECTION_LASSO_BORDER_COLOR`, `SELECTION_LASSO_BORDER_WIDTH_PX`, `SELECTION_CLICK_RADIUS_PX`, `SELECTION_DRAG_THRESHOLD_PX`

### 4.7 Dependencies

Tasks 1–3 complete.

### 4.8 Estimated Agent Time

45–60 minutes

### 4.9 Risk Areas

- **State machine collapse:** Agent implements selection directly in `mousePressEvent` without a state machine, making drag-vs-click ambiguous. The `PENDING` state is essential — don't let agent skip it.
- **Selection in view, not model:** Agent stores `is_selected` on the `EventRect` (view) instead of in `TimelineModel.selection`. After a repaint, selection is lost. Enforce: `model.selection` is the only selection state.
- **Lasso Y coordinate confusion:** Agent lasso may not account for `scroll_y`. Lasso start/end must be converted to model coordinates before comparing to event positions.
- **Overlapping event hit priority:** Agent may return any event under cursor, not the topmost. Must iterate layers in display order (topmost layer last in paint order = highest priority in hit test).

### 4.10 EZ1 Reference Files

- `events/items.py` — selection flag and visual state (lines 1–150)
- `types.py` — `DragState` enum for state machine reference

---

## Task 5: Drag + Snap

### 5.1 Description

Move selected events by dragging. Snap magnetism: when a dragged event's start/end is within `SNAP_MAGNETISM_RADIUS_PX` of a snap target, it pulls to the target with visual indicator. Show ghost position during drag (event at current cursor position, semi-transparent). Show snap target indicator (vertical line at snap point). Snap commits on release. Supports: snap to grid, snap to other event onsets, snap to markers.

### 5.2 Input

- Full codebase from Tasks 1–4
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\timing\snap_calculator.py` — EZ1 snap logic (reuse algorithm, not implementation)
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\events\movement_controller.py` — EZ1 drag state machine (full reference)

### 5.3 Output

```
data/
└── snap_calculator.py         # SnapCalculator: finds nearest snap target (pure Python)
ui/input/
└── drag_controller.py         # DragController: manages drag state machine
ui/renderers/
└── drag_renderer.py           # DragRenderer: ghost events + snap indicator line
```

**`FEEL.py` entries to add:**

```python
# Snap
SNAP_ENABLED_DEFAULT = True
SNAP_MAGNETISM_RADIUS_PX = 12        # Screen pixels — snap activates within this radius
SNAP_TO_GRID = True
SNAP_TO_EVENT_ONSETS = True
SNAP_TO_MARKERS = True

# Drag visual
DRAG_GHOST_ALPHA = 120                # 0-255, opacity of ghost event rect
DRAG_GHOST_BORDER_COLOR = "#66AAFF"
SNAP_INDICATOR_COLOR = "#FFCC00"      # Vertical line color at snap target
SNAP_INDICATOR_WIDTH_PX = 1
SNAP_INDICATOR_HEIGHT = 1.0           # 1.0 = full track area height
DRAG_THRESHOLD_PX = 4                 # Min pixels before drag begins (from SelectionController)
```

### 5.4 Acceptance Criteria

**Visual:**
- [ ] Dragging an event: original position fades (ghost at drag position), moves with cursor
- [ ] When near a snap target: yellow vertical line appears at snap point
- [ ] Ghost position shows current drag position (cursor-relative), NOT the snapped position
- [ ] On release: event commits to snapped position (or cursor position if no snap)
- [ ] Multi-event drag: all selected events move together, maintaining relative offsets
- [ ] Dragging multiple events: snap is computed from the *primary* (first clicked) event only

**Functional:**
- [ ] `SnapCalculator.find_nearest(time, candidates, radius_seconds) -> float | None`
- [ ] Snap candidates include: grid lines at current zoom, all event start/end times (excluding dragged events), marker positions
- [ ] `DragController` is handed off from `SelectionController` when drag threshold exceeded
- [ ] `DragController` emits `drag_committed(deltas: dict[str, float])` on release — a dict of `{event_id: time_delta}` for undo stack integration
- [ ] Pressing `Escape` during drag cancels drag, returns events to original positions

**Edge cases:**
- [ ] Drag must not move events to negative time (clamp start_time >= 0)
- [ ] Drag on a non-selected event first selects it (replacing selection), then drags it

### 5.5 Agent Instructions

> **`SnapCalculator` (pure Python, `data/snap_calculator.py`):**
> ```python
> class SnapCalculator:
>     def find_nearest(
>         self,
>         time: float,
>         candidates: list[float],  # All snap target times in seconds
>         radius_seconds: float     # Computed from SNAP_MAGNETISM_RADIUS_PX / pps
>     ) -> float | None:
>         """Returns nearest candidate within radius, or None if none within radius."""
>         nearest = None
>         nearest_dist = radius_seconds
>         for t in candidates:
>             d = abs(t - time)
>             if d < nearest_dist:
>                 nearest = t
>                 nearest_dist = d
>         return nearest
>
>     def gather_candidates(
>         self,
>         model: TimelineModel,
>         grid_layout: GridLayout,
>         excluded_ids: set[str]     # Don't snap to the events being dragged
>     ) -> list[float]:
>         """Collect all snap times: grid ticks + event edges + markers."""
>         candidates = []
>         if SNAP_TO_GRID:
>             candidates += [tick.time for tick in grid_layout.get_snap_ticks()]
>         if SNAP_TO_EVENT_ONSETS:
>             for event in model.events:
>                 if event.id not in excluded_ids:
>                     candidates.append(event.start_time)
>                     candidates.append(event.start_time + event.duration)
>         if SNAP_TO_MARKERS:
>             candidates += [m.time for m in model.markers]
>         return candidates
> ```
> The radius conversion: `radius_seconds = SNAP_MAGNETISM_RADIUS_PX / viewport.pps`.
>
> **`DragController` state machine:**
> ```
> IDLE → (drag_started signal from SelectionController) → DRAGGING → (mouse_release) → IDLE
>                                                               ↑
>                                                     (mouse_move updates preview)
> ```
> During `DRAGGING`:
> - Compute `time_delta = (current_x - press_x) / pps` on every mouse_move
> - Apply snap: compute preview start = `original_start + time_delta`, find nearest snap, adjust delta
> - Emit `preview_updated(snap_target: float | None, ghost_rects: list[EventRect])` so renderer knows where to draw
>
> On `mouse_release`: emit `drag_committed({id: final_time_delta for id in selected})`. Do NOT mutate the model — the undo stack (future task) handles model mutation.
>
> On `Escape`: emit `drag_cancelled()`. Canvas resets to pre-drag state.
>
> **`DragRenderer` (`ui/renderers/drag_renderer.py`):**
> ```python
> class DragRenderer:
>     def draw(self, painter, ghost_rects, snap_target_x: float | None): ...
> ```
> Ghost: draw all `ghost_rects` as semi-transparent event rects. Use `DRAG_GHOST_ALPHA` for the fill alpha. Snap indicator: if `snap_target_x` is not None, draw a vertical line at that x from top of track area to bottom.

### 5.6 FEEL.py Parameters Used

`SNAP_ENABLED_DEFAULT`, `SNAP_MAGNETISM_RADIUS_PX`, `SNAP_TO_GRID`, `SNAP_TO_EVENT_ONSETS`, `SNAP_TO_MARKERS`, `DRAG_GHOST_ALPHA`, `DRAG_GHOST_BORDER_COLOR`, `SNAP_INDICATOR_COLOR`, `SNAP_INDICATOR_WIDTH_PX`, `DRAG_THRESHOLD_PX`

### 5.7 Dependencies

Tasks 1–4 complete.

### 5.8 Estimated Agent Time

75–105 minutes (state machine + snap math + ghost rendering)

### 5.9 Risk Areas

- **Ghost shows snapped position, not cursor position:** Agent snaps the ghost to the snap target. The ghost must track the cursor exactly. The snap indicator shows where it WILL snap; the ghost shows where it IS. These are different.
- **Snap candidates include dragged events:** Agent includes the dragged event's start in candidates, causing it to "snap to itself." Must pass `excluded_ids` to `gather_candidates`.
- **Multi-event drag offset:** Agent moves all events to the same time instead of maintaining relative offsets. Each event needs its own `time_delta` (same delta for all, applied to each event's original position).
- **Model mutated during drag preview:** Agent writes `event.start_time = preview_time` during drag. Model must not change until `drag_committed` is emitted. Preview is view-only state.
- **No Escape cancel:** Agents frequently forget the escape handler. Check that pressing Escape mid-drag returns events to original visual positions.

### 5.10 EZ1 Reference Files

- `timing/snap_calculator.py` — exact snap algorithm to port (clean up, remove Qt coupling)
- `events/movement_controller.py` — full drag state machine (adapt state names, keep logic)

---

## Task 6: Event Resize

### 6.1 Description

Resize events by dragging their left or right edges. Edge hit zones are `RESIZE_HANDLE_WIDTH_PX` wide. Cursor changes to `SizeHorCursor` when hovering over an edge. Minimum event duration is enforced. Proportional resize option (Shift+drag): maintains start/end ratio relative to event center.

### 6.2 Input

- Full codebase from Tasks 1–5
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\events\items.py` — search for "resize" in file for EZ1 resize logic
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\constants.py` — `RESIZE_HANDLE_WIDTH`, `MIN_RESIZE_HANDLE_WIDTH`

### 6.3 Output

```
ui/input/
└── resize_controller.py    # ResizeController: edge hit detection + resize drag
ui/renderers/
└── resize_renderer.py      # ResizeRenderer: hover handles visualization
```

Modify `hit_test.py` to add `edge_at(x, y, rects) -> tuple[EventModel, 'left'|'right'] | None`.

**`FEEL.py` entries to add:**

```python
# Resize
RESIZE_HANDLE_WIDTH_PX = 6           # Hit zone width on each edge (matches EZ1)
RESIZE_HANDLE_MIN_WIDTH_PX = 3       # Minimum handle width for narrow events
RESIZE_HANDLE_PERCENT = 0.15         # Handle as % of event width (for small events)
RESIZE_MIN_DURATION_S = 0.05         # Minimum event duration (50ms)
RESIZE_HOVER_COLOR = "#FFFFFF"       # Handle highlight color on hover
RESIZE_HOVER_ALPHA = 60              # 0-255
RESIZE_SNAP_ENABLED = True           # Snap also applies to resize
```

### 6.4 Acceptance Criteria

**Visual:**
- [ ] Hovering over the left or right edge of an event shows a `SizeHorCursor`
- [ ] A subtle highlight (lighter fill or border) appears at the hovered edge
- [ ] During resize: event visually updates in real-time as drag proceeds
- [ ] The non-resized edge stays fixed; only the dragged edge moves

**Functional:**
- [ ] `HitTester.edge_at(x, y)` returns the correct edge or `None`; handle width uses `max(RESIZE_HANDLE_WIDTH_PX, event_width * RESIZE_HANDLE_PERCENT, RESIZE_HANDLE_MIN_WIDTH_PX)`
- [ ] Resize respects snap (same `SnapCalculator` as Task 5)
- [ ] Duration never falls below `RESIZE_MIN_DURATION_S`
- [ ] Left-edge drag: `start_time` changes, `end_time` stays fixed
- [ ] Right-edge drag: `end_time` changes (duration changes), `start_time` stays fixed
- [ ] Shift+drag proportional resize: both edges move, center stays fixed

**Cursor:**
- [ ] `mouseMoveEvent` checks edge_at on every move (when not dragging) — cursor updates immediately
- [ ] Cursor reverts to arrow when not over an edge

### 6.5 Agent Instructions

> **Edge priority in hit test:** If cursor is within `RESIZE_HANDLE_WIDTH_PX` of an event's left or right edge, return that edge. Left edge check: `abs(x - event_rect.x) <= handle_width`. Right edge check: `abs(x - (event_rect.x + event_rect.w)) <= handle_width`. For events narrower than `2 * RESIZE_HANDLE_MIN_WIDTH_PX`, split the event in half: left half = left handle, right half = right handle.
>
> **`ResizeController` state machine:**
> ```
> IDLE → (edge_at returns result + mouse_press) → RESIZING → (mouse_release) → IDLE
> ```
> In `RESIZING`:
> - Track `press_x`, `original_start`, `original_duration`, `edge` ('left' or 'right')
> - On move: `time_delta = (current_x - press_x) / pps`
> - If `edge == 'right'`: `new_duration = max(RESIZE_MIN_DURATION_S, original_duration + time_delta)`. Apply snap to `original_start + new_duration`.
> - If `edge == 'left'`: `new_start = min(original_start + time_delta, original_start + original_duration - RESIZE_MIN_DURATION_S)`. Apply snap to `new_start`.
> - Emit `resize_preview(event_id, new_start, new_duration)` — canvas re-renders
>
> On release: emit `resize_committed(event_id, new_start, new_duration)` for undo stack.
>
> **Proportional resize (Shift held):** Both edges move. `scale = (original_duration + time_delta) / original_duration`. New duration = `original_duration * scale`. New start = `event_center - (new_duration / 2)`. Use original center.
>
> **Cursor management:** In `TimelineCanvas.mouseMoveEvent`, after checking `DragController` (returns if currently dragging), check `ResizeController.get_hover_edge(x, y)`. If edge found: `self.setCursor(Qt.CursorShape.SizeHorCursor)`. Else: `self.setCursor(Qt.CursorShape.ArrowCursor)`.
>
> **The canvas re-routes mouse events:** When `ResizeController` is active (RESIZING), it consumes all `mouseMoveEvent` and `mouseReleaseEvent` calls. When `DragController` is active, it consumes them. Only one controller is active at a time. Route in `TimelineCanvas.mouseMoveEvent`:
> ```python
> if self._resize_ctrl.is_active(): self._resize_ctrl.on_move(e); return
> if self._drag_ctrl.is_active():   self._drag_ctrl.on_move(e);   return
> # hover cursor check
> ```

### 6.6 FEEL.py Parameters Used

`RESIZE_HANDLE_WIDTH_PX`, `RESIZE_HANDLE_MIN_WIDTH_PX`, `RESIZE_HANDLE_PERCENT`, `RESIZE_MIN_DURATION_S`, `RESIZE_HOVER_COLOR`, `RESIZE_HOVER_ALPHA`, `RESIZE_SNAP_ENABLED`, `SNAP_MAGNETISM_RADIUS_PX`

### 6.7 Dependencies

Tasks 1–5 complete.

### 6.8 Estimated Agent Time

45–60 minutes

### 6.9 Risk Areas

- **Edge detection ignores scroll:** Agent computes `event_rect.x` without accounting for `scroll_x`. The `EventRect.x` must already be in screen coordinates (which it should be if layout engine is correct). Verify: drag-scroll right, then try to resize — edge handles must still track correctly.
- **Left-edge resize moves both edges:** Agent adjusts `start_time` but recomputes duration as `duration + delta` instead of `end_time - new_start_time`. End time must be fixed during left-edge resize.
- **Proportional resize not gated on Shift:** Agent always does proportional resize. Must check `Qt.KeyboardModifier.ShiftModifier`.
- **Cursor not cleared on leave:** Agent sets `SizeHorCursor` but never resets to `ArrowCursor` when moving off the edge. Result: resize cursor sticks everywhere. Ensure cursor reset on every `mouseMoveEvent` call.

### 6.10 EZ1 Reference Files

- `events/items.py` — search for "resize", "RESIZE_HANDLE", "handle_width" (the hit zone math)
- `constants.py` — `RESIZE_HANDLE_WIDTH`, `MIN_RESIZE_HANDLE_WIDTH`, `RESIZE_HANDLE_PERCENT`

---

## Task 7: Waveform Rendering

### 7.1 Description

Three-level LOD (Level of Detail) waveform system:
- **Sample level** (> `WAVEFORM_LOD_SAMPLE_PX_THRESHOLD` px/sample): draw individual sample values
- **Envelope level** (> `WAVEFORM_LOD_ENVELOPE_PX_THRESHOLD` px/sample): draw peak+RMS envelope as filled polygon
- **Overview level** (≤ threshold): draw pre-computed thumbnail as a single QImage blit

Pre-compute envelopes at load time. Cache rendered QImage tiles. Progressive rendering: show overview immediately, upgrade to envelope/samples when tile is ready. Smooth LOD transition: cross-fade between levels.

### 7.2 Input

- Full codebase from Tasks 1–6
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\events\waveform_widget.py` — EZ1 waveform display (data structure reference)
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\events\waveform_simple.py` — simple waveform draw approach

### 7.3 Output

```
data/
├── waveform_data.py          # WaveformData: samples, pre-computed envelope, overview QImage
└── waveform_cache.py         # WaveformCache: LRU cache of rendered QImage tiles
ui/renderers/
└── waveform_renderer.py      # WaveformRenderer: LOD selection + drawing
```

Modify `models.py` to add `waveform: WaveformData | None` field to `EventModel`.

**`FEEL.py` entries to add:**

```python
# Waveform LOD thresholds (px per sample)
WAVEFORM_LOD_SAMPLE_PX_THRESHOLD = 4.0      # Above this: draw individual samples
WAVEFORM_LOD_ENVELOPE_PX_THRESHOLD = 0.5    # Above this: draw envelope; below: overview

# Waveform visuals
WAVEFORM_PEAK_COLOR = "#4A8FC4"
WAVEFORM_RMS_COLOR = "#2A6FA0"
WAVEFORM_SAMPLE_COLOR = "#66AADD"
WAVEFORM_OVERVIEW_COLOR = "#2A5A80"
WAVEFORM_ALPHA = 200                         # 0-255 base alpha
WAVEFORM_LOD_CROSSFADE_MS = 150              # Transition duration between LOD levels

# Cache
WAVEFORM_CACHE_MAX_TILES = 128               # Max cached QImage tiles (LRU eviction)
WAVEFORM_TILE_WIDTH_PX = 256                 # Width of each cache tile in pixels
```

### 7.4 Acceptance Criteria

**Visual:**
- [ ] At high zoom (> 4px/sample): individual sample waveform visible with dot/line per sample
- [ ] At medium zoom: filled envelope polygon (peak outer, RMS inner fill) — no individual samples
- [ ] At low zoom: pre-computed thumbnail image, fast blit
- [ ] LOD transitions are smooth (brief cross-fade, not a sudden swap)
- [ ] Waveform is clipped to event rect bounds — does not render outside event

**Functional:**
- [ ] `WaveformData.compute_envelope(chunk_size=256)` generates `peaks: np.ndarray`, `rms: np.ndarray`
- [ ] `WaveformData.compute_overview(width_px=128)` generates `overview: QImage` (grayscale-ish)
- [ ] `WaveformCache.get_tile(event_id, tile_idx, lod_level, pps)` returns cached `QImage | None`
- [ ] Cache invalidation: changing `pps` by > 20% triggers tile re-render
- [ ] Progressive loading: if envelope not computed, show overview image immediately; upgrade when ready

**Performance:**
- [ ] With 200 visible events at envelope LOD: waveform pass completes in < 5ms
- [ ] Cache hit rate > 90% during normal scroll (tiles stay cached across scrolls)
- [ ] Tile computation happens in a `QThread`, not the paint thread

### 7.5 Agent Instructions

> **`WaveformData` (pure Python + numpy, `data/waveform_data.py`):**
> ```python
> @dataclass
> class WaveformData:
>     samples: np.ndarray         # float32, mono, shape (N,)
>     sample_rate: float
>     # Pre-computed (call compute() at load time)
>     peaks: np.ndarray | None = None     # shape (N//chunk_size, 2) — [min, max] per chunk
>     rms: np.ndarray | None = None       # shape (N//chunk_size,)
>     overview: 'QImage | None' = None    # Pre-rendered thumbnail (small)
>     is_ready: bool = False              # True after compute() completes
>
>     def compute(self, chunk_size: int = 256) -> None: ...   # Call in thread
>     def get_lod_level(self, px_per_sample: float) -> str: ...  # 'samples'|'envelope'|'overview'
> ```
> `px_per_sample = pps / sample_rate`. Note: this is different from `pps` (pixels per second).
>
> **Envelope polygon construction (critical for performance):**
> Don't draw N individual bars. Build two lists of points — top (peaks) and bottom (peaks mirrored) — and draw as a single `QPainterPath`:
> ```python
> path = QPainterPath()
> path.moveTo(x_start, center_y)
> for i, (peak_max, peak_min) in enumerate(zip(peaks_max, peaks_min)):
>     x = x_start + i * chunk_px
>     path.lineTo(x, center_y - peak_max * half_h)
> # reverse for bottom
> for i in range(len(peaks_min) - 1, -1, -1):
>     x = x_start + i * chunk_px
>     path.lineTo(x, center_y - peaks_min[i] * half_h)  # already negative
> path.closeSubpath()
> painter.fillPath(path, waveform_brush)
> ```
> Then draw RMS as a second filled path on top with `WAVEFORM_RMS_COLOR`.
>
> **LOD selection formula:**
> ```python
> px_per_sample = pps / waveform_data.sample_rate
> if px_per_sample >= WAVEFORM_LOD_SAMPLE_PX_THRESHOLD:
>     lod = 'samples'
> elif px_per_sample >= WAVEFORM_LOD_ENVELOPE_PX_THRESHOLD:
>     lod = 'envelope'
> else:
>     lod = 'overview'
> ```
>
> **Tile caching:** Each tile covers `WAVEFORM_TILE_WIDTH_PX` screen pixels at a given `pps`. Tile index = `int(scroll_x / WAVEFORM_TILE_WIDTH_PX)`. Cache key = `(event_id, tile_idx, round(pps, -1))` (round pps to nearest 10 to avoid cache misses on minor zoom changes).
>
> **Thread safety:** Tile computation runs in `QRunnable` submitted to `QThreadPool`. When complete, emits a signal that causes canvas `update()`. During computation, draw overview fallback.
>
> **Do NOT use `QImage` in the data layer.** `WaveformData.overview` is computed in the renderer thread (which is OK to use Qt), not in pure Python data layer. Alternatively, store raw bytes and reconstruct QImage in renderer.

### 7.6 FEEL.py Parameters Used

`WAVEFORM_LOD_SAMPLE_PX_THRESHOLD`, `WAVEFORM_LOD_ENVELOPE_PX_THRESHOLD`, `WAVEFORM_PEAK_COLOR`, `WAVEFORM_RMS_COLOR`, `WAVEFORM_SAMPLE_COLOR`, `WAVEFORM_OVERVIEW_COLOR`, `WAVEFORM_ALPHA`, `WAVEFORM_LOD_CROSSFADE_MS`, `WAVEFORM_CACHE_MAX_TILES`, `WAVEFORM_TILE_WIDTH_PX`

### 7.7 Dependencies

Tasks 1–6 complete. (Waveform rendering is independent of selection/drag but needs event layout.)

### 7.8 Estimated Agent Time

90–120 minutes (LOD system + tile cache + threading are all non-trivial)

### 7.9 Risk Areas

- **Blocking paint thread:** Agent calls `librosa.load()` or `compute_envelope()` in `paintEvent`. This freezes the UI for seconds on large files. Envelope computation MUST be in a thread. Enforce: paint thread checks `is_ready` flag; if False, draws overview placeholder.
- **QPainterPath performance:** Agent draws waveform as N individual `drawLine` calls instead of a single `fillPath`. At 10,000 chunks, this is catastrophically slow. Verify: profile waveform pass at medium zoom with 200 events.
- **Cache key collisions:** Agent uses `pps` as float cache key. Floating point pps causes cache misses on every wheel tick. Round pps: `round(pps / 10) * 10`.
- **QImage data race:** Agent creates `QImage` from numpy in background thread and reads it in paint thread without a lock. Use `QMutex` or pass the image via Qt signal.
- **LOD pop (no crossfade):** Agent changes LOD level instantly. Implement a simple opacity crossfade: track `lod_transition_alpha` (0–255) and animate it via a one-shot QTimer.

### 7.10 EZ1 Reference Files

- `events/waveform_widget.py` — waveform data loading + QPainterPath construction approach
- `events/waveform_simple.py` — simpler draw method (may be closer to what's needed)

---

## Task 8: Playhead + Scrub

### 8.1 Description

A smooth playhead that tracks audio clock position via interpolation (not a timer that polls position). Click on ruler to seek. Drag playhead or ruler to scrub. Scrub sends IPC seek commands at a throttled rate. Playhead renders as a colored vertical line with a triangle head in the ruler area.

### 8.2 Input

- Full codebase from Tasks 1–7
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\playback\playhead.py` — EZ1 playhead (adapt to non-QGraphicsItem)
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\playback\controller.py` — EZ1 playback controller (clock integration)

### 8.3 Output

```
data/
└── audio_clock.py           # AudioClock: interpolated position from core (pure Python)
ui/
├── renderers/
│   └── playhead_renderer.py # PlayheadRenderer: draws line + triangle head
└── input/
    └── playhead_controller.py  # PlayheadController: click-seek + scrub
```

**`FEEL.py` entries to add:**

```python
# Playhead
PLAYHEAD_COLOR = "#FF4444"
PLAYHEAD_WIDTH_PX = 2
PLAYHEAD_HEAD_HEIGHT_PX = 10          # Triangle height in ruler
PLAYHEAD_HEAD_WIDTH_PX = 10           # Triangle base width
PLAYHEAD_ALPHA = 230                  # 0-255

# Scrub
SCRUB_IPC_THROTTLE_MS = 50           # Min interval between IPC seek messages
SCRUB_DRAG_THRESHOLD_PX = 2          # Min drag before scrub (prevent accidental seek)

# Playback repaint
PLAYBACK_REPAINT_INTERVAL_MS = 16    # ~60fps repaint while playing
```

### 8.4 Acceptance Criteria

**Visual:**
- [ ] Playhead is a sharp vertical line of width `PLAYHEAD_WIDTH_PX` from ruler to bottom of track area
- [ ] Triangle head in ruler: filled with `PLAYHEAD_COLOR`, pointing downward, centered on playhead x
- [ ] Playhead moves smoothly at 60fps when playing (no jitter, no drift)
- [ ] During scrub-drag: playhead follows cursor in real time (no lag)

**Functional:**
- [ ] `AudioClock.now() -> float` returns interpolated playhead position in seconds using: `position_at_sync + (time.perf_counter() - sync_timestamp) * playback_rate`
- [ ] Click anywhere on ruler (not on playhead head) → seeks to clicked time (emits IPC seek)
- [ ] Drag ruler → scrubs; IPC seek throttled to `SCRUB_IPC_THROTTLE_MS`
- [ ] Drag playhead head → same scrub behavior
- [ ] `PlaybackController` drives a `QTimer` at `PLAYBACK_REPAINT_INTERVAL_MS` that calls `canvas.update()` during playback
- [ ] Timer stops when playback stops
- [ ] Playhead position is always computed from `AudioClock.now()` in `paintEvent`, never stored as a pixel coordinate

**No-timer interpolation check:** Play audio, kill the repaint timer, manually call a single `repaint()` 500ms later. The playhead should render at the correct position for that time, proving it's computed from clock, not cached pixel position.

### 8.5 Agent Instructions

> **`AudioClock` (pure Python, `data/audio_clock.py`):**
> ```python
> class AudioClock:
>     def __init__(self):
>         self._position: float = 0.0       # seconds at last sync
>         self._sync_at: float = 0.0        # perf_counter() at last sync
>         self._rate: float = 0.0           # 0 = stopped, 1.0 = playing
>
>     def sync(self, position: float, rate: float = 1.0) -> None:
>         """Call when IPC message arrives with new position."""
>         self._position = position
>         self._sync_at = time.perf_counter()
>         self._rate = rate
>
>     def stop(self) -> None:
>         self._position = self.now()
>         self._rate = 0.0
>
>     def now(self) -> float:
>         """Current interpolated position in seconds."""
>         if self._rate == 0.0:
>             return self._position
>         elapsed = time.perf_counter() - self._sync_at
>         return self._position + elapsed * self._rate
> ```
> Import `time` (standard library). No Qt.
>
> **`PlayheadRenderer` (`ui/renderers/playhead_renderer.py`):**
> ```python
> class PlayheadRenderer:
>     def draw(self, painter: QPainter, position_seconds: float, viewport: ViewportState,
>              widget_height: int, ruler_height: int) -> None:
>         x = coord.time_to_screen(position_seconds, viewport.pps, viewport.scroll_x)
>         if x < 0 or x > widget_width: return   # off-screen culling
>
>         # Triangle in ruler
>         head_tip_y = ruler_height
>         head_left_x = x - PLAYHEAD_HEAD_WIDTH_PX / 2
>         head_right_x = x + PLAYHEAD_HEAD_WIDTH_PX / 2
>         head_top_y = ruler_height - PLAYHEAD_HEAD_HEIGHT_PX
>         triangle = QPolygonF([
>             QPointF(x, head_tip_y),
>             QPointF(head_left_x, head_top_y),
>             QPointF(head_right_x, head_top_y)
>         ])
>         painter.setBrush(QColor(PLAYHEAD_COLOR))
>         painter.setPen(Qt.PenStyle.NoPen)
>         painter.drawPolygon(triangle)
>
>         # Vertical line
>         pen = QPen(QColor(PLAYHEAD_COLOR), PLAYHEAD_WIDTH_PX)
>         painter.setPen(pen)
>         painter.drawLine(QPointF(x, ruler_height), QPointF(x, widget_height))
> ```
>
> **`PlayheadController` (`ui/input/playhead_controller.py`):**
> - In `TimelineCanvas.mousePressEvent`: if `y < RULER_HEIGHT_PX`, check if `abs(x - playhead_x) <= PLAYHEAD_HEAD_WIDTH_PX`: enter `PLAYHEAD_DRAG` mode. Else: emit `seek_requested(time_at_cursor)` immediately.
> - In `mouseMoveEvent` during `PLAYHEAD_DRAG`: throttle IPC calls. Track `_last_scrub_at`. Only emit `scrub_requested(time)` if `time.perf_counter() - _last_scrub_at > SCRUB_IPC_THROTTLE_MS / 1000`.
> - Repaint timer: `QTimer` fires every `PLAYBACK_REPAINT_INTERVAL_MS` ms. On tick: `canvas.update()`. Started by `PlaybackController.on_play()`, stopped by `PlaybackController.on_stop()`.
>
> **In `paintEvent`**, the playhead position is always: `position = self._audio_clock.now()`. Never cache it elsewhere.

### 8.6 FEEL.py Parameters Used

`PLAYHEAD_COLOR`, `PLAYHEAD_WIDTH_PX`, `PLAYHEAD_HEAD_HEIGHT_PX`, `PLAYHEAD_HEAD_WIDTH_PX`, `PLAYHEAD_ALPHA`, `SCRUB_IPC_THROTTLE_MS`, `SCRUB_DRAG_THRESHOLD_PX`, `PLAYBACK_REPAINT_INTERVAL_MS`, `RULER_HEIGHT_PX`

### 8.7 Dependencies

Tasks 1–7 complete. (Can start after Task 3 — playhead rendering is independent of selection/drag/waveform.)

### 8.8 Estimated Agent Time

45–60 minutes

### 8.9 Risk Areas

- **Timer-driven position (not clock-driven):** Agent stores `self._playhead_x` and updates it in the timer. Playhead drifts relative to audio over time. Enforce: `playhead_renderer.draw()` must call `audio_clock.now()` — it must not accept a pre-computed pixel position.
- **Scrub fires IPC every mouseMoveEvent:** Without throttling, this floods the IPC channel with hundreds of seek messages per second. Verify: add a counter to IPC calls during a 1-second scrub — should be < 25 messages.
- **Triangle not centered on playhead line:** Agent offsets the triangle incorrectly. Check: at any position, the triangle tip should align exactly with the vertical line.
- **Repaint timer runs when stopped:** Agent starts the QTimer and never stops it. Should be `timer.start()` on play, `timer.stop()` on stop.
- **Click on ruler during playback seeks to wrong time:** Agent uses pixel X directly without accounting for scroll_x. The `time_at_cursor = screen_to_time(event.position().x(), pps, scroll_x)`.

### 8.10 EZ1 Reference Files

- `playback/playhead.py` — triangle + line drawing pattern (adapt from QGraphicsItem to QPainter)
- `playback/controller.py` — clock sync and repaint timer pattern

---

## Task 9: Layer Panel + Track Management

### 9.1 Description

Add a fixed-width left panel showing layer (track) labels. Each layer has: label text, mute button (M), solo button (S), color swatch (click to pick), collapse/expand button (▶/▼). Layers can be drag-reordered by dragging the label. Right-click on a layer shows context menu: Rename, Duplicate, Delete, Move Up, Move Down.

The track area (event canvas) accounts for the panel width via `scroll_x` and layout Y offsets.

### 9.2 Input

- Full codebase from Tasks 1–8
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\events\layer_manager.py` — EZ1 layer management
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\settings\panel.py` — EZ1 settings panel (for UI structure reference only)

### 9.3 Output

```
ui/
├── layer_panel.py              # LayerPanel(QWidget): the left-side track header panel
└── renderers/
    └── layer_renderer.py       # LayerRenderer: draws layer rows in panel
```

Modify `canvas.py` to offset event rendering by `LAYER_PANEL_WIDTH_PX`.  
Modify `models.py` to add `is_muted`, `is_solo` to `LayerModel`.

**`FEEL.py` entries to add:**

```python
# Layer Panel
LAYER_PANEL_WIDTH_PX = 160
LAYER_ROW_HEIGHT_PX = 48            # Must match TRACK_HEIGHT_PX
LAYER_ROW_COLLAPSED_HEIGHT_PX = 8   # Must match TRACK_COLLAPSED_HEIGHT_PX
LAYER_PANEL_BG_COLOR = "#0E1820"
LAYER_LABEL_COLOR = "#AABBCC"
LAYER_LABEL_FONT_SIZE_PX = 12
LAYER_LABEL_FONT_BOLD = True
LAYER_SWATCH_WIDTH_PX = 14
LAYER_SWATCH_HEIGHT_PX = 14
LAYER_BUTTON_SIZE_PX = 20           # M / S button size
LAYER_MUTE_COLOR_ACTIVE = "#FF8800"
LAYER_SOLO_COLOR_ACTIVE = "#FFDD00"
LAYER_BUTTON_COLOR_INACTIVE = "#2A3848"
LAYER_DRAG_INDICATOR_COLOR = "#4499FF"
LAYER_DRAG_INDICATOR_WIDTH_PX = 2
```

### 9.4 Acceptance Criteria

**Visual:**
- [ ] Left panel is `LAYER_PANEL_WIDTH_PX` wide, dark background, fixed (does not scroll with track area)
- [ ] Each layer row height matches track area height (visual alignment between panel and canvas)
- [ ] Layer name displayed in bold, truncated with `...` if too long
- [ ] Color swatch is a small square filled with layer color
- [ ] M/S buttons render distinctly when active (orange/yellow) vs inactive
- [ ] Collapse/expand: ▶ when collapsed (narrow row), ▼ when expanded
- [ ] Drag reorder: blue horizontal line shows where the dragged layer will land

**Functional:**
- [ ] Mute: muted layer's events are rendered with `EVENT_MUTED_ALPHA` opacity (add to FEEL.py: `EVENT_MUTED_ALPHA = 80`)
- [ ] Solo: only soloed layers render at full opacity; all others render at muted alpha
- [ ] Collapse: row height changes to `LAYER_ROW_COLLAPSED_HEIGHT_PX`; events hidden in that track
- [ ] Color swatch click: opens `QColorDialog`; on accept, updates `layer.color`
- [ ] Right-click context menu: Rename (inline QLineEdit), Duplicate, Delete (with confirmation if has events), Move Up, Move Down
- [ ] Drag reorder: drag a layer row up or down; drop target shown by indicator line; `model.layers` reordered on drop
- [ ] Panel scrolls vertically in sync with the track area canvas (same `scroll_y`)

### 9.5 Agent Instructions

> **`LayerPanel(QWidget)` is a separate widget, NOT drawn inside the canvas.** The timeline container is a `QHBoxLayout` with `LayerPanel` on the left and `TimelineCanvas` on the right. They share the same `ViewportState` reference (specifically `scroll_y`).
>
> **`LayerPanel` renders with `QPainter` in its own `paintEvent`** (same pattern as `TimelineCanvas`). It does not use `QLabel`, `QPushButton` etc. for individual rows — it draws everything in `paintEvent` for performance and alignment precision.
>
> **Hit testing in `LayerPanel`:** In `mousePressEvent`, determine which row was clicked and which "zone" (swatch, M button, S button, expand button, drag handle, or label text). Use a `LayerHitTester` similar to `HitTester` in Task 4.
>
> **Button rects layout (left to right within each row):**
> ```
> [drag_handle 8px] [swatch 14px] [4px gap] [label text, expands] [S btn 20px] [M btn 20px] [4px margin]
> ```
>
> **Drag reorder:** On `mousePress` in drag handle zone: enter `LAYER_DRAGGING` state. On `mouseMove`: compute `target_index = y_offset // LAYER_ROW_HEIGHT_PX`. Render a blue `LAYER_DRAG_INDICATOR_WIDTH_PX` line between rows at `target_index`. On `mouseRelease`: call `model.reorder_layer(from_index, target_index)` and trigger canvas `update()`.
>
> **Inline rename:** On right-click > Rename: position a `QLineEdit` over the label area of the row with the current name pre-filled. On `returnPressed` or focus loss: commit rename, hide the `QLineEdit`.
>
> **Sync `scroll_y` with canvas:** `LayerPanel` and `TimelineCanvas` both receive `scroll_y` from a shared `ViewportState`. When the canvas scrolls vertically, call `layer_panel.update()` to repaint with the new scroll offset. Layer rows' Y positions in the panel must exactly match the canvas track Y positions.

### 9.6 FEEL.py Parameters Used

`LAYER_PANEL_WIDTH_PX`, `LAYER_ROW_HEIGHT_PX`, `LAYER_ROW_COLLAPSED_HEIGHT_PX`, `LAYER_PANEL_BG_COLOR`, `LAYER_LABEL_COLOR`, `LAYER_LABEL_FONT_SIZE_PX`, `LAYER_LABEL_FONT_BOLD`, `LAYER_SWATCH_WIDTH_PX`, `LAYER_SWATCH_HEIGHT_PX`, `LAYER_BUTTON_SIZE_PX`, `LAYER_MUTE_COLOR_ACTIVE`, `LAYER_SOLO_COLOR_ACTIVE`, `LAYER_BUTTON_COLOR_INACTIVE`, `LAYER_DRAG_INDICATOR_COLOR`, `LAYER_DRAG_INDICATOR_WIDTH_PX`, `EVENT_MUTED_ALPHA`, `TRACK_HEIGHT_PX`, `TRACK_COLLAPSED_HEIGHT_PX`

### 9.7 Dependencies

Tasks 1–8 complete. (Can begin after Task 2 — layer model exists; Tasks 3–8 can be integrated after.)

### 9.8 Estimated Agent Time

90–120 minutes (panel layout + drag reorder + sync are all complexity)

### 9.9 Risk Areas

- **Panel/canvas misalignment:** The most common bug. Layer rows in the panel don't visually align with track rows in the canvas because they use different Y calculations. Both must use the same Y offset computation from `ViewportState.scroll_y` and the same `LAYER_ROW_HEIGHT_PX` + `TRACK_SPACING_PX`.
- **QWidget buttons in each row:** Agent creates actual `QPushButton` widgets per row. This creates dozens of child widgets, making the panel slow and hard to manage. Enforce: everything drawn in `paintEvent` via QPainter; only the rename `QLineEdit` is an actual widget.
- **Solo logic:** Agent just mutes non-soloed layers at model level (changing `is_muted`). Solo should be a render-time decision, not a model mutation — `is_muted` and `is_solo` are independent flags. The renderer checks: if any layer has `is_solo=True`, render non-soloed layers at muted alpha.
- **Context menu position:** Agent shows right-click menu at wrong position. Use `event.globalPosition().toPoint()` not `event.position()`.

### 9.10 EZ1 Reference Files

- `events/layer_manager.py` — layer CRUD, solo/mute logic, reorder
- `settings/panel.py` — reference only for Qt widget layout patterns (do not copy QGraphicsScene code)

---

## Task 10: Polish Pass

### 10.1 Description

Complete the "real tool" feel by implementing: hover states on events and buttons, cursor shape changes everywhere, context-sensitive right-click menus, keyboard shortcut completeness, undo/redo flash animation on affected events, and smooth state transition animations for expand/collapse and mute/solo.

This task is not optional — it's the difference between a working tool and a professional one.

### 10.2 Input

- Full codebase from Tasks 1–9
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\settings\shortcuts.py` — EZ1 keyboard shortcuts (reference for which keys to implement)
- `C:\Users\griff\EchoZero\ui\qt_gui\widgets\timeline\core\style.py` — EZ1 color definitions

### 10.3 Output

```
ui/
├── input/
│   ├── keyboard_controller.py     # All keyboard shortcuts
│   └── hover_controller.py        # Hover state tracking + cursor management
├── renderers/
│   └── hover_renderer.py          # Hover overlays on events
└── animations/
    ├── __init__.py
    ├── undo_flash.py               # Flash animation for undo/redo
    └── state_animator.py           # Generic QTimer-driven animation values
```

Modify `canvas.py` to wire all controllers.  
Add `FEEL.py` entries for every new constant (see below).

**`FEEL.py` entries to add:**

```python
# Hover
HOVER_EVENT_ALPHA_OVERLAY = 25       # 0-255, white overlay on hover
HOVER_BUTTON_ALPHA_OVERLAY = 40
HOVER_FADE_DURATION_MS = 80          # How fast hover fades in/out
HOVER_RESIZE_HANDLE_VISIBLE = True   # Show subtle edge indicator on hover

# Undo flash
UNDO_FLASH_COLOR = "#FFFFFF"
UNDO_FLASH_ALPHA_PEAK = 80           # Peak opacity of flash
UNDO_FLASH_DURATION_MS = 300         # Total flash duration
UNDO_FLASH_FADE_CURVE = "ease_out"   # Easing curve name

# State animations
COLLAPSE_ANIMATION_DURATION_MS = 180
MUTE_FADE_DURATION_MS = 120

# Cursor rules (documentation — agents read these to know what to show where)
# CURSOR_DEFAULT = ArrowCursor
# CURSOR_ON_EVENT = OpenHandCursor
# CURSOR_ON_RESIZE_EDGE = SizeHorCursor
# CURSOR_ON_RULER = IBeamCursor (for click-seek)
# CURSOR_DRAGGING = ClosedHandCursor
# CURSOR_LASSO = CrossCursor

# Keyboard shortcuts (documentation)
# DELETE / BACKSPACE = delete selected events
# SPACE = play/pause
# HOME = seek to start
# END = seek to end
# Ctrl+A = select all
# Ctrl+Z = undo
# Ctrl+Shift+Z or Ctrl+Y = redo
# Ctrl+D = duplicate selected
# Ctrl+G = toggle snap
# Ctrl+0 = reset zoom
# Ctrl+= / Ctrl+- = zoom in/out
# Ctrl+Shift+F = zoom to fit selection
# Escape = deselect all / cancel drag
# Arrow keys = nudge selected events (Left/Right = 1 grid unit, Shift = 10 units)
```

### 10.4 Acceptance Criteria

**Hover states:**
- [ ] Hovering an event: subtle white overlay appears (fade in `HOVER_FADE_DURATION_MS`)
- [ ] Moving off event: overlay fades out
- [ ] Hovering resize edge: cursor changes to `SizeHorCursor`, no delay
- [ ] Hovering over M/S buttons in layer panel: button brightens
- [ ] Cursor is always the correct shape for the hovered zone (per the CURSOR_* rules in FEEL.py)

**Keyboard shortcuts:**
- [ ] All shortcuts from FEEL.py documentation block are implemented and functional
- [ ] Arrow key nudge: moves selected events by exactly 1 minor grid interval (from `GridLayout`), or 10 intervals with Shift
- [ ] Delete: removes events from model (with undo)
- [ ] Ctrl+D: duplicates selected events, places them offset by 1 minor grid interval, selects new copies
- [ ] Escape: if dragging → cancel drag; else if selection → deselect all; else → no-op

**Context menus:**
- [ ] Right-click on event: "Cut", "Copy", "Paste", "Duplicate", "Delete", separator, "Properties..."
- [ ] Right-click on empty track area: "Paste here", "Add Marker", separator, "Track Settings..."
- [ ] Right-click on ruler: "Add Marker at [time]", "Clear All Markers"
- [ ] Right-click on layer panel track: (already handled in Task 9, verify it still works)
- [ ] All context menus: items are grayed out when not applicable (e.g., Paste grayed when clipboard empty)

**Undo flash:**
- [ ] After Ctrl+Z: events affected by the undo briefly flash white, then return to normal
- [ ] Flash uses `UNDO_FLASH_ALPHA_PEAK` at peak, fades out over `UNDO_FLASH_DURATION_MS`
- [ ] Flash does not affect events not in the undo action

**Collapse animation:**
- [ ] Expanding/collapsing a track animates over `COLLAPSE_ANIMATION_DURATION_MS` — height changes smoothly
- [ ] During animation, events in the collapsing track fade out

**Performance:**
- [ ] Hover tracking: `mouseMoveEvent` completes in < 0.5ms (hover detection must be O(visible events), not O(all events))
- [ ] Animations use `QTimer` driving `canvas.update()` — not `QPropertyAnimation` (which fights QPainter)

### 10.5 Agent Instructions

> **`HoverController` (`ui/input/hover_controller.py`):**
> Tracks which event (by ID) is currently hovered. On `mouseMoveEvent`, call `HitTester.event_at(x, y)` using the current layout's event rects. If result changes from previous hover: start a `QTimer` to drive a `hover_alpha` from 0 → `HOVER_EVENT_ALPHA_OVERLAY` over `HOVER_FADE_DURATION_MS`. On mouse leave: drive alpha back to 0.
>
> `HoverRenderer.draw(painter, hovered_event_id, hover_alpha, rects)`: for the hovered rect, draw a `QColor(255, 255, 255, hover_alpha)` filled rect on top.
>
> **`KeyboardController` (`ui/input/keyboard_controller.py`):**
> Implement in `TimelineCanvas.keyPressEvent`. Route to `KeyboardController.handle(event, model, viewport, drag_ctrl, selection_ctrl)`. Return `True` if handled (call `event.accept()`).
>
> Arrow nudge logic:
> ```python
> def nudge(model, selection, direction: int, modifiers):
>     interval = grid_layout.get_minor_interval(viewport.pps)  # in seconds
>     delta = interval * direction * (10 if Shift else 1)
>     for event_id in selection:
>         event = model.get_event(event_id)
>         new_start = max(0, event.start_time + delta)
>         undo_stack.push(MoveEventCommand(event_id, new_start))
> ```
>
> **`UndoFlash` animation (`ui/animations/undo_flash.py`):**
> ```python
> class UndoFlash:
>     def __init__(self, canvas):
>         self._canvas = canvas
>         self._event_ids: set[str] = set()
>         self._alpha: float = 0.0
>         self._timer = QTimer()
>         self._timer.timeout.connect(self._tick)
>         self._elapsed_ms: float = 0.0
>
>     def trigger(self, event_ids: set[str]):
>         self._event_ids = event_ids
>         self._elapsed_ms = 0.0
>         self._timer.start(16)   # 60fps
>
>     def _tick(self):
>         self._elapsed_ms += 16
>         progress = self._elapsed_ms / UNDO_FLASH_DURATION_MS
>         if progress >= 1.0:
>             self._alpha = 0.0
>             self._timer.stop()
>         else:
>             # ease_out: fast rise, slow fall — approximate with: 1 - (2*p-1)^2 for p<0.5
>             self._alpha = ease_out_curve(progress) * UNDO_FLASH_ALPHA_PEAK
>         self._canvas.update()
>
>     def get_flash_alpha(self, event_id: str) -> float:
>         if event_id in self._event_ids:
>             return self._alpha
>         return 0.0
> ```
>
> **Context menus:** Use `QMenu` with `addAction`. Wire actions to model operations (which push to undo stack). Use `action.setEnabled(False)` for grayed items.
>
> **Collapse animation:** Add `_collapse_progress: dict[str, float]` to `ViewportState` (0.0 = fully collapsed, 1.0 = fully expanded). `StateAnimator` drives these from 0→1 or 1→0. `EventLayout.compute()` reads the progress to compute intermediate track height. `LayerPanel.paintEvent` reads it for row height.

### 10.6 FEEL.py Parameters Used

`HOVER_EVENT_ALPHA_OVERLAY`, `HOVER_BUTTON_ALPHA_OVERLAY`, `HOVER_FADE_DURATION_MS`, `HOVER_RESIZE_HANDLE_VISIBLE`, `UNDO_FLASH_COLOR`, `UNDO_FLASH_ALPHA_PEAK`, `UNDO_FLASH_DURATION_MS`, `UNDO_FLASH_FADE_CURVE`, `COLLAPSE_ANIMATION_DURATION_MS`, `MUTE_FADE_DURATION_MS`

### 10.7 Dependencies

Tasks 1–9 complete.

### 10.8 Estimated Agent Time

120–150 minutes (most tasks in this pass are small individually but there are many of them)

### 10.9 Risk Areas

- **Hover tracking expensive:** Agent calls `HitTester.event_at()` on every pixel of mouse movement, but `HitTester` iterates all events (not just visible ones). `HoverController` must only test against the already-computed `visible_rects` from the last layout pass — these are already cached.
- **`QPropertyAnimation` on custom paint:** Agent uses `QPropertyAnimation` on a Python property and hooks into `update()`. This often creates timing conflicts with `paintEvent`. Use `QTimer` + manual alpha tracking.
- **Context menu "Properties...":** Agent creates a full `QDialog` immediately. This is fine, but the dialog must not block the canvas repaint. Use `.show()` (non-modal) or `.exec()` (modal, but acceptable for a settings dialog).
- **Undo flash targets wrong events:** Agent flashes all selected events on any undo, instead of only the events affected by the undone command. The undo command itself must know which event IDs it touched.
- **Arrow nudge bypasses undo stack:** Agent mutates model directly in `keyPressEvent` without going through `undo_stack.push(...)`. Every mutation must go through the undo stack. Verify: nudge an event, then Ctrl+Z — it should come back.
- **Collapse animation stalls at 0.5:** Agent's easing function computes `NaN` or clamps incorrectly. Test: rapidly toggle collapse on the same track. The progress should always end cleanly at 0.0 or 1.0.

### 10.10 EZ1 Reference Files

- `settings/shortcuts.py` — full keyboard shortcut list (verify coverage)
- `core/style.py` — color/theme definitions (use as visual reference for Polish aesthetics)
- `events/items.py` — hover state visual patterns in EZ1 (search for "hover", "highlight")

---

## Appendix A: FEEL.py Complete Skeleton

The following is the complete `FEEL.py` file that should exist before Task 1 begins. Agents populate entries; Griff tunes values.

```python
"""
FEEL.py — EchoZero 2 Timeline Craft Constants

Human-tunable parameters. This file defines the FEEL of the timeline.
AI coding agents NEVER edit default values here. They only ADD new entries
with their own documented names and sensible starting values.

Griff tunes these by running the app and editing this file.
No restart required — changes take effect on next repaint.
"""

# ============================================================
# GRID
# ============================================================
RULER_HEIGHT_PX = 30
GRID_MIN_MINOR_SPACING_PX = 12
GRID_MIN_MAJOR_SPACING_PX = 80
GRID_MAX_LINES = 500
GRID_MINOR_LINE_COLOR = "#1E2A38"
GRID_MAJOR_LINE_COLOR = "#2A3848"
GRID_RULER_BG_COLOR = "#141E28"
GRID_RULER_TEXT_COLOR = "#8899AA"
GRID_RULER_FONT_SIZE_PX = 11

# ============================================================
# ZOOM
# ============================================================
ZOOM_DEFAULT_PPS = 100.0
ZOOM_MIN_PPS = 10.0
ZOOM_MAX_PPS = 2000.0
ZOOM_PIXEL_SENSITIVITY = 0.003
ZOOM_ANGLE_SENSITIVITY = 0.12
ZOOM_ACCUMULATOR_THRESHOLD = 0.005
ZOOM_ANCHOR_TO_CURSOR = True

# ============================================================
# SCROLL / INERTIA
# ============================================================
SCROLL_INERTIA_DECAY = 0.88
SCROLL_INERTIA_TIMER_MS = 16
SCROLL_MIN_VELOCITY_PX = 0.8
SCROLL_PIXEL_SENSITIVITY = 1.0
SCROLL_ANGLE_SENSITIVITY = 0.5

# ============================================================
# TRACKS / LAYERS
# ============================================================
TRACK_HEIGHT_PX = 48
TRACK_SPACING_PX = 2
TRACK_COLLAPSED_HEIGHT_PX = 8
TRACK_LABEL_WIDTH_PX = 160

# ============================================================
# EVENTS
# ============================================================
EVENT_HEIGHT_PX = 40
EVENT_CORNER_RADIUS_PX = 4
EVENT_BORDER_WIDTH_PX = 1
EVENT_LABEL_FONT_SIZE_PX = 11
EVENT_LABEL_PADDING_PX = 6
EVENT_LABEL_MIN_WIDTH_PX = 30
EVENT_DEFAULT_COLOR = "#3A6EA5"
EVENT_SELECTED_COLOR = "#0066FF"
EVENT_SELECTED_BORDER_COLOR = "#66AAFF"
EVENT_BORDER_COLOR = "#2A5080"
EVENT_TEXT_COLOR = "#DDEEFF"
EVENT_TEXT_SELECTED_COLOR = "#FFFFFF"
EVENT_MUTED_ALPHA = 80

# ============================================================
# SELECTION
# ============================================================
SELECTION_LASSO_FILL_COLOR = "#0066FF"
SELECTION_LASSO_FILL_ALPHA = 30
SELECTION_LASSO_BORDER_COLOR = "#4499FF"
SELECTION_LASSO_BORDER_WIDTH_PX = 1
SELECTION_CLICK_RADIUS_PX = 4
SELECTION_DRAG_THRESHOLD_PX = 4

# ============================================================
# DRAG + SNAP
# ============================================================
SNAP_ENABLED_DEFAULT = True
SNAP_MAGNETISM_RADIUS_PX = 12
SNAP_TO_GRID = True
SNAP_TO_EVENT_ONSETS = True
SNAP_TO_MARKERS = True
DRAG_GHOST_ALPHA = 120
DRAG_GHOST_BORDER_COLOR = "#66AAFF"
SNAP_INDICATOR_COLOR = "#FFCC00"
SNAP_INDICATOR_WIDTH_PX = 1
SNAP_INDICATOR_HEIGHT = 1.0
DRAG_THRESHOLD_PX = 4

# ============================================================
# RESIZE
# ============================================================
RESIZE_HANDLE_WIDTH_PX = 6
RESIZE_HANDLE_MIN_WIDTH_PX = 3
RESIZE_HANDLE_PERCENT = 0.15
RESIZE_MIN_DURATION_S = 0.05
RESIZE_HOVER_COLOR = "#FFFFFF"
RESIZE_HOVER_ALPHA = 60
RESIZE_SNAP_ENABLED = True

# ============================================================
# WAVEFORM
# ============================================================
WAVEFORM_LOD_SAMPLE_PX_THRESHOLD = 4.0
WAVEFORM_LOD_ENVELOPE_PX_THRESHOLD = 0.5
WAVEFORM_PEAK_COLOR = "#4A8FC4"
WAVEFORM_RMS_COLOR = "#2A6FA0"
WAVEFORM_SAMPLE_COLOR = "#66AADD"
WAVEFORM_OVERVIEW_COLOR = "#2A5A80"
WAVEFORM_ALPHA = 200
WAVEFORM_LOD_CROSSFADE_MS = 150
WAVEFORM_CACHE_MAX_TILES = 128
WAVEFORM_TILE_WIDTH_PX = 256

# ============================================================
# PLAYHEAD
# ============================================================
PLAYHEAD_COLOR = "#FF4444"
PLAYHEAD_WIDTH_PX = 2
PLAYHEAD_HEAD_HEIGHT_PX = 10
PLAYHEAD_HEAD_WIDTH_PX = 10
PLAYHEAD_ALPHA = 230
SCRUB_IPC_THROTTLE_MS = 50
SCRUB_DRAG_THRESHOLD_PX = 2
PLAYBACK_REPAINT_INTERVAL_MS = 16

# ============================================================
# LAYER PANEL
# ============================================================
LAYER_PANEL_WIDTH_PX = 160
LAYER_ROW_HEIGHT_PX = 48
LAYER_ROW_COLLAPSED_HEIGHT_PX = 8
LAYER_PANEL_BG_COLOR = "#0E1820"
LAYER_LABEL_COLOR = "#AABBCC"
LAYER_LABEL_FONT_SIZE_PX = 12
LAYER_LABEL_FONT_BOLD = True
LAYER_SWATCH_WIDTH_PX = 14
LAYER_SWATCH_HEIGHT_PX = 14
LAYER_BUTTON_SIZE_PX = 20
LAYER_MUTE_COLOR_ACTIVE = "#FF8800"
LAYER_SOLO_COLOR_ACTIVE = "#FFDD00"
LAYER_BUTTON_COLOR_INACTIVE = "#2A3848"
LAYER_DRAG_INDICATOR_COLOR = "#4499FF"
LAYER_DRAG_INDICATOR_WIDTH_PX = 2

# ============================================================
# HOVER + POLISH
# ============================================================
HOVER_EVENT_ALPHA_OVERLAY = 25
HOVER_BUTTON_ALPHA_OVERLAY = 40
HOVER_FADE_DURATION_MS = 80
HOVER_RESIZE_HANDLE_VISIBLE = True

UNDO_FLASH_COLOR = "#FFFFFF"
UNDO_FLASH_ALPHA_PEAK = 80
UNDO_FLASH_DURATION_MS = 300
UNDO_FLASH_FADE_CURVE = "ease_out"

COLLAPSE_ANIMATION_DURATION_MS = 180
MUTE_FADE_DURATION_MS = 120
```

---

## Appendix B: File Structure Overview

Final directory structure after all 10 tasks:

```
C:\Users\griff\EchoZero\ui\timeline_v2\
├── FEEL.py
├── data/
│   ├── models.py              # EventModel, LayerModel, TimelineModel
│   ├── coord_transform.py     # CoordTransform (time ↔ screen)
│   ├── spatial_index.py       # SpatialIndex (interval tree)
│   ├── snap_calculator.py     # SnapCalculator
│   ├── hit_test.py            # HitTester
│   ├── audio_clock.py         # AudioClock (interpolated position)
│   └── waveform_data.py       # WaveformData + WaveformCache
├── layout/
│   ├── grid_layout.py         # GridLayout (tick positions)
│   └── event_layout.py        # EventLayout (event screen rects)
├── ui/
│   ├── canvas.py              # TimelineCanvas(QWidget) — main widget
│   ├── layer_panel.py         # LayerPanel(QWidget) — left panel
│   ├── renderers/
│   │   ├── grid_renderer.py
│   │   ├── event_renderer.py
│   │   ├── waveform_renderer.py
│   │   ├── playhead_renderer.py
│   │   ├── selection_renderer.py
│   │   ├── drag_renderer.py
│   │   ├── resize_renderer.py
│   │   ├── hover_renderer.py
│   │   └── layer_renderer.py
│   ├── input/
│   │   ├── scroll_controller.py
│   │   ├── zoom_controller.py
│   │   ├── selection_controller.py
│   │   ├── drag_controller.py
│   │   ├── resize_controller.py
│   │   ├── playhead_controller.py
│   │   ├── keyboard_controller.py
│   │   └── hover_controller.py
│   └── animations/
│       ├── undo_flash.py
│       └── state_animator.py
└── tests/
    ├── test_coord_transform.py
    ├── test_spatial_index.py
    └── test_event_layout.py
```

---

## Appendix C: Task Sequence + Estimated Timeline

| # | Task | Est. Agent Time | Hard Dependency | Can Parallelize With |
|---|------|----------------|-----------------|---------------------|
| 1 | Canvas + Grid | 45–75 min | — | — |
| 2 | Event Rendering | 60–90 min | Task 1 | — |
| 3 | Scroll + Zoom | 60–90 min | Tasks 1–2 | — |
| 4 | Selection | 45–60 min | Tasks 1–3 | — |
| 5 | Drag + Snap | 75–105 min | Tasks 1–4 | — |
| 6 | Resize | 45–60 min | Tasks 1–5 | — |
| 7 | Waveform | 90–120 min | Tasks 1–6 | Could start after Task 2 |
| 8 | Playhead | 45–60 min | Tasks 1–3 | Could start after Task 3 |
| 9 | Layer Panel | 90–120 min | Tasks 1–8 | Could start after Task 2 |
| 10 | Polish | 120–150 min | Tasks 1–9 | — |
| **Total** | | **~11–15 agent-hours** | | |

---

*Document authored by Chonch (OpenClaw) for EchoZero 2 development.*  
*Architecture principles sourced from S.A.S.S. Panel: echozero-timeline-feasibility.*
