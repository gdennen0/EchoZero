# EchoZero Timeline Prototype

A standalone visual test harness for the EchoZero 2 timeline UI.

## Files

- **prototype.py** — Main entry point. Run with `python prototype.py`
- **canvas.py** — Core timeline canvas (QWidget with paintEvent, NOT QGraphicsView)
- **model.py** — Pure Python data model (no Qt imports). Events, layers, state, viewport culling
- **ruler.py** — Time ruler widget at top with adaptive ticks and playhead triangle
- **layers_panel.py** — Layer label panel on the left. Click to select layer
- **input.py** — Mouse/keyboard handling: zoom, pan, select, drag, resize
- **FEEL.py** — Visual constants. Tweak these to adjust look/feel
- **README.md** — This file

## Requirements

- Python 3.7+
- PyQt5 (install: `pip install PyQt5`)

## Running

```bash
python prototype.py
```

A window opens showing 500 fake events across 10 fake layers.

## Controls

| Action | Control |
|--------|---------|
| **Zoom** | Mouse wheel |
| **Pan** | Middle-drag or Alt+Left-drag |
| **Select** | Click event |
| **Multi-select** | Shift+Click |
| **Move event** | Drag (with snap indicator) |
| **Resize event** | Drag from edge (left/right) |
| **Deselect** | Click empty space or Esc |
| **Delete selected** | Delete key |
| **Play/Pause** | Space bar |

## Architecture

### Model (model.py)

Pure Python — no Qt imports. Use anywhere.

- `TimelineEvent`: id, time, duration, layer_id, label, color, classifications
- `TimelineLayer`: id, name, color, order, collapsed, height
- `TimelineState`: events, layers, zoom, scroll, playhead, selection
- `ViewportRect`: visible time/layer range
- `visible_events(state, viewport)`: Fast culling with bisect

### Canvas (canvas.py)

QWidget subclass with QPainter rendering pipeline:

1. Grid background (adaptive based on zoom)
2. Section regions (colored bands)
3. Event backgrounds (batched)
4. Event borders (selected/unselected)
5. Event labels (with clipping)
6. Playhead
7. Snap indicator

### Input (input.py)

InputHandler class:

- Wheel → zoom (cursor-anchored)
- Middle/Alt-drag → pan
- Click → select
- Drag selected → move (snaps to grid from FEEL.py)
- Drag edge → resize
- Escape/Delete → clear selection / delete

### Ruler & Layers Panel

- `TimeRuler`: Time labels, tick marks, playhead triangle, click to move playhead
- `LayersPanel`: Layer names + colors, click to select, scrolls with canvas

### Feel (FEEL.py)

ALL visual constants live here. No hardcoded pixel values elsewhere.

Examples:
- `ZOOM_MIN`, `ZOOM_MAX`, `ZOOM_STEP` — zoom behavior
- `EVENT_HEIGHT`, `EVENT_RADIUS` — event appearance
- `PLAYHEAD_COLOR`, `PLAYHEAD_WIDTH` — playhead style
- `SNAP_GRID_SECONDS` — snap grid resolution
- `LAYER_ROW_HEIGHT` — vertical spacing

Change these to evaluate feel/UX without touching code.

## Performance

- 500 events + 10 layers
- Spatial culling: only visible events render (bisect + viewport)
- Batch rendering: events rendered in passes (backgrounds → borders → labels)
- Target: 60fps (16ms refresh via timer)

## Future

This is a **prototype**, not the final product. Upgrade paths:

- Replace bisect + sorted list with interval tree for very large datasets
- Add undo/redo history
- Persist state to disk
- Real audio waveform rendering
- Audio playback sync
- Scripting/automation layer

For now: Griff runs this, tweaks FEEL.py, evaluates the feel, and builds the real app from these insights.

---

**Status:** Ready to run. PyQt5 must be installed and working (see Requirements).
