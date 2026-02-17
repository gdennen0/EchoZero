# Timeline Widget

Qt-based timeline for event editing and visualization.

## Overview

The timeline provides:
- Event visualization and editing
- Layer management
- Time-based navigation
- Playback integration

## Architecture

```
timeline/
├── timeline_widget.py     # Main widget
├── timeline_scene.py      # QGraphicsScene
├── timeline_view.py       # QGraphicsView
├── event_item.py          # Event graphics items
├── layer_manager.py       # Layer handling
├── grid_system.py         # Time grid
├── movement_controller.py # Drag/drop
└── playback_controller.py # Playback cursor
```

## Key Components

- **TimelineWidget** - Container widget
- **TimelineScene** - Manages events and layers
- **TimelineView** - Viewport with zoom/pan
- **EventItem** - Individual event visualization

## Usage

```python
from ui.qt_gui.widgets.timeline import TimelineWidget

timeline = TimelineWidget()
timeline.set_events(events)
timeline.set_layers(layers)
```

## Related

- [Encyclopedia: Timeline](../../../../docs/encyclopedia/04-ui/timeline/README.md)
