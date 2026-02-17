"""
Timeline Widget Package
=======================

A professional DAW-style timeline widget for PyQt6 applications.
Designed to be **standalone and reusable** - can be dropped into any PyQt6 project.

Directory Structure
-------------------
- core/       - Main widget components (TimelineWidget, TimelineScene, TimelineView, style)
- events/     - Event handling (items, layers, movement, inspector)
- timing/     - Grid and snap calculations
- playback/   - Playback control and playhead
- settings/   - Settings UI and storage

Import Examples
---------------
    from ui.qt_gui.widgets.timeline.core import TimelineWidget, TimelineScene
    from ui.qt_gui.widgets.timeline.events import LayerManager, BlockEventItem
    from ui.qt_gui.widgets.timeline.timing import GridCalculator, TimebaseMode
    from ui.qt_gui.widgets.timeline.playback import PlaybackController, PlayheadItem
    from ui.qt_gui.widgets.timeline.settings import SettingsPanel
    from ui.qt_gui.widgets.timeline.types import TimelineEvent, TimelineLayer
    from ui.qt_gui.widgets.timeline.interfaces import PlaybackInterface
    from ui.qt_gui.widgets.timeline.grid_system import GridSystem

Features
--------
- Multiple event layers/tracks with independent layer management
- Block events (with duration) and marker events (instant)
- 60 FPS playhead synchronized with audio playback
- Event editing: create, move (time + layer), resize, delete
- Multi-select with Shift+Click, rubber-band, Ctrl+A
- Snap-to-grid with configurable intervals
- Configurable time display (timecode, seconds, frames)
- Zoom/pan navigation (Ctrl+scroll, middle-click drag)
- Event Inspector panel (right side, resizable)
"""

__version__ = "2.1.0"
