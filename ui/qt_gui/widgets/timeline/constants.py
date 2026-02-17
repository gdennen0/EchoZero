"""
Timeline Constants

Central location for timeline dimensions and timing constants.
Colors are defined in style.py for centralized styling.
"""

# =============================================================================
# Dimensions
# =============================================================================

DEFAULT_PIXELS_PER_SECOND = 100  # Default scale level
MIN_PIXELS_PER_SECOND = 10  # Minimum zoom (very zoomed out)
MAX_PIXELS_PER_SECOND = 2000  # Maximum zoom (very zoomed in)
ZOOM_FACTOR = 1.15  # Zoom increment per scroll step (DAW-standard)

# Smooth zoom sensitivity constants (POC-verified)
# These control how sensitive zoom is to scroll input
PIXEL_ZOOM_SENSITIVITY = 0.002   # For trackpads (pixelDelta) - higher = more sensitive
ANGLE_ZOOM_SENSITIVITY = 0.001   # For mice (angleDelta) - lower = smoother
ZOOM_ACCUMULATOR_THRESHOLD = 0.005  # Minimum accumulated delta before applying zoom

RULER_HEIGHT = 30
TRACK_HEIGHT = 40
TRACK_SPACING = 4
EVENT_HEIGHT = 32
MARKER_WIDTH = 13
RESIZE_HANDLE_WIDTH = 6  # Default resize handle width (pixels)
MIN_RESIZE_HANDLE_WIDTH = 3  # Minimum resize handle width for small events
MIN_MOVE_AREA_WIDTH = 8  # Minimum width of move area in center of event
RESIZE_HANDLE_PERCENT = 0.15  # Percentage of event width for resize handles (for small events)

# Playhead
PLAYHEAD_WIDTH = 1
PLAYHEAD_HEAD_SIZE = 10

# =============================================================================
# Timing
# =============================================================================

# Target 60 FPS for smooth playhead animation
PLAYHEAD_UPDATE_INTERVAL_MS = 16  # ~60 FPS (1000ms / 60 = 16.67ms)

# =============================================================================
# Snap Grid
# =============================================================================

# Default snap intervals in seconds
DEFAULT_SNAP_INTERVALS = [
    0.001,   # 1ms
    0.01,    # 10ms
    0.1,     # 100ms
    0.5,     # 500ms
    1.0,     # 1 second
    5.0,     # 5 seconds
    10.0,    # 10 seconds
    30.0,    # 30 seconds
    60.0,    # 1 minute
]

# =============================================================================
# Grid Rendering (POC-verified)
# =============================================================================

# Minimum pixel spacing for grid lines (adaptive density)
# Lines closer than this will be skipped for readability
MIN_MINOR_LINE_SPACING_PX = 12   # Minimum spacing between minor grid lines
MIN_MAJOR_LINE_SPACING_PX = 80   # Minimum spacing between major grid lines (with labels)

# Maximum number of grid lines to draw (performance limit)
MAX_GRID_LINES = 500
