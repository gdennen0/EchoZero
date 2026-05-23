"""
FEEL.py: Human-tunable constants for all UI behavior and interaction.
Exists because craft lives in constants, not code — Griff tunes these, agents build the machinery.
Referenced by all UI rendering, interaction, and animation code. No magic numbers elsewhere.
"""

# =============================================================================
# SNAP BEHAVIOR
# =============================================================================

SNAP_MAGNETISM_RADIUS_PX = 12  # How close (px) before snap activates
SNAP_EASE_DURATION_MS = 60  # How long the snap animation takes
SNAP_EASE_CURVE = "ease_out_quad"  # Easing function name
SNAP_SHOW_INDICATOR = True  # Show snap target indicator before committing

# =============================================================================
# SCROLL & ZOOM
# =============================================================================

SCROLL_INERTIA_DECAY = 0.92  # Per-frame multiplier (0.9 = fast stop, 0.99 = long glide)
SCROLL_MIN_VELOCITY_PX = 0.5  # Stop inertia below this speed
ZOOM_ANCHOR_TO_CURSOR = True  # Zoom keeps the point under cursor fixed
ZOOM_SPEED_MULTIPLIER = 1.0  # Scale factor for zoom sensitivity
ZOOM_MIN_SCALE = 0.001  # Minimum zoom level (full song overview)
ZOOM_MAX_SCALE = 100.0  # Maximum zoom level (individual samples)
TIMELINE_ZOOM_STEP_FACTOR = 1.12  # Per-notch timeline zoom multiplier
TIMELINE_ZOOM_MIN_PPS = 5.0  # Minimum timeline zoom clamp in pixels/second
TIMELINE_ZOOM_MAX_PPS = 720.0  # Maximum timeline zoom clamp in pixels/second
TIMELINE_RUNTIME_TICK_ACTIVE_MS = 16  # Runtime timer cadence while transport is actively moving
TIMELINE_RUNTIME_TICK_TEXT_INPUT_MS = (
    48  # Runtime cadence while transport moves and the operator is typing
)
TIMELINE_RUNTIME_TICK_IDLE_MS = 64  # Runtime timer cadence while idle (reduced overhead)
TIMELINE_MIX_SYNC_DEBOUNCE_MS = (
    24  # Coalesce non-structural runtime mix sync off the action hot path
)
TIMELINE_STRUCTURAL_SYNC_DEBOUNCE_MS = (
    120  # Coalesce structural playback rebuilds while editing during playback
)

# =============================================================================
# PLAYHEAD
# =============================================================================

PLAYHEAD_WIDTH_PX = 1  # Rendered width
PLAYHEAD_COLOR = "#93A0B1"  # Base color (theme can override)
PLAYHEAD_HEAD_HEIGHT_PX = 10  # Triangle height at top of ruler
PLAYHEAD_HEAD_WIDTH_PX = 8  # Triangle width at top of ruler

# =============================================================================
# WAVEFORM
# =============================================================================

WAVEFORM_LOD_SAMPLE_PX_THRESHOLD = 4.0  # Above this px/sample: render individual samples
WAVEFORM_LOD_ENVELOPE_PX_THRESHOLD = 0.5  # Below this: render overview thumbnail
WAVEFORM_COLOR = "#8f8a84"  # Base waveform color
WAVEFORM_RMS_COLOR = "#454145"  # RMS fill color (darker)
WAVEFORM_ANTIALIAS_OUTLINE = True  # Anti-alias envelope outline
WAVEFORM_ANTIALIAS_FILL = False  # Don't anti-alias fill (performance)
WAVEFORM_COLUMN_STEP_REFERENCE_PPS = (
    20.0  # Below this zoom level, bucket waveform columns more aggressively
)
WAVEFORM_COLUMN_STEP_MAX_PX = 4  # Maximum horizontal bucket size when heavily zoomed out
NOTE_CONTOUR_ROW_PADDING_PX = 8  # Vertical padding for note contour overlays in audio lanes
NOTE_CONTOUR_PEN_WIDTH_PX = 2.0  # Stroke width for note contour overlays
NOTE_CONTOUR_ALPHA = 205  # Opacity for note contour overlays

# =============================================================================
# EVENTS (Timeline)
# =============================================================================

EVENT_DEFAULT_COLOR = "#8f3a2f"  # Unclassified event color (bright coral red - visible on dark bg)
EVENT_HOVER_ALPHA = 30  # Overlay alpha on hover (0-255)
EVENT_SELECTION_COLOR = "#CC8844"  # Selected event highlight
EVENT_SELECTION_BORDER_PX = 2  # Selection border width
EVENT_SELECTION_OUTLINE_EXPAND_PX = 1.0  # Expand selection outline outside event bounds
EVENT_SELECTION_TINY_WIDTH_THRESHOLD_PX = (
    10.0  # Treat selected events narrower than this as zoomed-out
)
EVENT_SELECTION_TINY_WIDTH_EXTRA_PX = 1  # Extra outline width for tiny selected events
EVENT_MIN_VISIBLE_WIDTH_PX = 4  # Minimum rendered width (even for zero-duration)
EVENT_MIN_HIT_WIDTH_PX = 10  # Minimum clickable width for zoomed-out events
EVENT_LABEL_MIN_WIDTH_PX = 40  # Don't render labels on events narrower than this
EVENT_FALSE_BORDER_STYLE = "dash"  # Visual for false/uncertain events ("dash" or "dot")
TIMELINE_ADD_MODE_DEFAULT_EVENT_DURATION_SECONDS = 0.5  # Draw-mode quick-add event length

# =============================================================================
# INTERACTION
# =============================================================================

RESIZE_HANDLE_WIDTH_PX = 6  # Hit zone for edge resize cursor
DRAG_THRESHOLD_PX = 4  # Minimum drag distance before drag starts
DOUBLE_CLICK_MS = 400  # Double-click time window
HOVER_DELAY_MS = 150  # Delay before hover effects appear
SECTION_MOVE_EVENT_HIT_MIN_WIDTH_PX = 12  # Minimum section cue hit width while moving cues
MOVE_DRAG_PREVIEW_LINE_WIDTH_PX = 2  # Width for the move-mode vertical drag preview bar
MOVE_DRAG_PREVIEW_LINE_ALPHA = 180  # Opacity for the move-mode vertical drag preview bar
MOVE_DRAG_SNAP_LOCK_MULTIPLIER = (
    1.75  # Sticky-radius multiplier to avoid snap jitter while dragging
)

# =============================================================================
# TIME RULER
# =============================================================================

TIMELINE_TRANSPORT_HEIGHT_PX = 46  # Playback strip height below the timeline viewport
TIMELINE_TRANSPORT_BUTTON_HEIGHT_PX = 26  # Painted play/stop button height in transport
TIMELINE_TRANSPORT_TOP_GAP_PX = 1  # Vertical gap between timeline scroller and transport strip
TIMELINE_EDITOR_BAR_PADDING_X_PX = 4  # Horizontal padding around the editor toolbar strip
TIMELINE_EDITOR_BAR_PADDING_Y_PX = 1  # Vertical padding around the editor toolbar strip
TIMELINE_EDITOR_GROUP_PADDING_X_PX = 2  # Horizontal padding inside each toolbar group
TIMELINE_EDITOR_GROUP_PADDING_Y_PX = 1  # Vertical padding inside each toolbar group
TIMELINE_EDITOR_GROUP_SPACING_PX = 3  # Gap between toolbar labels and controls
TIMELINE_EDITOR_BUTTON_MIN_HEIGHT_PX = 18  # Dense minimum button height in the editor toolbar
TIMELINE_LAUNCHER_LOGO_HEIGHT_PX = 22  # Display height for the EZ logo in the menu bar
TIMELINE_LAUNCHER_LOGO_CONTAINER_HEIGHT_PX = 30  # Menu-bar logo hit/visual height
TIMELINE_RULER_TOP_GAP_PX = 3  # Small breathing room between editor toolbar and ruler/title bar
TIMELINE_LAUNCHER_SUBMENU_ARROW_SIZE_PX = 6  # Compact submenu chevron size in launcher menus
TIMELINE_LAUNCHER_SUBMENU_ARROW_RIGHT_PADDING_PX = 8  # Right gutter for launcher submenu chevrons
TIMELINE_OBJECT_INFO_METADATA_MIN_HEIGHT_PX = (
    72  # Compact starting body height for inspector facts
)
TIMELINE_OBJECT_INFO_METADATA_DEFAULT_HEIGHT_PX = 143  # Initial inspector metadata pane height
TIMELINE_OBJECT_INFO_SPLITTER_HANDLE_PX = 4  # Grab area for resizing inspector metadata vs actions
RULER_HEIGHT_PX = 20  # Ruler bar height
RULER_MIN_TICK_SPACING_PX = 60  # Minimum pixels between major ticks
RULER_MINOR_TICKS_PER_MAJOR = 4  # Subdivision density
RULER_FONT_SIZE = 10  # Tick label font size
RULER_TICK_COLOR = "#685f67"  # Tick mark color

# =============================================================================
# LAYERS
# =============================================================================

LAYER_HEADER_WIDTH_PX = 300  # Default width of layer name sidebar
LAYER_HEADER_MIN_WIDTH_PX = 220  # Minimum layer name sidebar width
LAYER_HEADER_MAX_WIDTH_PX = 560  # Maximum layer name sidebar width
LAYER_HEADER_RESIZE_HANDLE_HALF_WIDTH_PX = (
    6  # Horizontal hit radius around header divider drag target
)
LAYER_ROW_HEIGHT_PX = 40  # Fallback default height per main layer row
TAKE_ROW_HEIGHT_PX = 36  # Height of subordinate take rows
LAYER_HEADER_TOP_PADDING_PX = 4  # Top offset before first lane row
EVENT_BAR_HEIGHT_PX = 18  # Event pill/bar height inside rows
TIMELINE_RIGHT_PADDING_PX = 240  # Extra right scroll padding beyond content span
LAYER_ROW_MIN_HEIGHT_PX = 40  # Fallback minimum for resizable main rows
LAYER_SEPARATOR_PX = 1  # Divider line between layers
LAYER_ACTIVE_HIGHLIGHT_ALPHA = 15  # Subtle highlight on focused layer

# =============================================================================
# GRID
# =============================================================================

GRID_LINE_COLOR = "#3a383a"  # Grid line color (pixel-snapped, no anti-alias)
GRID_LINE_ALPHA = 40  # Grid line opacity (0-255)
GRID_BEAT_LINE_ALPHA = 80  # Beat-aligned grid lines (stronger)
GRID_BAR_LINE_ALPHA = 120  # Bar-aligned grid lines (strongest)

# =============================================================================
# GRAPH VIEW (Node Editor)
# =============================================================================

BLOCK_WIDTH_PX = 200  # Default block rectangle width
BLOCK_HEIGHT_PX = 80  # Default block rectangle height
BLOCK_CORNER_RADIUS_PX = 8  # Rounded corners
BLOCK_PORT_RADIUS_PX = 6  # Port circle radius
BLOCK_PORT_SPACING_PX = 20  # Vertical spacing between ports
BLOCK_GRID_SNAP_PX = 20  # Grid snap for block placement
CONNECTION_BEZIER_TENSION = 0.5  # Bezier curve control point tension (0 = straight, 1 = extreme)

# Port type colors
PORT_COLOR_AUDIO = "#8f8a84"  # Blue
PORT_COLOR_EVENT = "#7fd1ae"  # Green
PORT_COLOR_OSC = "#ffb84d"  # Orange
PORT_COLOR_CONTROL = "#8f8a84"  # Purple

# Block category colors
BLOCK_COLOR_PROCESSOR = "#3d3b3d"  # Blue-gray
BLOCK_COLOR_WORKSPACE = "#2b2117"  # Amber
BLOCK_COLOR_PLAYBACK = "#1c3428"  # Green

# =============================================================================
# ANIMATION
# =============================================================================

UNDO_FLASH_DURATION_MS = 200  # Flash duration on undo/redo affected events
UNDO_FLASH_COLOR = "#ffb84d"  # Flash color
EXECUTION_PROGRESS_COLOR = "#7fd1ae"  # Progress bar color during execution

# =============================================================================
# CLASSIFICATION COLORS
# =============================================================================
# Map classification labels to display colors. Extend as models are added.

CLASSIFICATION_COLORS = {
    "kick": "#a6533e",
    "snare": "#91a19a",
    "hihat": "#b49a63",
    "tom": "#a07a68",
    "cymbal": "#9a8f83",
    "clap": "#9a8f83",
    "percussion": "#8f8a84",
    "vocal": "#b56f61",
    "bass": "#7f9a72",
    "melody": "#86a0ad",
    "other": "#8f8a84",
}
