# FEEL.py â€” Visual constants for EchoZero Timeline Prototype
# Tweak these to adjust the feel. Everything visual lives here.

from PyQt6.QtGui import QColor

# â”€â”€â”€ Background & Grid â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BG_COLOR                = QColor(22, 22, 26)
GRID_MINOR_COLOR        = QColor(38, 38, 46)
GRID_MAJOR_COLOR        = QColor(52, 52, 62)
GRID_MINOR_WIDTH        = 1
GRID_MAJOR_WIDTH        = 1

# â”€â”€â”€ Ruler â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RULER_HEIGHT            = 32
RULER_BG_COLOR          = QColor(18, 18, 22)
RULER_BORDER_COLOR      = QColor(60, 60, 75)
RULER_TICK_MAJOR_COLOR  = QColor(140, 140, 160)
RULER_TICK_MINOR_COLOR  = QColor(75, 75, 90)
RULER_LABEL_COLOR       = QColor(170, 170, 190)
RULER_TICK_MAJOR_HEIGHT = 12
RULER_TICK_MINOR_HEIGHT = 6
RULER_FONT_SIZE         = 10
RULER_FONT_FAMILY       = "Consolas"

# â”€â”€â”€ Playhead â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PLAYHEAD_COLOR          = QColor(255, 80, 80)
PLAYHEAD_WIDTH          = 2
PLAYHEAD_TRIANGLE_SIZE  = 8

# â”€â”€â”€ Layers Panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LAYERS_PANEL_WIDTH      = 140
LAYERS_PANEL_BG_COLOR   = QColor(20, 20, 24)
LAYERS_PANEL_BORDER_COLOR = QColor(50, 50, 60)
LAYER_LABEL_COLOR       = QColor(190, 190, 210)
LAYER_SELECTED_BG       = QColor(40, 40, 55)
LAYER_HOVER_BG          = QColor(32, 32, 42)
LAYER_FONT_SIZE         = 11
LAYER_FONT_FAMILY       = "Segoe UI"
LAYER_SWATCH_WIDTH      = 6
LAYER_SWATCH_MARGIN     = 6
LAYER_TEXT_PADDING      = 8

# â”€â”€â”€ Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
EVENT_HEIGHT            = 28          # default event block height in pixels
EVENT_RADIUS            = 4           # corner radius
EVENT_ALPHA             = 180         # background alpha (0-255)
EVENT_BORDER_ALPHA      = 220
EVENT_BORDER_WIDTH      = 1
EVENT_SELECTED_BORDER_COLOR = QColor(255, 220, 80)
EVENT_SELECTED_BORDER_WIDTH = 2
EVENT_LABEL_COLOR       = QColor(240, 240, 255)
EVENT_LABEL_ALPHA       = 230
EVENT_FONT_SIZE         = 10
EVENT_FONT_FAMILY       = "Segoe UI"
EVENT_MIN_LABEL_WIDTH   = 30          # min px width before hiding label
EVENT_VERTICAL_PADDING  = 3          # space above/below event within layer row

# â”€â”€â”€ Selection Overlay â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SELECTION_RECT_COLOR    = QColor(100, 160, 255, 40)
SELECTION_RECT_BORDER   = QColor(100, 160, 255, 140)

# â”€â”€â”€ Snap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SNAP_INDICATOR_COLOR    = QColor(255, 220, 80, 180)
SNAP_INDICATOR_WIDTH    = 2
SNAP_THRESHOLD_PX       = 8           # pixels within which snap activates
SNAP_GRID_SECONDS       = 0.25        # snap grid resolution in seconds

# â”€â”€â”€ Section Regions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SECTION_ALPHA           = 25          # very subtle background bands

# â”€â”€â”€ Zoom â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ZOOM_MIN                = 10.0        # px per second (zoomed out)
ZOOM_MAX                = 2000.0      # px per second (zoomed in)
ZOOM_STEP               = 0.12        # zoom factor per wheel tick (fraction)

# â”€â”€â”€ Scroll â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LAYER_ROW_HEIGHT        = 40          # pixels per layer row (matches EVENT_HEIGHT + padding)

