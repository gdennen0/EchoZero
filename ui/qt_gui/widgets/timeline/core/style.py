"""
Timeline Style Configuration

Style constants for the timeline widget. Shared tokens (backgrounds, text,
borders, accents) are synced from the global design system whenever the
theme changes via ``TimelineStyle.apply_theme()``.

Timeline-specific tokens (playhead, selection, snap, grid, layer colors)
remain independently configurable.
"""

from PyQt6.QtGui import QColor, QFont


class TimelineStyle:
    """
    Style configuration for timeline widgets.
    
    Shared tokens are synced from the global ``Colors`` via ``apply_theme()``.
    Timeline-specific tokens are defined here and can be overridden directly.
    """
    
    # =========================================================================
    # Background Colors (synced from global theme)
    # =========================================================================
    BG_COLOR = QColor(30, 30, 35)
    BG_DARK = QColor(30, 30, 35)      # Alias for compatibility
    BG_MEDIUM = QColor(45, 45, 50)
    BG_LIGHT = QColor(60, 60, 65)
    BG_LIGHTER = QColor(45, 45, 50)   # Alias
    BG_LIGHTEST = QColor(60, 60, 65)  # Alias
    
    # =========================================================================
    # Text Colors (synced from global theme)
    # =========================================================================
    TEXT_PRIMARY = QColor(240, 240, 245)
    TEXT_SECONDARY = QColor(180, 180, 185)
    TEXT_DISABLED = QColor(120, 120, 125)
    
    # =========================================================================
    # UI Element Colors (synced from global theme)
    # =========================================================================
    BORDER = QColor(80, 80, 85)
    HOVER = QColor(70, 70, 75)
    SELECTED = QColor(90, 90, 95)
    
    # =========================================================================
    # Accent Colors (synced from global theme)
    # =========================================================================
    ACCENT_BLUE = QColor(70, 130, 220)
    ACCENT_GREEN = QColor(80, 180, 120)
    ACCENT_RED = QColor(220, 80, 80)
    ACCENT_YELLOW = QColor(220, 180, 60)
    
    @classmethod
    def apply_theme(cls):
        """
        Sync shared tokens from the global design system Colors.
        
        Called automatically by ``Colors.apply_theme()`` after the global
        theme is updated, so timeline widgets always reflect the active theme.
        """
        from ui.qt_gui.design_system import Colors as GlobalColors
        
        # Backgrounds
        cls.BG_COLOR = GlobalColors.BG_DARK
        cls.BG_DARK = GlobalColors.BG_DARK
        cls.BG_MEDIUM = GlobalColors.BG_MEDIUM
        cls.BG_LIGHT = GlobalColors.BG_LIGHT
        cls.BG_LIGHTER = GlobalColors.BG_MEDIUM
        cls.BG_LIGHTEST = GlobalColors.BG_LIGHT
        
        # Text
        cls.TEXT_PRIMARY = GlobalColors.TEXT_PRIMARY
        cls.TEXT_SECONDARY = GlobalColors.TEXT_SECONDARY
        cls.TEXT_DISABLED = GlobalColors.TEXT_DISABLED
        
        # UI elements
        cls.BORDER = GlobalColors.BORDER
        cls.HOVER = GlobalColors.HOVER
        cls.SELECTED = GlobalColors.SELECTED
        
        # Accents
        cls.ACCENT_BLUE = GlobalColors.ACCENT_BLUE
        cls.ACCENT_GREEN = GlobalColors.ACCENT_GREEN
        cls.ACCENT_RED = GlobalColors.ACCENT_RED
        cls.ACCENT_YELLOW = GlobalColors.ACCENT_YELLOW
    
    # =========================================================================
    # Timeline-Specific Colors
    # =========================================================================
    PLAYHEAD_COLOR = QColor(150, 150, 150)  # Grey
    SELECTION_COLOR = QColor(255, 200, 60)  # Yellow/gold
    SNAP_LINE_COLOR = QColor(100, 180, 255, 100)  # Light blue, semi-transparent
    GRID_LINE_MAJOR = QColor(80, 80, 85)
    GRID_LINE_MINOR = QColor(50, 50, 55)
    
    # =========================================================================
    # Layer/Track Colors (cycles through for multiple layers)
    # =========================================================================
    LAYER_COLORS = [
        QColor(70, 130, 220),    # Blue
        QColor(80, 180, 120),    # Green
        QColor(220, 140, 70),    # Orange
        QColor(180, 120, 200),   # Purple
        QColor(220, 80, 140),    # Magenta
        QColor(100, 200, 200),   # Cyan
        QColor(200, 200, 80),    # Yellow
        QColor(200, 100, 100),   # Red
        QColor(150, 150, 180),   # Slate
        QColor(180, 140, 100),   # Brown
    ]
    
    @classmethod
    def get_layer_color(cls, index: int) -> QColor:
        """Get color for a layer by index (cycles through palette)"""
        return cls.LAYER_COLORS[index % len(cls.LAYER_COLORS)]
    
    # =========================================================================
    # Fonts
    # =========================================================================
    @staticmethod
    def default_font() -> QFont:
        """Default font for timeline text"""
        font = QFont()
        font.setFamily("SF Pro Text, Segoe UI, -apple-system, system-ui, sans-serif")
        font.setPixelSize(13)
        return font
    
    @staticmethod
    def small_font() -> QFont:
        """Small font for labels and timestamps"""
        font = QFont()
        font.setFamily("SF Pro Text, Segoe UI, -apple-system, system-ui, sans-serif")
        font.setPixelSize(10)
        return font
    
    @staticmethod
    def monospace_font() -> QFont:
        """Monospace font for timecode display"""
        font = QFont()
        font.setFamily("SF Mono, Consolas, Monaco, monospace")
        font.setPixelSize(11)
        return font
    
    @staticmethod
    def mono_font() -> QFont:
        """Alias for monospace_font"""
        return TimelineStyle.monospace_font()
    
    # =========================================================================
    # Spacing
    # =========================================================================
    SPACING_XS = 4
    SPACING_SM = 8
    SPACING_MD = 16
    SPACING_LG = 24
    
    # Aliases for compatibility
    XS = 4
    SM = 8
    MD = 16
    LG = 24
    XL = 32


# Convenience aliases for backward compatibility
Colors = TimelineStyle
Spacing = TimelineStyle
Typography = TimelineStyle

