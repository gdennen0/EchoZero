"""
Design System

Centralized design tokens for the entire UI.
Colors, spacing, typography, and visual constants.
"""
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from typing import Optional


# =============================================================================
# Theme Signals
# =============================================================================

class ThemeSignals(QObject):
    """Emits when the global theme changes so panels can refresh."""
    theme_changed = pyqtSignal()


_theme_signals: ThemeSignals = None  # Lazily created (needs QApplication)


def _get_theme_signals() -> ThemeSignals:
    """Get or create the singleton ThemeSignals instance."""
    global _theme_signals
    if _theme_signals is None:
        _theme_signals = ThemeSignals()
    return _theme_signals


def on_theme_changed(callback) -> None:
    """
    Connect a callback to the global theme_changed signal.
    
    Usage:
        from ui.qt_gui.design_system import on_theme_changed
        on_theme_changed(self._refresh_styling)
    """
    _get_theme_signals().theme_changed.connect(callback)


def disconnect_theme_changed(callback) -> None:
    """Disconnect a callback from the global theme_changed signal."""
    try:
        _get_theme_signals().theme_changed.disconnect(callback)
    except TypeError:
        pass  # Not connected


# =============================================================================
# Theme-Aware Mixin
# =============================================================================

class ThemeAwareMixin:
    """Mixin for QWidget / QDialog subclasses that need instant theme updates.
    
    Automatically clears stale child stylesheets on theme change so the
    global ``QApplication`` stylesheet takes effect, then calls the
    overridable ``_apply_local_styles()`` hook for variant overrides.
    
    Usage::
    
        class MyDialog(ThemeAwareMixin, QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self._init_theme_aware()   # <-- call after UI setup
            
            def _apply_local_styles(self):
                # Re-apply any variant-specific overrides here
                self.my_primary_btn.setStyleSheet(StyleFactory.button("primary"))
    """
    
    def _init_theme_aware(self):
        """Call once after UI is fully built to subscribe to theme changes."""
        on_theme_changed(self._on_theme_changed_mixin)
    
    def _on_theme_changed_mixin(self):
        """Clear stale child stylesheets, then let subclass re-apply overrides."""
        from PyQt6.QtWidgets import QWidget
        for child in self.findChildren(QWidget):
            if child.styleSheet():
                child.setStyleSheet("")
        # Clear own stylesheet too so global can cascade
        if hasattr(self, 'setStyleSheet') and self.styleSheet():
            self.setStyleSheet("")
        self._apply_local_styles()
    
    def _apply_local_styles(self):
        """Override to re-apply variant/context-specific styles after theme change."""
        pass
    
    def closeEvent(self, event):
        """Disconnect theme signal to prevent callbacks on deleted widgets."""
        try:
            disconnect_theme_changed(self._on_theme_changed_mixin)
        except (TypeError, RuntimeError):
            pass
        super().closeEvent(event)


# =============================================================================
# Sharp Corners Global Setting
# =============================================================================

_sharp_corners: bool = False


def set_sharp_corners(enabled: bool) -> None:
    """Set the global sharp corners preference. When True, all border-radius values become 0."""
    global _sharp_corners
    _sharp_corners = enabled
    # Also update the Sizes constant for QPainter-based corner radius
    Sizes.BLOCK_CORNER_RADIUS = 0 if enabled else Sizes._DEFAULT_BLOCK_CORNER_RADIUS


def is_sharp_corners() -> bool:
    """Check if sharp corners mode is currently active."""
    return _sharp_corners


def border_radius(px: int = 4) -> str:
    """
    Return border-radius CSS value respecting the global sharp corners setting.
    
    Args:
        px: The default border-radius in pixels when sharp corners is disabled.
        
    Returns:
        '0px' when sharp corners is enabled, otherwise '{px}px'.
    """
    return "0px" if _sharp_corners else f"{px}px"


class Colors:
    """Color palette for the application"""
    
    # Current theme (can be overridden)
    _current_theme = None
    
    # Background colors (refined for modern look)
    BG_DARK = QColor(28, 28, 32)  # Slightly darker for better contrast
    BG_MEDIUM = QColor(42, 42, 47)  # Refined for depth
    BG_LIGHT = QColor(56, 56, 62)  # Subtle highlight
    
    # UI element colors (enhanced)
    BORDER = QColor(75, 75, 80)  # Softer border
    HOVER = QColor(65, 65, 70)  # Refined hover state
    SELECTED = QColor(85, 85, 90)  # Better selection feedback
    
    # Text colors
    TEXT_PRIMARY = QColor(240, 240, 245)
    TEXT_SECONDARY = QColor(180, 180, 185)
    TEXT_DISABLED = QColor(120, 120, 125)
    
    # Accent colors
    ACCENT_BLUE = QColor(70, 130, 220)
    ACCENT_GREEN = QColor(80, 180, 120)
    ACCENT_RED = QColor(220, 80, 80)
    ACCENT_YELLOW = QColor(220, 180, 60)
    ACCENT_ORANGE = QColor(220, 135, 65)
    ACCENT_PURPLE = QColor(175, 115, 195)
    
    # Semantic status colors (derived from accents, overridable per theme)
    STATUS_SUCCESS = QColor(80, 180, 120)   # = ACCENT_GREEN
    STATUS_WARNING = QColor(220, 180, 60)   # = ACCENT_YELLOW
    STATUS_ERROR = QColor(220, 80, 80)      # = ACCENT_RED
    STATUS_INFO = QColor(70, 130, 220)      # = ACCENT_BLUE
    STATUS_INACTIVE = QColor(120, 120, 125) # = TEXT_DISABLED
    
    # Danger action colors (for destructive buttons / warnings)
    DANGER_BG = QColor(58, 32, 32)
    DANGER_FG = QColor(255, 107, 107)
    
    # Block type colors (refined, modern palette)
    BLOCK_LOAD = QColor(65, 125, 210)      # Rich blue - input
    BLOCK_ANALYZE = QColor(110, 195, 115)   # Fresh green - analysis
    BLOCK_TRANSFORM = QColor(215, 135, 65)  # Vibrant orange - transform
    BLOCK_EXPORT = QColor(95, 195, 195)    # Clean cyan - output
    BLOCK_EDITOR = QColor(175, 115, 195)    # Elegant purple - editing
    BLOCK_VISUALIZE = QColor(215, 75, 135)  # Bright magenta - viz
    BLOCK_UTILITY = QColor(135, 135, 140)   # Neutral gray - utility
    BLOCK_PLAYER = QColor(60, 180, 170)     # Teal - audio player/preview
    
    # Connection/line colors
    CONNECTION_NORMAL = QColor(120, 120, 125)
    CONNECTION_HOVER = QColor(180, 180, 185)
    CONNECTION_SELECTED = QColor(220, 180, 60)
    
    # Port colors by type (refined for modern aesthetics)
    PORT_INPUT = QColor(100, 200, 100)
    PORT_OUTPUT = QColor(200, 100, 100)
    PORT_AUDIO = QColor(70, 170, 220)      # Vibrant blue for audio
    PORT_EVENT = QColor(230, 150, 70)      # Warm orange for events
    PORT_MANIPULATOR = QColor(255, 140, 0) # Bright orange for manipulator (bidirectional)
    PORT_GENERIC = QColor(150, 150, 155)   # Refined gray for unknown
    
    # Port label colors
    PORT_LABEL_TEXT = QColor(180, 180, 185)
    PORT_TYPE_TEXT = QColor(140, 140, 145)
    
    @classmethod
    def get_port_color(cls, port_type_name: str) -> QColor:
        """Get color for a port based on its type"""
        type_lower = port_type_name.lower() if port_type_name else ""
        if "audio" in type_lower:
            return cls.PORT_AUDIO
        elif "event" in type_lower:
            return cls.PORT_EVENT
        elif "manipulator" in type_lower:
            return cls.PORT_MANIPULATOR
        return cls.PORT_GENERIC
    
    @classmethod
    def is_manipulator_port(cls, port_type_name: str) -> bool:
        """Check if port type is a manipulator (bidirectional command) port."""
        type_lower = port_type_name.lower() if port_type_name else ""
        return "manipulator" in type_lower
    
    @classmethod
    def get_block_color(cls, block_type: str) -> QColor:
        """Get color for a specific block type"""
        color_map = {
            'LoadAudio': cls.BLOCK_LOAD,
            'ExportAudio': cls.BLOCK_EXPORT,
            'DetectOnsets': cls.BLOCK_ANALYZE,
            'TranscribeNote': cls.BLOCK_ANALYZE,
            'TranscribeLib': cls.BLOCK_ANALYZE,
            'Separator': cls.BLOCK_TRANSFORM,
            'Editor': cls.BLOCK_EDITOR,
            'EditorV2': cls.BLOCK_EDITOR,
            'CommandSequencer': cls.BLOCK_UTILITY,
            'AudioPlayer': cls.BLOCK_PLAYER,
            'AudioFilter': cls.BLOCK_TRANSFORM,
        }
        return color_map.get(block_type, cls.BLOCK_UTILITY)
    
    @classmethod
    def apply_theme(cls, theme_name: Optional[str] = None):
        """
        Apply a theme to the Colors class.
        
        Also reads the sharp_corners setting and applies it globally.
        
        Args:
            theme_name: Name of the theme to apply (case-insensitive).
                       If None, uses the default theme from settings.
        """
        from ui.qt_gui.theme_registry import ThemeRegistry
        
        if theme_name is None:
            # Try to get from QApplication's property (set by MainWindow)
            try:
                from PyQt6.QtWidgets import QApplication
                app = QApplication.instance()
                if app and hasattr(app, 'property') and app.property('app_settings'):
                    settings_manager = app.property('app_settings')
                    theme_name = settings_manager.theme_preset
                    # Also sync sharp_corners setting
                    set_sharp_corners(settings_manager.sharp_corners)
                else:
                    theme_name = "default dark"
            except Exception:
                # Settings manager might not be initialized yet
                theme_name = "default dark"
        
        theme = ThemeRegistry.get_theme(theme_name)
        if not theme:
            # Fallback to default dark if theme not found
            theme = ThemeRegistry.get_theme("default dark")
            if not theme:
                return  # Can't apply theme
        
        cls._current_theme = theme
        
        # Update all color attributes
        cls.BG_DARK = theme.bg_dark
        cls.BG_MEDIUM = theme.bg_medium
        cls.BG_LIGHT = theme.bg_light
        cls.BORDER = theme.border
        cls.HOVER = theme.hover
        cls.SELECTED = theme.selected
        cls.TEXT_PRIMARY = theme.text_primary
        cls.TEXT_SECONDARY = theme.text_secondary
        cls.TEXT_DISABLED = theme.text_disabled
        cls.ACCENT_BLUE = theme.accent_blue
        cls.ACCENT_GREEN = theme.accent_green
        cls.ACCENT_RED = theme.accent_red
        cls.ACCENT_YELLOW = theme.accent_yellow
        cls.BLOCK_LOAD = theme.block_load
        cls.BLOCK_ANALYZE = theme.block_analyze
        cls.BLOCK_TRANSFORM = theme.block_transform
        cls.BLOCK_EXPORT = theme.block_export
        cls.BLOCK_EDITOR = theme.block_editor
        cls.BLOCK_VISUALIZE = theme.block_visualize
        cls.BLOCK_UTILITY = theme.block_utility
        cls.CONNECTION_NORMAL = theme.connection_normal
        cls.CONNECTION_HOVER = theme.connection_hover
        cls.CONNECTION_SELECTED = theme.connection_selected
        cls.PORT_INPUT = theme.port_input
        cls.PORT_OUTPUT = theme.port_output
        cls.PORT_AUDIO = theme.port_audio
        cls.PORT_EVENT = theme.port_event
        cls.PORT_GENERIC = theme.port_generic
        
        # Accent colors that may not be in all themes
        cls.ACCENT_ORANGE = getattr(theme, 'accent_orange', None) or QColor(220, 135, 65)
        cls.ACCENT_PURPLE = getattr(theme, 'accent_purple', None) or QColor(175, 115, 195)
        
        # Semantic status colors (use theme overrides if present, else derive)
        cls.STATUS_SUCCESS = getattr(theme, 'status_success', None) or cls.ACCENT_GREEN
        cls.STATUS_WARNING = getattr(theme, 'status_warning', None) or cls.ACCENT_YELLOW
        cls.STATUS_ERROR = getattr(theme, 'status_error', None) or cls.ACCENT_RED
        cls.STATUS_INFO = getattr(theme, 'status_info', None) or cls.ACCENT_BLUE
        cls.STATUS_INACTIVE = getattr(theme, 'status_inactive', None) or cls.TEXT_DISABLED
        
        # Danger colors (use theme overrides if present, else derive)
        cls.DANGER_BG = getattr(theme, 'danger_bg', None) or QColor(
            max(0, cls.BG_DARK.red() + 30), cls.BG_DARK.green(), cls.BG_DARK.blue()
        )
        cls.DANGER_FG = getattr(theme, 'danger_fg', None) or QColor(255, 107, 107)
        
        # Sync timeline style with current theme
        try:
            from ui.qt_gui.widgets.timeline.core.style import TimelineStyle
            TimelineStyle.apply_theme()
        except ImportError:
            pass  # Timeline module may not be available
        
        # Emit theme_changed signal so panels can refresh
        try:
            _get_theme_signals().theme_changed.emit()
        except RuntimeError:
            pass  # QApplication may not exist yet during startup
    
    @classmethod
    def apply_theme_from_dict(cls, color_dict: dict):
        """Apply colors from a flat ``{attr_name: "#hex"}`` dict for live preview.
        
        This does NOT require a registered Theme -- it is used by the settings
        dialog to preview edits before the user saves a custom preset.
        """
        from ui.qt_gui.theme_registry import ThemeRegistry
        
        # Build a temporary Theme object from the dict
        theme = ThemeRegistry.theme_from_dict("_live_preview", "Live preview", color_dict)
        cls._current_theme = theme
        
        # Apply every color field present in the dict (or from the fallback)
        attr_map = {
            "bg_dark": "BG_DARK", "bg_medium": "BG_MEDIUM", "bg_light": "BG_LIGHT",
            "border": "BORDER", "hover": "HOVER", "selected": "SELECTED",
            "text_primary": "TEXT_PRIMARY", "text_secondary": "TEXT_SECONDARY",
            "text_disabled": "TEXT_DISABLED",
            "accent_blue": "ACCENT_BLUE", "accent_green": "ACCENT_GREEN",
            "accent_red": "ACCENT_RED", "accent_yellow": "ACCENT_YELLOW",
            "block_load": "BLOCK_LOAD", "block_analyze": "BLOCK_ANALYZE",
            "block_transform": "BLOCK_TRANSFORM", "block_export": "BLOCK_EXPORT",
            "block_editor": "BLOCK_EDITOR", "block_visualize": "BLOCK_VISUALIZE",
            "block_utility": "BLOCK_UTILITY",
            "connection_normal": "CONNECTION_NORMAL", "connection_hover": "CONNECTION_HOVER",
            "connection_selected": "CONNECTION_SELECTED",
            "port_input": "PORT_INPUT", "port_output": "PORT_OUTPUT",
            "port_audio": "PORT_AUDIO", "port_event": "PORT_EVENT",
            "port_manipulator": "PORT_MANIPULATOR", "port_generic": "PORT_GENERIC",
        }
        for theme_field, cls_attr in attr_map.items():
            val = getattr(theme, theme_field, None)
            if val is not None:
                setattr(cls, cls_attr, val)
        
        # Derived / optional fields
        cls.ACCENT_ORANGE = getattr(theme, 'accent_orange', None) or QColor(220, 135, 65)
        cls.ACCENT_PURPLE = getattr(theme, 'accent_purple', None) or QColor(175, 115, 195)
        cls.STATUS_SUCCESS = getattr(theme, 'status_success', None) or cls.ACCENT_GREEN
        cls.STATUS_WARNING = getattr(theme, 'status_warning', None) or cls.ACCENT_YELLOW
        cls.STATUS_ERROR = getattr(theme, 'status_error', None) or cls.ACCENT_RED
        cls.STATUS_INFO = getattr(theme, 'status_info', None) or cls.ACCENT_BLUE
        cls.STATUS_INACTIVE = getattr(theme, 'status_inactive', None) or cls.TEXT_DISABLED
        cls.DANGER_BG = getattr(theme, 'danger_bg', None) or QColor(
            max(0, cls.BG_DARK.red() + 30), cls.BG_DARK.green(), cls.BG_DARK.blue()
        )
        cls.DANGER_FG = getattr(theme, 'danger_fg', None) or QColor(255, 107, 107)
        
        # Sync timeline and emit signal
        try:
            from ui.qt_gui.widgets.timeline.core.style import TimelineStyle
            TimelineStyle.apply_theme()
        except ImportError:
            pass
        try:
            _get_theme_signals().theme_changed.emit()
        except RuntimeError:
            pass


class Spacing:
    """Spacing constants"""
    XS = 4
    SM = 8
    MD = 16
    LG = 24
    XL = 32
    XXL = 48


class Typography:
    """Font definitions"""
    
    @staticmethod
    def default_font() -> QFont:
        font = QFont()
        font.setFamily("SF Pro Text, Segoe UI, -apple-system, system-ui")
        font.setPixelSize(13)
        return font
    
    @staticmethod
    def heading_font() -> QFont:
        font = QFont()
        font.setFamily("SF Pro Display, Segoe UI, -apple-system, system-ui")
        font.setPixelSize(16)
        font.setWeight(QFont.Weight.DemiBold)
        return font
    
    @staticmethod
    def mono_font() -> QFont:
        font = QFont()
        font.setFamily("SF Mono, Consolas, Monaco, monospace")
        font.setPixelSize(12)
        return font


class Sizes:
    """Size constants for UI elements"""
    
    # Node/Block sizes - compact, less rectangular proportions
    BLOCK_WIDTH = 150
    BLOCK_HEIGHT = 100
    BLOCK_MIN_HEIGHT = 70  # Minimum height even with few ports
    BLOCK_HEADER_HEIGHT = 28
    _DEFAULT_BLOCK_CORNER_RADIUS = 0  # Sharp corners by default
    BLOCK_CORNER_RADIUS = 0
    BLOCK_BODY_PADDING = 6
    
    # Port sizes
    PORT_RADIUS = 4
    PORT_LABEL_OFFSET = 10  # Distance from port to label
    PORT_VERTICAL_SPACING = 20  # Vertical space between ports
    PORT_ZONE_HEIGHT = 22  # Height of each port row
    
    # Connection line width
    CONNECTION_WIDTH = 2
    CONNECTION_WIDTH_SELECTED = 3
    CONNECTION_WIDTH_HOVER = 2.5
    
    # Grid
    GRID_SIZE = 20
    GRID_MAJOR_EVERY = 5
    
    # Context menu
    CONTEXT_MENU_MIN_WIDTH = 180
    
    # Audio player embedded controls
    PLAYER_BLOCK_WIDTH = 350
    PLAYER_CONTROL_HEIGHT = 200
    
    # Audio filter: height is computed dynamically in audio_filter_block_item.py
    # (compact for simple filters, expanded for shelf/peak with gain/Q row)
    FILTER_CONTROL_HEIGHT = 110  # fallback / legacy

    # EQ Bands: no fixed constant -- height is computed dynamically
    # based on band count in eq_bands_block_item.py

    # Audio negate embedded controls (wider node to fit knob rows)
    NEGATE_BLOCK_WIDTH = 210
    # Negate control height is computed dynamically in audio_negate_block_item.py


class Effects:
    """Visual effect constants"""
    
    # Shadows
    SHADOW_OFFSET_X = 0
    SHADOW_OFFSET_Y = 2
    SHADOW_BLUR_RADIUS = 8
    SHADOW_COLOR = QColor(0, 0, 0, 60)
    
    # Animations
    ANIMATION_DURATION_FAST = 150  # ms
    ANIMATION_DURATION_NORMAL = 250  # ms
    ANIMATION_DURATION_SLOW = 350  # ms


def get_stylesheet() -> str:
    """
    Generate comprehensive global stylesheet for the application using current theme.
    
    This stylesheet is the single source of truth for all widget styling.
    When theme or sharp_corners changes, child widget stylesheets are cleared
    so that these rules take effect everywhere.
    """
    # Ensure theme is applied
    Colors.apply_theme()
    
    br = border_radius  # Short alias for readability
    
    stylesheet = f"""
    /* === Base === */
    QMainWindow, QWidget {{
        background-color: {Colors.BG_DARK.name()};
        color: {Colors.TEXT_PRIMARY.name()};
        font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
        font-size: 13px;
    }}
    
    /* === Menu Bar === */
    QMenuBar {{
        background-color: {Colors.BG_MEDIUM.name()};
        border-bottom: 1px solid {Colors.BORDER.name()};
        padding: 4px;
    }}
    QMenuBar::item {{
        padding: 4px 12px;
        background: transparent;
        border-radius: {br(4)};
    }}
    QMenuBar::item:selected {{
        background-color: {Colors.HOVER.name()};
    }}
    
    /* === Menus === */
    QMenu {{
        background-color: {Colors.BG_MEDIUM.name()};
        border: 1px solid {Colors.BORDER.name()};
        border-radius: {br(4)};
        padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 24px 6px 12px;
        border-radius: {br(3)};
    }}
    QMenu::item:selected {{
        background-color: {Colors.HOVER.name()};
    }}
    QMenu::separator {{
        height: 1px;
        background-color: {Colors.BORDER.name()};
        margin: 4px 8px;
    }}
    
    /* === Toolbar === */
    QToolBar {{
        background-color: {Colors.BG_MEDIUM.name()};
        border-bottom: 1px solid {Colors.BORDER.name()};
        spacing: 4px;
        padding: 4px;
    }}
    QToolButton {{
        background-color: transparent;
        border: none;
        border-radius: {br(4)};
        padding: 6px 12px;
        color: {Colors.TEXT_PRIMARY.name()};
    }}
    QToolButton:hover {{
        background-color: {Colors.HOVER.name()};
    }}
    QToolButton:pressed {{
        background-color: {Colors.SELECTED.name()};
    }}
    
    /* === Buttons === */
    QPushButton {{
        background-color: {Colors.BG_MEDIUM.name()};
        border: 1px solid {Colors.BORDER.name()};
        border-radius: {br(4)};
        padding: 6px 16px;
        color: {Colors.TEXT_PRIMARY.name()};
        min-width: 60px;
    }}
    QPushButton:hover {{
        background-color: {Colors.HOVER.name()};
    }}
    QPushButton:pressed {{
        background-color: {Colors.SELECTED.name()};
    }}
    QPushButton:disabled {{
        color: {Colors.TEXT_DISABLED.name()};
        background-color: {Colors.BG_DARK.name()};
        border-color: {Colors.BG_LIGHT.name()};
    }}
    
    /* === Input Widgets === */
    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {Colors.BG_MEDIUM.name()};
        border: 1px solid {Colors.BORDER.name()};
        border-radius: {br(4)};
        padding: 4px 8px;
        color: {Colors.TEXT_PRIMARY.name()};
        selection-background-color: {Colors.ACCENT_BLUE.name()};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {Colors.ACCENT_BLUE.name()};
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        color: {Colors.TEXT_DISABLED.name()};
        background-color: {Colors.BG_DARK.name()};
    }}
    
    /* === Combo Box === */
    QComboBox {{
        background-color: {Colors.BG_MEDIUM.name()};
        border: 1px solid {Colors.BORDER.name()};
        border-radius: {br(4)};
        padding: 4px 8px;
        color: {Colors.TEXT_PRIMARY.name()};
        min-width: 80px;
    }}
    QComboBox:focus {{
        border-color: {Colors.ACCENT_BLUE.name()};
    }}
    QComboBox::drop-down {{
        border: none;
        padding-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {Colors.BG_MEDIUM.name()};
        border: 1px solid {Colors.BORDER.name()};
        color: {Colors.TEXT_PRIMARY.name()};
        selection-background-color: {Colors.HOVER.name()};
    }}
    
    /* === Check Box / Radio Button === */
    QCheckBox {{
        spacing: 8px;
        color: {Colors.TEXT_PRIMARY.name()};
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {Colors.BORDER.name()};
        border-radius: {br(3)};
        background-color: {Colors.BG_MEDIUM.name()};
    }}
    QCheckBox::indicator:checked {{
        background-color: {Colors.ACCENT_BLUE.name()};
        border-color: {Colors.ACCENT_BLUE.name()};
    }}
    QRadioButton {{
        spacing: 8px;
        color: {Colors.TEXT_PRIMARY.name()};
    }}
    
    /* === Text Editors === */
    QTextEdit, QPlainTextEdit {{
        background-color: {Colors.BG_MEDIUM.name()};
        border: 1px solid {Colors.BORDER.name()};
        border-radius: {br(4)};
        color: {Colors.TEXT_PRIMARY.name()};
        selection-background-color: {Colors.ACCENT_BLUE.name()};
    }}
    
    /* === Tabs === */
    QTabWidget::pane {{
        border: 1px solid {Colors.BORDER.name()};
        background-color: {Colors.BG_DARK.name()};
    }}
    QTabBar::tab {{
        background-color: {Colors.BG_MEDIUM.name()};
        color: {Colors.TEXT_SECONDARY.name()};
        border: 1px solid {Colors.BORDER.name()};
        border-bottom: none;
        padding: 6px 16px;
    }}
    QTabBar::tab:selected {{
        background-color: {Colors.BG_DARK.name()};
        color: {Colors.TEXT_PRIMARY.name()};
        border-bottom: 2px solid {Colors.ACCENT_BLUE.name()};
    }}
    QTabBar::tab:hover:!selected {{
        background-color: {Colors.HOVER.name()};
    }}
    
    /* === Dock Widgets === */
    QDockWidget {{
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }}
    QDockWidget::title {{
        background-color: {Colors.BG_MEDIUM.name()};
        color: {Colors.TEXT_SECONDARY.name()};
        border: 1px solid {Colors.BORDER.name()};
        border-bottom: none;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 500;
    }}
    QDockWidget::title:hover {{
        background-color: {Colors.HOVER.name()};
        color: {Colors.TEXT_PRIMARY.name()};
    }}
    
    /* === Group Box === */
    QGroupBox {{
        border: 1px solid {Colors.BORDER.name()};
        border-radius: {br(4)};
        margin-top: 8px;
        padding-top: 16px;
        color: {Colors.TEXT_PRIMARY.name()};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        padding: 0 4px;
        color: {Colors.TEXT_SECONDARY.name()};
    }}
    
    /* === Scroll Bars === */
    QScrollBar:vertical {{
        background-color: {Colors.BG_DARK.name()};
        width: 12px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background-color: {Colors.BG_LIGHT.name()};
        border-radius: {br(4)};
        min-height: 30px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {Colors.HOVER.name()};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background-color: {Colors.BG_DARK.name()};
        height: 12px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {Colors.BG_LIGHT.name()};
        border-radius: {br(4)};
        min-width: 30px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background-color: {Colors.HOVER.name()};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}
    
    /* === Progress Bar === */
    QProgressBar {{
        background-color: {Colors.BG_MEDIUM.name()};
        border: 1px solid {Colors.BORDER.name()};
        border-radius: {br(4)};
        text-align: center;
        color: {Colors.TEXT_PRIMARY.name()};
    }}
    QProgressBar::chunk {{
        background-color: {Colors.ACCENT_BLUE.name()};
        border-radius: {br(3)};
    }}
    
    /* === Tree / List / Table Views === */
    QTreeView, QListView, QTableView,
    QTreeWidget, QListWidget, QTableWidget {{
        background-color: {Colors.BG_MEDIUM.name()};
        border: 1px solid {Colors.BORDER.name()};
        border-radius: {br(4)};
        color: {Colors.TEXT_PRIMARY.name()};
        alternate-background-color: {Colors.BG_LIGHT.name()};
        selection-background-color: {Colors.HOVER.name()};
        selection-color: {Colors.TEXT_PRIMARY.name()};
    }}
    QHeaderView::section {{
        background-color: {Colors.BG_MEDIUM.name()};
        color: {Colors.TEXT_SECONDARY.name()};
        border: 1px solid {Colors.BORDER.name()};
        padding: 4px 8px;
        font-weight: 500;
    }}
    
    /* === Scroll Area === */
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    
    /* === Tooltip === */
    QToolTip {{
        background-color: {Colors.BG_MEDIUM.name()};
        color: {Colors.TEXT_PRIMARY.name()};
        border: 1px solid {Colors.BORDER.name()};
        border-radius: {br(3)};
        padding: 4px 8px;
    }}
    
    /* === Graphics View === */
    QGraphicsView {{
        border: none;
        background-color: {Colors.BG_DARK.name()};
    }}
    
    /* === Labels === */
    QLabel {{
        color: {Colors.TEXT_PRIMARY.name()};
    }}
    
    /* === Status Bar === */
    QStatusBar {{
        background-color: {Colors.BG_MEDIUM.name()};
        border-top: 1px solid {Colors.BORDER.name()};
    }}
    
    /* === Dialogs === */
    QDialog {{
        background-color: {Colors.BG_DARK.name()};
        color: {Colors.TEXT_PRIMARY.name()};
    }}
    """
    
    return stylesheet

