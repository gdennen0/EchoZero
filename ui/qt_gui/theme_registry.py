"""
Theme Registry

Centralized theme presets for the EchoZero application.
Each theme defines a complete color palette and visual style.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
from PyQt6.QtGui import QColor


@dataclass
class Theme:
    """Complete theme definition with all color values"""
    name: str
    description: str
    
    # Background colors
    bg_dark: QColor
    bg_medium: QColor
    bg_light: QColor
    
    # UI element colors
    border: QColor
    hover: QColor
    selected: QColor
    
    # Text colors
    text_primary: QColor
    text_secondary: QColor
    text_disabled: QColor
    
    # Accent colors
    accent_blue: QColor
    accent_green: QColor
    accent_red: QColor
    accent_yellow: QColor
    
    # Block type colors
    block_load: QColor
    block_analyze: QColor
    block_transform: QColor
    block_export: QColor
    block_editor: QColor
    block_visualize: QColor
    block_utility: QColor
    
    # Connection colors
    connection_normal: QColor
    connection_hover: QColor
    connection_selected: QColor
    
    # Port colors
    port_input: QColor
    port_output: QColor
    port_audio: QColor
    port_event: QColor
    port_manipulator: QColor  # Bright orange for bidirectional command ports
    port_generic: QColor
    
    # Optional extended accent colors (default: derived by Colors.apply_theme)
    accent_orange: Optional[QColor] = None
    accent_purple: Optional[QColor] = None
    
    # Optional semantic status colors (default: derived from accents)
    status_success: Optional[QColor] = None
    status_warning: Optional[QColor] = None
    status_error: Optional[QColor] = None
    status_info: Optional[QColor] = None
    status_inactive: Optional[QColor] = None
    
    # Optional danger colors (default: derived from bg + red)
    danger_bg: Optional[QColor] = None
    danger_fg: Optional[QColor] = None

    # Optional overlay and domain-specific tokens
    overlay_subtle: Optional[QColor] = None
    overlay_feint: Optional[QColor] = None
    overlay_dim: Optional[QColor] = None
    overlay_very_subtle: Optional[QColor] = None
    text_on_light: Optional[QColor] = None
    text_on_dark: Optional[QColor] = None
    filter_shelf: Optional[QColor] = None
    filter_peak: Optional[QColor] = None
    grid_line: Optional[QColor] = None

    # Node editor
    node_border: Optional[QColor] = None

    # Timeline-specific
    timeline_playhead: Optional[QColor] = None
    timeline_selection: Optional[QColor] = None
    timeline_grid_major: Optional[QColor] = None
    timeline_grid_minor: Optional[QColor] = None
    timeline_track_alt: Optional[QColor] = None


class ThemeRegistry:
    """Registry of all available themes"""
    
    _themes: Dict[str, Theme] = {}
    _builtin_names: set = set()
    
    # All color fields on the Theme dataclass (used for serialization).
    # Order matches the dataclass definition.
    COLOR_FIELDS = [
        "bg_dark", "bg_medium", "bg_light",
        "border", "hover", "selected",
        "text_primary", "text_secondary", "text_disabled",
        "accent_blue", "accent_green", "accent_red", "accent_yellow",
        "block_load", "block_analyze", "block_transform", "block_export",
        "block_editor", "block_visualize", "block_utility",
        "connection_normal", "connection_hover", "connection_selected",
        "port_input", "port_output", "port_audio", "port_event",
        "port_manipulator", "port_generic",
        # Optional fields
        "accent_orange", "accent_purple",
        "status_success", "status_warning", "status_error",
        "status_info", "status_inactive",
        "danger_bg", "danger_fg",
        "overlay_subtle", "overlay_feint", "overlay_dim", "overlay_very_subtle",
        "text_on_light", "text_on_dark",
        "filter_shelf", "filter_peak",
        "grid_line",
        "node_border",
        "timeline_playhead", "timeline_selection",
        "timeline_grid_major", "timeline_grid_minor",
        "timeline_track_alt",
    ]
    
    @classmethod
    def register_theme(cls, theme: Theme):
        """Register a theme"""
        cls._themes[theme.name.lower()] = theme
    
    @classmethod
    def get_theme(cls, name: str) -> Theme:
        """Get a theme by name (case-insensitive)"""
        return cls._themes.get(name.lower())
    
    @classmethod
    def get_all_themes(cls) -> Dict[str, Theme]:
        """Get all registered themes"""
        return cls._themes.copy()
    
    @classmethod
    def get_theme_names(cls):
        """Get list of all theme names"""
        return sorted(cls._themes.keys())
    
    @classmethod
    def is_builtin(cls, name: str) -> bool:
        """Check whether a theme is a built-in preset."""
        return name.lower() in cls._builtin_names
    
    @classmethod
    def unregister_theme(cls, name: str):
        """Remove a theme from the registry. Cannot remove built-in themes."""
        key = name.lower()
        if key in cls._builtin_names:
            return  # Protect built-in themes
        cls._themes.pop(key, None)
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    @classmethod
    def theme_to_dict(cls, theme: Theme) -> Dict[str, str]:
        """Serialize a Theme's colors to ``{field_name: "#hex"}``."""
        result = {}
        for field_name in cls.COLOR_FIELDS:
            color = getattr(theme, field_name, None)
            if color is not None:
                result[field_name] = color.name()
        return result
    
    @classmethod
    def theme_from_dict(cls, name: str, description: str, color_dict: Dict[str, str]) -> Theme:
        """Deserialize a Theme from a flat color dict.
        
        Missing required fields are filled from the Default Dark preset.
        """
        fallback = cls.get_theme("default dark")
        kwargs = {"name": name, "description": description}
        for field_name in cls.COLOR_FIELDS:
            hex_val = color_dict.get(field_name)
            if hex_val:
                kwargs[field_name] = QColor(hex_val)
            else:
                # Required fields must come from somewhere
                fb_val = getattr(fallback, field_name, None) if fallback else None
                if fb_val is not None:
                    kwargs[field_name] = QColor(fb_val)
                # Optional fields can stay None (their dataclass default)
        return Theme(**kwargs)
    
    @classmethod
    def register_custom_theme(cls, name: str, description: str, color_dict: Dict[str, str]):
        """Create a Theme from a color dict, register it, and return it."""
        theme = cls.theme_from_dict(name, description, color_dict)
        cls.register_theme(theme)
        return theme


def _create_themes():
    """Create and register all theme presets"""
    
    # Default Dark (current default)
    ThemeRegistry.register_theme(Theme(
        name="Default Dark",
        description="Classic dark theme with blue accents",
        bg_dark=QColor(28, 28, 32),
        bg_medium=QColor(42, 42, 47),
        bg_light=QColor(56, 56, 62),
        border=QColor(75, 75, 80),
        hover=QColor(65, 65, 70),
        selected=QColor(85, 85, 90),
        text_primary=QColor(240, 240, 245),
        text_secondary=QColor(180, 180, 185),
        text_disabled=QColor(120, 120, 125),
        accent_blue=QColor(70, 130, 220),
        accent_green=QColor(80, 180, 120),
        accent_red=QColor(220, 80, 80),
        accent_yellow=QColor(220, 180, 60),
        block_load=QColor(65, 125, 210),
        block_analyze=QColor(110, 195, 115),
        block_transform=QColor(215, 135, 65),
        block_export=QColor(95, 195, 195),
        block_editor=QColor(175, 115, 195),
        block_visualize=QColor(215, 75, 135),
        block_utility=QColor(135, 135, 140),
        connection_normal=QColor(120, 120, 125),
        connection_hover=QColor(180, 180, 185),
        connection_selected=QColor(220, 180, 60),
        port_input=QColor(100, 200, 100),
        port_output=QColor(200, 100, 100),
        port_audio=QColor(70, 170, 220),
        port_event=QColor(230, 150, 70),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(150, 150, 155),
    ))
    
    # NASA Theme
    ThemeRegistry.register_theme(Theme(
        name="NASA",
        description="Inspired by NASA mission control with deep blues and warm oranges",
        bg_dark=QColor(15, 20, 35),
        bg_medium=QColor(25, 35, 50),
        bg_light=QColor(35, 45, 65),
        border=QColor(50, 70, 90),
        hover=QColor(40, 55, 75),
        selected=QColor(60, 85, 110),
        text_primary=QColor(240, 245, 250),
        text_secondary=QColor(180, 195, 210),
        text_disabled=QColor(100, 120, 140),
        accent_blue=QColor(0, 150, 255),
        accent_green=QColor(0, 200, 150),
        accent_red=QColor(255, 80, 80),
        accent_yellow=QColor(255, 200, 0),
        block_load=QColor(0, 150, 255),
        block_analyze=QColor(0, 200, 150),
        block_transform=QColor(255, 150, 0),
        block_export=QColor(100, 200, 255),
        block_editor=QColor(200, 100, 255),
        block_visualize=QColor(255, 100, 150),
        block_utility=QColor(150, 150, 170),
        connection_normal=QColor(100, 130, 160),
        connection_hover=QColor(150, 180, 210),
        connection_selected=QColor(255, 200, 0),
        port_input=QColor(0, 200, 150),
        port_output=QColor(255, 100, 100),
        port_audio=QColor(0, 150, 255),
        port_event=QColor(255, 150, 0),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(150, 170, 190),
    ))
    
    # SPACEX Theme
    ThemeRegistry.register_theme(Theme(
        name="SPACEX",
        description="Modern SpaceX-inspired theme with sleek grays and vibrant accents",
        bg_dark=QColor(20, 20, 25),
        bg_medium=QColor(30, 30, 35),
        bg_light=QColor(40, 40, 45),
        border=QColor(60, 60, 65),
        hover=QColor(50, 50, 55),
        selected=QColor(70, 70, 75),
        text_primary=QColor(255, 255, 255),
        text_secondary=QColor(200, 200, 200),
        text_disabled=QColor(120, 120, 120),
        accent_blue=QColor(0, 200, 255),
        accent_green=QColor(0, 255, 150),
        accent_red=QColor(255, 50, 50),
        accent_yellow=QColor(255, 220, 0),
        block_load=QColor(0, 200, 255),
        block_analyze=QColor(0, 255, 150),
        block_transform=QColor(255, 150, 0),
        block_export=QColor(100, 220, 255),
        block_editor=QColor(200, 100, 255),
        block_visualize=QColor(255, 100, 200),
        block_utility=QColor(150, 150, 150),
        connection_normal=QColor(100, 100, 100),
        connection_hover=QColor(180, 180, 180),
        connection_selected=QColor(255, 220, 0),
        port_input=QColor(0, 255, 150),
        port_output=QColor(255, 100, 100),
        port_audio=QColor(0, 200, 255),
        port_event=QColor(255, 150, 0),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(180, 180, 180),
    ))
    
    # Midnight Blue
    ThemeRegistry.register_theme(Theme(
        name="Midnight Blue",
        description="Deep midnight blue with elegant contrast",
        bg_dark=QColor(15, 20, 35),
        bg_medium=QColor(25, 30, 45),
        bg_light=QColor(35, 40, 55),
        border=QColor(50, 60, 75),
        hover=QColor(40, 50, 65),
        selected=QColor(60, 70, 85),
        text_primary=QColor(240, 245, 255),
        text_secondary=QColor(180, 190, 210),
        text_disabled=QColor(100, 110, 130),
        accent_blue=QColor(100, 150, 255),
        accent_green=QColor(80, 200, 120),
        accent_red=QColor(255, 100, 100),
        accent_yellow=QColor(255, 200, 80),
        block_load=QColor(100, 150, 255),
        block_analyze=QColor(80, 200, 120),
        block_transform=QColor(255, 150, 80),
        block_export=QColor(100, 200, 255),
        block_editor=QColor(200, 120, 255),
        block_visualize=QColor(255, 120, 180),
        block_utility=QColor(140, 150, 170),
        connection_normal=QColor(100, 120, 150),
        connection_hover=QColor(150, 170, 200),
        connection_selected=QColor(255, 200, 80),
        port_input=QColor(80, 200, 120),
        port_output=QColor(255, 100, 100),
        port_audio=QColor(100, 150, 255),
        port_event=QColor(255, 150, 80),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(150, 160, 180),
    ))
    
    # Forest Green
    ThemeRegistry.register_theme(Theme(
        name="Forest Green",
        description="Natural forest tones with earthy greens",
        bg_dark=QColor(20, 28, 20),
        bg_medium=QColor(30, 40, 30),
        bg_light=QColor(40, 52, 40),
        border=QColor(60, 75, 60),
        hover=QColor(50, 65, 50),
        selected=QColor(70, 85, 70),
        text_primary=QColor(240, 250, 240),
        text_secondary=QColor(180, 200, 180),
        text_disabled=QColor(100, 130, 100),
        accent_blue=QColor(80, 150, 200),
        accent_green=QColor(100, 200, 120),
        accent_red=QColor(220, 100, 100),
        accent_yellow=QColor(220, 200, 80),
        block_load=QColor(80, 150, 200),
        block_analyze=QColor(100, 200, 120),
        block_transform=QColor(200, 150, 80),
        block_export=QColor(100, 200, 180),
        block_editor=QColor(180, 120, 200),
        block_visualize=QColor(220, 120, 160),
        block_utility=QColor(130, 150, 130),
        connection_normal=QColor(100, 130, 100),
        connection_hover=QColor(150, 180, 150),
        connection_selected=QColor(220, 200, 80),
        port_input=QColor(100, 200, 120),
        port_output=QColor(220, 100, 100),
        port_audio=QColor(80, 150, 200),
        port_event=QColor(200, 150, 80),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(140, 160, 140),
    ))
    
    # Sunset Orange
    ThemeRegistry.register_theme(Theme(
        name="Sunset Orange",
        description="Warm sunset colors with orange and purple tones",
        bg_dark=QColor(35, 25, 30),
        bg_medium=QColor(50, 35, 40),
        bg_light=QColor(65, 45, 50),
        border=QColor(90, 65, 75),
        hover=QColor(75, 55, 65),
        selected=QColor(105, 75, 85),
        text_primary=QColor(255, 245, 240),
        text_secondary=QColor(220, 190, 180),
        text_disabled=QColor(140, 110, 100),
        accent_blue=QColor(100, 150, 220),
        accent_green=QColor(100, 200, 120),
        accent_red=QColor(255, 120, 80),
        accent_yellow=QColor(255, 200, 100),
        block_load=QColor(100, 150, 220),
        block_analyze=QColor(100, 200, 120),
        block_transform=QColor(255, 150, 80),
        block_export=QColor(255, 180, 150),
        block_editor=QColor(220, 120, 200),
        block_visualize=QColor(255, 120, 180),
        block_utility=QColor(170, 140, 150),
        connection_normal=QColor(150, 120, 130),
        connection_hover=QColor(200, 170, 180),
        connection_selected=QColor(255, 200, 100),
        port_input=QColor(100, 200, 120),
        port_output=QColor(255, 120, 80),
        port_audio=QColor(100, 150, 220),
        port_event=QColor(255, 150, 80),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(180, 150, 160),
    ))
    
    # Ocean Depths
    ThemeRegistry.register_theme(Theme(
        name="Ocean Depths",
        description="Deep ocean blues with teal accents",
        bg_dark=QColor(10, 25, 35),
        bg_medium=QColor(20, 40, 50),
        bg_light=QColor(30, 55, 65),
        border=QColor(50, 80, 90),
        hover=QColor(40, 65, 75),
        selected=QColor(60, 95, 105),
        text_primary=QColor(240, 250, 255),
        text_secondary=QColor(180, 210, 220),
        text_disabled=QColor(100, 130, 140),
        accent_blue=QColor(0, 180, 255),
        accent_green=QColor(0, 220, 180),
        accent_red=QColor(255, 100, 100),
        accent_yellow=QColor(255, 220, 100),
        block_load=QColor(0, 180, 255),
        block_analyze=QColor(0, 220, 180),
        block_transform=QColor(255, 180, 100),
        block_export=QColor(100, 220, 255),
        block_editor=QColor(200, 120, 255),
        block_visualize=QColor(255, 120, 200),
        block_utility=QColor(120, 160, 180),
        connection_normal=QColor(80, 140, 160),
        connection_hover=QColor(130, 190, 210),
        connection_selected=QColor(255, 220, 100),
        port_input=QColor(0, 220, 180),
        port_output=QColor(255, 100, 100),
        port_audio=QColor(0, 180, 255),
        port_event=QColor(255, 180, 100),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(140, 180, 200),
    ))
    
    # Purple Haze
    ThemeRegistry.register_theme(Theme(
        name="Purple Haze",
        description="Rich purple tones with vibrant accents",
        bg_dark=QColor(30, 20, 40),
        bg_medium=QColor(45, 30, 55),
        bg_light=QColor(60, 40, 70),
        border=QColor(85, 65, 95),
        hover=QColor(70, 50, 80),
        selected=QColor(100, 75, 110),
        text_primary=QColor(255, 240, 255),
        text_secondary=QColor(220, 180, 220),
        text_disabled=QColor(140, 100, 140),
        accent_blue=QColor(120, 150, 255),
        accent_green=QColor(120, 220, 150),
        accent_red=QColor(255, 120, 120),
        accent_yellow=QColor(255, 220, 120),
        block_load=QColor(120, 150, 255),
        block_analyze=QColor(120, 220, 150),
        block_transform=QColor(255, 180, 120),
        block_export=QColor(180, 200, 255),
        block_editor=QColor(220, 120, 255),
        block_visualize=QColor(255, 120, 220),
        block_utility=QColor(160, 140, 180),
        connection_normal=QColor(140, 120, 160),
        connection_hover=QColor(190, 170, 210),
        connection_selected=QColor(255, 220, 120),
        port_input=QColor(120, 220, 150),
        port_output=QColor(255, 120, 120),
        port_audio=QColor(120, 150, 255),
        port_event=QColor(255, 180, 120),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(180, 160, 200),
    ))
    
    # Matrix Green
    ThemeRegistry.register_theme(Theme(
        name="Matrix Green",
        description="Classic Matrix-style green on black",
        bg_dark=QColor(5, 10, 5),
        bg_medium=QColor(10, 20, 10),
        bg_light=QColor(15, 30, 15),
        border=QColor(20, 50, 20),
        hover=QColor(15, 40, 15),
        selected=QColor(25, 60, 25),
        text_primary=QColor(0, 255, 100),
        text_secondary=QColor(0, 200, 80),
        text_disabled=QColor(0, 100, 40),
        accent_blue=QColor(0, 200, 255),
        accent_green=QColor(0, 255, 150),
        accent_red=QColor(255, 50, 50),
        accent_yellow=QColor(255, 255, 0),
        block_load=QColor(0, 200, 255),
        block_analyze=QColor(0, 255, 150),
        block_transform=QColor(255, 200, 0),
        block_export=QColor(0, 255, 200),
        block_editor=QColor(200, 0, 255),
        block_visualize=QColor(255, 0, 200),
        block_utility=QColor(0, 150, 100),
        connection_normal=QColor(0, 150, 80),
        connection_hover=QColor(0, 220, 120),
        connection_selected=QColor(255, 255, 0),
        port_input=QColor(0, 255, 150),
        port_output=QColor(255, 50, 50),
        port_audio=QColor(0, 200, 255),
        port_event=QColor(255, 200, 0),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(0, 180, 120),
    ))
    
    # Fire Red
    ThemeRegistry.register_theme(Theme(
        name="Fire Red",
        description="Bold reds and oranges like flames",
        bg_dark=QColor(35, 15, 15),
        bg_medium=QColor(50, 25, 25),
        bg_light=QColor(65, 35, 35),
        border=QColor(90, 50, 50),
        hover=QColor(75, 40, 40),
        selected=QColor(105, 60, 60),
        text_primary=QColor(255, 240, 240),
        text_secondary=QColor(220, 180, 180),
        text_disabled=QColor(140, 100, 100),
        accent_blue=QColor(100, 150, 255),
        accent_green=QColor(100, 220, 120),
        accent_red=QColor(255, 80, 80),
        accent_yellow=QColor(255, 200, 100),
        block_load=QColor(100, 150, 255),
        block_analyze=QColor(100, 220, 120),
        block_transform=QColor(255, 150, 80),
        block_export=QColor(255, 180, 150),
        block_editor=QColor(220, 120, 200),
        block_visualize=QColor(255, 120, 150),
        block_utility=QColor(170, 120, 120),
        connection_normal=QColor(150, 100, 100),
        connection_hover=QColor(200, 150, 150),
        connection_selected=QColor(255, 200, 100),
        port_input=QColor(100, 220, 120),
        port_output=QColor(255, 80, 80),
        port_audio=QColor(100, 150, 255),
        port_event=QColor(255, 150, 80),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(180, 130, 130),
    ))
    
    # Arctic White
    ThemeRegistry.register_theme(Theme(
        name="Arctic White",
        description="Cool whites and light grays",
        bg_dark=QColor(240, 245, 250),
        bg_medium=QColor(220, 230, 240),
        bg_light=QColor(200, 215, 230),
        border=QColor(180, 190, 200),
        hover=QColor(190, 200, 210),
        selected=QColor(160, 170, 180),
        text_primary=QColor(20, 25, 35),
        text_secondary=QColor(60, 70, 80),
        text_disabled=QColor(120, 130, 140),
        accent_blue=QColor(50, 120, 200),
        accent_green=QColor(50, 180, 100),
        accent_red=QColor(220, 80, 80),
        accent_yellow=QColor(220, 180, 60),
        block_load=QColor(50, 120, 200),
        block_analyze=QColor(50, 180, 100),
        block_transform=QColor(220, 140, 60),
        block_export=QColor(50, 180, 200),
        block_editor=QColor(180, 100, 200),
        block_visualize=QColor(220, 80, 140),
        block_utility=QColor(140, 150, 160),
        connection_normal=QColor(120, 130, 140),
        connection_hover=QColor(80, 90, 100),
        connection_selected=QColor(220, 180, 60),
        port_input=QColor(50, 180, 100),
        port_output=QColor(220, 80, 80),
        port_audio=QColor(50, 120, 200),
        port_event=QColor(220, 140, 60),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(160, 170, 180),
    ))
    
    # Cyberpunk
    ThemeRegistry.register_theme(Theme(
        name="Cyberpunk",
        description="Neon cyberpunk aesthetic with vibrant pinks and cyans",
        bg_dark=QColor(10, 5, 20),
        bg_medium=QColor(20, 10, 30),
        bg_light=QColor(30, 15, 40),
        border=QColor(60, 30, 80),
        hover=QColor(50, 25, 70),
        selected=QColor(70, 35, 90),
        text_primary=QColor(255, 200, 255),
        text_secondary=QColor(200, 150, 220),
        text_disabled=QColor(120, 80, 140),
        accent_blue=QColor(0, 255, 255),
        accent_green=QColor(0, 255, 150),
        accent_red=QColor(255, 50, 100),
        accent_yellow=QColor(255, 255, 0),
        block_load=QColor(0, 255, 255),
        block_analyze=QColor(0, 255, 150),
        block_transform=QColor(255, 150, 0),
        block_export=QColor(100, 255, 255),
        block_editor=QColor(255, 100, 255),
        block_visualize=QColor(255, 50, 200),
        block_utility=QColor(150, 100, 200),
        connection_normal=QColor(100, 50, 150),
        connection_hover=QColor(150, 100, 200),
        connection_selected=QColor(255, 255, 0),
        port_input=QColor(0, 255, 150),
        port_output=QColor(255, 50, 100),
        port_audio=QColor(0, 255, 255),
        port_event=QColor(255, 150, 0),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(180, 120, 220),
    ))
    
    # Desert Sand
    ThemeRegistry.register_theme(Theme(
        name="Desert Sand",
        description="Warm desert tones with sandy beiges",
        bg_dark=QColor(45, 40, 35),
        bg_medium=QColor(60, 55, 50),
        bg_light=QColor(75, 70, 65),
        border=QColor(100, 90, 80),
        hover=QColor(85, 75, 65),
        selected=QColor(115, 105, 95),
        text_primary=QColor(255, 250, 240),
        text_secondary=QColor(220, 200, 180),
        text_disabled=QColor(140, 120, 100),
        accent_blue=QColor(100, 150, 220),
        accent_green=QColor(120, 180, 100),
        accent_red=QColor(220, 100, 80),
        accent_yellow=QColor(255, 200, 100),
        block_load=QColor(100, 150, 220),
        block_analyze=QColor(120, 180, 100),
        block_transform=QColor(255, 180, 100),
        block_export=QColor(200, 200, 150),
        block_editor=QColor(220, 150, 180),
        block_visualize=QColor(255, 150, 120),
        block_utility=QColor(170, 160, 150),
        connection_normal=QColor(150, 130, 120),
        connection_hover=QColor(200, 180, 170),
        connection_selected=QColor(255, 200, 100),
        port_input=QColor(120, 180, 100),
        port_output=QColor(220, 100, 80),
        port_audio=QColor(100, 150, 220),
        port_event=QColor(255, 180, 100),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(180, 170, 160),
    ))
    
    # Emerald
    ThemeRegistry.register_theme(Theme(
        name="Emerald",
        description="Rich emerald greens with gold accents",
        bg_dark=QColor(15, 30, 20),
        bg_medium=QColor(25, 45, 30),
        bg_light=QColor(35, 60, 40),
        border=QColor(60, 90, 70),
        hover=QColor(50, 75, 60),
        selected=QColor(70, 105, 80),
        text_primary=QColor(240, 255, 245),
        text_secondary=QColor(180, 220, 190),
        text_disabled=QColor(100, 140, 110),
        accent_blue=QColor(80, 150, 220),
        accent_green=QColor(50, 200, 100),
        accent_red=QColor(220, 100, 100),
        accent_yellow=QColor(255, 220, 80),
        block_load=QColor(80, 150, 220),
        block_analyze=QColor(50, 200, 100),
        block_transform=QColor(255, 180, 80),
        block_export=QColor(100, 220, 180),
        block_editor=QColor(200, 120, 220),
        block_visualize=QColor(255, 120, 180),
        block_utility=QColor(130, 170, 140),
        connection_normal=QColor(100, 150, 120),
        connection_hover=QColor(150, 200, 170),
        connection_selected=QColor(255, 220, 80),
        port_input=QColor(50, 200, 100),
        port_output=QColor(220, 100, 100),
        port_audio=QColor(80, 150, 220),
        port_event=QColor(255, 180, 80),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(140, 180, 150),
    ))
    
    # Lavender Dreams
    ThemeRegistry.register_theme(Theme(
        name="Lavender Dreams",
        description="Soft lavender and pastel purples",
        bg_dark=QColor(40, 35, 50),
        bg_medium=QColor(55, 50, 65),
        bg_light=QColor(70, 65, 80),
        border=QColor(95, 85, 105),
        hover=QColor(80, 70, 90),
        selected=QColor(110, 100, 120),
        text_primary=QColor(255, 250, 255),
        text_secondary=QColor(220, 200, 230),
        text_disabled=QColor(140, 120, 150),
        accent_blue=QColor(150, 180, 255),
        accent_green=QColor(150, 220, 180),
        accent_red=QColor(255, 150, 150),
        accent_yellow=QColor(255, 240, 150),
        block_load=QColor(150, 180, 255),
        block_analyze=QColor(150, 220, 180),
        block_transform=QColor(255, 200, 150),
        block_export=QColor(200, 220, 255),
        block_editor=QColor(240, 150, 255),
        block_visualize=QColor(255, 150, 220),
        block_utility=QColor(180, 170, 200),
        connection_normal=QColor(160, 150, 180),
        connection_hover=QColor(210, 200, 230),
        connection_selected=QColor(255, 240, 150),
        port_input=QColor(150, 220, 180),
        port_output=QColor(255, 150, 150),
        port_audio=QColor(150, 180, 255),
        port_event=QColor(255, 200, 150),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(200, 190, 220),
    ))
    
    # Steel Gray
    ThemeRegistry.register_theme(Theme(
        name="Steel Gray",
        description="Industrial steel grays with blue accents",
        bg_dark=QColor(30, 32, 35),
        bg_medium=QColor(45, 47, 50),
        bg_light=QColor(60, 62, 65),
        border=QColor(85, 87, 90),
        hover=QColor(70, 72, 75),
        selected=QColor(100, 102, 105),
        text_primary=QColor(240, 245, 250),
        text_secondary=QColor(190, 195, 200),
        text_disabled=QColor(130, 135, 140),
        accent_blue=QColor(80, 150, 220),
        accent_green=QColor(100, 200, 120),
        accent_red=QColor(220, 100, 100),
        accent_yellow=QColor(220, 200, 80),
        block_load=QColor(80, 150, 220),
        block_analyze=QColor(100, 200, 120),
        block_transform=QColor(220, 150, 80),
        block_export=QColor(100, 200, 220),
        block_editor=QColor(200, 120, 220),
        block_visualize=QColor(220, 100, 180),
        block_utility=QColor(150, 155, 160),
        connection_normal=QColor(120, 125, 130),
        connection_hover=QColor(170, 175, 180),
        connection_selected=QColor(220, 200, 80),
        port_input=QColor(100, 200, 120),
        port_output=QColor(220, 100, 100),
        port_audio=QColor(80, 150, 220),
        port_event=QColor(220, 150, 80),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(170, 175, 180),
    ))
    
    # Cherry Blossom
    ThemeRegistry.register_theme(Theme(
        name="Cherry Blossom",
        description="Soft pink cherry blossom tones",
        bg_dark=QColor(50, 35, 40),
        bg_medium=QColor(65, 50, 55),
        bg_light=QColor(80, 65, 70),
        border=QColor(110, 90, 95),
        hover=QColor(95, 75, 80),
        selected=QColor(125, 105, 110),
        text_primary=QColor(255, 245, 250),
        text_secondary=QColor(230, 200, 210),
        text_disabled=QColor(150, 120, 130),
        accent_blue=QColor(150, 180, 255),
        accent_green=QColor(150, 220, 180),
        accent_red=QColor(255, 150, 150),
        accent_yellow=QColor(255, 230, 150),
        block_load=QColor(150, 180, 255),
        block_analyze=QColor(150, 220, 180),
        block_transform=QColor(255, 200, 150),
        block_export=QColor(255, 200, 220),
        block_editor=QColor(240, 150, 220),
        block_visualize=QColor(255, 150, 200),
        block_utility=QColor(190, 170, 180),
        connection_normal=QColor(170, 150, 160),
        connection_hover=QColor(220, 200, 210),
        connection_selected=QColor(255, 230, 150),
        port_input=QColor(150, 220, 180),
        port_output=QColor(255, 150, 150),
        port_audio=QColor(150, 180, 255),
        port_event=QColor(255, 200, 150),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(210, 190, 200),
    ))
    
    # Golden Hour
    ThemeRegistry.register_theme(Theme(
        name="Golden Hour",
        description="Warm golden sunset colors",
        bg_dark=QColor(40, 30, 20),
        bg_medium=QColor(55, 45, 35),
        bg_light=QColor(70, 60, 50),
        border=QColor(95, 80, 65),
        hover=QColor(80, 65, 50),
        selected=QColor(110, 95, 80),
        text_primary=QColor(255, 250, 240),
        text_secondary=QColor(230, 210, 180),
        text_disabled=QColor(150, 130, 100),
        accent_blue=QColor(100, 150, 220),
        accent_green=QColor(120, 200, 100),
        accent_red=QColor(255, 120, 100),
        accent_yellow=QColor(255, 220, 100),
        block_load=QColor(100, 150, 220),
        block_analyze=QColor(120, 200, 100),
        block_transform=QColor(255, 180, 100),
        block_export=QColor(255, 220, 150),
        block_editor=QColor(240, 150, 220),
        block_visualize=QColor(255, 150, 180),
        block_utility=QColor(190, 170, 150),
        connection_normal=QColor(170, 150, 130),
        connection_hover=QColor(220, 200, 180),
        connection_selected=QColor(255, 220, 100),
        port_input=QColor(120, 200, 100),
        port_output=QColor(255, 120, 100),
        port_audio=QColor(100, 150, 220),
        port_event=QColor(255, 180, 100),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(210, 190, 170),
    ))
    
    # Neon Nights
    ThemeRegistry.register_theme(Theme(
        name="Neon Nights",
        description="Vibrant neon colors on dark background",
        bg_dark=QColor(5, 5, 15),
        bg_medium=QColor(15, 10, 25),
        bg_light=QColor(25, 15, 35),
        border=QColor(50, 30, 60),
        hover=QColor(40, 25, 50),
        selected=QColor(60, 35, 70),
        text_primary=QColor(255, 255, 255),
        text_secondary=QColor(200, 200, 220),
        text_disabled=QColor(120, 100, 140),
        accent_blue=QColor(0, 200, 255),
        accent_green=QColor(0, 255, 150),
        accent_red=QColor(255, 0, 100),
        accent_yellow=QColor(255, 255, 0),
        block_load=QColor(0, 200, 255),
        block_analyze=QColor(0, 255, 150),
        block_transform=QColor(255, 200, 0),
        block_export=QColor(100, 255, 255),
        block_editor=QColor(255, 100, 255),
        block_visualize=QColor(255, 0, 200),
        block_utility=QColor(150, 100, 200),
        connection_normal=QColor(100, 50, 150),
        connection_hover=QColor(150, 100, 200),
        connection_selected=QColor(255, 255, 0),
        port_input=QColor(0, 255, 150),
        port_output=QColor(255, 0, 100),
        port_audio=QColor(0, 200, 255),
        port_event=QColor(255, 200, 0),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(180, 120, 220),
    ))
    
    # Monochrome
    ThemeRegistry.register_theme(Theme(
        name="Monochrome",
        description="Pure black and white with grayscale",
        bg_dark=QColor(20, 20, 20),
        bg_medium=QColor(40, 40, 40),
        bg_light=QColor(60, 60, 60),
        border=QColor(100, 100, 100),
        hover=QColor(80, 80, 80),
        selected=QColor(120, 120, 120),
        text_primary=QColor(255, 255, 255),
        text_secondary=QColor(200, 200, 200),
        text_disabled=QColor(120, 120, 120),
        accent_blue=QColor(200, 200, 200),
        accent_green=QColor(180, 180, 180),
        accent_red=QColor(220, 220, 220),
        accent_yellow=QColor(240, 240, 240),
        block_load=QColor(180, 180, 180),
        block_analyze=QColor(200, 200, 200),
        block_transform=QColor(160, 160, 160),
        block_export=QColor(190, 190, 190),
        block_editor=QColor(170, 170, 170),
        block_visualize=QColor(210, 210, 210),
        block_utility=QColor(150, 150, 150),
        connection_normal=QColor(120, 120, 120),
        connection_hover=QColor(180, 180, 180),
        connection_selected=QColor(255, 255, 255),
        port_input=QColor(200, 200, 200),
        port_output=QColor(180, 180, 180),
        port_audio=QColor(190, 190, 190),
        port_event=QColor(170, 170, 170),
        port_manipulator=QColor(220, 220, 220),
        port_generic=QColor(160, 160, 160),
    ))

    # Solarized Dark
    ThemeRegistry.register_theme(Theme(
        name="Solarized Dark",
        description="Ethan Schoonover's Solarized palette on a warm dark base",
        bg_dark=QColor(0, 43, 54),
        bg_medium=QColor(7, 54, 66),
        bg_light=QColor(22, 68, 78),
        border=QColor(88, 110, 117),
        hover=QColor(38, 78, 88),
        selected=QColor(58, 98, 108),
        text_primary=QColor(253, 246, 227),
        text_secondary=QColor(147, 161, 161),
        text_disabled=QColor(88, 110, 117),
        accent_blue=QColor(38, 139, 210),
        accent_green=QColor(133, 153, 0),
        accent_red=QColor(220, 50, 47),
        accent_yellow=QColor(181, 137, 0),
        block_load=QColor(38, 139, 210),
        block_analyze=QColor(133, 153, 0),
        block_transform=QColor(203, 75, 22),
        block_export=QColor(42, 161, 152),
        block_editor=QColor(108, 113, 196),
        block_visualize=QColor(211, 54, 130),
        block_utility=QColor(88, 110, 117),
        connection_normal=QColor(88, 110, 117),
        connection_hover=QColor(147, 161, 161),
        connection_selected=QColor(181, 137, 0),
        port_input=QColor(133, 153, 0),
        port_output=QColor(220, 50, 47),
        port_audio=QColor(38, 139, 210),
        port_event=QColor(203, 75, 22),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(101, 123, 131),
    ))

    # Solarized Light
    ThemeRegistry.register_theme(Theme(
        name="Solarized Light",
        description="Solarized palette on a warm cream-white base",
        bg_dark=QColor(253, 246, 227),
        bg_medium=QColor(238, 232, 213),
        bg_light=QColor(220, 215, 198),
        border=QColor(147, 161, 161),
        hover=QColor(228, 222, 203),
        selected=QColor(198, 192, 173),
        text_primary=QColor(0, 43, 54),
        text_secondary=QColor(88, 110, 117),
        text_disabled=QColor(147, 161, 161),
        accent_blue=QColor(38, 139, 210),
        accent_green=QColor(133, 153, 0),
        accent_red=QColor(220, 50, 47),
        accent_yellow=QColor(181, 137, 0),
        block_load=QColor(38, 139, 210),
        block_analyze=QColor(133, 153, 0),
        block_transform=QColor(203, 75, 22),
        block_export=QColor(42, 161, 152),
        block_editor=QColor(108, 113, 196),
        block_visualize=QColor(211, 54, 130),
        block_utility=QColor(130, 140, 145),
        connection_normal=QColor(147, 161, 161),
        connection_hover=QColor(88, 110, 117),
        connection_selected=QColor(181, 137, 0),
        port_input=QColor(133, 153, 0),
        port_output=QColor(220, 50, 47),
        port_audio=QColor(38, 139, 210),
        port_event=QColor(203, 75, 22),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(160, 172, 175),
    ))

    # Nord
    ThemeRegistry.register_theme(Theme(
        name="Nord",
        description="Arctic north-bluish palette inspired by polar landscapes",
        bg_dark=QColor(46, 52, 64),
        bg_medium=QColor(59, 66, 82),
        bg_light=QColor(67, 76, 94),
        border=QColor(76, 86, 106),
        hover=QColor(72, 80, 98),
        selected=QColor(86, 95, 114),
        text_primary=QColor(236, 239, 244),
        text_secondary=QColor(216, 222, 233),
        text_disabled=QColor(120, 130, 148),
        accent_blue=QColor(136, 192, 208),
        accent_green=QColor(163, 190, 140),
        accent_red=QColor(191, 97, 106),
        accent_yellow=QColor(235, 203, 139),
        block_load=QColor(129, 161, 193),
        block_analyze=QColor(163, 190, 140),
        block_transform=QColor(208, 135, 112),
        block_export=QColor(136, 192, 208),
        block_editor=QColor(180, 142, 173),
        block_visualize=QColor(191, 97, 106),
        block_utility=QColor(110, 120, 138),
        connection_normal=QColor(76, 86, 106),
        connection_hover=QColor(143, 156, 178),
        connection_selected=QColor(235, 203, 139),
        port_input=QColor(163, 190, 140),
        port_output=QColor(191, 97, 106),
        port_audio=QColor(136, 192, 208),
        port_event=QColor(208, 135, 112),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(120, 130, 148),
    ))

    # Dracula
    ThemeRegistry.register_theme(Theme(
        name="Dracula",
        description="Famous dark theme with vivid purple, pink, and cyan accents",
        bg_dark=QColor(40, 42, 54),
        bg_medium=QColor(50, 52, 66),
        bg_light=QColor(62, 64, 80),
        border=QColor(80, 82, 98),
        hover=QColor(68, 71, 90),
        selected=QColor(90, 92, 110),
        text_primary=QColor(248, 248, 242),
        text_secondary=QColor(189, 147, 249),
        text_disabled=QColor(98, 114, 164),
        accent_blue=QColor(139, 233, 253),
        accent_green=QColor(80, 250, 123),
        accent_red=QColor(255, 85, 85),
        accent_yellow=QColor(241, 250, 140),
        block_load=QColor(139, 233, 253),
        block_analyze=QColor(80, 250, 123),
        block_transform=QColor(255, 184, 108),
        block_export=QColor(139, 233, 253),
        block_editor=QColor(189, 147, 249),
        block_visualize=QColor(255, 121, 198),
        block_utility=QColor(98, 114, 164),
        connection_normal=QColor(98, 114, 164),
        connection_hover=QColor(189, 147, 249),
        connection_selected=QColor(241, 250, 140),
        port_input=QColor(80, 250, 123),
        port_output=QColor(255, 85, 85),
        port_audio=QColor(139, 233, 253),
        port_event=QColor(255, 184, 108),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(130, 145, 182),
    ))

    # Gruvbox Dark
    ThemeRegistry.register_theme(Theme(
        name="Gruvbox Dark",
        description="Retro groove palette with warm earthy colors",
        bg_dark=QColor(40, 40, 40),
        bg_medium=QColor(50, 48, 47),
        bg_light=QColor(60, 56, 54),
        border=QColor(80, 73, 69),
        hover=QColor(70, 65, 61),
        selected=QColor(102, 92, 84),
        text_primary=QColor(235, 219, 178),
        text_secondary=QColor(189, 174, 147),
        text_disabled=QColor(124, 111, 100),
        accent_blue=QColor(131, 165, 152),
        accent_green=QColor(184, 187, 38),
        accent_red=QColor(251, 73, 52),
        accent_yellow=QColor(250, 189, 47),
        block_load=QColor(131, 165, 152),
        block_analyze=QColor(184, 187, 38),
        block_transform=QColor(254, 128, 25),
        block_export=QColor(142, 192, 124),
        block_editor=QColor(211, 134, 155),
        block_visualize=QColor(251, 73, 52),
        block_utility=QColor(146, 131, 116),
        connection_normal=QColor(124, 111, 100),
        connection_hover=QColor(189, 174, 147),
        connection_selected=QColor(250, 189, 47),
        port_input=QColor(184, 187, 38),
        port_output=QColor(251, 73, 52),
        port_audio=QColor(131, 165, 152),
        port_event=QColor(254, 128, 25),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(168, 153, 132),
    ))

    # Tokyo Night
    ThemeRegistry.register_theme(Theme(
        name="Tokyo Night",
        description="Storm variant inspired by neon-lit Tokyo evenings",
        bg_dark=QColor(26, 27, 38),
        bg_medium=QColor(36, 40, 59),
        bg_light=QColor(48, 52, 70),
        border=QColor(65, 72, 104),
        hover=QColor(55, 60, 85),
        selected=QColor(75, 82, 114),
        text_primary=QColor(192, 202, 245),
        text_secondary=QColor(134, 154, 220),
        text_disabled=QColor(86, 95, 137),
        accent_blue=QColor(125, 174, 247),
        accent_green=QColor(158, 206, 106),
        accent_red=QColor(247, 118, 142),
        accent_yellow=QColor(224, 175, 104),
        block_load=QColor(125, 174, 247),
        block_analyze=QColor(158, 206, 106),
        block_transform=QColor(255, 158, 100),
        block_export=QColor(115, 218, 202),
        block_editor=QColor(187, 154, 247),
        block_visualize=QColor(247, 118, 142),
        block_utility=QColor(86, 95, 137),
        connection_normal=QColor(65, 72, 104),
        connection_hover=QColor(134, 154, 220),
        connection_selected=QColor(224, 175, 104),
        port_input=QColor(158, 206, 106),
        port_output=QColor(247, 118, 142),
        port_audio=QColor(125, 174, 247),
        port_event=QColor(255, 158, 100),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(110, 120, 160),
    ))

    # Catppuccin Mocha
    ThemeRegistry.register_theme(Theme(
        name="Catppuccin Mocha",
        description="Warm pastel dark theme with soothing colors",
        bg_dark=QColor(30, 30, 46),
        bg_medium=QColor(49, 50, 68),
        bg_light=QColor(69, 71, 90),
        border=QColor(88, 91, 112),
        hover=QColor(58, 60, 78),
        selected=QColor(98, 101, 122),
        text_primary=QColor(205, 214, 244),
        text_secondary=QColor(166, 173, 200),
        text_disabled=QColor(108, 112, 134),
        accent_blue=QColor(137, 180, 250),
        accent_green=QColor(166, 227, 161),
        accent_red=QColor(243, 139, 168),
        accent_yellow=QColor(249, 226, 175),
        block_load=QColor(137, 180, 250),
        block_analyze=QColor(166, 227, 161),
        block_transform=QColor(250, 179, 135),
        block_export=QColor(148, 226, 213),
        block_editor=QColor(203, 166, 247),
        block_visualize=QColor(245, 194, 231),
        block_utility=QColor(108, 112, 134),
        connection_normal=QColor(88, 91, 112),
        connection_hover=QColor(166, 173, 200),
        connection_selected=QColor(249, 226, 175),
        port_input=QColor(166, 227, 161),
        port_output=QColor(243, 139, 168),
        port_audio=QColor(137, 180, 250),
        port_event=QColor(250, 179, 135),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(127, 132, 156),
    ))

    # One Dark
    ThemeRegistry.register_theme(Theme(
        name="One Dark",
        description="Atom-inspired dark theme with balanced syntax colors",
        bg_dark=QColor(40, 44, 52),
        bg_medium=QColor(50, 55, 65),
        bg_light=QColor(60, 66, 78),
        border=QColor(76, 82, 96),
        hover=QColor(55, 61, 73),
        selected=QColor(80, 88, 105),
        text_primary=QColor(171, 178, 191),
        text_secondary=QColor(140, 148, 162),
        text_disabled=QColor(92, 99, 112),
        accent_blue=QColor(97, 175, 239),
        accent_green=QColor(152, 195, 121),
        accent_red=QColor(224, 108, 117),
        accent_yellow=QColor(229, 192, 123),
        block_load=QColor(97, 175, 239),
        block_analyze=QColor(152, 195, 121),
        block_transform=QColor(209, 154, 102),
        block_export=QColor(86, 182, 194),
        block_editor=QColor(198, 120, 221),
        block_visualize=QColor(224, 108, 117),
        block_utility=QColor(92, 99, 112),
        connection_normal=QColor(76, 82, 96),
        connection_hover=QColor(140, 148, 162),
        connection_selected=QColor(229, 192, 123),
        port_input=QColor(152, 195, 121),
        port_output=QColor(224, 108, 117),
        port_audio=QColor(97, 175, 239),
        port_event=QColor(209, 154, 102),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(115, 122, 136),
    ))

    # Rose Pine
    ThemeRegistry.register_theme(Theme(
        name="Rose Pine",
        description="Natural pine with warm rosy highlights",
        bg_dark=QColor(25, 23, 36),
        bg_medium=QColor(38, 35, 53),
        bg_light=QColor(52, 49, 72),
        border=QColor(68, 65, 90),
        hover=QColor(45, 42, 62),
        selected=QColor(78, 75, 100),
        text_primary=QColor(224, 222, 244),
        text_secondary=QColor(144, 140, 170),
        text_disabled=QColor(110, 106, 134),
        accent_blue=QColor(49, 116, 143),
        accent_green=QColor(156, 207, 216),
        accent_red=QColor(235, 111, 146),
        accent_yellow=QColor(246, 193, 119),
        block_load=QColor(49, 116, 143),
        block_analyze=QColor(156, 207, 216),
        block_transform=QColor(246, 193, 119),
        block_export=QColor(156, 207, 216),
        block_editor=QColor(196, 167, 231),
        block_visualize=QColor(235, 111, 146),
        block_utility=QColor(110, 106, 134),
        connection_normal=QColor(68, 65, 90),
        connection_hover=QColor(144, 140, 170),
        connection_selected=QColor(246, 193, 119),
        port_input=QColor(156, 207, 216),
        port_output=QColor(235, 111, 146),
        port_audio=QColor(49, 116, 143),
        port_event=QColor(246, 193, 119),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(128, 124, 152),
    ))

    # Cobalt
    ThemeRegistry.register_theme(Theme(
        name="Cobalt",
        description="Deep rich cobalt blue workbench",
        bg_dark=QColor(0, 27, 52),
        bg_medium=QColor(10, 40, 68),
        bg_light=QColor(22, 54, 82),
        border=QColor(40, 75, 105),
        hover=QColor(18, 50, 78),
        selected=QColor(50, 85, 115),
        text_primary=QColor(255, 255, 255),
        text_secondary=QColor(150, 190, 225),
        text_disabled=QColor(80, 120, 155),
        accent_blue=QColor(0, 170, 255),
        accent_green=QColor(60, 230, 130),
        accent_red=QColor(255, 90, 90),
        accent_yellow=QColor(255, 210, 80),
        block_load=QColor(0, 170, 255),
        block_analyze=QColor(60, 230, 130),
        block_transform=QColor(255, 180, 60),
        block_export=QColor(80, 210, 240),
        block_editor=QColor(200, 130, 255),
        block_visualize=QColor(255, 100, 170),
        block_utility=QColor(100, 140, 170),
        connection_normal=QColor(60, 100, 140),
        connection_hover=QColor(120, 170, 210),
        connection_selected=QColor(255, 210, 80),
        port_input=QColor(60, 230, 130),
        port_output=QColor(255, 90, 90),
        port_audio=QColor(0, 170, 255),
        port_event=QColor(255, 180, 60),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(110, 150, 185),
    ))

    # Synthwave 84
    ThemeRegistry.register_theme(Theme(
        name="Synthwave 84",
        description="Retro 80s synthwave with hot pinks and electric blues",
        bg_dark=QColor(36, 18, 54),
        bg_medium=QColor(48, 26, 70),
        bg_light=QColor(60, 34, 86),
        border=QColor(80, 50, 110),
        hover=QColor(55, 30, 80),
        selected=QColor(90, 58, 120),
        text_primary=QColor(255, 240, 255),
        text_secondary=QColor(200, 170, 230),
        text_disabled=QColor(120, 90, 150),
        accent_blue=QColor(54, 240, 255),
        accent_green=QColor(114, 241, 117),
        accent_red=QColor(254, 78, 152),
        accent_yellow=QColor(254, 255, 98),
        block_load=QColor(54, 240, 255),
        block_analyze=QColor(114, 241, 117),
        block_transform=QColor(255, 177, 60),
        block_export=QColor(72, 216, 230),
        block_editor=QColor(255, 125, 233),
        block_visualize=QColor(254, 78, 152),
        block_utility=QColor(140, 110, 170),
        connection_normal=QColor(100, 70, 140),
        connection_hover=QColor(170, 130, 220),
        connection_selected=QColor(254, 255, 98),
        port_input=QColor(114, 241, 117),
        port_output=QColor(254, 78, 152),
        port_audio=QColor(54, 240, 255),
        port_event=QColor(255, 177, 60),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(150, 120, 190),
    ))

    # Ayu Dark
    ThemeRegistry.register_theme(Theme(
        name="Ayu Dark",
        description="Minimal dark theme with warm orange accents",
        bg_dark=QColor(10, 14, 20),
        bg_medium=QColor(20, 25, 32),
        bg_light=QColor(30, 36, 44),
        border=QColor(48, 56, 66),
        hover=QColor(26, 32, 40),
        selected=QColor(55, 65, 78),
        text_primary=QColor(203, 204, 198),
        text_secondary=QColor(150, 155, 150),
        text_disabled=QColor(90, 98, 105),
        accent_blue=QColor(57, 186, 230),
        accent_green=QColor(170, 215, 100),
        accent_red=QColor(255, 51, 51),
        accent_yellow=QColor(255, 180, 84),
        block_load=QColor(57, 186, 230),
        block_analyze=QColor(170, 215, 100),
        block_transform=QColor(255, 160, 75),
        block_export=QColor(90, 207, 207),
        block_editor=QColor(210, 163, 255),
        block_visualize=QColor(255, 80, 120),
        block_utility=QColor(90, 100, 112),
        connection_normal=QColor(60, 70, 82),
        connection_hover=QColor(130, 140, 150),
        connection_selected=QColor(255, 180, 84),
        port_input=QColor(170, 215, 100),
        port_output=QColor(255, 51, 51),
        port_audio=QColor(57, 186, 230),
        port_event=QColor(255, 160, 75),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(100, 112, 124),
    ))

    # Kanagawa
    ThemeRegistry.register_theme(Theme(
        name="Kanagawa",
        description="Inspired by Katsushika Hokusai's The Great Wave",
        bg_dark=QColor(22, 22, 29),
        bg_medium=QColor(35, 35, 44),
        bg_light=QColor(48, 48, 60),
        border=QColor(68, 68, 82),
        hover=QColor(42, 42, 54),
        selected=QColor(78, 78, 92),
        text_primary=QColor(220, 215, 186),
        text_secondary=QColor(168, 162, 138),
        text_disabled=QColor(98, 95, 82),
        accent_blue=QColor(127, 180, 202),
        accent_green=QColor(152, 187, 108),
        accent_red=QColor(195, 64, 67),
        accent_yellow=QColor(228, 164, 60),
        block_load=QColor(127, 180, 202),
        block_analyze=QColor(152, 187, 108),
        block_transform=QColor(255, 160, 102),
        block_export=QColor(106, 149, 137),
        block_editor=QColor(149, 127, 184),
        block_visualize=QColor(210, 126, 153),
        block_utility=QColor(98, 95, 82),
        connection_normal=QColor(68, 68, 82),
        connection_hover=QColor(168, 162, 138),
        connection_selected=QColor(228, 164, 60),
        port_input=QColor(152, 187, 108),
        port_output=QColor(195, 64, 67),
        port_audio=QColor(127, 180, 202),
        port_event=QColor(255, 160, 102),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(120, 115, 100),
    ))

    # Amber Terminal
    ThemeRegistry.register_theme(Theme(
        name="Amber Terminal",
        description="Retro amber phosphor CRT monitor aesthetic",
        bg_dark=QColor(15, 12, 5),
        bg_medium=QColor(25, 20, 10),
        bg_light=QColor(35, 28, 15),
        border=QColor(55, 45, 25),
        hover=QColor(40, 32, 18),
        selected=QColor(65, 52, 28),
        text_primary=QColor(255, 176, 0),
        text_secondary=QColor(200, 140, 0),
        text_disabled=QColor(100, 70, 0),
        accent_blue=QColor(255, 200, 60),
        accent_green=QColor(255, 190, 40),
        accent_red=QColor(255, 100, 30),
        accent_yellow=QColor(255, 220, 80),
        block_load=QColor(255, 200, 60),
        block_analyze=QColor(230, 180, 40),
        block_transform=QColor(255, 140, 20),
        block_export=QColor(200, 170, 50),
        block_editor=QColor(240, 160, 80),
        block_visualize=QColor(255, 120, 40),
        block_utility=QColor(140, 110, 50),
        connection_normal=QColor(100, 80, 30),
        connection_hover=QColor(180, 140, 50),
        connection_selected=QColor(255, 220, 80),
        port_input=QColor(230, 180, 40),
        port_output=QColor(255, 100, 30),
        port_audio=QColor(255, 200, 60),
        port_event=QColor(255, 140, 20),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(160, 125, 50),
    ))

    # High Contrast
    ThemeRegistry.register_theme(Theme(
        name="High Contrast",
        description="Maximum contrast for accessibility (WCAG-friendly)",
        bg_dark=QColor(0, 0, 0),
        bg_medium=QColor(18, 18, 18),
        bg_light=QColor(36, 36, 36),
        border=QColor(200, 200, 200),
        hover=QColor(50, 50, 50),
        selected=QColor(70, 70, 70),
        text_primary=QColor(255, 255, 255),
        text_secondary=QColor(230, 230, 230),
        text_disabled=QColor(140, 140, 140),
        accent_blue=QColor(60, 160, 255),
        accent_green=QColor(0, 230, 118),
        accent_red=QColor(255, 82, 82),
        accent_yellow=QColor(255, 230, 50),
        block_load=QColor(60, 160, 255),
        block_analyze=QColor(0, 230, 118),
        block_transform=QColor(255, 170, 40),
        block_export=QColor(0, 210, 210),
        block_editor=QColor(220, 130, 255),
        block_visualize=QColor(255, 70, 140),
        block_utility=QColor(180, 180, 180),
        connection_normal=QColor(160, 160, 160),
        connection_hover=QColor(220, 220, 220),
        connection_selected=QColor(255, 230, 50),
        port_input=QColor(0, 230, 118),
        port_output=QColor(255, 82, 82),
        port_audio=QColor(60, 160, 255),
        port_event=QColor(255, 170, 40),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(190, 190, 190),
    ))

    # Copper Patina
    ThemeRegistry.register_theme(Theme(
        name="Copper Patina",
        description="Weathered copper and verdigris green-blue",
        bg_dark=QColor(28, 32, 30),
        bg_medium=QColor(38, 44, 40),
        bg_light=QColor(50, 58, 52),
        border=QColor(72, 82, 74),
        hover=QColor(44, 52, 46),
        selected=QColor(82, 92, 84),
        text_primary=QColor(230, 228, 218),
        text_secondary=QColor(175, 172, 162),
        text_disabled=QColor(110, 108, 100),
        accent_blue=QColor(90, 165, 155),
        accent_green=QColor(120, 180, 130),
        accent_red=QColor(200, 95, 75),
        accent_yellow=QColor(210, 175, 90),
        block_load=QColor(90, 165, 155),
        block_analyze=QColor(120, 180, 130),
        block_transform=QColor(200, 140, 80),
        block_export=QColor(110, 185, 170),
        block_editor=QColor(170, 125, 165),
        block_visualize=QColor(200, 95, 75),
        block_utility=QColor(120, 128, 118),
        connection_normal=QColor(90, 100, 92),
        connection_hover=QColor(150, 158, 148),
        connection_selected=QColor(210, 175, 90),
        port_input=QColor(120, 180, 130),
        port_output=QColor(200, 95, 75),
        port_audio=QColor(90, 165, 155),
        port_event=QColor(200, 140, 80),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(130, 138, 128),
    ))

    # Slate
    ThemeRegistry.register_theme(Theme(
        name="Slate",
        description="Cool blue-gray slate stone tones",
        bg_dark=QColor(15, 23, 42),
        bg_medium=QColor(30, 41, 59),
        bg_light=QColor(51, 65, 85),
        border=QColor(71, 85, 105),
        hover=QColor(40, 53, 72),
        selected=QColor(81, 95, 115),
        text_primary=QColor(241, 245, 249),
        text_secondary=QColor(148, 163, 184),
        text_disabled=QColor(100, 116, 139),
        accent_blue=QColor(56, 189, 248),
        accent_green=QColor(74, 222, 128),
        accent_red=QColor(248, 113, 113),
        accent_yellow=QColor(250, 204, 21),
        block_load=QColor(56, 189, 248),
        block_analyze=QColor(74, 222, 128),
        block_transform=QColor(251, 146, 60),
        block_export=QColor(45, 212, 191),
        block_editor=QColor(167, 139, 250),
        block_visualize=QColor(244, 114, 182),
        block_utility=QColor(100, 116, 139),
        connection_normal=QColor(71, 85, 105),
        connection_hover=QColor(148, 163, 184),
        connection_selected=QColor(250, 204, 21),
        port_input=QColor(74, 222, 128),
        port_output=QColor(248, 113, 113),
        port_audio=QColor(56, 189, 248),
        port_event=QColor(251, 146, 60),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(120, 135, 158),
    ))

    # Coral Reef
    ThemeRegistry.register_theme(Theme(
        name="Coral Reef",
        description="Tropical underwater palette with coral and aquamarine",
        bg_dark=QColor(16, 28, 36),
        bg_medium=QColor(24, 40, 50),
        bg_light=QColor(34, 54, 66),
        border=QColor(52, 76, 90),
        hover=QColor(30, 48, 60),
        selected=QColor(60, 86, 100),
        text_primary=QColor(245, 250, 252),
        text_secondary=QColor(180, 210, 220),
        text_disabled=QColor(100, 130, 142),
        accent_blue=QColor(70, 200, 220),
        accent_green=QColor(80, 210, 160),
        accent_red=QColor(255, 110, 100),
        accent_yellow=QColor(255, 215, 100),
        block_load=QColor(70, 200, 220),
        block_analyze=QColor(80, 210, 160),
        block_transform=QColor(255, 140, 90),
        block_export=QColor(100, 220, 200),
        block_editor=QColor(200, 140, 220),
        block_visualize=QColor(255, 110, 130),
        block_utility=QColor(110, 145, 160),
        connection_normal=QColor(70, 100, 115),
        connection_hover=QColor(140, 180, 195),
        connection_selected=QColor(255, 215, 100),
        port_input=QColor(80, 210, 160),
        port_output=QColor(255, 110, 100),
        port_audio=QColor(70, 200, 220),
        port_event=QColor(255, 140, 90),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(120, 158, 175),
    ))

    # Frost
    ThemeRegistry.register_theme(Theme(
        name="Frost",
        description="Icy blue and white with crystalline frost tones",
        bg_dark=QColor(220, 235, 245),
        bg_medium=QColor(200, 220, 235),
        bg_light=QColor(180, 205, 225),
        border=QColor(140, 170, 195),
        hover=QColor(190, 212, 230),
        selected=QColor(160, 185, 210),
        text_primary=QColor(15, 30, 50),
        text_secondary=QColor(50, 70, 100),
        text_disabled=QColor(120, 145, 170),
        accent_blue=QColor(30, 120, 210),
        accent_green=QColor(40, 160, 100),
        accent_red=QColor(210, 60, 60),
        accent_yellow=QColor(200, 160, 40),
        block_load=QColor(30, 120, 210),
        block_analyze=QColor(40, 160, 100),
        block_transform=QColor(210, 120, 40),
        block_export=QColor(30, 155, 175),
        block_editor=QColor(140, 90, 190),
        block_visualize=QColor(200, 60, 110),
        block_utility=QColor(120, 140, 160),
        connection_normal=QColor(140, 165, 185),
        connection_hover=QColor(80, 110, 145),
        connection_selected=QColor(200, 160, 40),
        port_input=QColor(40, 160, 100),
        port_output=QColor(210, 60, 60),
        port_audio=QColor(30, 120, 210),
        port_event=QColor(210, 120, 40),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(140, 162, 182),
    ))

    # Volcanic
    ThemeRegistry.register_theme(Theme(
        name="Volcanic",
        description="Deep volcanic dark with glowing lava accents",
        bg_dark=QColor(18, 12, 12),
        bg_medium=QColor(30, 20, 18),
        bg_light=QColor(44, 30, 26),
        border=QColor(65, 45, 38),
        hover=QColor(38, 26, 22),
        selected=QColor(75, 52, 44),
        text_primary=QColor(250, 235, 225),
        text_secondary=QColor(195, 170, 155),
        text_disabled=QColor(115, 95, 85),
        accent_blue=QColor(90, 150, 210),
        accent_green=QColor(120, 195, 100),
        accent_red=QColor(255, 70, 40),
        accent_yellow=QColor(255, 190, 50),
        block_load=QColor(90, 150, 210),
        block_analyze=QColor(120, 195, 100),
        block_transform=QColor(255, 130, 40),
        block_export=QColor(100, 190, 180),
        block_editor=QColor(200, 120, 180),
        block_visualize=QColor(255, 70, 90),
        block_utility=QColor(120, 100, 90),
        connection_normal=QColor(85, 65, 55),
        connection_hover=QColor(160, 130, 115),
        connection_selected=QColor(255, 190, 50),
        port_input=QColor(120, 195, 100),
        port_output=QColor(255, 70, 40),
        port_audio=QColor(90, 150, 210),
        port_event=QColor(255, 130, 40),
        port_manipulator=QColor(255, 140, 0),
        port_generic=QColor(140, 118, 108),
    ))


# Initialize themes on module import
_create_themes()
# Snapshot builtin theme names so custom themes are distinguishable
ThemeRegistry._builtin_names = set(ThemeRegistry._themes.keys())

