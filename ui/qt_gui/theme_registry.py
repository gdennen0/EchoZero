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


# Initialize themes on module import
_create_themes()
# Snapshot builtin theme names so custom themes are distinguishable
ThemeRegistry._builtin_names = set(ThemeRegistry._themes.keys())

