"""
Timeline Settings Storage

Single source of truth for timeline widget UI settings.
Uses the application's PreferencesRepository for persistence.

Settings are:
- App-wide (not project-specific)
- Persist across sessions
- Auto-saved on change
- Type-safe via dataclass schema

Usage:
    # Get the settings manager (singleton per widget instance)
    settings = TimelineSettingsManager(preferences_repo)
    
    # Read settings
    width = settings.layer_column_width
    
    # Write settings (auto-persists)
    settings.layer_column_width = 150
    
    # Listen for changes
    settings.settings_changed.connect(my_handler)
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from PyQt6.QtGui import QColor

from src.application.settings.base_settings import BaseSettings, BaseSettingsManager
from ..constants import TRACK_HEIGHT, DEFAULT_PIXELS_PER_SECOND

if TYPE_CHECKING:
    from src.infrastructure.persistence.sqlite.preferences_repository_impl import PreferencesRepository


# =============================================================================
# Settings Schema (Dataclass)
# =============================================================================

@dataclass
class TimelineSettings(BaseSettings):
    """
    Timeline widget settings schema.
    
    Add new settings here - they will automatically be saved/loaded.
    All fields should have default values for backwards compatibility.
    """
    
    # Layer panel settings
    layer_column_width: int = 120
    layer_column_min_width: int = 80
    layer_column_max_width: int = 300
    
    # Default layer settings
    default_layer_height: int = 40
    
    # Grid/snap settings
    snap_enabled: bool = True
    snap_to_grid: bool = True
    # Snap interval mode: "auto" uses current grid, or explicit frame values like "1f", "5f", "10f"
    snap_interval_mode: str = "auto"  # Options: "auto", "1f", "2f", "5f", "10f", "1s"
    
    # Zoom settings
    default_pixels_per_second: float = 100.0
    min_pixels_per_second: float = 10.0
    max_pixels_per_second: float = 1000.0
    
    # Playback settings
    playhead_follow_mode: str = "page"  # "off", "page", "smooth", "center"
    
    # Visual settings
    show_grid_lines: bool = True
    show_waveform: bool = True
    waveform_opacity: float = 0.5
    show_waveforms_in_timeline: bool = False  # Show waveforms in clip events on timeline (disabled by default for performance)
    waveform_resolution: int = 50  # Number of points per waveform (low resolution for performance)
    waveform_min_width: int = 30  # Minimum event width (in pixels) to display waveforms (LOD threshold)
    show_event_labels: bool = True  # Show event classification labels
    show_event_duration_labels: bool = False  # Show event duration labels
    highlight_current_event: bool = True  # Highlight events at playhead position
    
    # Note: Grid line intervals are automatically calculated from timebase/FPS settings.
    # Users can toggle snap on/off but cannot manually adjust grid line frequency.
    
    # Scrollbar settings
    vertical_scrollbar_always_visible: bool = True  # Always show vs show as needed
    horizontal_scrollbar_always_visible: bool = True
    
    # Scroll position (restored on load)
    last_scroll_x: int = 0  # Horizontal scroll position
    last_scroll_y: int = 0  # Vertical scroll position (0 = top)
    restore_scroll_position: bool = True  # Whether to restore last position or always start at top-left
    
    # Inspector panel
    inspector_visible: bool = True
    inspector_width: int = 220
    
    # Recent layer colors (for custom assignments)
    recent_layer_colors: List[str] = field(default_factory=list)
    
    # Keyboard shortcuts (stored as string representations: "Key_Left", "Ctrl+Key_Left", etc.)
    shortcut_move_event_left: str = "Key_Left"  # Move selected events left by 1 unit
    shortcut_move_event_right: str = "Key_Right"  # Move selected events right by 1 unit
    shortcut_move_event_up_layer: str = "Ctrl+Key_Up"  # Move selected events up one layer
    shortcut_move_event_down_layer: str = "Ctrl+Key_Down"  # Move selected events down one layer
    
    # Event time styling (ruler text)
    event_time_font_size: int = 10  # Font size in pixels
    event_time_font_family: str = "monospace"  # "monospace", "default", "small"
    event_time_major_color: str = "#F0F0F5"  # Hex color for major tick labels
    event_time_minor_color: str = "#78787D"  # Hex color for minor tick labels
    
    # Block event styling (notes/events with duration - QGraphicsRectItem properties)
    block_event_height: int = 32  # Height of block events in pixels
    block_event_border_radius: int = 3  # Border radius in pixels
    block_event_border_width: int = 1  # Border width in pixels
    block_event_label_font_size: int = 10  # Font size for event labels
    block_event_border_darken_percent: int = 150  # Percentage to darken border color (100 = same, 150 = darker)
    
    # Block event Qt Graphics Item properties
    block_event_opacity: float = 1.0  # Opacity (0.0 = transparent, 1.0 = opaque)
    block_event_z_value: float = 0.0  # Stacking order (higher = drawn on top)
    block_event_rotation: float = 0.0  # Rotation in degrees
    block_event_scale: float = 1.0  # Scale factor (1.0 = normal size)
    
    # Block event graphics effects
    block_event_drop_shadow_enabled: bool = False  # Enable drop shadow effect
    block_event_drop_shadow_blur_radius: float = 5.0  # Shadow blur radius
    block_event_drop_shadow_offset_x: float = 2.0  # Shadow horizontal offset
    block_event_drop_shadow_offset_y: float = 2.0  # Shadow vertical offset
    block_event_drop_shadow_color: str = "#000000"  # Shadow color (hex)
    block_event_drop_shadow_opacity: float = 0.5  # Shadow opacity (0.0-1.0)
    
    # Marker event styling (one-shot/instant events)
    marker_event_shape: str = "diamond"  # Shape style: "diamond", "circle", "square", "triangle_up", "triangle_down", "triangle_left", "triangle_right", "arrow_up", "arrow_down", "arrow_left", "arrow_right", "star", "cross", "plus"
    marker_event_width: int = 13  # Width/size of marker events in pixels
    marker_event_border_width: int = 1  # Border width in pixels
    marker_event_border_darken_percent: int = 150  # Percentage to darken border color
    
    # Marker event Qt Graphics Item properties
    marker_event_opacity: float = 1.0  # Opacity (0.0 = transparent, 1.0 = opaque)
    marker_event_z_value: float = 0.0  # Stacking order (higher = drawn on top)
    marker_event_rotation: float = 0.0  # Rotation in degrees
    marker_event_scale: float = 1.0  # Scale factor (1.0 = normal size)
    
    # Marker event graphics effects
    marker_event_drop_shadow_enabled: bool = False  # Enable drop shadow effect
    marker_event_drop_shadow_blur_radius: float = 5.0  # Shadow blur radius
    marker_event_drop_shadow_offset_x: float = 2.0  # Shadow horizontal offset
    marker_event_drop_shadow_offset_y: float = 2.0  # Shadow vertical offset
    marker_event_drop_shadow_color: str = "#000000"  # Shadow color (hex)
    marker_event_drop_shadow_opacity: float = 0.5  # Shadow opacity (0.0-1.0)


# =============================================================================
# Settings Manager (Inherits from BaseSettingsManager)
# =============================================================================

class TimelineSettingsManager(BaseSettingsManager):
    """
    Manager for timeline widget settings.
    
    Inherits from BaseSettingsManager for standardized persistence.
    Provides typed property accessors for all timeline settings.
    
    Signals (inherited):
        settings_changed(str): Emitted when a setting changes (setting name)
        settings_loaded(): Emitted when settings are loaded from storage
    """
    
    NAMESPACE = "timeline"
    SETTINGS_CLASS = TimelineSettings
    
    def __init__(self, preferences_repo: Optional['PreferencesRepository'] = None, parent=None):
        """
        Initialize the settings manager.
        
        Args:
            preferences_repo: Repository for persistence (if None, settings are in-memory only)
            parent: Parent QObject
        """
        super().__init__(preferences_repo, parent)
    
    # =========================================================================
    # Public API - Property Access (Type-Safe)
    # =========================================================================
    
    @property
    def layer_column_width(self) -> int:
        return self._settings.layer_column_width
    
    @layer_column_width.setter
    def layer_column_width(self, value: int):
        if value != self._settings.layer_column_width:
            self._settings.layer_column_width = max(
                self._settings.layer_column_min_width,
                min(value, self._settings.layer_column_max_width)
            )
            self._save_setting('layer_column_width')
    
    @property
    def default_layer_height(self) -> int:
        return self._settings.default_layer_height
    
    @default_layer_height.setter
    def default_layer_height(self, value: int):
        if value != self._settings.default_layer_height:
            self._settings.default_layer_height = max(20, min(value, 200))
            self._save_setting('default_layer_height')
    
    @property
    def snap_enabled(self) -> bool:
        return self._settings.snap_enabled
    
    @snap_enabled.setter
    def snap_enabled(self, value: bool):
        if value != self._settings.snap_enabled:
            self._settings.snap_enabled = value
            self._save_setting('snap_enabled')
    
    @property
    def snap_to_grid(self) -> bool:
        return self._settings.snap_to_grid
    
    @snap_to_grid.setter
    def snap_to_grid(self, value: bool):
        if value != self._settings.snap_to_grid:
            self._settings.snap_to_grid = value
            self._save_setting('snap_to_grid')
    
    @property
    def snap_interval_mode(self) -> str:
        return self._settings.snap_interval_mode
    
    @snap_interval_mode.setter
    def snap_interval_mode(self, value: str):
        if value != self._settings.snap_interval_mode:
            valid_modes = {"auto", "1f", "2f", "5f", "10f", "1s"}
            if value not in valid_modes:
                raise ValueError(
                    f"Invalid snap interval mode: '{value}'. "
                    f"Valid options: {', '.join(sorted(valid_modes))}"
                )
            self._settings.snap_interval_mode = value
            self._save_setting('snap_interval_mode')
    
    @property
    def default_pixels_per_second(self) -> float:
        return self._settings.default_pixels_per_second
    
    @default_pixels_per_second.setter
    def default_pixels_per_second(self, value: float):
        if value != self._settings.default_pixels_per_second:
            self._settings.default_pixels_per_second = max(
                self._settings.min_pixels_per_second,
                min(value, self._settings.max_pixels_per_second)
            )
            self._save_setting('default_pixels_per_second')
    
    @property
    def playhead_follow_mode(self) -> str:
        return self._settings.playhead_follow_mode
    
    @playhead_follow_mode.setter
    def playhead_follow_mode(self, value: str):
        if value != self._settings.playhead_follow_mode:
            valid_modes = {"off", "page", "smooth", "center"}
            if value not in valid_modes:
                raise ValueError(
                    f"Invalid follow mode: '{value}'. "
                    f"Valid options: {', '.join(sorted(valid_modes))}"
                )
            self._settings.playhead_follow_mode = value
            self._save_setting('playhead_follow_mode')
    
    @property
    def show_grid_lines(self) -> bool:
        return self._settings.show_grid_lines
    
    @show_grid_lines.setter
    def show_grid_lines(self, value: bool):
        if value != self._settings.show_grid_lines:
            self._settings.show_grid_lines = value
            self._save_setting('show_grid_lines')
    
    # Note: grid_major_interval_multiplier and grid_minor_interval_multiplier properties
    # have been removed. Grid intervals are now automatically calculated from timebase/FPS.
    # Use snap_interval_mode for explicit snap intervals (auto, 1f, 5f, 10f, 1s).
    
    @property
    def show_waveform(self) -> bool:
        return self._settings.show_waveform
    
    @show_waveform.setter
    def show_waveform(self, value: bool):
        if value != self._settings.show_waveform:
            self._settings.show_waveform = value
            self._save_setting('show_waveform')
    
    @property
    def waveform_opacity(self) -> float:
        return self._settings.waveform_opacity
    
    @waveform_opacity.setter
    def waveform_opacity(self, value: float):
        if value != self._settings.waveform_opacity:
            self._settings.waveform_opacity = max(0.0, min(value, 1.0))
            self._save_setting('waveform_opacity')
    
    @property
    def show_waveforms_in_timeline(self) -> bool:
        return self._settings.show_waveforms_in_timeline
    
    @show_waveforms_in_timeline.setter
    def show_waveforms_in_timeline(self, value: bool):
        if value != self._settings.show_waveforms_in_timeline:
            self._settings.show_waveforms_in_timeline = value
            self._save_setting('show_waveforms_in_timeline')
    
    @property
    def waveform_resolution(self) -> int:
        return self._settings.waveform_resolution
    
    @waveform_resolution.setter
    def waveform_resolution(self, value: int):
        if value != self._settings.waveform_resolution:
            self._settings.waveform_resolution = max(5, min(value, 1000))  # Clamp between 5 and 1000
            self._save_setting('waveform_resolution')
    
    @property
    def waveform_min_width(self) -> int:
        return self._settings.waveform_min_width
    
    @waveform_min_width.setter
    def waveform_min_width(self, value: int):
        if value != self._settings.waveform_min_width:
            self._settings.waveform_min_width = max(5, min(value, 200))  # Clamp between 5 and 200 pixels
            self._save_setting('waveform_min_width')
    
    @property
    def show_event_labels(self) -> bool:
        return self._settings.show_event_labels
    
    @show_event_labels.setter
    def show_event_labels(self, value: bool):
        if value != self._settings.show_event_labels:
            self._settings.show_event_labels = value
            self._save_setting('show_event_labels')
    
    @property
    def show_event_duration_labels(self) -> bool:
        return self._settings.show_event_duration_labels
    
    @show_event_duration_labels.setter
    def show_event_duration_labels(self, value: bool):
        if value != self._settings.show_event_duration_labels:
            self._settings.show_event_duration_labels = value
            self._save_setting('show_event_duration_labels')
    
    @property
    def highlight_current_event(self) -> bool:
        return self._settings.highlight_current_event
    
    @highlight_current_event.setter
    def highlight_current_event(self, value: bool):
        if value != self._settings.highlight_current_event:
            self._settings.highlight_current_event = value
            self._save_setting('highlight_current_event')
    
    @property
    def vertical_scrollbar_always_visible(self) -> bool:
        return self._settings.vertical_scrollbar_always_visible
    
    @vertical_scrollbar_always_visible.setter
    def vertical_scrollbar_always_visible(self, value: bool):
        if value != self._settings.vertical_scrollbar_always_visible:
            self._settings.vertical_scrollbar_always_visible = value
            self._save_setting('vertical_scrollbar_always_visible')
    
    @property
    def horizontal_scrollbar_always_visible(self) -> bool:
        return self._settings.horizontal_scrollbar_always_visible
    
    @horizontal_scrollbar_always_visible.setter
    def horizontal_scrollbar_always_visible(self, value: bool):
        if value != self._settings.horizontal_scrollbar_always_visible:
            self._settings.horizontal_scrollbar_always_visible = value
            self._save_setting('horizontal_scrollbar_always_visible')
    
    @property
    def last_scroll_x(self) -> int:
        return self._settings.last_scroll_x
    
    @last_scroll_x.setter
    def last_scroll_x(self, value: int):
        if value != self._settings.last_scroll_x:
            self._settings.last_scroll_x = max(0, value)
            self._save_setting('last_scroll_x')
    
    @property
    def last_scroll_y(self) -> int:
        return self._settings.last_scroll_y
    
    @last_scroll_y.setter
    def last_scroll_y(self, value: int):
        if value != self._settings.last_scroll_y:
            self._settings.last_scroll_y = max(0, value)
            self._save_setting('last_scroll_y')
    
    @property
    def restore_scroll_position(self) -> bool:
        return self._settings.restore_scroll_position
    
    @restore_scroll_position.setter
    def restore_scroll_position(self, value: bool):
        if value != self._settings.restore_scroll_position:
            self._settings.restore_scroll_position = value
            self._save_setting('restore_scroll_position')
    
    @property
    def inspector_visible(self) -> bool:
        return self._settings.inspector_visible
    
    @inspector_visible.setter
    def inspector_visible(self, value: bool):
        if value != self._settings.inspector_visible:
            self._settings.inspector_visible = value
            self._save_setting('inspector_visible')
    
    @property
    def inspector_width(self) -> int:
        return self._settings.inspector_width
    
    @inspector_width.setter
    def inspector_width(self, value: int):
        if value != self._settings.inspector_width:
            self._settings.inspector_width = max(150, min(value, 500))
            self._save_setting('inspector_width')
    
    @property
    def recent_layer_colors(self) -> list:
        return self._settings.recent_layer_colors.copy()
    
    def add_recent_layer_color(self, color: str):
        """Add a color to the recent colors list."""
        colors = self._settings.recent_layer_colors
        if color in colors:
            colors.remove(color)
        colors.insert(0, color)
        # Keep only last 10
        self._settings.recent_layer_colors = colors[:10]
        self._save_setting('recent_layer_colors')
    
    # =========================================================================
    # Event Time Styling Properties
    # =========================================================================
    
    @property
    def event_time_font_size(self) -> int:
        return self._settings.event_time_font_size
    
    @event_time_font_size.setter
    def event_time_font_size(self, value: int):
        if value != self._settings.event_time_font_size:
            self._settings.event_time_font_size = max(6, min(value, 24))
            self._save_setting('event_time_font_size')
    
    @property
    def event_time_font_family(self) -> str:
        return self._settings.event_time_font_family
    
    @event_time_font_family.setter
    def event_time_font_family(self, value: str):
        if value != self._settings.event_time_font_family:
            valid_families = {"monospace", "default", "small"}
            if value not in valid_families:
                raise ValueError(
                    f"Invalid font family: '{value}'. "
                    f"Valid options: {', '.join(sorted(valid_families))}"
                )
            self._settings.event_time_font_family = value
            self._save_setting('event_time_font_family')
    
    @property
    def event_time_major_color(self) -> str:
        return self._settings.event_time_major_color
    
    @event_time_major_color.setter
    def event_time_major_color(self, value: str):
        if value != self._settings.event_time_major_color:
            # Validate hex color format
            if not isinstance(value, str) or not value.startswith('#'):
                raise ValueError(f"Invalid color format: {value}. Must be hex color (e.g., '#F0F0F5')")
            try:
                QColor(value)  # Validate color can be parsed
            except (ValueError, TypeError):
                raise ValueError(f"Invalid color value: {value}. Must be valid hex color.")
            self._settings.event_time_major_color = value
            self._save_setting('event_time_major_color')
    
    @property
    def event_time_minor_color(self) -> str:
        return self._settings.event_time_minor_color
    
    @event_time_minor_color.setter
    def event_time_minor_color(self, value: str):
        if value != self._settings.event_time_minor_color:
            # Validate hex color format
            if not isinstance(value, str) or not value.startswith('#'):
                raise ValueError(f"Invalid color format: {value}. Must be hex color (e.g., '#78787D')")
            try:
                QColor(value)  # Validate color can be parsed
            except (ValueError, TypeError):
                raise ValueError(f"Invalid color value: {value}. Must be valid hex color.")
            self._settings.event_time_minor_color = value
            self._save_setting('event_time_minor_color')
    
    # =========================================================================
    # Block Event Styling Properties (Notes/Events with Duration)
    # =========================================================================
    
    @property
    def block_event_height(self) -> int:
        return self._settings.block_event_height
    
    @block_event_height.setter
    def block_event_height(self, value: int):
        if value != self._settings.block_event_height:
            self._settings.block_event_height = max(16, min(value, 100))
            self._save_setting('block_event_height')
    
    @property
    def block_event_border_radius(self) -> int:
        return self._settings.block_event_border_radius
    
    @block_event_border_radius.setter
    def block_event_border_radius(self, value: int):
        if value != self._settings.block_event_border_radius:
            self._settings.block_event_border_radius = max(0, min(value, 20))
            self._save_setting('block_event_border_radius')
    
    @property
    def block_event_border_width(self) -> int:
        return self._settings.block_event_border_width
    
    @block_event_border_width.setter
    def block_event_border_width(self, value: int):
        if value != self._settings.block_event_border_width:
            self._settings.block_event_border_width = max(0, min(value, 5))
            self._save_setting('block_event_border_width')
    
    @property
    def block_event_label_font_size(self) -> int:
        return self._settings.block_event_label_font_size
    
    @block_event_label_font_size.setter
    def block_event_label_font_size(self, value: int):
        if value != self._settings.block_event_label_font_size:
            self._settings.block_event_label_font_size = max(6, min(value, 24))
            self._save_setting('block_event_label_font_size')
    
    @property
    def block_event_border_darken_percent(self) -> int:
        return self._settings.block_event_border_darken_percent
    
    @block_event_border_darken_percent.setter
    def block_event_border_darken_percent(self, value: int):
        if value != self._settings.block_event_border_darken_percent:
            self._settings.block_event_border_darken_percent = max(100, min(value, 200))
            self._save_setting('block_event_border_darken_percent')
    
    # Block Event Qt Graphics Item Properties
    @property
    def block_event_opacity(self) -> float:
        return self._settings.block_event_opacity
    
    @block_event_opacity.setter
    def block_event_opacity(self, value: float):
        if value != self._settings.block_event_opacity:
            self._settings.block_event_opacity = max(0.0, min(value, 1.0))
            self._save_setting('block_event_opacity')
    
    @property
    def block_event_z_value(self) -> float:
        return self._settings.block_event_z_value
    
    @block_event_z_value.setter
    def block_event_z_value(self, value: float):
        if value != self._settings.block_event_z_value:
            self._settings.block_event_z_value = max(-1000.0, min(value, 1000.0))
            self._save_setting('block_event_z_value')
    
    @property
    def block_event_rotation(self) -> float:
        return self._settings.block_event_rotation
    
    @block_event_rotation.setter
    def block_event_rotation(self, value: float):
        if value != self._settings.block_event_rotation:
            self._settings.block_event_rotation = value % 360.0  # Normalize to 0-360
            self._save_setting('block_event_rotation')
    
    @property
    def block_event_scale(self) -> float:
        return self._settings.block_event_scale
    
    @block_event_scale.setter
    def block_event_scale(self, value: float):
        if value != self._settings.block_event_scale:
            self._settings.block_event_scale = max(0.1, min(value, 5.0))
            self._save_setting('block_event_scale')
    
    # Block Event Graphics Effects
    @property
    def block_event_drop_shadow_enabled(self) -> bool:
        return self._settings.block_event_drop_shadow_enabled
    
    @block_event_drop_shadow_enabled.setter
    def block_event_drop_shadow_enabled(self, value: bool):
        if value != self._settings.block_event_drop_shadow_enabled:
            self._settings.block_event_drop_shadow_enabled = value
            self._save_setting('block_event_drop_shadow_enabled')
    
    @property
    def block_event_drop_shadow_blur_radius(self) -> float:
        return self._settings.block_event_drop_shadow_blur_radius
    
    @block_event_drop_shadow_blur_radius.setter
    def block_event_drop_shadow_blur_radius(self, value: float):
        if value != self._settings.block_event_drop_shadow_blur_radius:
            self._settings.block_event_drop_shadow_blur_radius = max(0.0, min(value, 50.0))
            self._save_setting('block_event_drop_shadow_blur_radius')
    
    @property
    def block_event_drop_shadow_offset_x(self) -> float:
        return self._settings.block_event_drop_shadow_offset_x
    
    @block_event_drop_shadow_offset_x.setter
    def block_event_drop_shadow_offset_x(self, value: float):
        if value != self._settings.block_event_drop_shadow_offset_x:
            self._settings.block_event_drop_shadow_offset_x = max(-50.0, min(value, 50.0))
            self._save_setting('block_event_drop_shadow_offset_x')
    
    @property
    def block_event_drop_shadow_offset_y(self) -> float:
        return self._settings.block_event_drop_shadow_offset_y
    
    @block_event_drop_shadow_offset_y.setter
    def block_event_drop_shadow_offset_y(self, value: float):
        if value != self._settings.block_event_drop_shadow_offset_y:
            self._settings.block_event_drop_shadow_offset_y = max(-50.0, min(value, 50.0))
            self._save_setting('block_event_drop_shadow_offset_y')
    
    @property
    def block_event_drop_shadow_color(self) -> str:
        return self._settings.block_event_drop_shadow_color
    
    @block_event_drop_shadow_color.setter
    def block_event_drop_shadow_color(self, value: str):
        if value != self._settings.block_event_drop_shadow_color:
            # Validate hex color format
            if not isinstance(value, str) or not value.startswith('#'):
                raise ValueError(f"Invalid color format: {value}. Must be hex color (e.g., '#000000')")
            try:
                QColor(value)  # Validate color can be parsed
            except (ValueError, TypeError):
                raise ValueError(f"Invalid color value: {value}. Must be valid hex color.")
            self._settings.block_event_drop_shadow_color = value
            self._save_setting('block_event_drop_shadow_color')
    
    @property
    def block_event_drop_shadow_opacity(self) -> float:
        return self._settings.block_event_drop_shadow_opacity
    
    @block_event_drop_shadow_opacity.setter
    def block_event_drop_shadow_opacity(self, value: float):
        if value != self._settings.block_event_drop_shadow_opacity:
            self._settings.block_event_drop_shadow_opacity = max(0.0, min(value, 1.0))
            self._save_setting('block_event_drop_shadow_opacity')
    
    # =========================================================================
    # Marker Event Styling Properties (One-Shot/Instant Events)
    # =========================================================================
    
    @property
    def marker_event_shape(self) -> str:
        return self._settings.marker_event_shape
    
    @marker_event_shape.setter
    def marker_event_shape(self, value: str):
        if value != self._settings.marker_event_shape:
            valid_shapes = {
                "diamond", "circle", "square", "triangle_up", "triangle_down",
                "triangle_left", "triangle_right", "arrow_up", "arrow_down",
                "arrow_left", "arrow_right", "star", "cross", "plus"
            }
            if value not in valid_shapes:
                raise ValueError(
                    f"Invalid marker shape: '{value}'. "
                    f"Valid options: {', '.join(sorted(valid_shapes))}"
                )
            self._settings.marker_event_shape = value
            self._save_setting('marker_event_shape')
    
    @property
    def marker_event_width(self) -> int:
        return self._settings.marker_event_width
    
    @marker_event_width.setter
    def marker_event_width(self, value: int):
        if value != self._settings.marker_event_width:
            self._settings.marker_event_width = max(4, min(value, 50))
            self._save_setting('marker_event_width')
    
    @property
    def marker_event_border_width(self) -> int:
        return self._settings.marker_event_border_width
    
    @marker_event_border_width.setter
    def marker_event_border_width(self, value: int):
        if value != self._settings.marker_event_border_width:
            self._settings.marker_event_border_width = max(0, min(value, 5))
            self._save_setting('marker_event_border_width')
    
    @property
    def marker_event_border_darken_percent(self) -> int:
        return self._settings.marker_event_border_darken_percent
    
    @marker_event_border_darken_percent.setter
    def marker_event_border_darken_percent(self, value: int):
        if value != self._settings.marker_event_border_darken_percent:
            self._settings.marker_event_border_darken_percent = max(100, min(value, 200))
            self._save_setting('marker_event_border_darken_percent')
    
    # Marker Event Qt Graphics Item Properties
    @property
    def marker_event_opacity(self) -> float:
        return self._settings.marker_event_opacity
    
    @marker_event_opacity.setter
    def marker_event_opacity(self, value: float):
        if value != self._settings.marker_event_opacity:
            self._settings.marker_event_opacity = max(0.0, min(value, 1.0))
            self._save_setting('marker_event_opacity')
    
    @property
    def marker_event_z_value(self) -> float:
        return self._settings.marker_event_z_value
    
    @marker_event_z_value.setter
    def marker_event_z_value(self, value: float):
        if value != self._settings.marker_event_z_value:
            self._settings.marker_event_z_value = max(-1000.0, min(value, 1000.0))
            self._save_setting('marker_event_z_value')
    
    @property
    def marker_event_rotation(self) -> float:
        return self._settings.marker_event_rotation
    
    @marker_event_rotation.setter
    def marker_event_rotation(self, value: float):
        if value != self._settings.marker_event_rotation:
            self._settings.marker_event_rotation = value % 360.0  # Normalize to 0-360
            self._save_setting('marker_event_rotation')
    
    @property
    def marker_event_scale(self) -> float:
        return self._settings.marker_event_scale
    
    @marker_event_scale.setter
    def marker_event_scale(self, value: float):
        if value != self._settings.marker_event_scale:
            self._settings.marker_event_scale = max(0.1, min(value, 5.0))
            self._save_setting('marker_event_scale')
    
    # Marker Event Graphics Effects
    @property
    def marker_event_drop_shadow_enabled(self) -> bool:
        return self._settings.marker_event_drop_shadow_enabled
    
    @marker_event_drop_shadow_enabled.setter
    def marker_event_drop_shadow_enabled(self, value: bool):
        if value != self._settings.marker_event_drop_shadow_enabled:
            self._settings.marker_event_drop_shadow_enabled = value
            self._save_setting('marker_event_drop_shadow_enabled')
    
    @property
    def marker_event_drop_shadow_blur_radius(self) -> float:
        return self._settings.marker_event_drop_shadow_blur_radius
    
    @marker_event_drop_shadow_blur_radius.setter
    def marker_event_drop_shadow_blur_radius(self, value: float):
        if value != self._settings.marker_event_drop_shadow_blur_radius:
            self._settings.marker_event_drop_shadow_blur_radius = max(0.0, min(value, 50.0))
            self._save_setting('marker_event_drop_shadow_blur_radius')
    
    @property
    def marker_event_drop_shadow_offset_x(self) -> float:
        return self._settings.marker_event_drop_shadow_offset_x
    
    @marker_event_drop_shadow_offset_x.setter
    def marker_event_drop_shadow_offset_x(self, value: float):
        if value != self._settings.marker_event_drop_shadow_offset_x:
            self._settings.marker_event_drop_shadow_offset_x = max(-50.0, min(value, 50.0))
            self._save_setting('marker_event_drop_shadow_offset_x')
    
    @property
    def marker_event_drop_shadow_offset_y(self) -> float:
        return self._settings.marker_event_drop_shadow_offset_y
    
    @marker_event_drop_shadow_offset_y.setter
    def marker_event_drop_shadow_offset_y(self, value: float):
        if value != self._settings.marker_event_drop_shadow_offset_y:
            self._settings.marker_event_drop_shadow_offset_y = max(-50.0, min(value, 50.0))
            self._save_setting('marker_event_drop_shadow_offset_y')
    
    @property
    def marker_event_drop_shadow_color(self) -> str:
        return self._settings.marker_event_drop_shadow_color
    
    @marker_event_drop_shadow_color.setter
    def marker_event_drop_shadow_color(self, value: str):
        if value != self._settings.marker_event_drop_shadow_color:
            # Validate hex color format
            if not isinstance(value, str) or not value.startswith('#'):
                raise ValueError(f"Invalid color format: {value}. Must be hex color (e.g., '#000000')")
            try:
                QColor(value)  # Validate color can be parsed
            except (ValueError, TypeError):
                raise ValueError(f"Invalid color value: {value}. Must be valid hex color.")
            self._settings.marker_event_drop_shadow_color = value
            self._save_setting('marker_event_drop_shadow_color')
    
    @property
    def marker_event_drop_shadow_opacity(self) -> float:
        return self._settings.marker_event_drop_shadow_opacity
    
    @marker_event_drop_shadow_opacity.setter
    def marker_event_drop_shadow_opacity(self, value: float):
        if value != self._settings.marker_event_drop_shadow_opacity:
            self._settings.marker_event_drop_shadow_opacity = max(0.0, min(value, 1.0))
            self._save_setting('marker_event_drop_shadow_opacity')


# =============================================================================
# Global Instance (Optional - for simpler access)
# =============================================================================

_global_settings_manager: Optional[TimelineSettingsManager] = None


def get_timeline_settings_manager() -> Optional[TimelineSettingsManager]:
    """Get the global timeline settings manager if one has been set."""
    return _global_settings_manager


def set_timeline_settings_manager(manager: TimelineSettingsManager):
    """Set the global timeline settings manager."""
    global _global_settings_manager
    _global_settings_manager = manager



