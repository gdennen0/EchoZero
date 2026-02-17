"""
Timeline Settings Panel

Modern DAW-style settings panel for timeline configuration.
Features collapsible sections, toggle switches, and a polished UI.
"""

from enum import Enum, auto
from typing import Optional, Dict, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QScrollArea, QPushButton,
    QSlider, QToolButton, QSizePolicy, QColorDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QSize
from PyQt6.QtGui import QIcon, QFont, QColor

# Local imports
from ..core.style import TimelineStyle as Colors, TimelineStyle as Spacing
from ..grid_system import GridSystem, TimebaseMode
from ui.qt_gui.design_system import border_radius


class PlayheadFollowMode(Enum):
    """Playhead follow/scroll behavior modes"""
    OFF = auto()           # No auto-scroll, playhead can go off screen
    PAGE = auto()          # Jump to next page when playhead reaches edge
    SMOOTH = auto()        # Smoothly scroll to keep playhead at ~75% position
    CENTER = auto()        # Keep playhead centered (like Reaper's center scroll)


class TimelineSettings:
    """Container for all timeline settings."""
    
    def __init__(self):
        self.follow_mode = PlayheadFollowMode.PAGE
        self.follow_during_playback_only = True
        self.scroll_margin = 0.15
        self.snap_enabled = True
        # Note: snap_interval removed - snapping now uses current grid interval
        self.show_grid_lines = True
        self.timebase_mode = TimebaseMode.TIMECODE
        self.frame_rate = 30.0
        self.show_event_labels = True
        self.show_event_duration_labels = False
        self.show_playhead_time = True
        self.highlight_current_event = True


class ToggleSwitch(QCheckBox):
    """Modern toggle switch widget."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(44, 24)
        self._update_style()
        self.stateChanged.connect(self._update_style)
    
    def _update_style(self):
        checked = self.isChecked()
        bg = Colors.ACCENT_BLUE.name() if checked else Colors.BG_MEDIUM.name()
        knob_x = "22px" if checked else "2px"
        
        self.setStyleSheet(f"""
            QCheckBox {{
                background-color: {bg};
                border-radius: {border_radius(12)};
                padding: 0;
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border-radius: {border_radius(10)};
                background-color: {Colors.TEXT_PRIMARY.name()};
                margin-left: {knob_x};
                margin-top: 2px;
                border: none;
            }}
        """)


class CollapsibleSection(QWidget):
    """Collapsible section with header and content."""
    
    def __init__(self, title: str, icon: str = "", parent=None):
        super().__init__(parent)
        self._expanded = True
        self._title = title
        self._icon = icon
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        self._header = QFrame()
        self._header.setFixedHeight(36)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_MEDIUM.name()};
                border-radius: {border_radius(6)};
            }}
            QFrame:hover {{
                background-color: {Colors.HOVER.name()};
            }}
        """)
        
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(8)
        
        # Expand/collapse arrow
        self._arrow = QLabel("▼")
        self._arrow.setFixedWidth(16)
        self._arrow.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10px;")
        header_layout.addWidget(self._arrow)
        
        # Icon (emoji)
        if self._icon:
            icon_label = QLabel(self._icon)
            icon_label.setStyleSheet("font-size: 14px;")
            header_layout.addWidget(icon_label)
        
        # Title
        title_label = QLabel(self._title)
        title_label.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY.name()};
            font-weight: 600;
            font-size: 12px;
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addWidget(self._header)
        
        # Content container
        self._content = QFrame()
        self._content.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_DARK.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-top: none;
                border-radius: 0 0 {border_radius(6)} {border_radius(6)};
            }}
        """)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 12, 12, 12)
        self._content_layout.setSpacing(12)
        
        layout.addWidget(self._content)
        
        # Click handler
        self._header.mousePressEvent = self._toggle
    
    def _toggle(self, event):
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._arrow.setText("▼" if self._expanded else "▶")
    
    def add_widget(self, widget: QWidget):
        self._content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        self._content_layout.addLayout(layout)
    
    def content_layout(self):
        return self._content_layout


class ColorPickerButton(QPushButton):
    """Color picker button that shows current color and opens color dialog."""
    
    color_changed = pyqtSignal(str)  # Emits hex color string
    
    def __init__(self, initial_color: str = None, parent=None):
        if initial_color is None:
            initial_color = Colors.TEXT_PRIMARY.name()
        super().__init__(parent)
        self._color = QColor(initial_color)
        self.setFixedSize(60, 28)
        self._update_color()
        self.clicked.connect(self._pick_color)
    
    def _update_color(self):
        """Update button appearance to show current color."""
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._color.name()};
                border: 2px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
            }}
            QPushButton:hover {{
                border-color: {Colors.ACCENT_BLUE.name()};
            }}
        """)
    
    def _pick_color(self):
        """Open color dialog and update color."""
        new_color = QColorDialog.getColor(self._color, self, "Select Color")
        if new_color.isValid() and new_color != self._color:
            self._color = new_color
            self._update_color()
            self.color_changed.emit(self._color.name())
    
    def set_color(self, hex_color: str):
        """Set color from hex string."""
        color = QColor(hex_color)
        if color.isValid():
            self._color = color
            self._update_color()
    
    def get_color(self) -> str:
        """Get current color as hex string."""
        return self._color.name()


class SettingRow(QWidget):
    """Single setting row with label and control."""
    
    def __init__(self, label: str, control: QWidget, description: str = "", parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Main row
        row = QHBoxLayout()
        row.setSpacing(12)
        
        # Label
        label_widget = QLabel(label)
        label_widget.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 12px;")
        row.addWidget(label_widget)
        row.addStretch()
        row.addWidget(control)
        
        layout.addLayout(row)
        
        # Description (if provided)
        if description:
            desc = QLabel(description)
            desc.setStyleSheet(f"color: {Colors.TEXT_DISABLED.name()}; font-size: 10px;")
            desc.setWordWrap(True)
            layout.addWidget(desc)


class SettingsPanel(QWidget):
    """
    Modern DAW-style settings panel for timeline configuration.
    
    Signals:
        settings_changed(): Emitted when any setting changes
        follow_mode_changed(PlayheadFollowMode): Follow mode changed
    """
    
    settings_changed = pyqtSignal()
    follow_mode_changed = pyqtSignal(object)
    
    def __init__(self, grid_system: GridSystem, timeline_widget=None, parent=None):
        super().__init__(parent)
        
        self._grid_system = grid_system
        self._timeline_widget = timeline_widget
        self._settings = TimelineSettings()
        self._updating = False
        self._layer_height_spinboxes: Dict[int, QSpinBox] = {}
        
        self._setup_ui()
        self._sync_from_grid_system()
        
        if self._timeline_widget:
            QTimer.singleShot(100, self._update_layer_controls)
    
    @property
    def settings(self) -> TimelineSettings:
        return self._settings
    
    def _create_combo(self) -> QComboBox:
        """Create a styled combo box."""
        combo = QComboBox()
        combo.setFixedHeight(28)
        combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {Colors.BG_LIGHT.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 4px 8px;
                min-width: 120px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {Colors.TEXT_SECONDARY.name()};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                selection-background-color: {Colors.ACCENT_BLUE.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 4px;
            }}
        """)
        return combo
    
    def _create_spinbox(self, min_val: int = 0, max_val: int = 100, suffix: str = "") -> QSpinBox:
        """Create a styled spin box."""
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setFixedHeight(28)
        spin.setFixedWidth(80)
        if suffix:
            spin.setSuffix(suffix)
        spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: {Colors.BG_LIGHT.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 4px 8px;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                width: 16px;
                border: none;
                background-color: {Colors.BG_MEDIUM.name()};
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                background-color: {Colors.HOVER.name()};
            }}
        """)
        return spin
    
    def _create_double_spinbox(self, min_val: float = 0.0, max_val: float = 100.0, step: float = 0.1, decimals: int = 1) -> QDoubleSpinBox:
        """Create a styled double spin box."""
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setFixedHeight(28)
        spin.setFixedWidth(80)
        spin.setStyleSheet(f"""
            QDoubleSpinBox {{
                background-color: {Colors.BG_LIGHT.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 4px 8px;
            }}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width: 16px;
                border: none;
                background-color: {Colors.BG_MEDIUM.name()};
            }}
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: {Colors.HOVER.name()};
            }}
        """)
        return spin
    
    def _create_button(self, text: str, accent: bool = False) -> QPushButton:
        """Create a styled button."""
        btn = QPushButton(text)
        btn.setFixedHeight(28)
        
        if accent:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Colors.ACCENT_BLUE.name()};
                    color: {Colors.TEXT_PRIMARY.name()};
                    border: none;
                    border-radius: {border_radius(4)};
                    padding: 4px 12px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: {Colors.ACCENT_BLUE.lighter(110).name()};
                }}
                QPushButton:pressed {{
                    background-color: {Colors.ACCENT_BLUE.darker(110).name()};
                }}
            """)
        else:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Colors.BG_LIGHT.name()};
                    color: {Colors.TEXT_PRIMARY.name()};
                    border: 1px solid {Colors.BORDER.name()};
                    border-radius: {border_radius(4)};
                    padding: 4px 12px;
                }}
                QPushButton:hover {{
                    background-color: {Colors.HOVER.name()};
                }}
                QPushButton:pressed {{
                    background-color: {Colors.BG_DARK.name()};
                }}
            """)
        return btn
    
    def _setup_ui(self):
        """Setup the settings panel UI."""
        # Set fixed width to prevent stretching beyond content
        self.setFixedWidth(320)
        
        # Set size policy to prevent horizontal stretching
        size_policy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.setSizePolicy(size_policy)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = QFrame()
        header.setFixedHeight(44)
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_MEDIUM.name()};
                border-bottom: 1px solid {Colors.BORDER.name()};
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 0, 16, 0)
        
        title = QLabel("Timeline Settings")
        title.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY.name()};
            font-weight: 600;
            font-size: 14px;
        """)
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # Reset button
        reset_btn = self._create_button("Reset")
        reset_btn.setFixedWidth(60)
        reset_btn.clicked.connect(self._reset_to_defaults)
        header_layout.addWidget(reset_btn)
        
        main_layout.addWidget(header)
        
        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {Colors.BG_DARK.name()};
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: {Colors.BG_DARK.name()};
                width: 8px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {Colors.TEXT_DISABLED.name()};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)
        
        content = QWidget()
        content.setStyleSheet(f"background-color: {Colors.BG_DARK.name()};")
        # Prevent content widget from stretching horizontally
        content_size_policy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        content.setSizePolicy(content_size_policy)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(12)
        
        # ==================== Playback Section ====================
        playback_section = CollapsibleSection("Playback", "▶")
        
        # Follow mode
        self._follow_combo = self._create_combo()
        self._follow_combo.addItem("Off - Manual scroll", PlayheadFollowMode.OFF)
        self._follow_combo.addItem("Page - Jump at edge", PlayheadFollowMode.PAGE)
        self._follow_combo.addItem("Smooth - Continuous", PlayheadFollowMode.SMOOTH)
        self._follow_combo.addItem("Center - Keep centered", PlayheadFollowMode.CENTER)
        self._follow_combo.setCurrentIndex(1)
        self._follow_combo.currentIndexChanged.connect(self._on_follow_mode_changed)
        playback_section.add_widget(SettingRow("Follow Mode", self._follow_combo))
        
        # Playback only toggle
        self._playback_only_toggle = ToggleSwitch()
        self._playback_only_toggle.setChecked(True)
        self._playback_only_toggle.stateChanged.connect(self._on_setting_changed)
        playback_section.add_widget(SettingRow(
            "Only During Playback", 
            self._playback_only_toggle,
            "Auto-scroll only while playing"
        ))
        
        content_layout.addWidget(playback_section)
        
        # ==================== Grid & Snap Section ====================
        grid_section = CollapsibleSection("Grid & Snap", "⊞")
        
        # Snap toggle
        self._snap_toggle = ToggleSwitch()
        self._snap_toggle.setChecked(True)
        self._snap_toggle.stateChanged.connect(self._on_snap_changed)
        grid_section.add_widget(SettingRow(
            "Snap to Grid", 
            self._snap_toggle,
            "Snap events to the current grid lines"
        ))
        
        # Snap interval mode (like DAW - 1f, 5f, 10f, etc.)
        self._snap_interval_combo = self._create_combo()
        self._snap_interval_combo.addItem("Auto (Grid)", "auto")
        self._snap_interval_combo.addItem("1 Frame", "1f")
        self._snap_interval_combo.addItem("2 Frames", "2f")
        self._snap_interval_combo.addItem("5 Frames", "5f")
        self._snap_interval_combo.addItem("10 Frames", "10f")
        self._snap_interval_combo.addItem("1 Second", "1s")
        self._snap_interval_combo.currentIndexChanged.connect(self._on_snap_interval_changed)
        grid_section.add_widget(SettingRow(
            "Snap Interval",
            self._snap_interval_combo,
            "Snap granularity (frames based on current FPS)"
        ))
        
        # Grid lines toggle
        self._grid_toggle = ToggleSwitch()
        self._grid_toggle.setChecked(True)
        self._grid_toggle.stateChanged.connect(self._on_grid_changed)
        grid_section.add_widget(SettingRow("Show Grid Lines", self._grid_toggle))
        
        # Note: Grid line intervals are automatically calculated from timebase/FPS settings.
        # No manual frequency controls needed - grid adapts to zoom level.
        
        content_layout.addWidget(grid_section)
        
        # ==================== Time Display Section ====================
        time_section = CollapsibleSection("Time Display", "")
        
        # Timebase format
        self._timebase_combo = self._create_combo()
        self._timebase_combo.addItem("Timecode (HH:MM:SS)", TimebaseMode.TIMECODE)
        self._timebase_combo.addItem("Seconds (0.000)", TimebaseMode.SECONDS)
        self._timebase_combo.addItem("Frames", TimebaseMode.FRAMES)
        self._timebase_combo.addItem("Milliseconds", TimebaseMode.MILLISECONDS)
        self._timebase_combo.currentIndexChanged.connect(self._on_timebase_changed)
        time_section.add_widget(SettingRow("Time Format", self._timebase_combo))
        
        # Frame rate
        self._fps_combo = self._create_combo()
        for name, fps in self._grid_system.get_frame_rate_options():
            self._fps_combo.addItem(name, fps)
        self._fps_combo.setCurrentIndex(4)
        self._fps_combo.currentIndexChanged.connect(self._on_fps_changed)
        time_section.add_widget(SettingRow("Frame Rate", self._fps_combo))
        
        # Event time styling controls
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            settings_mgr = self._timeline_widget.settings_manager
            
            # Font size
            self._time_font_size_spin = self._create_spinbox(6, 24, " px")
            self._time_font_size_spin.setValue(settings_mgr.event_time_font_size)
            self._time_font_size_spin.valueChanged.connect(self._on_time_font_size_changed)
            time_section.add_widget(SettingRow("Font Size", self._time_font_size_spin))
            
            # Font family
            self._time_font_family_combo = self._create_combo()
            self._time_font_family_combo.addItem("Monospace", "monospace")
            self._time_font_family_combo.addItem("Default", "default")
            self._time_font_family_combo.addItem("Small", "small")
            current_family = settings_mgr.event_time_font_family
            for i in range(self._time_font_family_combo.count()):
                if self._time_font_family_combo.itemData(i) == current_family:
                    self._time_font_family_combo.setCurrentIndex(i)
                    break
            self._time_font_family_combo.currentIndexChanged.connect(self._on_time_font_family_changed)
            time_section.add_widget(SettingRow("Font Family", self._time_font_family_combo))
            
            # Major tick color
            self._time_major_color_btn = ColorPickerButton(settings_mgr.event_time_major_color)
            self._time_major_color_btn.color_changed.connect(self._on_time_major_color_changed)
            time_section.add_widget(SettingRow("Major Tick Color", self._time_major_color_btn))
            
            # Minor tick color
            self._time_minor_color_btn = ColorPickerButton(settings_mgr.event_time_minor_color)
            self._time_minor_color_btn.color_changed.connect(self._on_time_minor_color_changed)
            time_section.add_widget(SettingRow("Minor Tick Color", self._time_minor_color_btn))
        
        content_layout.addWidget(time_section)
        
        # ==================== Block Event Styling Section (Notes) ====================
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            settings_mgr = self._timeline_widget.settings_manager
            block_section = CollapsibleSection("Block Events (Notes)", "")
            
            # Block event height
            self._block_event_height_spin = self._create_spinbox(16, 100, " px")
            self._block_event_height_spin.setValue(settings_mgr.block_event_height)
            self._block_event_height_spin.valueChanged.connect(self._on_block_event_height_changed)
            block_section.add_widget(SettingRow("Height", self._block_event_height_spin, "Height of block events"))
            
            # Border radius
            self._block_event_border_radius_spin = self._create_spinbox(0, 20, " px")
            self._block_event_border_radius_spin.setValue(settings_mgr.block_event_border_radius)
            self._block_event_border_radius_spin.valueChanged.connect(self._on_block_event_border_radius_changed)
            block_section.add_widget(SettingRow("Border Radius", self._block_event_border_radius_spin, "Corner rounding"))
            
            # Border width
            self._block_event_border_width_spin = self._create_spinbox(0, 5, " px")
            self._block_event_border_width_spin.setValue(settings_mgr.block_event_border_width)
            self._block_event_border_width_spin.valueChanged.connect(self._on_block_event_border_width_changed)
            block_section.add_widget(SettingRow("Border Width", self._block_event_border_width_spin, "Event border thickness"))
            
            # Label font size
            self._block_event_label_font_size_spin = self._create_spinbox(6, 24, " px")
            self._block_event_label_font_size_spin.setValue(settings_mgr.block_event_label_font_size)
            self._block_event_label_font_size_spin.valueChanged.connect(self._on_block_event_label_font_size_changed)
            block_section.add_widget(SettingRow("Label Font Size", self._block_event_label_font_size_spin, "Text size in events"))
            
            # Border darken percent
            self._block_event_border_darken_spin = self._create_spinbox(100, 200, "%")
            self._block_event_border_darken_spin.setValue(settings_mgr.block_event_border_darken_percent)
            self._block_event_border_darken_spin.valueChanged.connect(self._on_block_event_border_darken_changed)
            block_section.add_widget(SettingRow("Border Darkness", self._block_event_border_darken_spin, "How much darker the border is (100 = same, 200 = very dark)"))
            
            # Separator
            sep1 = QFrame()
            sep1.setFixedHeight(1)
            sep1.setStyleSheet(f"background-color: {Colors.BORDER.name()};")
            block_section.add_widget(sep1)
            
            # Qt Graphics Item Properties
            # Opacity
            self._block_event_opacity_spin = QDoubleSpinBox()
            self._block_event_opacity_spin.setRange(0.0, 1.0)
            self._block_event_opacity_spin.setSingleStep(0.1)
            self._block_event_opacity_spin.setDecimals(1)
            self._block_event_opacity_spin.setFixedHeight(28)
            self._block_event_opacity_spin.setFixedWidth(80)
            self._block_event_opacity_spin.setValue(settings_mgr.block_event_opacity)
            self._block_event_opacity_spin.valueChanged.connect(self._on_block_event_opacity_changed)
            self._block_event_opacity_spin.setStyleSheet(self._create_spinbox().styleSheet())
            block_section.add_widget(SettingRow("Opacity", self._block_event_opacity_spin, "Transparency (0.0 = transparent, 1.0 = opaque)"))
            
            # Z-value
            self._block_event_z_value_spin = QDoubleSpinBox()
            self._block_event_z_value_spin.setRange(-1000.0, 1000.0)
            self._block_event_z_value_spin.setSingleStep(1.0)
            self._block_event_z_value_spin.setDecimals(0)
            self._block_event_z_value_spin.setFixedHeight(28)
            self._block_event_z_value_spin.setFixedWidth(80)
            self._block_event_z_value_spin.setValue(settings_mgr.block_event_z_value)
            self._block_event_z_value_spin.valueChanged.connect(self._on_block_event_z_value_changed)
            self._block_event_z_value_spin.setStyleSheet(self._create_spinbox().styleSheet())
            block_section.add_widget(SettingRow("Stacking Order", self._block_event_z_value_spin, "Z-value (higher = drawn on top)"))
            
            # Rotation
            self._block_event_rotation_spin = QDoubleSpinBox()
            self._block_event_rotation_spin.setRange(0.0, 360.0)
            self._block_event_rotation_spin.setSingleStep(1.0)
            self._block_event_rotation_spin.setDecimals(0)
            self._block_event_rotation_spin.setSuffix("°")
            self._block_event_rotation_spin.setFixedHeight(28)
            self._block_event_rotation_spin.setFixedWidth(80)
            self._block_event_rotation_spin.setValue(settings_mgr.block_event_rotation)
            self._block_event_rotation_spin.valueChanged.connect(self._on_block_event_rotation_changed)
            self._block_event_rotation_spin.setStyleSheet(self._create_spinbox().styleSheet())
            block_section.add_widget(SettingRow("Rotation", self._block_event_rotation_spin, "Rotation in degrees"))
            
            # Scale
            self._block_event_scale_spin = QDoubleSpinBox()
            self._block_event_scale_spin.setRange(0.1, 5.0)
            self._block_event_scale_spin.setSingleStep(0.1)
            self._block_event_scale_spin.setDecimals(1)
            self._block_event_scale_spin.setFixedHeight(28)
            self._block_event_scale_spin.setFixedWidth(80)
            self._block_event_scale_spin.setValue(settings_mgr.block_event_scale)
            self._block_event_scale_spin.valueChanged.connect(self._on_block_event_scale_changed)
            self._block_event_scale_spin.setStyleSheet(self._create_spinbox().styleSheet())
            block_section.add_widget(SettingRow("Scale", self._block_event_scale_spin, "Size multiplier (1.0 = normal)"))
            
            # Separator
            sep2 = QFrame()
            sep2.setFixedHeight(1)
            sep2.setStyleSheet(f"background-color: {Colors.BORDER.name()};")
            block_section.add_widget(sep2)
            
            # Drop Shadow
            self._block_event_shadow_toggle = ToggleSwitch()
            self._block_event_shadow_toggle.setChecked(settings_mgr.block_event_drop_shadow_enabled)
            self._block_event_shadow_toggle.stateChanged.connect(self._on_block_event_shadow_enabled_changed)
            block_section.add_widget(SettingRow("Drop Shadow", self._block_event_shadow_toggle))
            
            # Shadow blur radius
            self._block_event_shadow_blur_spin = QDoubleSpinBox()
            self._block_event_shadow_blur_spin.setRange(0.0, 50.0)
            self._block_event_shadow_blur_spin.setSingleStep(1.0)
            self._block_event_shadow_blur_spin.setDecimals(1)
            self._block_event_shadow_blur_spin.setFixedHeight(28)
            self._block_event_shadow_blur_spin.setFixedWidth(80)
            self._block_event_shadow_blur_spin.setValue(settings_mgr.block_event_drop_shadow_blur_radius)
            self._block_event_shadow_blur_spin.valueChanged.connect(self._on_block_event_shadow_blur_changed)
            self._block_event_shadow_blur_spin.setStyleSheet(self._create_spinbox().styleSheet())
            block_section.add_widget(SettingRow("Shadow Blur", self._block_event_shadow_blur_spin, "Shadow blur radius"))
            
            # Shadow offset X
            self._block_event_shadow_offset_x_spin = QDoubleSpinBox()
            self._block_event_shadow_offset_x_spin.setRange(-50.0, 50.0)
            self._block_event_shadow_offset_x_spin.setSingleStep(1.0)
            self._block_event_shadow_offset_x_spin.setDecimals(1)
            self._block_event_shadow_offset_x_spin.setFixedHeight(28)
            self._block_event_shadow_offset_x_spin.setFixedWidth(80)
            self._block_event_shadow_offset_x_spin.setValue(settings_mgr.block_event_drop_shadow_offset_x)
            self._block_event_shadow_offset_x_spin.valueChanged.connect(self._on_block_event_shadow_offset_x_changed)
            self._block_event_shadow_offset_x_spin.setStyleSheet(self._create_spinbox().styleSheet())
            block_section.add_widget(SettingRow("Shadow Offset X", self._block_event_shadow_offset_x_spin))
            
            # Shadow offset Y
            self._block_event_shadow_offset_y_spin = QDoubleSpinBox()
            self._block_event_shadow_offset_y_spin.setRange(-50.0, 50.0)
            self._block_event_shadow_offset_y_spin.setSingleStep(1.0)
            self._block_event_shadow_offset_y_spin.setDecimals(1)
            self._block_event_shadow_offset_y_spin.setFixedHeight(28)
            self._block_event_shadow_offset_y_spin.setFixedWidth(80)
            self._block_event_shadow_offset_y_spin.setValue(settings_mgr.block_event_drop_shadow_offset_y)
            self._block_event_shadow_offset_y_spin.valueChanged.connect(self._on_block_event_shadow_offset_y_changed)
            self._block_event_shadow_offset_y_spin.setStyleSheet(self._create_spinbox().styleSheet())
            block_section.add_widget(SettingRow("Shadow Offset Y", self._block_event_shadow_offset_y_spin))
            
            # Shadow color
            self._block_event_shadow_color_btn = ColorPickerButton(settings_mgr.block_event_drop_shadow_color)
            self._block_event_shadow_color_btn.color_changed.connect(self._on_block_event_shadow_color_changed)
            block_section.add_widget(SettingRow("Shadow Color", self._block_event_shadow_color_btn))
            
            # Shadow opacity
            self._block_event_shadow_opacity_spin = QDoubleSpinBox()
            self._block_event_shadow_opacity_spin.setRange(0.0, 1.0)
            self._block_event_shadow_opacity_spin.setSingleStep(0.1)
            self._block_event_shadow_opacity_spin.setDecimals(1)
            self._block_event_shadow_opacity_spin.setFixedHeight(28)
            self._block_event_shadow_opacity_spin.setFixedWidth(80)
            self._block_event_shadow_opacity_spin.setValue(settings_mgr.block_event_drop_shadow_opacity)
            self._block_event_shadow_opacity_spin.valueChanged.connect(self._on_block_event_shadow_opacity_changed)
            self._block_event_shadow_opacity_spin.setStyleSheet(self._create_spinbox().styleSheet())
            block_section.add_widget(SettingRow("Shadow Opacity", self._block_event_shadow_opacity_spin))
            
            content_layout.addWidget(block_section)
            
            # ==================== Marker Event Styling Section (One-Shot) ====================
            marker_section = CollapsibleSection("Marker Events (One-Shot)", "")
            
            # Marker event shape
            self._marker_event_shape_combo = self._create_combo()
            self._marker_event_shape_combo.addItem("Diamond", "diamond")
            self._marker_event_shape_combo.addItem("Circle", "circle")
            self._marker_event_shape_combo.addItem("Square", "square")
            self._marker_event_shape_combo.addItem("Triangle Up", "triangle_up")
            self._marker_event_shape_combo.addItem("Triangle Down", "triangle_down")
            self._marker_event_shape_combo.addItem("Triangle Left", "triangle_left")
            self._marker_event_shape_combo.addItem("Triangle Right", "triangle_right")
            self._marker_event_shape_combo.addItem("Arrow Up", "arrow_up")
            self._marker_event_shape_combo.addItem("Arrow Down", "arrow_down")
            self._marker_event_shape_combo.addItem("Arrow Left", "arrow_left")
            self._marker_event_shape_combo.addItem("Arrow Right", "arrow_right")
            self._marker_event_shape_combo.addItem("Star", "star")
            self._marker_event_shape_combo.addItem("Cross (X)", "cross")
            self._marker_event_shape_combo.addItem("Plus (+)", "plus")
            current_shape = settings_mgr.marker_event_shape
            for i in range(self._marker_event_shape_combo.count()):
                if self._marker_event_shape_combo.itemData(i) == current_shape:
                    self._marker_event_shape_combo.setCurrentIndex(i)
                    break
            self._marker_event_shape_combo.currentIndexChanged.connect(self._on_marker_event_shape_changed)
            marker_section.add_widget(SettingRow("Shape", self._marker_event_shape_combo, "Visual shape of marker events"))
            
            # Marker event width
            self._marker_event_width_spin = self._create_spinbox(4, 50, " px")
            self._marker_event_width_spin.setValue(settings_mgr.marker_event_width)
            self._marker_event_width_spin.valueChanged.connect(self._on_marker_event_width_changed)
            marker_section.add_widget(SettingRow("Size", self._marker_event_width_spin, "Size of marker events"))
            
            # Border width
            self._marker_event_border_width_spin = self._create_spinbox(0, 5, " px")
            self._marker_event_border_width_spin.setValue(settings_mgr.marker_event_border_width)
            self._marker_event_border_width_spin.valueChanged.connect(self._on_marker_event_border_width_changed)
            marker_section.add_widget(SettingRow("Border Width", self._marker_event_border_width_spin, "Marker border thickness"))
            
            # Border darken percent
            self._marker_event_border_darken_spin = self._create_spinbox(100, 200, "%")
            self._marker_event_border_darken_spin.setValue(settings_mgr.marker_event_border_darken_percent)
            self._marker_event_border_darken_spin.valueChanged.connect(self._on_marker_event_border_darken_changed)
            marker_section.add_widget(SettingRow("Border Darkness", self._marker_event_border_darken_spin, "How much darker the border is (100 = same, 200 = very dark)"))
            
            # Separator
            sep3 = QFrame()
            sep3.setFixedHeight(1)
            sep3.setStyleSheet(f"background-color: {Colors.BORDER.name()};")
            marker_section.add_widget(sep3)
            
            # Qt Graphics Item Properties
            # Opacity
            self._marker_event_opacity_spin = QDoubleSpinBox()
            self._marker_event_opacity_spin.setRange(0.0, 1.0)
            self._marker_event_opacity_spin.setSingleStep(0.1)
            self._marker_event_opacity_spin.setDecimals(1)
            self._marker_event_opacity_spin.setFixedHeight(28)
            self._marker_event_opacity_spin.setFixedWidth(80)
            self._marker_event_opacity_spin.setValue(settings_mgr.marker_event_opacity)
            self._marker_event_opacity_spin.valueChanged.connect(self._on_marker_event_opacity_changed)
            self._marker_event_opacity_spin.setStyleSheet(self._create_spinbox().styleSheet())
            marker_section.add_widget(SettingRow("Opacity", self._marker_event_opacity_spin, "Transparency (0.0 = transparent, 1.0 = opaque)"))
            
            # Z-value
            self._marker_event_z_value_spin = QDoubleSpinBox()
            self._marker_event_z_value_spin.setRange(-1000.0, 1000.0)
            self._marker_event_z_value_spin.setSingleStep(1.0)
            self._marker_event_z_value_spin.setDecimals(0)
            self._marker_event_z_value_spin.setFixedHeight(28)
            self._marker_event_z_value_spin.setFixedWidth(80)
            self._marker_event_z_value_spin.setValue(settings_mgr.marker_event_z_value)
            self._marker_event_z_value_spin.valueChanged.connect(self._on_marker_event_z_value_changed)
            self._marker_event_z_value_spin.setStyleSheet(self._create_spinbox().styleSheet())
            marker_section.add_widget(SettingRow("Stacking Order", self._marker_event_z_value_spin, "Z-value (higher = drawn on top)"))
            
            # Rotation
            self._marker_event_rotation_spin = QDoubleSpinBox()
            self._marker_event_rotation_spin.setRange(0.0, 360.0)
            self._marker_event_rotation_spin.setSingleStep(1.0)
            self._marker_event_rotation_spin.setDecimals(0)
            self._marker_event_rotation_spin.setSuffix("°")
            self._marker_event_rotation_spin.setFixedHeight(28)
            self._marker_event_rotation_spin.setFixedWidth(80)
            self._marker_event_rotation_spin.setValue(settings_mgr.marker_event_rotation)
            self._marker_event_rotation_spin.valueChanged.connect(self._on_marker_event_rotation_changed)
            self._marker_event_rotation_spin.setStyleSheet(self._create_spinbox().styleSheet())
            marker_section.add_widget(SettingRow("Rotation", self._marker_event_rotation_spin, "Rotation in degrees"))
            
            # Scale
            self._marker_event_scale_spin = QDoubleSpinBox()
            self._marker_event_scale_spin.setRange(0.1, 5.0)
            self._marker_event_scale_spin.setSingleStep(0.1)
            self._marker_event_scale_spin.setDecimals(1)
            self._marker_event_scale_spin.setFixedHeight(28)
            self._marker_event_scale_spin.setFixedWidth(80)
            self._marker_event_scale_spin.setValue(settings_mgr.marker_event_scale)
            self._marker_event_scale_spin.valueChanged.connect(self._on_marker_event_scale_changed)
            self._marker_event_scale_spin.setStyleSheet(self._create_spinbox().styleSheet())
            marker_section.add_widget(SettingRow("Scale", self._marker_event_scale_spin, "Size multiplier (1.0 = normal)"))
            
            # Separator
            sep4 = QFrame()
            sep4.setFixedHeight(1)
            sep4.setStyleSheet(f"background-color: {Colors.BORDER.name()};")
            marker_section.add_widget(sep4)
            
            # Drop Shadow
            self._marker_event_shadow_toggle = ToggleSwitch()
            self._marker_event_shadow_toggle.setChecked(settings_mgr.marker_event_drop_shadow_enabled)
            self._marker_event_shadow_toggle.stateChanged.connect(self._on_marker_event_shadow_enabled_changed)
            marker_section.add_widget(SettingRow("Drop Shadow", self._marker_event_shadow_toggle))
            
            # Shadow blur radius
            self._marker_event_shadow_blur_spin = QDoubleSpinBox()
            self._marker_event_shadow_blur_spin.setRange(0.0, 50.0)
            self._marker_event_shadow_blur_spin.setSingleStep(1.0)
            self._marker_event_shadow_blur_spin.setDecimals(1)
            self._marker_event_shadow_blur_spin.setFixedHeight(28)
            self._marker_event_shadow_blur_spin.setFixedWidth(80)
            self._marker_event_shadow_blur_spin.setValue(settings_mgr.marker_event_drop_shadow_blur_radius)
            self._marker_event_shadow_blur_spin.valueChanged.connect(self._on_marker_event_shadow_blur_changed)
            self._marker_event_shadow_blur_spin.setStyleSheet(self._create_spinbox().styleSheet())
            marker_section.add_widget(SettingRow("Shadow Blur", self._marker_event_shadow_blur_spin, "Shadow blur radius"))
            
            # Shadow offset X
            self._marker_event_shadow_offset_x_spin = QDoubleSpinBox()
            self._marker_event_shadow_offset_x_spin.setRange(-50.0, 50.0)
            self._marker_event_shadow_offset_x_spin.setSingleStep(1.0)
            self._marker_event_shadow_offset_x_spin.setDecimals(1)
            self._marker_event_shadow_offset_x_spin.setFixedHeight(28)
            self._marker_event_shadow_offset_x_spin.setFixedWidth(80)
            self._marker_event_shadow_offset_x_spin.setValue(settings_mgr.marker_event_drop_shadow_offset_x)
            self._marker_event_shadow_offset_x_spin.valueChanged.connect(self._on_marker_event_shadow_offset_x_changed)
            self._marker_event_shadow_offset_x_spin.setStyleSheet(self._create_spinbox().styleSheet())
            marker_section.add_widget(SettingRow("Shadow Offset X", self._marker_event_shadow_offset_x_spin))
            
            # Shadow offset Y
            self._marker_event_shadow_offset_y_spin = QDoubleSpinBox()
            self._marker_event_shadow_offset_y_spin.setRange(-50.0, 50.0)
            self._marker_event_shadow_offset_y_spin.setSingleStep(1.0)
            self._marker_event_shadow_offset_y_spin.setDecimals(1)
            self._marker_event_shadow_offset_y_spin.setFixedHeight(28)
            self._marker_event_shadow_offset_y_spin.setFixedWidth(80)
            self._marker_event_shadow_offset_y_spin.setValue(settings_mgr.marker_event_drop_shadow_offset_y)
            self._marker_event_shadow_offset_y_spin.valueChanged.connect(self._on_marker_event_shadow_offset_y_changed)
            self._marker_event_shadow_offset_y_spin.setStyleSheet(self._create_spinbox().styleSheet())
            marker_section.add_widget(SettingRow("Shadow Offset Y", self._marker_event_shadow_offset_y_spin))
            
            # Shadow color
            self._marker_event_shadow_color_btn = ColorPickerButton(settings_mgr.marker_event_drop_shadow_color)
            self._marker_event_shadow_color_btn.color_changed.connect(self._on_marker_event_shadow_color_changed)
            marker_section.add_widget(SettingRow("Shadow Color", self._marker_event_shadow_color_btn))
            
            # Shadow opacity
            self._marker_event_shadow_opacity_spin = QDoubleSpinBox()
            self._marker_event_shadow_opacity_spin.setRange(0.0, 1.0)
            self._marker_event_shadow_opacity_spin.setSingleStep(0.1)
            self._marker_event_shadow_opacity_spin.setDecimals(1)
            self._marker_event_shadow_opacity_spin.setFixedHeight(28)
            self._marker_event_shadow_opacity_spin.setFixedWidth(80)
            self._marker_event_shadow_opacity_spin.setValue(settings_mgr.marker_event_drop_shadow_opacity)
            self._marker_event_shadow_opacity_spin.valueChanged.connect(self._on_marker_event_shadow_opacity_changed)
            self._marker_event_shadow_opacity_spin.setStyleSheet(self._create_spinbox().styleSheet())
            marker_section.add_widget(SettingRow("Shadow Opacity", self._marker_event_shadow_opacity_spin))
            
            content_layout.addWidget(marker_section)
        
        # ==================== Appearance Section ====================
        appearance_section = CollapsibleSection("Appearance", "")
        
        # Event labels
        self._event_labels_toggle = ToggleSwitch()
        # Get initial value from settings manager if available
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._event_labels_toggle.setChecked(self._timeline_widget.settings_manager.show_event_labels)
        else:
            self._event_labels_toggle.setChecked(True)
        self._event_labels_toggle.stateChanged.connect(self._on_setting_changed)
        appearance_section.add_widget(SettingRow("Event Labels", self._event_labels_toggle))
        
        # Duration labels
        self._duration_labels_toggle = ToggleSwitch()
        # Get initial value from settings manager if available
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._duration_labels_toggle.setChecked(self._timeline_widget.settings_manager.show_event_duration_labels)
        else:
            self._duration_labels_toggle.setChecked(False)
        self._duration_labels_toggle.stateChanged.connect(self._on_setting_changed)
        appearance_section.add_widget(SettingRow("Duration Labels", self._duration_labels_toggle))
        
        # Highlight at playhead
        self._highlight_toggle = ToggleSwitch()
        # Get initial value from settings manager if available
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._highlight_toggle.setChecked(self._timeline_widget.settings_manager.highlight_current_event)
        else:
            self._highlight_toggle.setChecked(True)
        self._highlight_toggle.stateChanged.connect(self._on_setting_changed)
        appearance_section.add_widget(SettingRow(
            "Highlight at Playhead", 
            self._highlight_toggle,
            "Highlight events under playhead"
        ))
        
        # Waveform settings (if settings manager available)
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            settings_mgr = self._timeline_widget.settings_manager
            
            # Show waveforms in timeline
            self._waveforms_timeline_toggle = ToggleSwitch()
            self._waveforms_timeline_toggle.setChecked(settings_mgr.show_waveforms_in_timeline)
            self._waveforms_timeline_toggle.stateChanged.connect(self._on_waveforms_timeline_changed)
            appearance_section.add_widget(SettingRow(
                "Show Waveforms in Clips",
                self._waveforms_timeline_toggle,
                "Display audio waveforms inside clip events on timeline"
            ))
            
            # Waveform opacity
            self._waveform_opacity_slider = QSlider(Qt.Orientation.Horizontal)
            self._waveform_opacity_slider.setRange(0, 100)
            self._waveform_opacity_slider.setValue(int(settings_mgr.waveform_opacity * 100))
            self._waveform_opacity_slider.setFixedWidth(120)
            self._waveform_opacity_slider.valueChanged.connect(self._on_waveform_opacity_changed)
            self._waveform_opacity_label = QLabel(f"{int(settings_mgr.waveform_opacity * 100)}%")
            self._waveform_opacity_label.setFixedWidth(40)
            self._waveform_opacity_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
            opacity_row = QHBoxLayout()
            opacity_row.addWidget(self._waveform_opacity_slider)
            opacity_row.addWidget(self._waveform_opacity_label)
            opacity_row.addStretch()
            opacity_widget = QWidget()
            opacity_widget.setLayout(opacity_row)
            appearance_section.add_widget(SettingRow(
                "Waveform Opacity",
                opacity_widget,
                "Transparency of waveform overlays (0-100%)"
            ))
            
            # Waveform resolution with Apply button
            resolution_widget = QWidget()
            resolution_layout = QHBoxLayout(resolution_widget)
            resolution_layout.setContentsMargins(0, 0, 0, 0)
            resolution_layout.setSpacing(8)
            
            self._waveform_resolution_spin = self._create_spinbox(5, 1000)
            self._waveform_resolution_spin.setValue(settings_mgr.waveform_resolution)
            self._waveform_resolution_spin.valueChanged.connect(self._on_waveform_resolution_changed)
            # Also handle editingFinished for manual typing
            self._waveform_resolution_spin.editingFinished.connect(self._on_waveform_resolution_editing_finished)
            resolution_layout.addWidget(self._waveform_resolution_spin)
            
            self._apply_resolution_btn = self._create_button("Apply", accent=True)
            self._apply_resolution_btn.setFixedWidth(60)
            self._apply_resolution_btn.clicked.connect(self._on_apply_waveform_resolution)
            resolution_layout.addWidget(self._apply_resolution_btn)
            
            appearance_section.add_widget(SettingRow(
                "Waveform Resolution",
                resolution_widget,
                "Points per second (5=blocky, 50=normal, 200+=detailed)"
            ))
            
            # Waveform minimum width (LOD threshold)
            self._waveform_min_width_spin = self._create_spinbox(5, 200)
            self._waveform_min_width_spin.setValue(settings_mgr.waveform_min_width)
            self._waveform_min_width_spin.valueChanged.connect(self._on_waveform_min_width_changed)
            appearance_section.add_widget(SettingRow(
                "Waveform Min Width",
                self._waveform_min_width_spin,
                "Minimum event width (pixels) to display waveforms. Events smaller than this will skip waveform rendering for performance."
            ))
        
        content_layout.addWidget(appearance_section)
        
        # ==================== Keyboard Shortcuts Section ====================
        shortcuts_section = CollapsibleSection("Keyboard Shortcuts", "⌨")
        shortcuts_btn = self._create_button("Configure Shortcuts...", accent=True)
        shortcuts_btn.clicked.connect(self._open_shortcuts_dialog)
        shortcuts_section.add_widget(SettingRow(
            "Shortcuts",
            shortcuts_btn,
            "Configure keyboard shortcuts for timeline operations"
        ))
        content_layout.addWidget(shortcuts_section)
        
        # ==================== Layers Section ====================
        self._layers_section = CollapsibleSection("Layers", "")
        
        # Default height
        default_row = QHBoxLayout()
        default_label = QLabel("Default Height")
        default_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 12px;")
        default_row.addWidget(default_label)
        default_row.addStretch()
        
        self._default_height_spin = self._create_spinbox(20, 200, " px")
        default_height = 40
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            default_height = self._timeline_widget.settings_manager.default_layer_height
        self._default_height_spin.setValue(default_height)
        self._default_height_spin.valueChanged.connect(self._on_default_height_changed)
        default_row.addWidget(self._default_height_spin)
        
        apply_btn = self._create_button("Apply All", accent=True)
        apply_btn.setFixedWidth(70)
        apply_btn.clicked.connect(self._apply_default_to_all)
        default_row.addWidget(apply_btn)
        
        default_widget = QWidget()
        default_widget.setLayout(default_row)
        self._layers_section.add_widget(default_widget)
        
        # Separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {Colors.BORDER.name()};")
        self._layers_section.add_widget(sep)
        
        # Layer list container
        self._layers_container = QWidget()
        self._layers_container_layout = QVBoxLayout(self._layers_container)
        self._layers_container_layout.setContentsMargins(0, 0, 0, 0)
        self._layers_container_layout.setSpacing(8)
        self._layers_section.add_widget(self._layers_container)
        
        content_layout.addWidget(self._layers_section)
        
        # Stretch at bottom
        content_layout.addStretch()
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll, 1)
    
    
    def _sync_from_grid_system(self):
        """Sync UI with current grid system settings."""
        self._updating = True

        self._snap_toggle.setChecked(self._grid_system.snap_enabled)
        
        # Sync snap interval combo from grid_system or settings_manager
        current_mode = self._grid_system.snap_interval_mode
        for i in range(self._snap_interval_combo.count()):
            if self._snap_interval_combo.itemData(i) == current_mode:
                self._snap_interval_combo.setCurrentIndex(i)
                break
        
        self._grid_toggle.setChecked(self._grid_system.settings.show_grid_lines)
        
        for i in range(self._timebase_combo.count()):
            if self._timebase_combo.itemData(i) == self._grid_system.timebase_mode:
                self._timebase_combo.setCurrentIndex(i)
                break
        
        for i in range(self._fps_combo.count()):
            if abs(self._fps_combo.itemData(i) - self._grid_system.frame_rate) < 0.01:
                self._fps_combo.setCurrentIndex(i)
                break
        
        self._updating = False
    
    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        self._updating = True
        
        self._follow_combo.setCurrentIndex(1)  # PAGE
        self._playback_only_toggle.setChecked(True)
        self._snap_toggle.setChecked(True)
        self._snap_interval_combo.setCurrentIndex(0)  # Auto (Grid)
        self._grid_toggle.setChecked(True)
        
        # Note: Grid intervals are automatically calculated - no manual reset needed
        
        self._timebase_combo.setCurrentIndex(0)  # Timecode
        self._fps_combo.setCurrentIndex(4)  # 30fps
        self._event_labels_toggle.setChecked(True)
        self._duration_labels_toggle.setChecked(False)
        self._highlight_toggle.setChecked(True)
        self._default_height_spin.setValue(40)
        
        # Reset event time styling if available
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            from ui.qt_gui.design_system import Colors as GlobalColors
            settings_mgr = self._timeline_widget.settings_manager
            settings_mgr.event_time_font_size = 10
            settings_mgr.event_time_font_family = "monospace"
            settings_mgr.event_time_major_color = GlobalColors.TEXT_PRIMARY.name()
            settings_mgr.event_time_minor_color = GlobalColors.TEXT_DISABLED.name()
            
            # Update UI controls
            if hasattr(self, '_time_font_size_spin'):
                self._time_font_size_spin.setValue(10)
            if hasattr(self, '_time_font_family_combo'):
                self._time_font_family_combo.setCurrentIndex(0)  # Monospace
            if hasattr(self, '_time_major_color_btn'):
                self._time_major_color_btn.set_color(GlobalColors.TEXT_PRIMARY.name())
            if hasattr(self, '_time_minor_color_btn'):
                self._time_minor_color_btn.set_color(GlobalColors.TEXT_DISABLED.name())
            
            # Reset block event styling
            settings_mgr.block_event_height = 32
            settings_mgr.block_event_border_radius = 3
            settings_mgr.block_event_border_width = 1
            settings_mgr.block_event_label_font_size = 10
            settings_mgr.block_event_border_darken_percent = 150
            settings_mgr.block_event_opacity = 1.0
            settings_mgr.block_event_z_value = 0.0
            settings_mgr.block_event_rotation = 0.0
            settings_mgr.block_event_scale = 1.0
            settings_mgr.block_event_drop_shadow_enabled = False
            settings_mgr.block_event_drop_shadow_blur_radius = 5.0
            settings_mgr.block_event_drop_shadow_offset_x = 2.0
            settings_mgr.block_event_drop_shadow_offset_y = 2.0
            settings_mgr.block_event_drop_shadow_color = "#000000"
            settings_mgr.block_event_drop_shadow_opacity = 0.5
            
            # Reset marker event styling
            settings_mgr.marker_event_shape = "diamond"
            settings_mgr.marker_event_width = 13
            settings_mgr.marker_event_border_width = 1
            settings_mgr.marker_event_border_darken_percent = 150
            settings_mgr.marker_event_opacity = 1.0
            settings_mgr.marker_event_z_value = 0.0
            settings_mgr.marker_event_rotation = 0.0
            settings_mgr.marker_event_scale = 1.0
            settings_mgr.marker_event_drop_shadow_enabled = False
            settings_mgr.marker_event_drop_shadow_blur_radius = 5.0
            settings_mgr.marker_event_drop_shadow_offset_x = 2.0
            settings_mgr.marker_event_drop_shadow_offset_y = 2.0
            settings_mgr.marker_event_drop_shadow_color = "#000000"
            settings_mgr.marker_event_drop_shadow_opacity = 0.5
            
            # Update UI controls - block events
            if hasattr(self, '_block_event_height_spin'):
                self._block_event_height_spin.setValue(32)
            if hasattr(self, '_block_event_border_radius_spin'):
                self._block_event_border_radius_spin.setValue(3)
            if hasattr(self, '_block_event_border_width_spin'):
                self._block_event_border_width_spin.setValue(1)
            if hasattr(self, '_block_event_label_font_size_spin'):
                self._block_event_label_font_size_spin.setValue(10)
            if hasattr(self, '_block_event_border_darken_spin'):
                self._block_event_border_darken_spin.setValue(150)
            if hasattr(self, '_block_event_opacity_spin'):
                self._block_event_opacity_spin.setValue(1.0)
            if hasattr(self, '_block_event_z_value_spin'):
                self._block_event_z_value_spin.setValue(0.0)
            if hasattr(self, '_block_event_rotation_spin'):
                self._block_event_rotation_spin.setValue(0.0)
            if hasattr(self, '_block_event_scale_spin'):
                self._block_event_scale_spin.setValue(1.0)
            if hasattr(self, '_block_event_shadow_toggle'):
                self._block_event_shadow_toggle.setChecked(False)
            if hasattr(self, '_block_event_shadow_blur_spin'):
                self._block_event_shadow_blur_spin.setValue(5.0)
            if hasattr(self, '_block_event_shadow_offset_x_spin'):
                self._block_event_shadow_offset_x_spin.setValue(2.0)
            if hasattr(self, '_block_event_shadow_offset_y_spin'):
                self._block_event_shadow_offset_y_spin.setValue(2.0)
            if hasattr(self, '_block_event_shadow_color_btn'):
                self._block_event_shadow_color_btn.set_color("#000000")
            if hasattr(self, '_block_event_shadow_opacity_spin'):
                self._block_event_shadow_opacity_spin.setValue(0.5)
            
            # Update UI controls - marker events
            if hasattr(self, '_marker_event_shape_combo'):
                for i in range(self._marker_event_shape_combo.count()):
                    if self._marker_event_shape_combo.itemData(i) == "diamond":
                        self._marker_event_shape_combo.setCurrentIndex(i)
                        break
            if hasattr(self, '_marker_event_width_spin'):
                self._marker_event_width_spin.setValue(13)
            if hasattr(self, '_marker_event_border_width_spin'):
                self._marker_event_border_width_spin.setValue(1)
            if hasattr(self, '_marker_event_border_darken_spin'):
                self._marker_event_border_darken_spin.setValue(150)
            if hasattr(self, '_marker_event_opacity_spin'):
                self._marker_event_opacity_spin.setValue(1.0)
            if hasattr(self, '_marker_event_z_value_spin'):
                self._marker_event_z_value_spin.setValue(0.0)
            if hasattr(self, '_marker_event_rotation_spin'):
                self._marker_event_rotation_spin.setValue(0.0)
            if hasattr(self, '_marker_event_scale_spin'):
                self._marker_event_scale_spin.setValue(1.0)
            if hasattr(self, '_marker_event_shadow_toggle'):
                self._marker_event_shadow_toggle.setChecked(False)
            if hasattr(self, '_marker_event_shadow_blur_spin'):
                self._marker_event_shadow_blur_spin.setValue(5.0)
            if hasattr(self, '_marker_event_shadow_offset_x_spin'):
                self._marker_event_shadow_offset_x_spin.setValue(2.0)
            if hasattr(self, '_marker_event_shadow_offset_y_spin'):
                self._marker_event_shadow_offset_y_spin.setValue(2.0)
            if hasattr(self, '_marker_event_shadow_color_btn'):
                self._marker_event_shadow_color_btn.set_color("#000000")
            if hasattr(self, '_marker_event_shadow_opacity_spin'):
                self._marker_event_shadow_opacity_spin.setValue(0.5)
            
            # Reset waveform settings
            settings_mgr.show_waveforms_in_timeline = False
            settings_mgr.waveform_opacity = 0.5
            settings_mgr.waveform_resolution = 50
            settings_mgr.waveform_min_width = 30
            
            if hasattr(self, '_waveforms_timeline_toggle'):
                self._waveforms_timeline_toggle.setChecked(False)
            if hasattr(self, '_waveform_opacity_slider'):
                self._waveform_opacity_slider.setValue(50)
            if hasattr(self, '_waveform_opacity_label'):
                self._waveform_opacity_label.setText("50%")
            if hasattr(self, '_waveform_resolution_spin'):
                self._waveform_resolution_spin.setValue(50)
            if hasattr(self, '_waveform_min_width_spin'):
                self._waveform_min_width_spin.setValue(30)
        
        self._updating = False
        self._emit_settings_changed()
    
    # ==================== Event Handlers ====================
    
    def _on_follow_mode_changed(self, index: int):
        mode = self._follow_combo.currentData()
        if mode:
            self._settings.follow_mode = mode
            self.follow_mode_changed.emit(mode)
            self._emit_settings_changed()
    
    def _on_snap_changed(self, state: int):
        enabled = self._snap_toggle.isChecked()
        self._settings.snap_enabled = enabled
        self._grid_system.snap_enabled = enabled
        self._emit_settings_changed()
    
    def _on_snap_interval_changed(self, index: int):
        mode = self._snap_interval_combo.currentData()
        if mode and self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.snap_interval_mode = mode
            # Update grid system with the new snap mode
            if hasattr(self._grid_system, 'set_snap_interval_mode'):
                self._grid_system.set_snap_interval_mode(mode)
            self._emit_settings_changed()
    
    def _on_grid_changed(self, state: int):
        enabled = self._grid_toggle.isChecked()
        self._settings.show_grid_lines = enabled
        self._grid_system.settings.show_grid_lines = enabled
        self._emit_settings_changed()
    
    # Note: Grid intervals are automatically calculated from timebase/FPS.
    # No manual multiplier handlers needed.
    
    def _on_timebase_changed(self, index: int):
        mode = self._timebase_combo.currentData()
        if mode:
            self._settings.timebase_mode = mode
            self._grid_system.timebase_mode = mode
            self._emit_settings_changed()
    
    def _on_fps_changed(self, index: int):
        fps = self._fps_combo.currentData()
        if fps:
            self._settings.frame_rate = fps
            self._grid_system.frame_rate = fps
            self._emit_settings_changed()
    
    def _on_time_font_size_changed(self, value: int):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.event_time_font_size = value
            self._emit_settings_changed()
    
    def _on_time_font_family_changed(self, index: int):
        font_family = self._time_font_family_combo.currentData()
        if font_family and self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.event_time_font_family = font_family
            self._emit_settings_changed()
    
    def _on_time_major_color_changed(self, hex_color: str):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.event_time_major_color = hex_color
            self._emit_settings_changed()
    
    def _on_time_minor_color_changed(self, hex_color: str):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.event_time_minor_color = hex_color
            self._emit_settings_changed()
    
    # Block event handlers
    def _on_block_event_height_changed(self, value: int):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_height = value
            self._emit_settings_changed()
    
    def _on_block_event_border_radius_changed(self, value: int):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_border_radius = value
            self._emit_settings_changed()
    
    def _on_block_event_border_width_changed(self, value: int):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_border_width = value
            self._emit_settings_changed()
    
    def _on_block_event_label_font_size_changed(self, value: int):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_label_font_size = value
            self._emit_settings_changed()
    
    def _on_block_event_border_darken_changed(self, value: int):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_border_darken_percent = value
            self._emit_settings_changed()
    
    def _on_block_event_opacity_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_opacity = value
            self._emit_settings_changed()
    
    def _on_block_event_z_value_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_z_value = value
            self._emit_settings_changed()
    
    def _on_block_event_rotation_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_rotation = value
            self._emit_settings_changed()
    
    def _on_block_event_scale_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_scale = value
            self._emit_settings_changed()
    
    def _on_block_event_shadow_enabled_changed(self, state: int):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_drop_shadow_enabled = self._block_event_shadow_toggle.isChecked()
            self._emit_settings_changed()
    
    def _on_block_event_shadow_blur_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_drop_shadow_blur_radius = value
            self._emit_settings_changed()
    
    def _on_block_event_shadow_offset_x_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_drop_shadow_offset_x = value
            self._emit_settings_changed()
    
    def _on_block_event_shadow_offset_y_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_drop_shadow_offset_y = value
            self._emit_settings_changed()
    
    def _on_block_event_shadow_color_changed(self, hex_color: str):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_drop_shadow_color = hex_color
            self._emit_settings_changed()
    
    def _on_block_event_shadow_opacity_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.block_event_drop_shadow_opacity = value
            self._emit_settings_changed()
    
    # Marker event handlers
    def _on_marker_event_shape_changed(self, index: int):
        shape = self._marker_event_shape_combo.currentData()
        if shape and self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.marker_event_shape = shape
            self._emit_settings_changed()
    
    def _on_marker_event_width_changed(self, value: int):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.marker_event_width = value
            self._emit_settings_changed()
    
    def _on_marker_event_border_width_changed(self, value: int):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.marker_event_border_width = value
            self._emit_settings_changed()
    
    def _on_marker_event_border_darken_changed(self, value: int):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.marker_event_border_darken_percent = value
            self._emit_settings_changed()
    
    def _on_marker_event_opacity_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.marker_event_opacity = value
            self._emit_settings_changed()
    
    def _on_marker_event_z_value_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.marker_event_z_value = value
            self._emit_settings_changed()
    
    def _on_marker_event_rotation_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.marker_event_rotation = value
            self._emit_settings_changed()
    
    def _on_marker_event_scale_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.marker_event_scale = value
            self._emit_settings_changed()
    
    def _on_marker_event_shadow_enabled_changed(self, state: int):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.marker_event_drop_shadow_enabled = self._marker_event_shadow_toggle.isChecked()
            self._emit_settings_changed()
    
    def _on_marker_event_shadow_blur_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.marker_event_drop_shadow_blur_radius = value
            self._emit_settings_changed()
    
    def _on_marker_event_shadow_offset_x_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.marker_event_drop_shadow_offset_x = value
            self._emit_settings_changed()
    
    def _on_marker_event_shadow_offset_y_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.marker_event_drop_shadow_offset_y = value
            self._emit_settings_changed()
    
    def _on_marker_event_shadow_color_changed(self, hex_color: str):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.marker_event_drop_shadow_color = hex_color
            self._emit_settings_changed()
    
    def _on_marker_event_shadow_opacity_changed(self, value: float):
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.marker_event_drop_shadow_opacity = value
    
    def _on_waveforms_timeline_changed(self, state: int):
        """Handle waveforms in timeline toggle change."""
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            enabled = state == Qt.CheckState.Checked.value
            self._timeline_widget.settings_manager.show_waveforms_in_timeline = enabled
            # Trigger scene update to show/hide waveforms
            if hasattr(self._timeline_widget, '_scene'):
                self._timeline_widget._scene.update()
            self.settings_changed.emit()
    
    def _on_waveform_opacity_changed(self, value: int):
        """Handle waveform opacity slider change."""
        if self._waveform_opacity_label:
            self._waveform_opacity_label.setText(f"{value}%")
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            opacity = value / 100.0
            self._timeline_widget.settings_manager.waveform_opacity = opacity
            # Trigger scene update to refresh waveform rendering
            if hasattr(self._timeline_widget, '_scene'):
                self._timeline_widget._scene.update()
            self.settings_changed.emit()
    
    def _on_waveform_resolution_changed(self, value: int):
        """Handle waveform resolution spinbox change (just saves setting, doesn't apply)."""
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.waveform_resolution = value
            self.settings_changed.emit()
            self._emit_settings_changed()
    
    def _on_waveform_resolution_editing_finished(self):
        """Handle waveform resolution spinbox editing finished (when user types and presses Enter)."""
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            value = self._waveform_resolution_spin.value()
            self._timeline_widget.settings_manager.waveform_resolution = value
            self.settings_changed.emit()
            self._emit_settings_changed()
    
    def _on_apply_waveform_resolution(self):
        """Apply current waveform resolution - regenerates all waveforms at new resolution."""
        if not self._timeline_widget or not hasattr(self._timeline_widget, '_scene'):
            return
        
        from ..events.waveform_simple import clear_cache
        from ..logging import TimelineLog as Log
        
        resolution = self._waveform_resolution_spin.value()
        
        # Save to settings manager
        if hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.waveform_resolution = resolution
        
        Log.info(f"Applying waveform resolution: {resolution} pts/sec -- regenerating all waveforms")
        
        # Disable button during regeneration
        self._apply_resolution_btn.setEnabled(False)
        self._apply_resolution_btn.setText("...")
        
        # Regenerate all waveforms at the new resolution
        regenerated = 0
        try:
            from src.shared.application.services.waveform_service import get_waveform_service
            waveform_service = get_waveform_service()
            regenerated = waveform_service.regenerate_all_waveforms_at_resolution(resolution)
            Log.info(f"Regenerated {regenerated} waveform(s) at {resolution} pts/sec")
        except Exception as e:
            Log.error(f"Failed to regenerate waveforms: {e}")
        
        # Clear UI caches
        clear_cache()
        
        scene = self._timeline_widget._scene
        for item in scene._event_items.values():
            if hasattr(item, '_cached_waveform_data'):
                item._cached_waveform_data = None
                item._cached_waveform_path = None
            if hasattr(item, '_cached_waveform_path'):
                item._cached_waveform_path = None
        
        # Reload waveforms
        if hasattr(self._timeline_widget, '_schedule_staged_waveform_loading'):
            self._timeline_widget._schedule_staged_waveform_loading()
        else:
            scene.update()
        
        # Re-enable button
        self._apply_resolution_btn.setEnabled(True)
        self._apply_resolution_btn.setText("Apply")
    
    def _on_waveform_min_width_changed(self, value: int):
        """Handle waveform minimum width change."""
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            self._timeline_widget.settings_manager.waveform_min_width = value
            # Trigger scene update to refresh waveform visibility based on new threshold
            if hasattr(self._timeline_widget, '_scene'):
                self._timeline_widget._scene.update()
            self.settings_changed.emit()
    
    def _on_setting_changed(self):
        self._settings.follow_during_playback_only = self._playback_only_toggle.isChecked()
        self._settings.show_event_labels = self._event_labels_toggle.isChecked()
        self._settings.show_event_duration_labels = self._duration_labels_toggle.isChecked()
        self._settings.highlight_current_event = self._highlight_toggle.isChecked()
        
        # Sync to settings manager if available
        if self._timeline_widget and hasattr(self._timeline_widget, 'settings_manager'):
            settings_mgr = self._timeline_widget.settings_manager
            settings_mgr.show_event_labels = self._settings.show_event_labels
            settings_mgr.show_event_duration_labels = self._settings.show_event_duration_labels
            settings_mgr.highlight_current_event = self._settings.highlight_current_event
        
        self._emit_settings_changed()
    
    def _emit_settings_changed(self):
        if not self._updating:
            self.settings_changed.emit()
    
    # ==================== Public API ====================
    
    def get_follow_mode(self) -> PlayheadFollowMode:
        return self._settings.follow_mode
    
    def set_follow_mode(self, mode: PlayheadFollowMode):
        for i in range(self._follow_combo.count()):
            if self._follow_combo.itemData(i) == mode:
                self._follow_combo.setCurrentIndex(i)
                break
    
    def set_timeline_widget(self, timeline_widget):
        self._timeline_widget = timeline_widget
        self._update_layer_controls()
    
    def _open_shortcuts_dialog(self):
        """Open the keyboard shortcuts configuration dialog."""
        from .shortcuts import ShortcutsSettingsDialog
        
        # Get current shortcuts from settings manager if available
        current_shortcuts = {}
        if self._timeline_widget and hasattr(self._timeline_widget, '_settings_manager'):
            settings = self._timeline_widget._settings_manager
            current_shortcuts = {
                "Move Event Left": getattr(settings, 'shortcut_move_event_left', 'Key_Left'),
                "Move Event Right": getattr(settings, 'shortcut_move_event_right', 'Key_Right'),
                "Move Event Up Layer": getattr(settings, 'shortcut_move_event_up_layer', 'Ctrl+Key_Up'),
                "Move Event Down Layer": getattr(settings, 'shortcut_move_event_down_layer', 'Ctrl+Key_Down'),
            }
        
        dialog = ShortcutsSettingsDialog(current_shortcuts, self)
        dialog.shortcuts_changed.connect(self._on_shortcuts_changed)
        dialog.exec()
    
    def _on_shortcuts_changed(self, shortcuts: Dict[str, str]):
        """Handle shortcuts changed from dialog."""
        if not self._timeline_widget or not hasattr(self._timeline_widget, '_settings_manager'):
            return
        
        settings = self._timeline_widget._settings_manager
        
        # Map action names to setting names
        mapping = {
            "Move Event Left": "shortcut_move_event_left",
            "Move Event Right": "shortcut_move_event_right",
            "Move Event Up Layer": "shortcut_move_event_up_layer",
            "Move Event Down Layer": "shortcut_move_event_down_layer",
        }
        
        # Update settings
        for action_name, setting_name in mapping.items():
            if action_name in shortcuts:
                # Convert QKeySequence format to our format
                shortcut_str = shortcuts[action_name]
                # Store as-is (will be normalized when used)
                setattr(settings, setting_name, shortcut_str)
        
        # Settings are auto-saved by the settings manager
    
    def _update_layer_controls(self):
        """Update layer height controls based on current layers."""
        if not self._timeline_widget:
            self._layers_section.setVisible(False)
            return
        
        # Clear existing
        for spinbox in self._layer_height_spinboxes.values():
            spinbox.setParent(None)
            spinbox.deleteLater()
        self._layer_height_spinboxes.clear()
        
        while self._layers_container_layout.count():
            item = self._layers_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        layers = self._timeline_widget.get_layers()
        if not layers:
            self._layers_section.setVisible(False)
            return
        
        self._layers_section.setVisible(True)
        
        for i, layer in enumerate(layers):
            layer_name = layer.name if hasattr(layer, 'name') else str(layer)
            
            row = QHBoxLayout()
            row.setSpacing(8)
            
            # Color indicator
            color_indicator = QFrame()
            color_indicator.setFixedSize(4, 20)
            layer_color = layer.color if hasattr(layer, 'color') and layer.color else Colors.ACCENT_BLUE.name()
            color_indicator.setStyleSheet(f"background-color: {layer_color}; border-radius: {border_radius(2)};")
            row.addWidget(color_indicator)
            
            # Layer name (truncated)
            display_name = layer_name if len(layer_name) <= 20 else layer_name[:18] + "..."
            label = QLabel(display_name)
            label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
            label.setToolTip(layer_name)
            row.addWidget(label)
            row.addStretch()
            
            # Height spinbox
            height_spin = self._create_spinbox(20, 200, " px")
            height_spin.setFixedWidth(70)
            current_height = self._timeline_widget.get_layer_height(i)
            height_spin.setValue(int(current_height))
            height_spin.valueChanged.connect(
                lambda value, idx=i: self._on_layer_height_changed(idx, value)
            )
            row.addWidget(height_spin)
            
            self._layer_height_spinboxes[i] = height_spin
            
            widget = QWidget()
            widget.setLayout(row)
            self._layers_container_layout.addWidget(widget)
    
    def _on_layer_height_changed(self, layer_index: int, height: int):
        if not self._timeline_widget or self._updating:
            return
        self._timeline_widget.set_layer_height(layer_index, float(height))
        self._emit_settings_changed()
    
    def _on_default_height_changed(self, value: int):
        pass  # Only apply when button clicked
    
    def _apply_default_to_all(self):
        if not self._timeline_widget:
            return
        
        default_height = self._default_height_spin.value()
        self._timeline_widget.set_default_layer_height(default_height)
        
        layers = self._timeline_widget.get_layers()
        
        self._updating = True
        for i in range(len(layers)):
            self._timeline_widget.set_layer_height(i, float(default_height))
            if i in self._layer_height_spinboxes:
                self._layer_height_spinboxes[i].setValue(default_height)
        self._updating = False
        
        self._emit_settings_changed()
