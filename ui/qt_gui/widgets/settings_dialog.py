"""
Global Settings Dialog

Provides a modal dialog for editing application-wide settings.
Settings are persisted to the database via AppSettingsManager.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QCheckBox, QSpinBox, QComboBox, QPushButton,
    QFormLayout, QLineEdit, QFileDialog, QGroupBox, QFrame,
    QScrollArea, QSizePolicy, QColorDialog, QInputDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from src.application.settings import AppSettingsManager
from src.utils.message import Log
from ui.qt_gui.design_system import Colors, Spacing, ThemeAwareMixin, border_radius


class SettingsDialog(ThemeAwareMixin, QDialog):
    """
    Global settings dialog with tabbed sections.
    
    Sections:
    - Startup: Autoload, welcome screen
    - Paths: Default directories
    - Editor: Grid, snapping, confirmations
    - Audio: Sample rate, buffer size
    - Theming: Theme presets
    - Advanced: Autosave, undo steps
    """
    
    settings_changed = pyqtSignal()
    visual_defaults_reset = pyqtSignal()  # Emitted when visual settings are reset to defaults
    
    def __init__(self, parent=None, app_settings_manager: AppSettingsManager = None):
        super().__init__(parent)
        self.setWindowTitle("Global Settings")
        self.setMinimumSize(500, 450)
        self.setModal(True)
        
        # Get settings manager from parent (MainWindow facade) or parameter
        if app_settings_manager is None and parent is not None:
            # Try to get from parent's facade
            if hasattr(parent, 'facade') and hasattr(parent.facade, 'app_settings'):
                app_settings_manager = parent.facade.app_settings
        
        if app_settings_manager is None:
            # Fallback: try to get from bootstrap/service container
            try:
                from src.application.bootstrap import get_service_container
                container = get_service_container()
                if hasattr(container, 'app_settings'):
                    app_settings_manager = container.app_settings
            except Exception:
                Log.warning("SettingsDialog: Could not access app_settings_manager")
        
        self._settings_manager = app_settings_manager
        
        # Track original values for cancel/apply logic
        self._original_values = {}
        self._widgets = {}
        self._preview_widgets = {}  # Initialize preview widgets dict
        
        # Custom color editing state
        self._custom_colors = {}  # {attr_name: "#hex"} for colors edited by the user
        self._colors_dirty = False  # True when any swatch has been modified
        self._original_theme_name = None  # Theme active when dialog opened (for cancel revert)
        
        self._setup_ui()
        self._load_settings()
        self._apply_styling()
        self._snapshot_original_theme()
        self._init_theme_aware()
    
    def _setup_ui(self):
        """Create the dialog UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_startup_tab(), "Startup")
        self.tabs.addTab(self._create_paths_tab(), "Paths")
        self.tabs.addTab(self._create_editor_tab(), "Editor")
        self.tabs.addTab(self._create_audio_tab(), "Audio")
        self.tabs.addTab(self._create_theming_tab(), "Theming")
        self.tabs.addTab(self._create_advanced_tab(), "Advanced")
        
        layout.addWidget(self.tabs)
        
        # Button bar
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(Spacing.MD, Spacing.SM, Spacing.MD, Spacing.MD)
        button_layout.addStretch()
        
        self.btn_restore = QPushButton("Restore Defaults")
        self.btn_restore.clicked.connect(self._on_restore_defaults)
        button_layout.addWidget(self.btn_restore)
        
        button_layout.addSpacing(Spacing.MD)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)
        
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self._on_apply)
        button_layout.addWidget(self.btn_apply)
        
        self.btn_ok = QPushButton("OK")
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self._on_ok)
        button_layout.addWidget(self.btn_ok)
        
        layout.addLayout(button_layout)
    
    def _create_section_label(self, text: str) -> QLabel:
        """Create a section header label"""
        label = QLabel(text)
        font = label.font()
        font.setWeight(QFont.Weight.DemiBold)
        label.setFont(font)
        label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding: 8px 0 4px 0;")
        return label
    
    def _create_description_label(self, text: str) -> QLabel:
        """Create a description label"""
        label = QLabel(text)
        label.setStyleSheet(f"color: {Colors.TEXT_DISABLED.name()}; font-size: 11px;")
        label.setWordWrap(True)
        return label
    
    def _create_startup_tab(self) -> QWidget:
        """Create the Startup settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.SM)
        
        # Project Loading section
        layout.addWidget(self._create_section_label("Project Loading"))
        
        self._widgets["restore_last_project"] = QCheckBox("Open most recent project on startup")
        layout.addWidget(self._widgets["restore_last_project"])
        layout.addWidget(self._create_description_label(
            "When enabled, automatically loads the last opened project when the application starts."))
        
        layout.addSpacing(Spacing.MD)
        
        # Welcome section
        layout.addWidget(self._create_section_label("Welcome"))
        
        self._widgets["show_welcome_on_startup"] = QCheckBox("Show welcome screen on startup")
        layout.addWidget(self._widgets["show_welcome_on_startup"])
        layout.addWidget(self._create_description_label(
            "Display tips and quick-start options when opening the application."))
        
        layout.addStretch()
        return tab
    
    def _create_paths_tab(self) -> QWidget:
        """Create the Paths settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.SM)
        
        # Default Directories section
        layout.addWidget(self._create_section_label("Default Directories"))
        
        # Default project directory
        path_layout = QHBoxLayout()
        path_layout.setSpacing(Spacing.SM)
        
        self._widgets["default_project_directory"] = QLineEdit()
        self._widgets["default_project_directory"].setPlaceholderText("Default project folder...")
        path_layout.addWidget(self._widgets["default_project_directory"])
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(lambda: self._browse_directory("default_project_directory"))
        path_layout.addWidget(browse_btn)
        
        layout.addWidget(QLabel("Default project folder:"))
        layout.addLayout(path_layout)
        layout.addWidget(self._create_description_label(
            "The default location for saving and opening projects."))
        
        layout.addStretch()
        return tab
    
    def _create_editor_tab(self) -> QWidget:
        """Create the Editor settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.SM)
        
        # Node Editor section
        layout.addWidget(self._create_section_label("Node Editor"))
        
        self._widgets["snap_to_grid"] = QCheckBox("Snap blocks to grid")
        layout.addWidget(self._widgets["snap_to_grid"])
        
        # Grid size
        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(Spacing.SM)
        grid_layout.addWidget(QLabel("Grid size:"))
        self._widgets["grid_size"] = QSpinBox()
        self._widgets["grid_size"].setRange(5, 50)
        self._widgets["grid_size"].setKeyboardTracking(False)
        self._widgets["grid_size"].setSuffix(" px")
        grid_layout.addWidget(self._widgets["grid_size"])
        grid_layout.addStretch()
        layout.addLayout(grid_layout)
        
        layout.addSpacing(Spacing.MD)
        
        # Block Behavior section
        layout.addWidget(self._create_section_label("Block Behavior"))
        
        self._widgets["auto_connect_blocks"] = QCheckBox("Auto-connect compatible ports when adding blocks")
        layout.addWidget(self._widgets["auto_connect_blocks"])
        layout.addWidget(self._create_description_label(
            "Automatically create connections between compatible ports when adding new blocks."))
        
        layout.addSpacing(Spacing.SM)
        
        self._widgets["confirm_block_deletion"] = QCheckBox("Confirm before deleting blocks")
        layout.addWidget(self._widgets["confirm_block_deletion"])
        
        layout.addStretch()
        return tab
    
    def _create_audio_tab(self) -> QWidget:
        """Create the Audio settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.SM)
        
        # Audio Devices section
        layout.addWidget(self._create_section_label("Audio Devices"))
        
        # Output device
        output_layout = QHBoxLayout()
        output_layout.setSpacing(Spacing.SM)
        output_layout.addWidget(QLabel("Output device:"))
        self._widgets["audio_output_device_id"] = QComboBox()
        self._widgets["audio_output_device_id"].setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        output_layout.addWidget(self._widgets["audio_output_device_id"])
        layout.addLayout(output_layout)
        
        layout.addSpacing(Spacing.XS)
        
        # Input device
        input_layout = QHBoxLayout()
        input_layout.setSpacing(Spacing.SM)
        input_layout.addWidget(QLabel("Input device:"))
        self._widgets["audio_input_device_id"] = QComboBox()
        self._widgets["audio_input_device_id"].setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        input_layout.addWidget(self._widgets["audio_input_device_id"])
        layout.addLayout(input_layout)
        
        layout.addSpacing(Spacing.XS)
        
        # Refresh button
        refresh_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.setToolTip("Re-scan available audio hardware")
        refresh_btn.clicked.connect(self._refresh_audio_devices)
        refresh_layout.addWidget(refresh_btn)
        refresh_layout.addStretch()
        layout.addLayout(refresh_layout)
        
        layout.addWidget(self._create_description_label(
            "Select audio devices for input and output. "
            "'System Default' uses whichever device your OS has selected."))
        
        # Populate device lists
        self._populate_audio_devices()
        
        layout.addSpacing(Spacing.LG)
        
        # Audio Processing section
        layout.addWidget(self._create_section_label("Audio Processing"))
        
        # Sample rate
        rate_layout = QHBoxLayout()
        rate_layout.setSpacing(Spacing.SM)
        rate_layout.addWidget(QLabel("Default sample rate:"))
        self._widgets["default_sample_rate"] = QComboBox()
        self._widgets["default_sample_rate"].addItems(["22050", "44100", "48000", "96000"])
        rate_layout.addWidget(self._widgets["default_sample_rate"])
        rate_layout.addWidget(QLabel("Hz"))
        rate_layout.addStretch()
        layout.addLayout(rate_layout)
        
        layout.addSpacing(Spacing.SM)
        
        # Buffer size
        buffer_layout = QHBoxLayout()
        buffer_layout.setSpacing(Spacing.SM)
        buffer_layout.addWidget(QLabel("Audio buffer size:"))
        self._widgets["audio_buffer_size"] = QComboBox()
        self._widgets["audio_buffer_size"].addItems(["128", "256", "512", "1024", "2048"])
        buffer_layout.addWidget(self._widgets["audio_buffer_size"])
        buffer_layout.addWidget(QLabel("samples"))
        buffer_layout.addStretch()
        layout.addLayout(buffer_layout)
        
        layout.addWidget(self._create_description_label(
            "Larger buffer sizes provide more stability but increase latency."))
        
        layout.addStretch()
        return tab
    
    def _populate_audio_devices(self):
        """Populate audio output and input device combo boxes from system hardware."""
        try:
            from PyQt6.QtMultimedia import QMediaDevices
        except ImportError:
            Log.warning("QtMultimedia not available -- cannot enumerate audio devices")
            self._widgets["audio_output_device_id"].addItem("System Default", "")
            self._widgets["audio_input_device_id"].addItem("System Default", "")
            return
        
        # Remember current selections so we can restore after repopulating
        out_combo = self._widgets["audio_output_device_id"]
        in_combo = self._widgets["audio_input_device_id"]
        
        prev_out = out_combo.currentData() if out_combo.count() > 0 else None
        prev_in = in_combo.currentData() if in_combo.count() > 0 else None
        
        out_combo.blockSignals(True)
        in_combo.blockSignals(True)
        out_combo.clear()
        in_combo.clear()
        
        # Output devices
        out_combo.addItem("System Default", "")
        for device in QMediaDevices.audioOutputs():
            device_id = device.id().data().decode("utf-8", errors="replace")
            out_combo.addItem(device.description(), device_id)
        
        # Input devices
        in_combo.addItem("System Default", "")
        for device in QMediaDevices.audioInputs():
            device_id = device.id().data().decode("utf-8", errors="replace")
            in_combo.addItem(device.description(), device_id)
        
        # Restore previous selections if they still exist
        if prev_out is not None:
            idx = out_combo.findData(prev_out)
            if idx >= 0:
                out_combo.setCurrentIndex(idx)
        
        if prev_in is not None:
            idx = in_combo.findData(prev_in)
            if idx >= 0:
                in_combo.setCurrentIndex(idx)
        
        out_combo.blockSignals(False)
        in_combo.blockSignals(False)
    
    def _refresh_audio_devices(self):
        """Re-scan audio hardware and repopulate device combo boxes."""
        self._populate_audio_devices()
        
        # Reselect saved device if settings manager is available
        if self._settings_manager:
            saved_out = self._settings_manager.audio_output_device_id
            saved_in = self._settings_manager.audio_input_device_id
            self._select_audio_device("audio_output_device_id", saved_out)
            self._select_audio_device("audio_input_device_id", saved_in)
    
    def _select_audio_device(self, widget_key: str, device_id: str):
        """Select a device in the combo box by its device ID string."""
        combo = self._widgets.get(widget_key)
        if not combo:
            return
        idx = combo.findData(device_id)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            # Device not found, fall back to System Default
            combo.setCurrentIndex(0)
    
    def _create_theming_tab(self) -> QWidget:
        """Create the Theming settings tab"""
        from ui.qt_gui.theme_registry import ThemeRegistry
        
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.MD)
        
        # Theme Preset section
        layout.addWidget(self._create_section_label("Theme Preset"))
        
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(Spacing.SM)
        preset_layout.addWidget(QLabel("Select theme:"))
        
        self._widgets["theme_preset"] = QComboBox()
        self._populate_theme_combo()
        
        preset_layout.addWidget(self._widgets["theme_preset"])
        preset_layout.addStretch()
        layout.addLayout(preset_layout)
        
        # Theme description + modified indicator
        desc_row = QHBoxLayout()
        self._theme_description = QLabel()
        self._theme_description.setStyleSheet(f"color: {Colors.TEXT_DISABLED.name()}; font-size: 11px; font-style: italic;")
        self._theme_description.setWordWrap(True)
        desc_row.addWidget(self._theme_description)
        
        self._modified_label = QLabel("(modified)")
        self._modified_label.setStyleSheet(f"color: {Colors.ACCENT_YELLOW.name()}; font-size: 11px; font-weight: bold;")
        self._modified_label.setVisible(False)
        desc_row.addWidget(self._modified_label)
        desc_row.addStretch()
        layout.addLayout(desc_row)
        
        # Update description when selection changes
        self._widgets["theme_preset"].currentIndexChanged.connect(self._on_theme_preset_changed)
        
        # Save / Delete preset buttons
        preset_btn_layout = QHBoxLayout()
        preset_btn_layout.setSpacing(Spacing.SM)
        
        self._save_preset_btn = QPushButton("Save as Preset...")
        self._save_preset_btn.setToolTip("Save the current colors as a named custom preset")
        self._save_preset_btn.clicked.connect(self._on_save_preset)
        preset_btn_layout.addWidget(self._save_preset_btn)
        
        self._delete_preset_btn = QPushButton("Delete Preset")
        self._delete_preset_btn.setToolTip("Delete the currently selected custom preset (built-in presets cannot be deleted)")
        self._delete_preset_btn.clicked.connect(self._on_delete_preset)
        preset_btn_layout.addWidget(self._delete_preset_btn)
        self._update_delete_btn_state()
        
        preset_btn_layout.addStretch()
        layout.addLayout(preset_btn_layout)
        
        # Appearance section
        layout.addSpacing(Spacing.MD)
        layout.addWidget(self._create_section_label("Appearance"))
        
        self._widgets["sharp_corners"] = QCheckBox("Sharp corners (no rounded edges)")
        layout.addWidget(self._widgets["sharp_corners"])
        layout.addWidget(self._create_description_label(
            "When enabled, all UI elements use square corners instead of rounded edges. "
            "Requires Apply or OK to take effect."))
        
        # Reset visual settings button
        layout.addSpacing(Spacing.MD)
        layout.addWidget(self._create_section_label("Reset"))
        
        reset_visual_btn = QPushButton("Reset All Visual Settings to Defaults")
        reset_visual_btn.setToolTip(
            "Resets theme preset, corner style, and all editor visualization "
            "settings (event styling, grid, labels, etc.) to factory defaults."
        )
        reset_visual_btn.clicked.connect(self._on_reset_visual_defaults)
        layout.addWidget(reset_visual_btn)
        layout.addWidget(self._create_description_label(
            "Resets the theme preset, appearance options, and all editor/timeline "
            "visualization settings to their original defaults. Takes effect when you click Apply or OK."))
        
        # Theme color editor section
        layout.addSpacing(Spacing.MD)
        layout.addWidget(self._create_section_label("Color Editor"))
        layout.addWidget(self._create_description_label(
            "Click any swatch to change its color. Changes preview instantly. "
            "Use 'Save as Preset...' to keep your changes."))
        
        preview_widget = QWidget()
        preview_widget.setMinimumHeight(400)
        preview_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
            }}
        """)
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        preview_layout.setSpacing(Spacing.SM)
        
        preview_label = QLabel("Preview of all theme colors (updates immediately)")
        preview_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-weight: bold;")
        preview_layout.addWidget(preview_label)
        
        # Scroll area for color swatches
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
        """)
        
        swatch_container = QWidget()
        swatch_layout = QVBoxLayout(swatch_container)
        swatch_layout.setSpacing(Spacing.MD)
        swatch_layout.setContentsMargins(0, 0, 0, 0)
        
        # Background colors
        bg_group = self._create_color_group("Backgrounds", [
            ("Dark", "bg_dark"),
            ("Medium", "bg_medium"),
            ("Light", "bg_light"),
        ])
        swatch_layout.addWidget(bg_group)
        
        # Text colors
        text_group = self._create_color_group("Text", [
            ("Primary", "text_primary"),
            ("Secondary", "text_secondary"),
            ("Disabled", "text_disabled"),
        ])
        swatch_layout.addWidget(text_group)
        
        # Accent colors
        accent_group = self._create_color_group("Accents", [
            ("Blue", "accent_blue"),
            ("Green", "accent_green"),
            ("Red", "accent_red"),
            ("Yellow", "accent_yellow"),
        ])
        swatch_layout.addWidget(accent_group)
        
        # Block colors
        block_group = self._create_color_group("Block Types", [
            ("Load", "block_load"),
            ("Analyze", "block_analyze"),
            ("Transform", "block_transform"),
            ("Export", "block_export"),
            ("Editor", "block_editor"),
            ("Visualize", "block_visualize"),
            ("Utility", "block_utility"),
        ])
        swatch_layout.addWidget(block_group)
        
        # Connection colors
        connection_group = self._create_color_group("Connections", [
            ("Normal", "connection_normal"),
            ("Hover", "connection_hover"),
            ("Selected", "connection_selected"),
        ])
        swatch_layout.addWidget(connection_group)
        
        # Port colors
        port_group = self._create_color_group("Ports", [
            ("Input", "port_input"),
            ("Output", "port_output"),
            ("Audio", "port_audio"),
            ("Event", "port_event"),
            ("Manipulator", "port_manipulator"),
            ("Generic", "port_generic"),
        ])
        swatch_layout.addWidget(port_group)
        
        # UI element colors
        ui_group = self._create_color_group("UI Elements", [
            ("Border", "border"),
            ("Hover", "hover"),
            ("Selected", "selected"),
        ])
        swatch_layout.addWidget(ui_group)
        
        swatch_layout.addStretch()
        scroll.setWidget(swatch_container)
        preview_layout.addWidget(scroll)
        
        layout.addWidget(preview_widget)
        
        layout.addStretch()
        return tab
    
    def _get_colors_attr_name(self, theme_attr_name: str) -> str:
        """Convert theme attribute name to Colors class attribute name"""
        # Map theme attribute names to Colors class constants
        mapping = {
            "bg_dark": "BG_DARK",
            "bg_medium": "BG_MEDIUM",
            "bg_light": "BG_LIGHT",
            "text_primary": "TEXT_PRIMARY",
            "text_secondary": "TEXT_SECONDARY",
            "text_disabled": "TEXT_DISABLED",
            "accent_blue": "ACCENT_BLUE",
            "accent_green": "ACCENT_GREEN",
            "accent_red": "ACCENT_RED",
            "accent_yellow": "ACCENT_YELLOW",
            "block_load": "BLOCK_LOAD",
            "block_analyze": "BLOCK_ANALYZE",
            "block_transform": "BLOCK_TRANSFORM",
            "block_export": "BLOCK_EXPORT",
            "block_editor": "BLOCK_EDITOR",
            "block_visualize": "BLOCK_VISUALIZE",
            "block_utility": "BLOCK_UTILITY",
            "connection_normal": "CONNECTION_NORMAL",
            "connection_hover": "CONNECTION_HOVER",
            "connection_selected": "CONNECTION_SELECTED",
            "port_input": "PORT_INPUT",
            "port_output": "PORT_OUTPUT",
            "port_audio": "PORT_AUDIO",
            "port_event": "PORT_EVENT",
            "port_manipulator": "PORT_MANIPULATOR",
            "port_generic": "PORT_GENERIC",
            "border": "BORDER",
            "hover": "HOVER",
            "selected": "SELECTED",
        }
        return mapping.get(theme_attr_name, "BG_DARK")
    
    def _create_color_group(self, title: str, colors: list) -> QWidget:
        """Create a group of clickable color swatches with labels"""
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.XS)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-weight: bold; font-size: 11px;")
        layout.addWidget(title_label)
        
        swatch_layout = QHBoxLayout()
        swatch_layout.setSpacing(Spacing.SM)
        
        for label, attr_name in colors:
            row = QHBoxLayout()
            row.setSpacing(Spacing.XS)
            
            label_widget = QLabel(f"{label}:")
            label_widget.setStyleSheet(f"color: {Colors.TEXT_DISABLED.name()}; font-size: 10px; min-width: 60px;")
            label_widget.setFixedWidth(60)
            row.addWidget(label_widget)
            
            swatch = QPushButton()
            swatch.setFixedSize(30, 30)
            swatch.setCursor(Qt.CursorShape.PointingHandCursor)
            swatch.setToolTip(f"Click to change {label} color")
            colors_attr = self._get_colors_attr_name(attr_name)
            color = getattr(Colors, colors_attr)
            self._set_swatch_color(swatch, color.name())
            
            # Connect click to open color picker for this attr_name
            swatch.clicked.connect(lambda checked=False, a=attr_name, s=swatch: self._on_swatch_clicked(a, s))
            row.addWidget(swatch)
            
            # Store reference for updates
            self._preview_widgets[attr_name] = swatch
            
            swatch_layout.addLayout(row)
        
        swatch_layout.addStretch()
        layout.addLayout(swatch_layout)
        
        return group
    
    def _set_swatch_color(self, swatch, hex_color: str):
        """Apply color to a swatch widget."""
        swatch.setStyleSheet(
            f"background-color: {hex_color}; border: 1px solid {Colors.BORDER.name()}; "
            f"border-radius: {border_radius(3)}; min-width: 0px; padding: 0px;"
        )
    
    def _on_swatch_clicked(self, attr_name: str, swatch):
        """Open QColorDialog when a swatch is clicked."""
        # Current color for the swatch
        current_hex = self._custom_colors.get(attr_name)
        if not current_hex:
            colors_attr = self._get_colors_attr_name(attr_name)
            current_hex = getattr(Colors, colors_attr).name()
        
        color = QColorDialog.getColor(
            QColor(current_hex), self, f"Choose color for {attr_name}"
        )
        if color.isValid():
            hex_val = color.name()
            self._custom_colors[attr_name] = hex_val
            self._set_swatch_color(swatch, hex_val)
            self._mark_colors_dirty(True)
            self._apply_live_preview()
    
    def _on_theme_preset_changed(self, index: int):
        """Update theme description and all preview colors when preset changes"""
        from ui.qt_gui.theme_registry import ThemeRegistry
        
        theme_name = self._widgets["theme_preset"].itemData(index)
        if theme_name:
            theme = ThemeRegistry.get_theme(theme_name)
            if theme:
                self._theme_description.setText(theme.description)
                
                # Clear custom edits -- switching preset resets the editor
                self._custom_colors.clear()
                self._mark_colors_dirty(False)
                
                # Build full color dict from the theme for swatches
                color_attr_map = {
                    "bg_dark": theme.bg_dark,
                    "bg_medium": theme.bg_medium,
                    "bg_light": theme.bg_light,
                    "text_primary": theme.text_primary,
                    "text_secondary": theme.text_secondary,
                    "text_disabled": theme.text_disabled,
                    "accent_blue": theme.accent_blue,
                    "accent_green": theme.accent_green,
                    "accent_red": theme.accent_red,
                    "accent_yellow": theme.accent_yellow,
                    "block_load": theme.block_load,
                    "block_analyze": theme.block_analyze,
                    "block_transform": theme.block_transform,
                    "block_export": theme.block_export,
                    "block_editor": theme.block_editor,
                    "block_visualize": theme.block_visualize,
                    "block_utility": theme.block_utility,
                    "connection_normal": theme.connection_normal,
                    "connection_hover": theme.connection_hover,
                    "connection_selected": theme.connection_selected,
                    "port_input": theme.port_input,
                    "port_output": theme.port_output,
                    "port_audio": theme.port_audio,
                    "port_event": theme.port_event,
                    "port_manipulator": theme.port_manipulator,
                    "port_generic": theme.port_generic,
                    "border": theme.border,
                    "hover": theme.hover,
                    "selected": theme.selected,
                }
                
                for attr_name, color in color_attr_map.items():
                    if attr_name in self._preview_widgets:
                        swatch = self._preview_widgets[attr_name]
                        self._set_swatch_color(swatch, color.name())
                
                self._update_delete_btn_state()
    
    # =========================================================================
    # Custom color editing helpers
    # =========================================================================
    
    def _snapshot_original_theme(self):
        """Store the theme name that was active when the dialog opened."""
        if self._settings_manager:
            self._original_theme_name = self._settings_manager.theme_preset
    
    def _mark_colors_dirty(self, dirty: bool):
        """Show/hide the (modified) indicator."""
        self._colors_dirty = dirty
        if hasattr(self, '_modified_label'):
            self._modified_label.setVisible(dirty)
    
    def _get_full_color_dict(self) -> dict:
        """Build a complete color dict by starting from the selected preset
        and overlaying any custom edits."""
        from ui.qt_gui.theme_registry import ThemeRegistry
        
        theme_name = self._widgets["theme_preset"].currentData()
        theme = ThemeRegistry.get_theme(theme_name) if theme_name else None
        if not theme:
            theme = ThemeRegistry.get_theme("default dark")
        
        base_dict = ThemeRegistry.theme_to_dict(theme) if theme else {}
        # Overlay custom edits
        base_dict.update(self._custom_colors)
        return base_dict
    
    def _apply_live_preview(self):
        """Push the current color edits to the UI for instant preview."""
        color_dict = self._get_full_color_dict()
        Colors.apply_theme_from_dict(color_dict)
    
    def _populate_theme_combo(self):
        """Fill the theme preset combo box from the registry."""
        from ui.qt_gui.theme_registry import ThemeRegistry
        
        combo = self._widgets["theme_preset"]
        combo.blockSignals(True)
        current_data = combo.currentData() if combo.count() > 0 else None
        combo.clear()
        themes = ThemeRegistry.get_all_themes()
        for theme_name in sorted(themes.keys()):
            theme = themes[theme_name]
            display = theme.name if ThemeRegistry.is_builtin(theme_name) else f"{theme.name} (custom)"
            combo.addItem(display, theme.name.lower())
        # Restore previous selection
        if current_data:
            idx = combo.findData(current_data)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)
    
    def _update_delete_btn_state(self):
        """Enable Delete button only for custom (non-builtin) presets."""
        from ui.qt_gui.theme_registry import ThemeRegistry
        
        if not hasattr(self, '_delete_preset_btn'):
            return
        theme_name = self._widgets["theme_preset"].currentData() or ""
        self._delete_preset_btn.setEnabled(not ThemeRegistry.is_builtin(theme_name))
    
    def _on_save_preset(self):
        """Save the current colors as a custom preset."""
        from ui.qt_gui.theme_registry import ThemeRegistry
        
        name, ok = QInputDialog.getText(
            self, "Save Custom Theme", "Preset name:",
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        
        # Prevent overwriting built-in themes
        if ThemeRegistry.is_builtin(name.lower()):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Cannot Overwrite",
                f"'{name}' is a built-in preset and cannot be overwritten."
            )
            return
        
        color_dict = self._get_full_color_dict()
        description = f"Custom theme: {name}"
        
        # Register in ThemeRegistry
        ThemeRegistry.register_custom_theme(name, description, color_dict)
        
        # Persist to DB
        if self._settings_manager:
            self._settings_manager.save_custom_theme(name, description, color_dict)
        
        # Refresh combo and select the new preset
        self._populate_theme_combo()
        idx = self._widgets["theme_preset"].findData(name.lower())
        if idx >= 0:
            self._widgets["theme_preset"].setCurrentIndex(idx)
        
        self._custom_colors.clear()
        self._mark_colors_dirty(False)
        Log.info(f"Custom theme '{name}' saved")
    
    def _on_delete_preset(self):
        """Delete the currently selected custom preset."""
        from ui.qt_gui.theme_registry import ThemeRegistry
        from PyQt6.QtWidgets import QMessageBox
        
        theme_name = self._widgets["theme_preset"].currentData()
        if not theme_name or ThemeRegistry.is_builtin(theme_name):
            return
        
        reply = QMessageBox.question(
            self, "Delete Custom Theme",
            f"Are you sure you want to delete the custom theme '{theme_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Remove from registry
        ThemeRegistry.unregister_theme(theme_name)
        
        # Remove from DB
        if self._settings_manager:
            self._settings_manager.delete_custom_theme(theme_name)
        
        # Refresh combo (will fallback to first item)
        self._populate_theme_combo()
        self._on_theme_preset_changed(self._widgets["theme_preset"].currentIndex())
        Log.info(f"Custom theme '{theme_name}' deleted")
    
    def _create_advanced_tab(self) -> QWidget:
        """Create the Advanced settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.SM)
        
        # Autosave section
        layout.addWidget(self._create_section_label("Autosave"))
        
        self._widgets["auto_save_enabled"] = QCheckBox("Enable autosave")
        layout.addWidget(self._widgets["auto_save_enabled"])
        
        autosave_layout = QHBoxLayout()
        autosave_layout.setSpacing(Spacing.SM)
        autosave_layout.addWidget(QLabel("Autosave interval:"))
        self._widgets["auto_save_interval_minutes"] = QSpinBox()
        self._widgets["auto_save_interval_minutes"].setRange(1, 60)
        self._widgets["auto_save_interval_minutes"].setKeyboardTracking(False)
        self._widgets["auto_save_interval_minutes"].setSuffix(" minutes")
        autosave_layout.addWidget(self._widgets["auto_save_interval_minutes"])
        autosave_layout.addStretch()
        layout.addLayout(autosave_layout)
        
        layout.addSpacing(Spacing.MD)
        
        # History section
        layout.addWidget(self._create_section_label("History"))
        
        undo_layout = QHBoxLayout()
        undo_layout.setSpacing(Spacing.SM)
        undo_layout.addWidget(QLabel("Maximum undo steps:"))
        self._widgets["max_undo_steps"] = QSpinBox()
        self._widgets["max_undo_steps"].setRange(10, 500)
        self._widgets["max_undo_steps"].setKeyboardTracking(False)
        undo_layout.addWidget(self._widgets["max_undo_steps"])
        undo_layout.addStretch()
        layout.addLayout(undo_layout)
        
        layout.addWidget(self._create_description_label(
            "Higher values use more memory but allow more undo history."))
        
        layout.addSpacing(Spacing.MD)
        
        # Execution section
        layout.addWidget(self._create_section_label("Execution"))
        self._widgets["use_subprocess_runner"] = QCheckBox("Run blocks in separate process")
        layout.addWidget(self._widgets["use_subprocess_runner"])
        layout.addWidget(self._create_description_label(
            "When enabled, block execution (e.g. PyTorch training) runs in a separate process. "
            "The UI stays responsive and training is not slowed by the main thread. Takes effect on next run."))
        
        layout.addSpacing(Spacing.MD)
        
        # Developer section
        layout.addWidget(self._create_section_label("Developer"))
        
        self._widgets["debug_mode"] = QCheckBox("Enable debug mode")
        layout.addWidget(self._widgets["debug_mode"])
        layout.addWidget(self._create_description_label(
            "Enable additional debugging features and verbose logging for development."))
        
        layout.addStretch()
        return tab
    
    def _load_settings(self):
        """Load current settings into widgets"""
        if not self._settings_manager:
            Log.warning("SettingsDialog: No settings manager available")
            return
        
        # Startup
        self._set_checkbox("restore_last_project", self._settings_manager.restore_last_project)
        self._set_checkbox("show_welcome_on_startup", self._settings_manager.show_welcome_on_startup)
        
        # Paths
        self._widgets["default_project_directory"].setText(
            self._settings_manager.default_project_directory or "")
        
        # Editor
        self._set_checkbox("snap_to_grid", self._settings_manager.snap_to_grid)
        self._widgets["grid_size"].setValue(self._settings_manager.grid_size)
        self._set_checkbox("auto_connect_blocks", self._settings_manager.auto_connect_blocks)
        self._set_checkbox("confirm_block_deletion", self._settings_manager.confirm_block_deletion)
        
        # Audio
        self._set_combobox("default_sample_rate", str(self._settings_manager.default_sample_rate))
        self._set_combobox("audio_buffer_size", str(self._settings_manager.audio_buffer_size))
        self._select_audio_device("audio_output_device_id",
                                  self._settings_manager.audio_output_device_id)
        self._select_audio_device("audio_input_device_id",
                                  self._settings_manager.audio_input_device_id)
        
        # Theming
        theme_preset = self._settings_manager.theme_preset or "default dark"
        self._set_combobox("theme_preset", theme_preset)
        # Trigger description update
        self._on_theme_preset_changed(self._widgets["theme_preset"].currentIndex())
        self._set_checkbox("sharp_corners", self._settings_manager.sharp_corners)
        
        # Advanced
        self._set_checkbox("auto_save_enabled", self._settings_manager.auto_save_enabled)
        # Convert seconds to minutes for display
        interval_minutes = self._settings_manager.auto_save_interval_seconds // 60
        self._widgets["auto_save_interval_minutes"].setValue(max(1, interval_minutes))
        self._widgets["max_undo_steps"].setValue(self._settings_manager.max_undo_steps)
        self._set_checkbox("use_subprocess_runner", self._settings_manager.use_subprocess_runner)
        self._set_checkbox("debug_mode", self._settings_manager.debug_mode)
        
        # Store original values for cancel
        self._store_original_values()
    
    def _store_original_values(self):
        """Store current widget values for cancel functionality"""
        for key, widget in self._widgets.items():
            if isinstance(widget, QCheckBox):
                self._original_values[key] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                self._original_values[key] = widget.value()
            elif isinstance(widget, QComboBox):
                self._original_values[key] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                self._original_values[key] = widget.text()
    
    def _set_checkbox(self, key: str, value: bool):
        """Set checkbox value"""
        if key in self._widgets:
            self._widgets[key].setChecked(bool(value))
    
    def _set_combobox(self, key: str, value: str):
        """Set combobox value"""
        if key in self._widgets:
            combo = self._widgets[key]
            # For theme_preset, match by data (lowercase) since items are stored with lowercase data
            if key == "theme_preset":
                idx = combo.findData(value.lower())
                if idx < 0:
                    # Fallback: try case-insensitive text match
                    for i in range(combo.count()):
                        if combo.itemText(i).lower() == value.lower():
                            idx = i
                            break
            else:
                # For other comboboxes, match by text
                idx = combo.findText(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)
    
    def _save_settings(self):
        """Save widget values to settings"""
        if not self._settings_manager:
            Log.warning("SettingsDialog: No settings manager available")
            return
        
        # Startup
        self._settings_manager.restore_last_project = self._widgets["restore_last_project"].isChecked()
        self._settings_manager.show_welcome_on_startup = self._widgets["show_welcome_on_startup"].isChecked()
        
        # Paths
        self._settings_manager.default_project_directory = self._widgets["default_project_directory"].text()
        
        # Editor
        self._settings_manager.snap_to_grid = self._widgets["snap_to_grid"].isChecked()
        self._settings_manager.grid_size = self._widgets["grid_size"].value()
        self._settings_manager.auto_connect_blocks = self._widgets["auto_connect_blocks"].isChecked()
        self._settings_manager.confirm_block_deletion = self._widgets["confirm_block_deletion"].isChecked()
        
        # Audio
        self._settings_manager.default_sample_rate = int(self._widgets["default_sample_rate"].currentText())
        self._settings_manager.audio_buffer_size = int(self._widgets["audio_buffer_size"].currentText())
        self._settings_manager.audio_output_device_id = (
            self._widgets["audio_output_device_id"].currentData() or "")
        self._settings_manager.audio_input_device_id = (
            self._widgets["audio_input_device_id"].currentData() or "")
        
        # Theming
        theme_preset = self._widgets["theme_preset"].currentData()
        if theme_preset:
            self._settings_manager.theme_preset = theme_preset
        self._settings_manager.sharp_corners = self._widgets["sharp_corners"].isChecked()
        
        # If colors were edited but not saved as a preset, auto-save as the
        # currently selected preset (only for custom presets) or create one
        if self._colors_dirty and self._custom_colors:
            from ui.qt_gui.theme_registry import ThemeRegistry
            if theme_preset and not ThemeRegistry.is_builtin(theme_preset):
                # Update the existing custom preset in-place
                color_dict = self._get_full_color_dict()
                theme = ThemeRegistry.get_theme(theme_preset)
                description = theme.description if theme else f"Custom theme: {theme_preset}"
                ThemeRegistry.register_custom_theme(theme_preset, description, color_dict)
                self._settings_manager.save_custom_theme(theme_preset, description, color_dict)
                self._custom_colors.clear()
                self._mark_colors_dirty(False)
        
        # Advanced
        self._settings_manager.auto_save_enabled = self._widgets["auto_save_enabled"].isChecked()
        # Convert minutes to seconds for storage
        self._settings_manager.auto_save_interval_seconds = self._widgets["auto_save_interval_minutes"].value() * 60
        self._settings_manager.max_undo_steps = self._widgets["max_undo_steps"].value()
        self._settings_manager.use_subprocess_runner = self._widgets["use_subprocess_runner"].isChecked()
        self._settings_manager.debug_mode = self._widgets["debug_mode"].isChecked()
        
        # Force save to ensure persistence
        self._settings_manager.force_save()
        
        # Apply theme and sharp corners immediately if changed
        from ui.qt_gui.design_system import set_sharp_corners
        set_sharp_corners(self._widgets["sharp_corners"].isChecked())
        
        if 'theme_preset' in self._widgets:
            theme_preset = self._widgets["theme_preset"].currentData()
            if theme_preset:
                if self._colors_dirty and self._custom_colors:
                    Colors.apply_theme_from_dict(self._get_full_color_dict())
                else:
                    Colors.apply_theme(theme_preset)
        
        Log.info("Global settings saved")
        self.settings_changed.emit()
    
    def _browse_directory(self, setting_key: str):
        """Open directory browser for a path setting"""
        current = self._widgets[setting_key].text()
        if not current:
            import os
            current = os.path.expanduser("~")
        
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", current)
        
        if directory:
            self._widgets[setting_key].setText(directory)
    
    def _on_restore_defaults(self):
        """Restore default settings"""
        from src.application.settings.app_settings import AppSettings
        
        # Get default values from AppSettings dataclass
        defaults = AppSettings()
        
        # Startup
        self._set_checkbox("restore_last_project", defaults.restore_last_project)
        self._set_checkbox("show_welcome_on_startup", defaults.show_welcome_on_startup)
        
        # Paths
        self._widgets["default_project_directory"].setText(defaults.default_project_directory)
        
        # Editor
        self._set_checkbox("snap_to_grid", defaults.snap_to_grid)
        self._widgets["grid_size"].setValue(defaults.grid_size)
        self._set_checkbox("auto_connect_blocks", defaults.auto_connect_blocks)
        self._set_checkbox("confirm_block_deletion", defaults.confirm_block_deletion)
        
        # Audio
        self._set_combobox("default_sample_rate", str(defaults.default_sample_rate))
        self._set_combobox("audio_buffer_size", str(defaults.audio_buffer_size))
        # Reset device selections to System Default (index 0)
        self._widgets["audio_output_device_id"].setCurrentIndex(0)
        self._widgets["audio_input_device_id"].setCurrentIndex(0)
        
        # Theming
        self._set_combobox("theme_preset", defaults.theme_preset)
        self._on_theme_preset_changed(self._widgets["theme_preset"].currentIndex())
        self._set_checkbox("sharp_corners", defaults.sharp_corners)
        
        # Advanced
        self._set_checkbox("auto_save_enabled", defaults.auto_save_enabled)
        self._widgets["auto_save_interval_minutes"].setValue(defaults.auto_save_interval_seconds // 60)
        self._widgets["max_undo_steps"].setValue(defaults.max_undo_steps)
        self._set_checkbox("use_subprocess_runner", defaults.use_subprocess_runner)
        self._set_checkbox("debug_mode", defaults.debug_mode)
    
    def _on_reset_visual_defaults(self):
        """Reset only visual/theming settings to defaults (theme preset, appearance, and timeline visualization)."""
        from src.application.settings.app_settings import AppSettings
        
        defaults = AppSettings()
        
        # Theme preset
        self._set_combobox("theme_preset", defaults.theme_preset)
        self._on_theme_preset_changed(self._widgets["theme_preset"].currentIndex())
        
        # Appearance
        self._set_checkbox("sharp_corners", defaults.sharp_corners)
        
        # Emit signal so editor panels / timeline settings panels can reset their visualization settings
        self.visual_defaults_reset.emit()
        
        Log.info("Visual settings reset to defaults (Apply or OK to persist)")
    
    def _on_apply(self):
        """Apply settings without closing dialog"""
        self._save_settings()
        self._store_original_values()
        # Snapshot the newly applied theme as the 'original' for future cancel
        self._snapshot_original_theme()
        # Update dialog styling to reflect theme changes
        self._apply_styling()
    
    def _on_ok(self):
        """Apply settings and close dialog"""
        self._save_settings()
        # Update dialog styling to reflect theme changes before closing
        self._apply_styling()
        self.accept()
    
    def reject(self):
        """Revert to original theme if user cancels after modifying colors."""
        if self._colors_dirty and self._original_theme_name:
            Colors.apply_theme(self._original_theme_name)
        super().reject()
    
    def _apply_local_styles(self):
        self._apply_styling()
        self._refresh_swatch_colors()
    
    def _refresh_swatch_colors(self):
        """Re-apply colors to all preview swatches after a theme change clears them."""
        from ui.qt_gui.theme_registry import ThemeRegistry
        
        combo = self._widgets.get("theme_preset")
        if not combo:
            return
        theme_name = combo.currentData()
        theme = ThemeRegistry.get_theme(theme_name) if theme_name else None
        if not theme:
            theme = ThemeRegistry.get_theme("default dark")
        if not theme:
            return
        
        theme_dict = ThemeRegistry.theme_to_dict(theme)
        
        for attr_name, swatch in self._preview_widgets.items():
            hex_color = self._custom_colors.get(attr_name) or theme_dict.get(attr_name)
            if hex_color:
                self._set_swatch_color(swatch, hex_color)

    def _apply_styling(self):
        """Apply dialog styling"""
        br = border_radius
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
            QTabWidget::pane {{
                border: 1px solid {Colors.BORDER.name()};
                background-color: {Colors.BG_DARK.name()};
            }}
            QTabBar::tab {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_SECONDARY.name()};
                padding: 8px 16px;
                border: 1px solid {Colors.BORDER.name()};
                border-bottom: none;
                min-width: 80px;
            }}
            QTabBar::tab:selected {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border-bottom: 2px solid {Colors.ACCENT_BLUE.name()};
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {Colors.HOVER.name()};
            }}
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
            QSpinBox, QComboBox, QLineEdit {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {br(4)};
                padding: 4px 8px;
                color: {Colors.TEXT_PRIMARY.name()};
                min-width: 80px;
            }}
            QSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
                border-color: {Colors.ACCENT_BLUE.name()};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}
            QPushButton {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {br(4)};
                padding: 6px 16px;
                color: {Colors.TEXT_PRIMARY.name()};
                min-width: 70px;
            }}
            QPushButton:hover {{
                background-color: {Colors.HOVER.name()};
            }}
            QPushButton:pressed {{
                background-color: {Colors.SELECTED.name()};
            }}
            QPushButton:default {{
                background-color: {Colors.ACCENT_BLUE.name()};
                border-color: {Colors.ACCENT_BLUE.name()};
            }}
        """)
