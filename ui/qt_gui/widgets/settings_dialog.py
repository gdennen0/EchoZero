"""
Global Settings Dialog

Provides a modal dialog for editing application-wide settings.
Settings are persisted to the database via AppSettingsManager.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QCheckBox, QSpinBox, QComboBox, QPushButton,
    QFormLayout, QLineEdit, QFileDialog, QGroupBox, QFrame,
    QScrollArea, QSizePolicy, QColorDialog, QInputDialog,
    QApplication,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtGui import QFontDatabase

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
        
        # Theme editing state
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
        """Create the Theming settings tab with scrollable content."""
        from ui.qt_gui.theme_registry import ThemeRegistry

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(400)

        content = QWidget()
        layout = QVBoxLayout(content)
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
        
        # UI font
        font_layout = QHBoxLayout()
        font_layout.setSpacing(Spacing.SM)
        font_layout.addWidget(QLabel("UI font:"))
        self._widgets["ui_font_family"] = QComboBox()
        self._widgets["ui_font_family"].setEditable(False)
        self._populate_font_combo()
        font_layout.addWidget(self._widgets["ui_font_family"])
        font_layout.addWidget(QLabel("Size:"))
        self._widgets["ui_font_size"] = QSpinBox()
        self._widgets["ui_font_size"].setRange(0, 48)
        self._widgets["ui_font_size"].setSpecialValueText("Default")
        self._widgets["ui_font_size"].setSuffix(" px")
        self._widgets["ui_font_size"].setKeyboardTracking(False)
        font_layout.addWidget(self._widgets["ui_font_size"])
        font_layout.addStretch()
        layout.addLayout(font_layout)
        layout.addWidget(self._create_description_label(
            "Font for the entire application UI. Set to Default (0) to use system default size (13px). "
            "Requires Apply or OK to take effect."))
        
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
        
        # Theme color editor - table-based
        layout.addSpacing(Spacing.MD)
        layout.addWidget(self._create_section_label("Theme Editor"))
        layout.addWidget(self._create_description_label(
            "Double-click any row to edit. Color changes preview instantly. "
            "Use 'Save as Preset...' to keep your changes."))

        from ui.qt_gui.widgets.theme_editor_table import ThemeEditorTable
        self._theme_editor_table = ThemeEditorTable()
        self._theme_editor_table.value_changed.connect(self._on_theme_table_changed)
        layout.addWidget(self._theme_editor_table)

        # Export to .qss button
        export_btn = QPushButton("Export to .qss")
        export_btn.setToolTip("Save current theme stylesheet to a file for sharing or debugging")
        export_btn.clicked.connect(self._on_export_qss)
        layout.addWidget(export_btn)

        layout.addStretch()
        content.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        scroll.setWidget(content)
        return scroll

    def _on_theme_table_changed(self):
        """Theme editor table value changed - apply live preview, sync widgets, mark dirty."""
        self._mark_colors_dirty(True)
        # Sync font/size/sharp from table to widgets so Apply saves correct values
        vals = self._theme_editor_table.get_values()
        if "ui_font_size" in vals and "ui_font_size" in self._widgets:
            self._widgets["ui_font_size"].setValue(vals["ui_font_size"])
        if "sharp_corners" in vals and "sharp_corners" in self._widgets:
            self._widgets["sharp_corners"].setChecked(vals["sharp_corners"])
        self._apply_live_preview()

    def _on_export_qss(self):
        """Export current theme stylesheet to a .qss file."""
        from ui.qt_gui.design_system import get_stylesheet

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Stylesheet", "", "QSS Files (*.qss);;All Files (*)"
        )
        if path:
            try:
                qss = get_stylesheet()
                with open(path, "w", encoding="utf-8") as f:
                    f.write(qss)
                Log.info(f"Exported stylesheet to {path}")
            except Exception as e:
                Log.error(f"Export failed: {e}")

    def _on_theme_preset_changed(self, index: int):
        """Update theme description and load theme into table when preset changes"""
        from ui.qt_gui.theme_registry import ThemeRegistry

        theme_name = self._widgets["theme_preset"].itemData(index)
        if theme_name:
            theme = ThemeRegistry.get_theme(theme_name)
            if theme:
                self._theme_description.setText(theme.description)
                self._mark_colors_dirty(False)
                if hasattr(self, "_theme_editor_table"):
                    self._theme_editor_table.set_colors_from_theme(theme)
                self._update_delete_btn_state()
    
    # =========================================================================
    # Custom color editing helpers
    # =========================================================================
    
    @staticmethod
    def _colors_dict_from_active() -> dict:
        """Build ``{field_name: "#hex"}`` from the currently active Colors.X values."""
        from ui.qt_gui.theme_registry import ThemeRegistry

        result = {}
        for field in ThemeRegistry.COLOR_FIELDS:
            attr = field.upper()
            color = getattr(Colors, attr, None)
            if color is not None and hasattr(color, 'name'):
                result[field] = color.name()
        return result

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
        """Build a complete color dict from the theme editor table."""
        from ui.qt_gui.theme_registry import ThemeRegistry

        if not hasattr(self, "_theme_editor_table"):
            return {}
        all_vals = self._theme_editor_table.get_values()
        return {k: v for k, v in all_vals.items() if k in ThemeRegistry.COLOR_FIELDS and isinstance(v, str)}
    
    def _apply_live_preview(self):
        """Push the current color edits to the UI for instant preview.

        Sets Colors.X via apply_theme_from_dict, then delegates to
        MainWindow._apply_theme() for the full refresh (palette, stylesheet,
        force_style_refresh, macOS title bar, node editor).  The
        _theme_applied_from_dict flag tells _apply_theme to skip reloading
        Colors from the registry.
        """
        color_dict = self._get_full_color_dict()
        Colors.apply_theme_from_dict(color_dict)

        app = QApplication.instance()
        if app:
            app.setProperty("_theme_applied_from_dict", True)

        main_window = self.parent()
        if main_window and hasattr(main_window, '_apply_theme'):
            main_window._apply_theme()
    
    def _populate_font_combo(self):
        """Populate the UI font combo with System Default plus available font families."""
        combo = self._widgets["ui_font_family"]
        combo.blockSignals(True)
        current = combo.currentData() if combo.count() > 0 else None
        combo.clear()
        combo.addItem("System Default", "")
        for name in sorted(QFontDatabase.families()):
            combo.addItem(name, name)
        if current is not None:
            idx = combo.findData(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)
    
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
        
        self._widgets["debug_mode"] = QCheckBox("Developer Mode (full UI, block editing, connection editing)")
        layout.addWidget(self._widgets["debug_mode"])
        layout.addWidget(self._create_description_label(
            "When enabled, all menus, toolbars, and editing features are available. "
            "When disabled, the application enters Production Mode with a simplified interface."))
        
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
        # Override table with the *actually active* Colors.X values so edits
        # applied to builtin themes (which aren't written back to the registry)
        # are reflected when the dialog reopens.
        if hasattr(self, "_theme_editor_table"):
            self._theme_editor_table.set_values(self._colors_dict_from_active())
        self._set_combobox("ui_font_family", self._settings_manager.ui_font_family)
        self._widgets["ui_font_size"].setValue(self._settings_manager.ui_font_size)
        self._set_checkbox("sharp_corners", self._settings_manager.sharp_corners)
        
        # Advanced
        self._set_checkbox("auto_save_enabled", self._settings_manager.auto_save_enabled)
        # Convert seconds to minutes for display
        interval_minutes = self._settings_manager.auto_save_interval_seconds // 60
        self._widgets["auto_save_interval_minutes"].setValue(max(1, interval_minutes))
        self._widgets["max_undo_steps"].setValue(self._settings_manager.max_undo_steps)
        self._set_checkbox("use_subprocess_runner", self._settings_manager.use_subprocess_runner)
        # Reflect the live app mode rather than the raw debug_mode setting
        parent = self.parent()
        mode_mgr = getattr(getattr(parent, 'facade', None), 'app_mode_manager', None)
        if mode_mgr is not None:
            self._set_checkbox("debug_mode", mode_mgr.is_developer)
        else:
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
                self._original_values[key] = widget.currentData() if key == "ui_font_family" else widget.currentText()
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
            elif key == "ui_font_family":
                # Match by data (font family string; "" = System Default)
                idx = combo.findData(value if value else "")
                if idx < 0:
                    idx = 0  # Fallback to System Default
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
        self._settings_manager.ui_font_family = self._widgets["ui_font_family"].currentData() or ""
        self._settings_manager.ui_font_size = self._widgets["ui_font_size"].value()
        self._settings_manager.sharp_corners = self._widgets["sharp_corners"].isChecked()
        
        # If colors were edited but not saved as a preset, auto-save as the
        # currently selected preset (only for custom presets) or create one
        if self._colors_dirty:
            from ui.qt_gui.theme_registry import ThemeRegistry
            if theme_preset and not ThemeRegistry.is_builtin(theme_preset):
                # Update the existing custom preset in-place
                color_dict = self._get_full_color_dict()
                theme = ThemeRegistry.get_theme(theme_preset)
                description = theme.description if theme else f"Custom theme: {theme_preset}"
                ThemeRegistry.register_custom_theme(theme_preset, description, color_dict)
                self._settings_manager.save_custom_theme(theme_preset, description, color_dict)
                self._mark_colors_dirty(False)
        
        # Advanced
        self._settings_manager.auto_save_enabled = self._widgets["auto_save_enabled"].isChecked()
        # Convert minutes to seconds for storage
        self._settings_manager.auto_save_interval_seconds = self._widgets["auto_save_interval_minutes"].value() * 60
        self._settings_manager.max_undo_steps = self._widgets["max_undo_steps"].value()
        self._settings_manager.use_subprocess_runner = self._widgets["use_subprocess_runner"].isChecked()
        new_debug = self._widgets["debug_mode"].isChecked()
        self._settings_manager.debug_mode = new_debug
        
        # Force save to ensure persistence
        self._settings_manager.force_save()
        
        # Propagate debug_mode change to the live AppModeManager
        parent = self.parent()
        mode_mgr = getattr(getattr(parent, 'facade', None), 'app_mode_manager', None)
        if mode_mgr is not None:
            from src.application.services.app_mode_manager import AppMode
            target = AppMode.DEVELOPER if new_debug else AppMode.PRODUCTION
            mode_mgr.switch_mode(target)
        
        # Apply theme and sharp corners immediately if changed
        from ui.qt_gui.design_system import set_sharp_corners
        set_sharp_corners(self._widgets["sharp_corners"].isChecked())
        
        theme_applied_from_dict = False
        if 'theme_preset' in self._widgets:
            theme_preset = self._widgets["theme_preset"].currentData()
            if theme_preset:
                if self._colors_dirty:
                    # Set flag BEFORE apply so theme_changed handlers
                    # (get_stylesheet, get_application_palette) skip reload
                    app = QApplication.instance()
                    if app:
                        app.setProperty("_theme_applied_from_dict", True)
                    Colors.apply_theme_from_dict(self._get_full_color_dict())
                    theme_applied_from_dict = True
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
        self._set_combobox("ui_font_family", defaults.ui_font_family)
        self._widgets["ui_font_size"].setValue(defaults.ui_font_size)
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
        self._set_combobox("ui_font_family", defaults.ui_font_family)
        self._widgets["ui_font_size"].setValue(defaults.ui_font_size)
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
        """Re-load theme editor table from current theme after external theme change.
        Skips refresh when _colors_dirty (user is editing in table) to avoid
        reverting in-progress edits when theme_changed fires from our own live preview.
        """
        if self._colors_dirty:
            return  # User's table edits are the source of truth; don't overwrite
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
        if hasattr(self, "_theme_editor_table"):
            self._theme_editor_table.set_colors_from_theme(theme)

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
