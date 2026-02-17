"""
Export Audio Dataset block panel.

Provides UI for configuring audio dataset export settings:
- Output directory selection
- Audio format (wav, mp3, flac, ogg)
- Clip naming scheme (index, timestamp, class+index)
- Zero-padding digits for index naming
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel,
    QPushButton, QFileDialog, QGroupBox, QComboBox, QSpinBox,
    QCheckBox, QLineEdit
)
from PyQt6.QtCore import Qt

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing, border_radius
from src.application.blocks.export_audio_dataset_block import (
    SUPPORTED_FORMATS, NAMING_SCHEMES,
)
from src.application.settings.export_audio_dataset_settings import (
    ExportAudioDatasetSettingsManager,
)
from src.utils.message import Log
from src.utils.settings import app_settings
from pathlib import Path


@register_block_panel("ExportAudioDataset")
class ExportAudioDatasetPanel(BlockPanelBase):
    """Panel for ExportAudioDataset block configuration."""

    def __init__(self, block_id: str, facade, parent=None):
        super().__init__(block_id, facade, parent)

        # Initialize settings manager AFTER parent init
        self._settings_manager = ExportAudioDatasetSettingsManager(
            facade, block_id, parent=self
        )

        # Connect to settings changes for UI updates
        self._settings_manager.settings_changed.connect(self._on_setting_changed)

        # Refresh UI now that settings manager is ready
        if self.block:
            self.refresh()

    def create_content_widget(self) -> QWidget:
        """Create ExportAudioDataset-specific UI."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)

        # -- Output directory group -------------------------------------------
        dir_group = QGroupBox("Output Directory")
        dir_layout = QVBoxLayout(dir_group)
        dir_layout.setSpacing(Spacing.SM)

        self.dir_path_label = QLabel("No directory selected")
        self.dir_path_label.setWordWrap(True)
        self.dir_path_label.setStyleSheet(f"""
            QLabel {{
                background-color: {Colors.BG_LIGHT.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: {Spacing.SM}px;
                color: {Colors.TEXT_SECONDARY.name()};
                font-family: monospace;
                font-size: 11px;
            }}
        """)
        dir_layout.addWidget(self.dir_path_label)

        browse_btn = QPushButton("Browse for Directory...")
        browse_btn.clicked.connect(self._on_browse_directory)
        dir_layout.addWidget(browse_btn)

        layout.addWidget(dir_group)

        # -- Export settings group --------------------------------------------
        settings_group = QGroupBox("Export Settings")
        settings_layout = QFormLayout(settings_group)
        settings_layout.setSpacing(Spacing.SM)

        # Audio format selector
        self.format_combo = QComboBox()
        self.format_combo.addItem("WAV (uncompressed)", "wav")
        self.format_combo.addItem("MP3 (compressed)", "mp3")
        self.format_combo.addItem("FLAC (lossless)", "flac")
        self.format_combo.addItem("OGG (compressed)", "ogg")
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        settings_layout.addRow("Format:", self.format_combo)

        # Naming scheme selector
        self.naming_combo = QComboBox()
        for scheme_id, info in NAMING_SCHEMES.items():
            self.naming_combo.addItem(info["name"], scheme_id)
        self.naming_combo.currentIndexChanged.connect(self._on_naming_changed)
        settings_layout.addRow("Naming:", self.naming_combo)

        # Naming description label
        self.naming_desc_label = QLabel()
        self.naming_desc_label.setWordWrap(True)
        self.naming_desc_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;"
        )
        settings_layout.addRow("", self.naming_desc_label)

        # Filename prefix (visible when naming scheme is "prefix")
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("clip")
        self.prefix_edit.setToolTip(
            "Custom prefix for exported filenames.\n"
            "e.g., 'kick' produces kick_0001.wav, kick_0002.wav, ..."
        )
        self.prefix_edit.textChanged.connect(self._on_prefix_changed)
        self.prefix_label = QLabel("Prefix:")
        settings_layout.addRow(self.prefix_label, self.prefix_edit)

        # Zero-pad digits spinner
        self.zero_pad_spin = QSpinBox()
        self.zero_pad_spin.setRange(1, 8)
        self.zero_pad_spin.setValue(4)
        self.zero_pad_spin.setToolTip(
            "Number of digits for zero-padding in index-based names.\n"
            "  3 = clip_001, clip_002, ...\n"
            "  4 = clip_0001, clip_0002, ... (default)\n"
            "  6 = clip_000001, clip_000002, ..."
        )
        self.zero_pad_spin.valueChanged.connect(self._on_zero_pad_changed)
        settings_layout.addRow("Zero-pad digits:", self.zero_pad_spin)

        layout.addWidget(settings_group)

        # -- Classification grouping group ------------------------------------
        self.class_group = QGroupBox("Classification Grouping")
        class_layout = QFormLayout(self.class_group)
        class_layout.setSpacing(Spacing.SM)

        self.group_by_class_check = QCheckBox("Group clips into subdirectories by classification")
        self.group_by_class_check.setToolTip(
            "When enabled, clips are saved into subdirectories\n"
            "named after each event's classification.\n"
            "e.g., output_dir/kick/clip_0001.wav\n"
            "      output_dir/snare/clip_0002.wav"
        )
        self.group_by_class_check.stateChanged.connect(self._on_group_by_class_changed)
        class_layout.addRow(self.group_by_class_check)

        self.unclassified_edit = QLineEdit()
        self.unclassified_edit.setPlaceholderText("unclassified")
        self.unclassified_edit.setToolTip(
            "Folder name for events that have no classification."
        )
        self.unclassified_edit.textChanged.connect(self._on_unclassified_folder_changed)
        self.unclassified_label = QLabel("Unclassified folder:")
        class_layout.addRow(self.unclassified_label, self.unclassified_edit)

        layout.addWidget(self.class_group)

        # Info note
        info_label = QLabel(
            "Extracts audio clips from source audio at each event's\n"
            "time region and saves them as individual files.\n"
            "Connect both 'audio' and 'events' inputs."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;"
        )
        layout.addWidget(info_label)

        layout.addStretch()
        return widget

    def refresh(self):
        """Update UI with current settings from settings manager."""
        if not hasattr(self, "_settings_manager") or not self._settings_manager:
            return

        if not self.block or not self._settings_manager.is_loaded():
            return

        try:
            output_dir = self._settings_manager.output_dir
            audio_format = self._settings_manager.audio_format
            naming_scheme = self._settings_manager.naming_scheme
            zero_pad_digits = self._settings_manager.zero_pad_digits
            filename_prefix = self._settings_manager.filename_prefix
            group_by_class = self._settings_manager.group_by_class
            unclassified_folder = self._settings_manager.unclassified_folder
        except Exception as e:
            Log.error(f"ExportAudioDatasetPanel: Failed to load settings: {e}")
            return

        # Block signals while updating
        self.format_combo.blockSignals(True)
        self.naming_combo.blockSignals(True)
        self.zero_pad_spin.blockSignals(True)
        self.prefix_edit.blockSignals(True)
        self.group_by_class_check.blockSignals(True)
        self.unclassified_edit.blockSignals(True)

        # Update directory display
        if output_dir:
            path = Path(output_dir)
            if path.exists():
                self.dir_path_label.setText(str(path))
                self.dir_path_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: {Colors.BG_LIGHT.name()};
                        border: 1px solid {Colors.ACCENT_GREEN.name()};
                        border-radius: {border_radius(4)};
                        padding: {Spacing.SM}px;
                        color: {Colors.TEXT_PRIMARY.name()};
                        font-family: monospace;
                        font-size: 11px;
                    }}
                """)
                self.set_status_message(f"Output: {path.name}")
            else:
                self.dir_path_label.setText(f"{output_dir}\n(Directory not found)")
                self.dir_path_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: {Colors.BG_LIGHT.name()};
                        border: 1px solid {Colors.ACCENT_RED.name()};
                        border-radius: {border_radius(4)};
                        padding: {Spacing.SM}px;
                        color: {Colors.ACCENT_RED.name()};
                        font-family: monospace;
                        font-size: 11px;
                    }}
                """)
                self.set_status_message("Directory not found", error=True)
        else:
            self.dir_path_label.setText("No directory selected")
            self.dir_path_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {Colors.BG_LIGHT.name()};
                    border: 1px solid {Colors.BORDER.name()};
                    border-radius: {border_radius(4)};
                    padding: {Spacing.SM}px;
                    color: {Colors.TEXT_SECONDARY.name()};
                    font-family: monospace;
                    font-size: 11px;
                }}
            """)
            self.set_status_message("No output directory set")

        # Set audio format
        for i in range(self.format_combo.count()):
            if self.format_combo.itemData(i) == audio_format:
                self.format_combo.setCurrentIndex(i)
                break

        # Set naming scheme
        for i in range(self.naming_combo.count()):
            if self.naming_combo.itemData(i) == naming_scheme:
                self.naming_combo.setCurrentIndex(i)
                break

        # Update naming description
        self._update_naming_description(naming_scheme)

        # Set prefix field
        self.prefix_edit.setText(filename_prefix)
        # Only show prefix field when "prefix" naming scheme is selected
        self.prefix_label.setVisible(naming_scheme == "prefix")
        self.prefix_edit.setVisible(naming_scheme == "prefix")

        # Set zero-pad digits
        self.zero_pad_spin.setValue(zero_pad_digits)

        # Set classification grouping
        self.group_by_class_check.setChecked(group_by_class)
        self.unclassified_edit.setText(unclassified_folder)
        # Only show the unclassified folder field when grouping is enabled
        self.unclassified_label.setVisible(group_by_class)
        self.unclassified_edit.setVisible(group_by_class)

        # Unblock signals
        self.format_combo.blockSignals(False)
        self.naming_combo.blockSignals(False)
        self.zero_pad_spin.blockSignals(False)
        self.prefix_edit.blockSignals(False)
        self.group_by_class_check.blockSignals(False)
        self.unclassified_edit.blockSignals(False)

        # Force Qt to update the widgets
        self.format_combo.update()
        self.naming_combo.update()
        self.zero_pad_spin.update()

    def _update_naming_description(self, scheme: str):
        """Update the naming scheme description label."""
        if scheme in NAMING_SCHEMES:
            self.naming_desc_label.setText(NAMING_SCHEMES[scheme]["description"])
        else:
            self.naming_desc_label.setText("")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_browse_directory(self):
        """Open dialog to select output directory."""
        current_dir = self._settings_manager.output_dir or ""
        start_dir = (
            current_dir
            if current_dir
            else app_settings.get_dialog_path("export_audio_dataset")
        )

        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", start_dir
        )

        if dir_path:
            app_settings.set_dialog_path("export_audio_dataset", dir_path)
            try:
                self._settings_manager.output_dir = dir_path
                self.set_status_message("Output directory set", error=False)
            except ValueError as e:
                self.set_status_message(str(e), error=True)
                self.refresh()

    def _on_format_changed(self, index: int):
        """Handle audio format change."""
        audio_format = self.format_combo.itemData(index)
        if not audio_format:
            return
        try:
            self._settings_manager.audio_format = audio_format
            self.set_status_message(
                f"Format set to {audio_format.upper()}", error=False
            )
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_naming_changed(self, index: int):
        """Handle naming scheme change."""
        scheme = self.naming_combo.itemData(index)
        if not scheme:
            return
        self._update_naming_description(scheme)
        # Show/hide the prefix field based on scheme
        self.prefix_label.setVisible(scheme == "prefix")
        self.prefix_edit.setVisible(scheme == "prefix")
        try:
            self._settings_manager.naming_scheme = scheme
            self.set_status_message(
                f"Naming set to {NAMING_SCHEMES[scheme]['name']}", error=False
            )
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_prefix_changed(self, text: str):
        """Handle filename prefix change."""
        try:
            self._settings_manager.filename_prefix = text
            display = text if text else "(default: clip)"
            self.set_status_message(f"Prefix: {display}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_zero_pad_changed(self, value: int):
        """Handle zero-pad digits change."""
        try:
            self._settings_manager.zero_pad_digits = value
            self.set_status_message(
                f"Zero-pad set to {value} digits", error=False
            )
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_group_by_class_changed(self, state: int):
        """Handle group-by-classification toggle."""
        enabled = state == Qt.CheckState.Checked.value
        try:
            self._settings_manager.group_by_class = enabled
            # Show/hide the unclassified folder field
            self.unclassified_label.setVisible(enabled)
            self.unclassified_edit.setVisible(enabled)
            label = "enabled" if enabled else "disabled"
            self.set_status_message(
                f"Group by classification {label}", error=False
            )
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_unclassified_folder_changed(self, text: str):
        """Handle unclassified folder name change."""
        try:
            self._settings_manager.unclassified_folder = text
            display = text if text else "(default: unclassified)"
            self.set_status_message(
                f"Unclassified folder: {display}", error=False
            )
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    # =========================================================================
    # External Updates
    # =========================================================================

    def refresh_for_undo(self):
        """Refresh panel after undo/redo operation."""
        if hasattr(self, "_settings_manager") and self._settings_manager:
            self._settings_manager.reload_from_storage()
        self.refresh()

    def _on_block_updated(self, event):
        """Handle block update event - reload settings and refresh UI."""
        updated_block_id = event.data.get("id")
        if updated_block_id == self.block_id:
            if self._is_saving:
                Log.debug(
                    f"ExportAudioDatasetPanel: Skipping refresh during save "
                    f"for {self.block_id}"
                )
                return

            Log.debug(
                f"ExportAudioDatasetPanel: Block {self.block_id} updated "
                f"externally, refreshing UI"
            )

            result = self.facade.describe_block(self.block_id)
            if result.success:
                self.block = result.data
                self._update_header()
            else:
                Log.warning(
                    f"ExportAudioDatasetPanel: Failed to reload block "
                    f"{self.block_id}"
                )
                return

            if hasattr(self, "_settings_manager") and self._settings_manager:
                self._settings_manager.reload_from_storage()
            else:
                Log.warning(
                    "ExportAudioDatasetPanel: Settings manager not available"
                )
                return

            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, self.refresh)

    def _on_setting_changed(self, setting_name: str):
        """React to settings changes from this panel's settings manager."""
        relevant = [
            "output_dir", "audio_format", "naming_scheme", "zero_pad_digits",
            "filename_prefix", "group_by_class", "unclassified_folder",
        ]
        if setting_name in relevant:
            self.refresh()
