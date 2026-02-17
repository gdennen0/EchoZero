"""
ExportAudio block panel.

Provides UI for configuring audio export settings.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel,
    QPushButton, QFileDialog, QGroupBox, QComboBox, QLineEdit
)
from PyQt6.QtCore import Qt

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing, border_radius
from src.application.settings.export_audio_settings import ExportAudioSettingsManager
from src.utils.message import Log
from src.utils.settings import app_settings
from pathlib import Path


@register_block_panel("ExportAudio")
class ExportAudioPanel(BlockPanelBase):
    """Panel for ExportAudio block configuration"""
    
    def __init__(self, block_id: str, facade, parent=None):
        # Call parent init first (sets up UI structure)
        # Note: parent.__init__ calls refresh() before _settings_manager exists,
        # so refresh() must be defensive
        super().__init__(block_id, facade, parent)
        
        # Initialize settings manager AFTER parent init
        self._settings_manager = ExportAudioSettingsManager(facade, block_id, parent=self)
        
        # Connect to settings changes for UI updates
        self._settings_manager.settings_changed.connect(self._on_setting_changed)
        
        # Refresh UI now that settings manager is ready
        # (parent's refresh() was called before manager existed)
        if self.block:
            self.refresh()
    
    def create_content_widget(self) -> QWidget:
        """Create ExportAudio-specific UI"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)
        
        # Output directory group
        dir_group = QGroupBox("Output Directory")
        dir_layout = QVBoxLayout(dir_group)
        dir_layout.setSpacing(Spacing.SM)
        
        # Directory path display
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
        
        # Browse button
        browse_btn = QPushButton("Browse for Directory...")
        browse_btn.clicked.connect(self._on_browse_directory)
        dir_layout.addWidget(browse_btn)
        
        layout.addWidget(dir_group)
        
        # Export settings group
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
        
        # Filename prefix
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("e.g., exported_")
        self.prefix_edit.textChanged.connect(self._on_prefix_changed)
        settings_layout.addRow("Filename Prefix:", self.prefix_edit)
        
        layout.addWidget(settings_group)
        
        # Info note
        info_label = QLabel(
            " Files will be saved as: [prefix][stem_name].[format]\n"
            "Example: exported_vocals.wav"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;")
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        return widget
    
    def refresh(self):
        """Update UI with current settings from settings manager"""
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return
        
        if not self.block or not self._settings_manager.is_loaded():
            return
        
        # Load settings from settings manager (single source of truth)
        try:
            output_dir = self._settings_manager.output_dir
            audio_format = self._settings_manager.audio_format
            filename_prefix = self._settings_manager.filename_prefix
        except Exception as e:
            Log.error(f"ExportAudioPanel: Failed to load settings: {e}")
            return
        
        # Block signals while updating
        self.format_combo.blockSignals(True)
        self.prefix_edit.blockSignals(True)
        
        # Get output directory
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
            self.set_status_message("No output directory set")
        
        # Set audio format
        format_found = False
        for i in range(self.format_combo.count()):
            if self.format_combo.itemData(i) == audio_format:
                self.format_combo.setCurrentIndex(i)
                format_found = True
                Log.debug(f"ExportAudioPanel: Set format combo to index {i} (value: {audio_format})")
                break
        if not format_found:
            Log.warning(f"ExportAudioPanel: Format '{audio_format}' not found in combo box")
        
        # Set filename prefix
        self.prefix_edit.setText(filename_prefix)
        Log.debug(f"ExportAudioPanel: Set filename_prefix to '{filename_prefix}'")
        
        # Unblock signals
        self.format_combo.blockSignals(False)
        self.prefix_edit.blockSignals(False)
        
        # Force Qt to update the widgets
        self.format_combo.update()
        self.prefix_edit.update()
    
    def _on_browse_directory(self):
        """Open dialog to select output directory (undoable via settings manager)"""
        # Get current directory if exists, fallback to remembered path
        current_dir = self._settings_manager.output_dir or ""
        start_dir = current_dir if current_dir else app_settings.get_dialog_path("export_audio")
        
        # Open directory dialog
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            start_dir
        )
        
        if dir_path:
            app_settings.set_dialog_path("export_audio", dir_path)
            
            # Update via settings manager (single pathway, auto-saves, undoable)
            try:
                self._settings_manager.output_dir = dir_path
                self.set_status_message("Output directory set", error=False)
            except ValueError as e:
                self.set_status_message(str(e), error=True)
                self.refresh()
    
    def _on_format_changed(self, index: int):
        """Handle audio format change (undoable via settings manager)"""
        audio_format = self.format_combo.itemData(index)
        if not audio_format:
            return
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.audio_format = audio_format
            self.set_status_message(f"Format set to {audio_format.upper()}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_prefix_changed(self, text: str):
        """Handle filename prefix change (undoable via settings manager)"""
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.filename_prefix = text
            prefix_display = text if text else "(none)"
            self.set_status_message(f"Prefix: {prefix_display}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def refresh_for_undo(self):
        """
        Refresh panel after undo/redo operation.
        
        Reloads settings from database to ensure UI reflects current state.
        Single source of truth: block.metadata in database.
        """
        # Reload settings manager from database (undo may have changed metadata)
        if hasattr(self, '_settings_manager') and self._settings_manager:
            self._settings_manager.reload_from_storage()
        
        # Refresh UI with current settings
        self.refresh()
    
    def _on_block_updated(self, event):
        """
        Handle block update event - reload settings and refresh UI.
        
        This ensures panel stays in sync when settings change via quick actions
        or other sources. Single source of truth: block.metadata in database.
        """
        updated_block_id = event.data.get('id')
        if updated_block_id == self.block_id:
            # Skip if we triggered this update (prevents refresh loop)
            if self._is_saving:
                Log.debug(f"ExportAudioPanel: Skipping refresh during save for {self.block_id}")
                return
            
            Log.debug(f"ExportAudioPanel: Block {self.block_id} updated externally, refreshing UI")
            
            # Reload block data from database (ensures self.block is current)
            result = self.facade.describe_block(self.block_id)
            if result.success:
                self.block = result.data
                self._update_header()
            else:
                Log.warning(f"ExportAudioPanel: Failed to reload block {self.block_id}")
                return
            
            # Reload settings from database (single source of truth)
            if hasattr(self, '_settings_manager') and self._settings_manager:
                self._settings_manager.reload_from_storage()
                Log.debug(f"ExportAudioPanel: Settings manager reloaded from database")
            else:
                Log.warning(f"ExportAudioPanel: Settings manager not available")
                return
            
            # Refresh UI to reflect changes (now that both block and settings are reloaded)
            # Use QTimer.singleShot to ensure refresh happens after event processing
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, self.refresh)
    
    def _on_setting_changed(self, setting_name: str):
        """
        React to settings changes from this panel's settings manager.
        
        Note: Changes from other sources (quick actions) are handled via
        _on_block_updated() which reloads from database.
        """
        if setting_name in ['output_dir', 'audio_format', 'filename_prefix']:
            # Refresh UI to reflect change
            self.refresh()

