"""
LoadAudio block panel.

Provides UI for selecting and viewing audio files.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel,
    QPushButton, QFileDialog, QGroupBox
)
from PyQt6.QtCore import Qt

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing, border_radius
from src.application.settings.load_audio_settings import LoadAudioSettingsManager
from src.utils.message import Log
from src.utils.settings import app_settings
from pathlib import Path


@register_block_panel("LoadAudio")
class LoadAudioPanel(BlockPanelBase):
    """Panel for LoadAudio block configuration"""
    
    def __init__(self, block_id: str, facade, parent=None):
        # Call parent init first (sets up UI structure)
        # Note: parent.__init__ calls refresh() before _settings_manager exists,
        # so refresh() must be defensive
        super().__init__(block_id, facade, parent)
        
        # Initialize settings manager AFTER parent init
        self._settings_manager = LoadAudioSettingsManager(facade, block_id, parent=self)
        
        # Connect to settings changes for UI updates
        self._settings_manager.settings_changed.connect(self._on_setting_changed)
        
        # Refresh UI now that settings manager is ready
        # (parent's refresh() was called before manager existed)
        if self.block:
            self.refresh()
    
    def create_content_widget(self) -> QWidget:
        """Create LoadAudio-specific UI"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)
        
        # Audio file selection group
        file_group = QGroupBox("Audio File")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(Spacing.SM)
        
        # File path display
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setStyleSheet(f"""
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
        file_layout.addWidget(self.file_path_label)
        
        # Browse button
        browse_btn = QPushButton("Browse for Audio File...")
        browse_btn.clicked.connect(self._on_browse)
        file_layout.addWidget(browse_btn)
        
        layout.addWidget(file_group)
        
        # Audio metadata group (read-only display)
        metadata_group = QGroupBox("Audio Information")
        metadata_layout = QFormLayout(metadata_group)
        metadata_layout.setSpacing(Spacing.SM)
        
        self.sample_rate_label = QLabel("—")
        self.duration_label = QLabel("—")
        self.channels_label = QLabel("—")
        self.file_size_label = QLabel("—")
        
        metadata_layout.addRow("Sample Rate:", self.sample_rate_label)
        metadata_layout.addRow("Duration:", self.duration_label)
        metadata_layout.addRow("Channels:", self.channels_label)
        metadata_layout.addRow("File Size:", self.file_size_label)
        
        layout.addWidget(metadata_group)
        
        # Info note
        info_label = QLabel(
            " Audio metadata will be available after the block is executed"
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
            audio_path = self._settings_manager.audio_path
        except Exception as e:
            Log.error(f"LoadAudioPanel: Failed to load settings: {e}")
            return
        
        # Get audio path from settings manager
        
        if audio_path:
            # Show file path
            path = Path(audio_path)
            if path.exists():
                self.file_path_label.setText(str(path))
                self.file_path_label.setStyleSheet(f"""
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
                
                # Show file size
                file_size = path.stat().st_size
                self.file_size_label.setText(self._format_file_size(file_size))
                
                self.set_status_message(f"Audio file: {path.name}")
            else:
                self.file_path_label.setText(f"{audio_path}\n(File not found)")
                self.file_path_label.setStyleSheet(f"""
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
                self.set_status_message("File not found", error=True)
        else:
            self.file_path_label.setText("No file selected")
            self.file_path_label.setStyleSheet(f"""
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
            self.set_status_message("No file selected")
        
        # Try to get audio metadata if available
        audio_metadata = self.block.metadata.get("audio_metadata", {})
        if audio_metadata:
            sample_rate = audio_metadata.get("sample_rate")
            duration = audio_metadata.get("duration")
            channels = audio_metadata.get("channels")
            
            if sample_rate:
                self.sample_rate_label.setText(f"{sample_rate} Hz")
            if duration:
                self.duration_label.setText(f"{duration:.2f} seconds")
            if channels:
                channel_text = "Mono" if channels == 1 else f"{channels} channels"
                self.channels_label.setText(channel_text)
    
    def _on_browse(self):
        """Open file dialog to select audio file (undoable via settings manager)"""
        # Get current path if exists, fallback to remembered path
        current_path = self._settings_manager.audio_path or ""
        start_dir = str(Path(current_path).parent) if current_path else app_settings.get_dialog_path("load_audio")
        
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            start_dir,
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*.*)"
        )
        
        if file_path:
            app_settings.set_dialog_path("load_audio", file_path)
            
            # Update via settings manager (single pathway, auto-saves, undoable)
            try:
                self._settings_manager.audio_path = file_path
                self.set_status_message(f"Audio file set: {Path(file_path).name}", error=False)
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
                Log.debug(f"LoadAudioPanel: Skipping refresh during save for {self.block_id}")
                return
            
            Log.debug(f"LoadAudioPanel: Block {self.block_id} updated externally, refreshing UI")
            
            # Reload block data from database (ensures self.block is current)
            result = self.facade.describe_block(self.block_id)
            if result.success:
                self.block = result.data
                self._update_header()
            else:
                Log.warning(f"LoadAudioPanel: Failed to reload block {self.block_id}")
                return
            
            # Reload settings from database (single source of truth)
            if hasattr(self, '_settings_manager') and self._settings_manager:
                self._settings_manager.reload_from_storage()
                Log.debug(f"LoadAudioPanel: Settings manager reloaded from database")
            else:
                Log.warning(f"LoadAudioPanel: Settings manager not available")
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
        if setting_name == 'audio_path':
            # Refresh UI to reflect change
            self.refresh()
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


