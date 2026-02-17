"""
TranscribeNote block panel.

Provides UI for configuring note transcription settings.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel,
    QGroupBox, QDoubleSpinBox, QSpinBox
)
from PyQt6.QtCore import Qt

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing
from src.application.settings.transcribe_note_settings import TranscribeNoteSettingsManager
from src.utils.message import Log


@register_block_panel("TranscribeNote")
class TranscribeNotePanel(BlockPanelBase):
    """Panel for TranscribeNote block configuration"""
    
    def __init__(self, block_id: str, facade, parent=None):
        # Call parent init first (sets up UI structure)
        # Note: parent.__init__ calls refresh() before _settings_manager exists,
        # so refresh() must be defensive
        super().__init__(block_id, facade, parent)
        
        # Initialize settings manager AFTER parent init
        self._settings_manager = TranscribeNoteSettingsManager(facade, block_id, parent=self)
        
        # Connect to settings changes for UI updates
        self._settings_manager.settings_changed.connect(self._on_setting_changed)
        
        # Refresh UI now that settings manager is ready
        # (parent's refresh() was called before manager existed)
        if self.block:
            self.refresh()
    
    def create_content_widget(self) -> QWidget:
        """Create TranscribeNote-specific UI"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)
        
        # Model info group
        model_group = QGroupBox("Model Information")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(Spacing.SM)
        
        info_label = QLabel(
            "TranscribeNote uses the basic-pitch model for note transcription.\n"
            "The model is automatically downloaded on first use."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11pt;")
        model_layout.addWidget(info_label)
        
        layout.addWidget(model_group)
        
        # Detection settings group
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QFormLayout(detection_group)
        detection_layout.setSpacing(Spacing.SM)
        
        # Onset threshold
        self.onset_spin = QDoubleSpinBox()
        self.onset_spin.setRange(0.0, 1.0)
        self.onset_spin.setSingleStep(0.05)
        self.onset_spin.setValue(0.5)
        self.onset_spin.setDecimals(2)
        self.onset_spin.valueChanged.connect(self._on_onset_changed)
        self.onset_spin.setToolTip("Lower = more notes detected (more false positives)\nHigher = fewer notes (may miss some)")
        detection_layout.addRow("Onset Threshold:", self.onset_spin)
        
        # Min duration
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.0, 5.0)
        self.min_duration_spin.setSingleStep(0.01)
        self.min_duration_spin.setValue(0.05)
        self.min_duration_spin.setDecimals(2)
        self.min_duration_spin.setSuffix(" sec")
        self.min_duration_spin.valueChanged.connect(self._on_min_duration_changed)
        self.min_duration_spin.setToolTip("Minimum note duration to detect")
        detection_layout.addRow("Min Duration:", self.min_duration_spin)
        
        layout.addWidget(detection_group)
        
        # Note range group
        range_group = QGroupBox("Note Range (MIDI)")
        range_layout = QFormLayout(range_group)
        range_layout.setSpacing(Spacing.SM)
        
        # Min note
        self.min_note_spin = QSpinBox()
        self.min_note_spin.setRange(0, 127)
        self.min_note_spin.setValue(21)  # A0
        self.min_note_spin.valueChanged.connect(self._on_note_range_changed)
        self.min_note_spin.setToolTip("Lowest MIDI note to detect (21 = A0)")
        range_layout.addRow("Min Note:", self.min_note_spin)
        
        # Max note
        self.max_note_spin = QSpinBox()
        self.max_note_spin.setRange(0, 127)
        self.max_note_spin.setValue(108)  # C8
        self.max_note_spin.valueChanged.connect(self._on_note_range_changed)
        self.max_note_spin.setToolTip("Highest MIDI note to detect (108 = C8)")
        range_layout.addRow("Max Note:", self.max_note_spin)
        
        # Note range info
        self.note_range_info = QLabel()
        self.note_range_info.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;")
        range_layout.addRow("Range:", self.note_range_info)
        
        layout.addWidget(range_group)
        
        # Info note
        info_label = QLabel(
            " Tip: Adjust onset threshold if too many or too few notes are detected.\n"
            "Lower threshold = more sensitive, higher threshold = more selective."
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
            onset_threshold = self._settings_manager.onset_threshold
            min_duration = self._settings_manager.min_duration
            min_note = self._settings_manager.min_note
            max_note = self._settings_manager.max_note
        except Exception as e:
            Log.error(f"TranscribeNotePanel: Failed to load settings: {e}")
            return
        
        # Block signals while updating
        self.onset_spin.blockSignals(True)
        self.min_duration_spin.blockSignals(True)
        self.min_note_spin.blockSignals(True)
        self.max_note_spin.blockSignals(True)
        
        # Set values
        self.onset_spin.setValue(onset_threshold)
        Log.debug(f"TranscribeNotePanel: Set onset_threshold to {onset_threshold}")
        
        self.min_duration_spin.setValue(min_duration)
        Log.debug(f"TranscribeNotePanel: Set min_duration to {min_duration}")
        
        self.min_note_spin.setValue(min_note)
        Log.debug(f"TranscribeNotePanel: Set min_note to {min_note}")
        
        self.max_note_spin.setValue(max_note)
        Log.debug(f"TranscribeNotePanel: Set max_note to {max_note}")
        
        # Update note range info
        self._update_note_range_info(min_note, max_note)
        
        # Unblock signals
        self.onset_spin.blockSignals(False)
        self.min_duration_spin.blockSignals(False)
        self.min_note_spin.blockSignals(False)
        self.max_note_spin.blockSignals(False)
        
        # Force Qt to update the widgets
        self.onset_spin.update()
        self.min_duration_spin.update()
        self.min_note_spin.update()
        self.max_note_spin.update()
        
        # Update status
        self.set_status_message("Settings loaded")
    
    def _update_note_range_info(self, min_note: int, max_note: int):
        """Update note range information label"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        def midi_to_name(midi: int) -> str:
            octave = (midi // 12) - 1
            note = note_names[midi % 12]
            return f"{note}{octave}"
        
        min_name = midi_to_name(min_note)
        max_name = midi_to_name(max_note)
        range_semitones = max_note - min_note + 1
        
        self.note_range_info.setText(f"{min_name} to {max_name} ({range_semitones} semitones)")
    
    def _on_onset_changed(self, value: float):
        """Handle onset threshold change (undoable via settings manager)"""
        # Update via settings manager (single pathway, auto-saves, undoable)
        # Settings manager handles change detection internally
        try:
            self._settings_manager.onset_threshold = value
            self.set_status_message(f"Onset threshold set to {value:.2f}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_min_duration_changed(self, value: float):
        """Handle min duration change (undoable via settings manager)"""
        # Update via settings manager (single pathway, auto-saves, undoable)
        # Settings manager handles change detection internally
        try:
            self._settings_manager.min_duration = value
            self.set_status_message(f"Min duration set to {value:.2f}s", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_note_range_changed(self):
        """Handle note range change (undoable via settings manager)"""
        min_note = self.min_note_spin.value()
        max_note = self.max_note_spin.value()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        # Settings manager validates range (min_note <= max_note)
        try:
            # Set min_note first (validates against current max_note)
            self._settings_manager.min_note = min_note
            # Then set max_note (validates against new min_note)
            self._settings_manager.max_note = max_note
            
            # Update info display
            self._update_note_range_info(min_note, max_note)
            self.set_status_message(f"Note range set to {min_note}-{max_note}", error=False)
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
                Log.debug(f"TranscribeNotePanel: Skipping refresh during save for {self.block_id}")
                return
            
            Log.debug(f"TranscribeNotePanel: Block {self.block_id} updated externally, refreshing UI")
            
            # Reload block data from database (ensures self.block is current)
            result = self.facade.describe_block(self.block_id)
            if result.success:
                self.block = result.data
                self._update_header()
            else:
                Log.warning(f"TranscribeNotePanel: Failed to reload block {self.block_id}")
                return
            
            # Reload settings from database (single source of truth)
            if hasattr(self, '_settings_manager') and self._settings_manager:
                self._settings_manager.reload_from_storage()
                Log.debug(f"TranscribeNotePanel: Settings manager reloaded from database")
            else:
                Log.warning(f"TranscribeNotePanel: Settings manager not available")
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
        if setting_name in ['onset_threshold', 'min_duration', 'min_note', 'max_note']:
            # Refresh UI to reflect change
            self.refresh()

