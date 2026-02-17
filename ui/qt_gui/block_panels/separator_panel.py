"""
Separator block panel.

Provides UI for configuring Demucs separator settings:
- Model selection
- Device selection
- Output format
- MP3 bitrate
- Two-stems mode

Uses SeparatorSettingsManager for unified settings handling.
"""

from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QLabel,
    QVBoxLayout, QGroupBox, QSpinBox
)
from PyQt6.QtCore import Qt

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Spacing, Colors
from src.utils.message import Log

# Import Demucs models info
from src.application.blocks.separator_block import DEMUCS_MODELS
from src.application.settings.separator_settings import SeparatorSettingsManager


@register_block_panel("Separator")
class SeparatorPanel(BlockPanelBase):
    """Panel for Separator block configuration"""
    
    def __init__(self, block_id: str, facade, parent=None):
        # Call parent init first (sets up UI structure)
        # Note: parent.__init__ calls refresh() before _settings_manager exists,
        # so refresh() must be defensive
        super().__init__(block_id, facade, parent)
        
        # Initialize settings manager AFTER parent init
        self._settings_manager = SeparatorSettingsManager(facade, block_id, parent=self)
        
        # Connect to settings changes for UI updates
        self._settings_manager.settings_changed.connect(self._on_setting_changed)
        
        # Refresh UI now that settings manager is ready
        # (parent's refresh() was called before manager existed)
        if self.block:
            self.refresh()
    
    def create_content_widget(self) -> QWidget:
        """Create Separator-specific UI"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)
        
        # Model settings group
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout(model_group)
        model_layout.setSpacing(Spacing.SM)
        
        # Model selector
        self.model_combo = QComboBox()
        for model_name, info in DEMUCS_MODELS.items():
            display_text = f"{model_name} - {info['quality']} quality, {info['speed']} speed"
            self.model_combo.addItem(display_text, model_name)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_layout.addRow("Model:", self.model_combo)
        
        # Model info label
        self.model_info_label = QLabel()
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;")
        model_layout.addRow("", self.model_info_label)
        
        layout.addWidget(model_group)
        
        # Processing settings group
        processing_group = QGroupBox("Processing Settings")
        processing_layout = QFormLayout(processing_group)
        processing_layout.setSpacing(Spacing.SM)
        
        # Device selector
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (recommended)", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("CUDA (NVIDIA GPU)", "cuda")
        self.device_combo.addItem("MPS (Apple Silicon)", "mps")
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        processing_layout.addRow("Device:", self.device_combo)
        
        # Two-stems mode
        self.two_stems_combo = QComboBox()
        self.two_stems_combo.addItem("All stems (4-way separation)", None)
        self.two_stems_combo.addItem("Vocals + No Vocals", "vocals")
        self.two_stems_combo.addItem("Drums + No Drums", "drums")
        self.two_stems_combo.addItem("Bass + No Bass", "bass")
        self.two_stems_combo.addItem("Other + No Other", "other")
        self.two_stems_combo.currentIndexChanged.connect(self._on_two_stems_changed)
        processing_layout.addRow("Separation Mode:", self.two_stems_combo)
        
        # Shifts (quality vs speed trade-off)
        self.shifts_spin = QSpinBox()
        self.shifts_spin.setRange(0, 100)  # Allow 0 to 100 (0=fastest, 1=default, 10=paper quality)
        self.shifts_spin.setValue(1)
        self.shifts_spin.setSingleStep(1)
        self.shifts_spin.valueChanged.connect(self._on_shifts_changed)
        self.shifts_spin.setToolTip(
            "Number of random shifts for quality (0-100):\n"
            "• 0 = Fastest (no shifts, lowest quality)\n"
            "• 1 = Default (good balance)\n"
            "• 10 = Paper recommendation (best quality, much slower)\n"
            "Higher values = better quality but slower processing"
        )
        processing_layout.addRow("Shifts:", self.shifts_spin)
        
        layout.addWidget(processing_group)
        
        # Output settings group
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout(output_group)
        output_layout.setSpacing(Spacing.SM)
        
        # Output format
        self.format_combo = QComboBox()
        self.format_combo.addItem("WAV (uncompressed)", "wav")
        self.format_combo.addItem("MP3 (compressed)", "mp3")
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        output_layout.addRow("Output Format:", self.format_combo)
        
        # MP3 bitrate (only shown when MP3 is selected)
        self.bitrate_label = QLabel("MP3 Bitrate:")
        self.bitrate_combo = QComboBox()
        self.bitrate_combo.addItem("320 kbps (highest quality)", "320")
        self.bitrate_combo.addItem("192 kbps (good quality)", "192")
        self.bitrate_combo.addItem("128 kbps (smaller files)", "128")
        self.bitrate_combo.currentIndexChanged.connect(self._on_bitrate_changed)
        output_layout.addRow(self.bitrate_label, self.bitrate_combo)
        
        layout.addWidget(output_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return widget
    
    def refresh(self):
        """Update UI with current block settings"""
        # Guard: settings manager might not be initialized yet (during __init__)
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return
        
        if not self.block or not self._settings_manager.is_loaded():
            return
        
        # Load settings from settings manager (single source of truth)
        try:
            model = self._settings_manager.model
            device = self._settings_manager.device
            output_format = self._settings_manager.output_format
            mp3_bitrate = self._settings_manager.mp3_bitrate
            two_stems = self._settings_manager.two_stems
            shifts = self._settings_manager.shifts
        except Exception as e:
            Log.error(f"SeparatorPanel: Failed to load settings: {e}")
            return
        
        # Block signals while updating
        self.model_combo.blockSignals(True)
        self.device_combo.blockSignals(True)
        self.format_combo.blockSignals(True)
        self.bitrate_combo.blockSignals(True)
        self.two_stems_combo.blockSignals(True)
        self.shifts_spin.blockSignals(True)
        
        # Set model
        model_found = False
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == model:
                self.model_combo.setCurrentIndex(i)
                model_found = True
                Log.debug(f"SeparatorPanel: Set model combo to index {i} (value: {model})")
                break
        if not model_found:
            Log.warning(f"SeparatorPanel: Model '{model}' not found in combo box")
        
        # Set device
        device_found = False
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == device:
                self.device_combo.setCurrentIndex(i)
                device_found = True
                Log.debug(f"SeparatorPanel: Set device combo to index {i} (value: {device})")
                break
        if not device_found:
            Log.warning(f"SeparatorPanel: Device '{device}' not found in combo box")
        
        # Set format
        format_found = False
        for i in range(self.format_combo.count()):
            if self.format_combo.itemData(i) == output_format:
                self.format_combo.setCurrentIndex(i)
                format_found = True
                Log.debug(f"SeparatorPanel: Set format combo to index {i} (value: {output_format})")
                break
        if not format_found:
            Log.warning(f"SeparatorPanel: Format '{output_format}' not found in combo box")
        
        # Set bitrate
        bitrate_found = False
        for i in range(self.bitrate_combo.count()):
            if self.bitrate_combo.itemData(i) == mp3_bitrate:
                self.bitrate_combo.setCurrentIndex(i)
                bitrate_found = True
                Log.debug(f"SeparatorPanel: Set bitrate combo to index {i} (value: {mp3_bitrate})")
                break
        if not bitrate_found:
            Log.warning(f"SeparatorPanel: Bitrate '{mp3_bitrate}' not found in combo box")
        
        # Set two-stems mode
        two_stems_found = False
        if two_stems is None:
            self.two_stems_combo.setCurrentIndex(0)
            two_stems_found = True
            Log.debug(f"SeparatorPanel: Set two_stems combo to index 0 (None/all stems)")
        else:
            for i in range(1, self.two_stems_combo.count()):
                if self.two_stems_combo.itemData(i) == two_stems:
                    self.two_stems_combo.setCurrentIndex(i)
                    two_stems_found = True
                    Log.debug(f"SeparatorPanel: Set two_stems combo to index {i} (value: {two_stems})")
                    break
        if not two_stems_found and two_stems is not None:
            Log.warning(f"SeparatorPanel: Two-stems '{two_stems}' not found in combo box")
        
        # Set shifts
        self.shifts_spin.setValue(shifts)
        Log.debug(f"SeparatorPanel: Set shifts spin to {shifts}")
        
        # Update model info
        self._update_model_info(model)
        
        # Show/hide bitrate based on format
        is_mp3 = output_format == "mp3"
        self.bitrate_label.setVisible(is_mp3)
        self.bitrate_combo.setVisible(is_mp3)
        
        # Unblock signals
        self.model_combo.blockSignals(False)
        self.device_combo.blockSignals(False)
        self.format_combo.blockSignals(False)
        self.bitrate_combo.blockSignals(False)
        self.two_stems_combo.blockSignals(False)
        self.shifts_spin.blockSignals(False)
        
        # Force Qt to update the combo box visuals (ensures dropdown shows correct selection)
        self.model_combo.update()
        self.device_combo.update()
        self.format_combo.update()
        self.bitrate_combo.update()
        self.two_stems_combo.update()
        
        # Update status
        self.set_status_message("Settings loaded")
    
    def _update_model_info(self, model_name: str):
        """Update model information label"""
        if model_name in DEMUCS_MODELS:
            info = DEMUCS_MODELS[model_name]
            self.model_info_label.setText(
                f"{info['description']} - {info['stems']} stems"
            )
        else:
            self.model_info_label.setText("")
    
    def _on_model_changed(self, index: int):
        """Handle model selection change (undoable via settings manager)"""
        model = self.model_combo.itemData(index)
        if not model:
            return
        
        # Update model info immediately for better UX
        self._update_model_info(model)
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.model = model
            self.set_status_message(f"Model set to {model}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            # Revert combo box to previous value
            self.refresh()
    
    def _on_device_changed(self, index: int):
        """Handle device selection change (undoable via settings manager)"""
        device = self.device_combo.itemData(index)
        if not device:
            return
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.device = device
            self.set_status_message(f"Device set to {device}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_format_changed(self, index: int):
        """Handle output format change (undoable via settings manager)"""
        output_format = self.format_combo.itemData(index)
        if not output_format:
            return
        
        # Show/hide bitrate based on format immediately for better UX
        is_mp3 = output_format == "mp3"
        self.bitrate_label.setVisible(is_mp3)
        self.bitrate_combo.setVisible(is_mp3)
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.output_format = output_format
            self.set_status_message(f"Format set to {output_format}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_bitrate_changed(self, index: int):
        """Handle MP3 bitrate change (undoable via settings manager)"""
        bitrate = self.bitrate_combo.itemData(index)
        if not bitrate:
            return
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.mp3_bitrate = bitrate
            self.set_status_message(f"MP3 bitrate set to {bitrate} kbps", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_two_stems_changed(self, index: int):
        """Handle two-stems mode change (undoable via settings manager)"""
        two_stems = self.two_stems_combo.itemData(index)
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.two_stems = two_stems
            mode_text = two_stems if two_stems else "All stems"
            self.set_status_message(f"Separation mode set to {mode_text}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_shifts_changed(self, value: int):
        """Handle shifts change (undoable via settings manager)"""
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.shifts = value
            quality_note = "fastest/lowest quality" if value == 0 else "default" if value == 1 else "higher quality/slower"
            self.set_status_message(f"Shifts set to {value} ({quality_note})", error=False)
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
                Log.debug(f"SeparatorPanel: Skipping refresh during save for {self.block_id}")
                return
            
            Log.debug(f"SeparatorPanel: Block {self.block_id} updated externally, refreshing UI")
            
            # Reload block data from database (ensures self.block is current)
            result = self.facade.describe_block(self.block_id)
            if result.success:
                self.block = result.data
                self._update_header()
            else:
                Log.warning(f"SeparatorPanel: Failed to reload block {self.block_id}")
                return
            
            # Reload settings from database (single source of truth)
            if hasattr(self, '_settings_manager') and self._settings_manager:
                self._settings_manager.reload_from_storage()
                Log.debug(f"SeparatorPanel: Settings manager reloaded from database")
            else:
                Log.warning(f"SeparatorPanel: Settings manager not available")
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
        if setting_name in ['model', 'device', 'output_format', 'mp3_bitrate', 'two_stems', 'shifts']:
            # Refresh UI to reflect change
            self.refresh()

