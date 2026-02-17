"""
Audio Filter block panel.

Provides UI for configuring audio filter settings:
- Filter type selection
- Cutoff frequency
- Upper cutoff frequency (bandpass/bandstop)
- Filter order (butterworth)
- Gain in dB (shelf/peak)
- Q factor (shelf/peak)

Uses AudioFilterSettingsManager for unified settings handling.
"""

from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QLabel,
    QVBoxLayout, QGroupBox, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Spacing, Colors
from src.utils.message import Log

from src.application.blocks.audio_filter_block import FILTER_TYPES
from src.application.settings.audio_filter_settings import AudioFilterSettingsManager


@register_block_panel("AudioFilter")
class AudioFilterPanel(BlockPanelBase):
    """Panel for AudioFilter block configuration."""

    def __init__(self, block_id: str, facade, parent=None):
        # Call parent init first (sets up UI structure)
        # Note: parent.__init__ calls refresh() before _settings_manager exists,
        # so refresh() must be defensive
        super().__init__(block_id, facade, parent)

        # Initialize settings manager AFTER parent init
        self._settings_manager = AudioFilterSettingsManager(facade, block_id, parent=self)

        # Connect to settings changes for UI updates
        self._settings_manager.settings_changed.connect(self._on_setting_changed)

        # Refresh UI now that settings manager is ready
        if self.block:
            self.refresh()

    def create_content_widget(self) -> QWidget:
        """Create AudioFilter-specific UI."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)

        # -- Filter Type Group --
        type_group = QGroupBox("Filter Type")
        type_layout = QFormLayout(type_group)
        type_layout.setSpacing(Spacing.SM)

        self.filter_type_combo = QComboBox()
        for type_id, info in FILTER_TYPES.items():
            self.filter_type_combo.addItem(f"{info['name']}", type_id)
        self.filter_type_combo.currentIndexChanged.connect(self._on_filter_type_changed)
        type_layout.addRow("Type:", self.filter_type_combo)

        # Filter description label
        self.filter_desc_label = QLabel()
        self.filter_desc_label.setWordWrap(True)
        self.filter_desc_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;"
        )
        type_layout.addRow("", self.filter_desc_label)

        layout.addWidget(type_group)

        # -- Frequency Settings Group --
        freq_group = QGroupBox("Frequency Settings")
        freq_layout = QFormLayout(freq_group)
        freq_layout.setSpacing(Spacing.SM)

        # Cutoff frequency
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(20.0, 20000.0)
        self.cutoff_spin.setValue(1000.0)
        self.cutoff_spin.setSingleStep(50.0)
        self.cutoff_spin.setDecimals(1)
        self.cutoff_spin.setSuffix(" Hz")
        self.cutoff_spin.setToolTip("Primary cutoff or center frequency (20-20000 Hz)")
        self.cutoff_spin.valueChanged.connect(self._on_cutoff_changed)
        self.cutoff_label = QLabel("Cutoff Freq:")
        freq_layout.addRow(self.cutoff_label, self.cutoff_spin)

        # Upper cutoff frequency (for bandpass/bandstop)
        self.cutoff_high_spin = QDoubleSpinBox()
        self.cutoff_high_spin.setRange(20.0, 20000.0)
        self.cutoff_high_spin.setValue(8000.0)
        self.cutoff_high_spin.setSingleStep(50.0)
        self.cutoff_high_spin.setDecimals(1)
        self.cutoff_high_spin.setSuffix(" Hz")
        self.cutoff_high_spin.setToolTip(
            "Upper cutoff frequency for band-pass and band-stop filters (20-20000 Hz)"
        )
        self.cutoff_high_spin.valueChanged.connect(self._on_cutoff_high_changed)
        self.cutoff_high_label = QLabel("Upper Freq:")
        freq_layout.addRow(self.cutoff_high_label, self.cutoff_high_spin)

        layout.addWidget(freq_group)

        # -- Butterworth Settings Group --
        self.order_group = QGroupBox("Butterworth Settings")
        order_layout = QFormLayout(self.order_group)
        order_layout.setSpacing(Spacing.SM)

        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 8)
        self.order_spin.setValue(4)
        self.order_spin.setSingleStep(1)
        self.order_spin.setToolTip(
            "Filter order (steepness):\n"
            "  1 = gentle slope (6 dB/octave)\n"
            "  4 = default (24 dB/octave)\n"
            "  8 = very steep (48 dB/octave)"
        )
        self.order_spin.valueChanged.connect(self._on_order_changed)
        order_layout.addRow("Order:", self.order_spin)

        layout.addWidget(self.order_group)

        # -- Shelf / Peak Settings Group --
        self.eq_group = QGroupBox("EQ Settings")
        eq_layout = QFormLayout(self.eq_group)
        eq_layout.setSpacing(Spacing.SM)

        # Gain
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(-24.0, 24.0)
        self.gain_spin.setValue(0.0)
        self.gain_spin.setSingleStep(0.5)
        self.gain_spin.setDecimals(1)
        self.gain_spin.setSuffix(" dB")
        self.gain_spin.setToolTip("Boost or cut in decibels (-24 to +24 dB)")
        self.gain_spin.valueChanged.connect(self._on_gain_changed)
        eq_layout.addRow("Gain:", self.gain_spin)

        # Q factor
        self.q_spin = QDoubleSpinBox()
        self.q_spin.setRange(0.1, 10.0)
        self.q_spin.setValue(0.707)
        self.q_spin.setSingleStep(0.1)
        self.q_spin.setDecimals(3)
        self.q_spin.setToolTip(
            "Q factor (bandwidth):\n"
            "  0.1 = very wide\n"
            "  0.707 = default (Butterworth)\n"
            "  10.0 = very narrow"
        )
        self.q_spin.valueChanged.connect(self._on_q_changed)
        eq_layout.addRow("Q Factor:", self.q_spin)

        layout.addWidget(self.eq_group)

        # Add stretch to push everything to top
        layout.addStretch()

        return widget

    def refresh(self):
        """Update UI with current block settings."""
        # Guard: settings manager might not be initialized yet (during __init__)
        if not hasattr(self, "_settings_manager") or not self._settings_manager:
            return

        if not self.block or not self._settings_manager.is_loaded():
            return

        # Load settings from settings manager (single source of truth)
        try:
            filter_type = self._settings_manager.filter_type
            cutoff_freq = self._settings_manager.cutoff_freq
            cutoff_freq_high = self._settings_manager.cutoff_freq_high
            order = self._settings_manager.order
            gain_db = self._settings_manager.gain_db
            q_factor = self._settings_manager.q_factor
        except Exception as e:
            Log.error(f"AudioFilterPanel: Failed to load settings: {e}")
            return

        # Block signals while updating
        self.filter_type_combo.blockSignals(True)
        self.cutoff_spin.blockSignals(True)
        self.cutoff_high_spin.blockSignals(True)
        self.order_spin.blockSignals(True)
        self.gain_spin.blockSignals(True)
        self.q_spin.blockSignals(True)

        # Set filter type
        for i in range(self.filter_type_combo.count()):
            if self.filter_type_combo.itemData(i) == filter_type:
                self.filter_type_combo.setCurrentIndex(i)
                break

        # Set frequency values
        self.cutoff_spin.setValue(cutoff_freq)
        self.cutoff_high_spin.setValue(cutoff_freq_high)

        # Set order
        self.order_spin.setValue(order)

        # Set gain and Q
        self.gain_spin.setValue(gain_db)
        self.q_spin.setValue(q_factor)

        # Update description
        self._update_filter_description(filter_type)

        # Show/hide controls based on filter type
        self._update_visibility(filter_type)

        # Unblock signals
        self.filter_type_combo.blockSignals(False)
        self.cutoff_spin.blockSignals(False)
        self.cutoff_high_spin.blockSignals(False)
        self.order_spin.blockSignals(False)
        self.gain_spin.blockSignals(False)
        self.q_spin.blockSignals(False)

        # Force Qt to update visuals
        self.filter_type_combo.update()

        # Update status
        self.set_status_message("Settings loaded")

    def _update_filter_description(self, filter_type: str):
        """Update the filter description label."""
        if filter_type in FILTER_TYPES:
            info = FILTER_TYPES[filter_type]
            self.filter_desc_label.setText(info["description"])
        else:
            self.filter_desc_label.setText("")

    def _update_visibility(self, filter_type: str):
        """Show/hide controls based on filter type."""
        if filter_type not in FILTER_TYPES:
            return

        info = FILTER_TYPES[filter_type]

        # Upper cutoff only for bandpass/bandstop
        uses_high = info["uses_cutoff_high"]
        self.cutoff_high_label.setVisible(uses_high)
        self.cutoff_high_spin.setVisible(uses_high)

        # Order only for butterworth types (lowpass, highpass, bandpass, bandstop)
        uses_order = not info["uses_gain"]
        self.order_group.setVisible(uses_order)

        # Gain and Q only for shelf/peak types
        uses_eq = info["uses_gain"]
        self.eq_group.setVisible(uses_eq)

        # Update cutoff label based on type
        if filter_type == "peak":
            self.cutoff_label.setText("Center Freq:")
        else:
            self.cutoff_label.setText("Cutoff Freq:")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_filter_type_changed(self, index: int):
        """Handle filter type selection change."""
        filter_type = self.filter_type_combo.itemData(index)
        if not filter_type:
            return

        self._update_filter_description(filter_type)
        self._update_visibility(filter_type)

        try:
            self._settings_manager.filter_type = filter_type
            name = FILTER_TYPES[filter_type]["name"]
            self.set_status_message(f"Filter type set to {name}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_cutoff_changed(self, value: float):
        """Handle cutoff frequency change."""
        try:
            self._settings_manager.cutoff_freq = value
            self.set_status_message(f"Cutoff set to {value:.1f} Hz", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_cutoff_high_changed(self, value: float):
        """Handle upper cutoff frequency change."""
        try:
            self._settings_manager.cutoff_freq_high = value
            self.set_status_message(f"Upper cutoff set to {value:.1f} Hz", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_order_changed(self, value: int):
        """Handle filter order change."""
        try:
            self._settings_manager.order = value
            db_per_oct = value * 6
            self.set_status_message(
                f"Order set to {value} ({db_per_oct} dB/octave)", error=False
            )
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_gain_changed(self, value: float):
        """Handle gain change."""
        try:
            self._settings_manager.gain_db = value
            self.set_status_message(f"Gain set to {value:+.1f} dB", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_q_changed(self, value: float):
        """Handle Q factor change."""
        try:
            self._settings_manager.q_factor = value
            self.set_status_message(f"Q factor set to {value:.3f}", error=False)
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
                    f"AudioFilterPanel: Skipping refresh during save for {self.block_id}"
                )
                return

            Log.debug(
                f"AudioFilterPanel: Block {self.block_id} updated externally, refreshing UI"
            )

            # Reload block data from database
            result = self.facade.describe_block(self.block_id)
            if result.success:
                self.block = result.data
                self._update_header()
            else:
                Log.warning(
                    f"AudioFilterPanel: Failed to reload block {self.block_id}"
                )
                return

            # Reload settings from database
            if hasattr(self, "_settings_manager") and self._settings_manager:
                self._settings_manager.reload_from_storage()
                Log.debug("AudioFilterPanel: Settings manager reloaded from database")
            else:
                Log.warning("AudioFilterPanel: Settings manager not available")
                return

            from PyQt6.QtCore import QTimer

            QTimer.singleShot(0, self.refresh)

    def _on_setting_changed(self, setting_name: str):
        """React to settings changes from this panel's settings manager."""
        relevant_settings = [
            "filter_type",
            "cutoff_freq",
            "cutoff_freq_high",
            "order",
            "gain_db",
            "q_factor",
        ]
        if setting_name in relevant_settings:
            self.refresh()
