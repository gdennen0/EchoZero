"""
Audio Negate block panel.

Provides UI for configuring audio negation settings:
- Negation mode (silence, attenuate, subtract)
- Crossfade duration at region edges
- Attenuation level (for attenuate mode)

Uses AudioNegateSettingsManager for unified settings handling.
"""

from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QLabel,
    QVBoxLayout, QGroupBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Spacing, Colors
from src.utils.message import Log

from src.application.blocks.audio_negate_block import NEGATE_MODES
from src.application.settings.audio_negate_settings import AudioNegateSettingsManager


@register_block_panel("AudioNegate")
class AudioNegatePanel(BlockPanelBase):
    """Panel for AudioNegate block configuration."""

    def __init__(self, block_id: str, facade, parent=None):
        # Call parent init first (sets up UI structure)
        # Note: parent.__init__ calls refresh() before _settings_manager exists,
        # so refresh() must be defensive
        super().__init__(block_id, facade, parent)

        # Initialize settings manager AFTER parent init
        self._settings_manager = AudioNegateSettingsManager(facade, block_id, parent=self)

        # Connect to settings changes for UI updates
        self._settings_manager.settings_changed.connect(self._on_setting_changed)

        # Refresh UI now that settings manager is ready
        if self.block:
            self.refresh()

    def create_content_widget(self) -> QWidget:
        """Create AudioNegate-specific UI."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.SM)

        # -- Mode Group --
        mode_group = QGroupBox("Negation Mode")
        mode_layout = QFormLayout(mode_group)
        mode_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        mode_layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        mode_layout.setSpacing(Spacing.XS)

        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumWidth(60)
        for mode_id, info in NEGATE_MODES.items():
            self.mode_combo.addItem(info["name"], mode_id)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addRow("Mode:", self.mode_combo)

        # Mode description label
        self.mode_desc_label = QLabel()
        self.mode_desc_label.setWordWrap(True)
        self.mode_desc_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;"
        )
        mode_layout.addRow(self.mode_desc_label)

        layout.addWidget(mode_group)

        # -- Crossfade Group --
        fade_group = QGroupBox("Crossfade")
        fade_layout = QFormLayout(fade_group)
        fade_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        fade_layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        fade_layout.setSpacing(Spacing.XS)

        self.fade_spin = QDoubleSpinBox()
        self.fade_spin.setMinimumWidth(60)
        self.fade_spin.setRange(0.0, 100.0)
        self.fade_spin.setValue(10.0)
        self.fade_spin.setSingleStep(1.0)
        self.fade_spin.setDecimals(1)
        self.fade_spin.setSuffix(" ms")
        self.fade_spin.setToolTip(
            "Crossfade duration at region boundaries to avoid clicks.\n"
            "  0 = hard cut (may click)\n"
            "  10 = default (smooth)\n"
            "  50-100 = very gradual transition"
        )
        self.fade_spin.valueChanged.connect(self._on_fade_changed)
        fade_layout.addRow("Fade:", self.fade_spin)

        layout.addWidget(fade_group)

        # -- Attenuation Group (only visible in attenuate mode) --
        self.attenuation_group = QGroupBox("Attenuation")
        atten_layout = QFormLayout(self.attenuation_group)
        atten_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        atten_layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        atten_layout.setSpacing(Spacing.XS)

        self.attenuation_spin = QDoubleSpinBox()
        self.attenuation_spin.setMinimumWidth(60)
        self.attenuation_spin.setRange(-60.0, 0.0)
        self.attenuation_spin.setValue(-20.0)
        self.attenuation_spin.setSingleStep(1.0)
        self.attenuation_spin.setDecimals(1)
        self.attenuation_spin.setSuffix(" dB")
        self.attenuation_spin.setToolTip(
            "Volume reduction at event regions (dB).\n"
            "  0 = no reduction\n"
            "  -20 = significant reduction (default)\n"
            "  -60 = near silence"
        )
        self.attenuation_spin.valueChanged.connect(self._on_attenuation_changed)
        atten_layout.addRow("Reduction:", self.attenuation_spin)

        layout.addWidget(self.attenuation_group)

        # -- Subtraction Group (only visible in subtract mode) --
        self.subtraction_group = QGroupBox("Subtraction")
        sub_layout = QFormLayout(self.subtraction_group)
        sub_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        sub_layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        sub_layout.setSpacing(Spacing.XS)

        self.subtract_gain_spin = QDoubleSpinBox()
        self.subtract_gain_spin.setMinimumWidth(60)
        self.subtract_gain_spin.setRange(1.0, 10.0)
        self.subtract_gain_spin.setValue(1.0)
        self.subtract_gain_spin.setSingleStep(0.5)
        self.subtract_gain_spin.setDecimals(1)
        self.subtract_gain_spin.setSuffix("x")
        self.subtract_gain_spin.setToolTip(
            "Multiplier for spectral subtraction strength.\n"
            "  1.0x = normal subtraction\n"
            "  2.0x = double strength (more aggressive)\n"
            "  5.0-10.0x = very aggressive removal"
        )
        self.subtract_gain_spin.valueChanged.connect(self._on_subtract_gain_changed)
        sub_layout.addRow("Gain:", self.subtract_gain_spin)

        self.onset_emphasis_spin = QDoubleSpinBox()
        self.onset_emphasis_spin.setMinimumWidth(60)
        self.onset_emphasis_spin.setRange(1.0, 5.0)
        self.onset_emphasis_spin.setValue(1.0)
        self.onset_emphasis_spin.setSingleStep(0.5)
        self.onset_emphasis_spin.setDecimals(1)
        self.onset_emphasis_spin.setSuffix("x")
        self.onset_emphasis_spin.setToolTip(
            "Extra emphasis on the onset/transient of each event.\n"
            "Front-loads the subtraction to target the attack peak.\n"
            "  1.0x = uniform subtraction across event\n"
            "  2.0x = onset subtracted 2x harder, decays to 1x\n"
            "  5.0x = very aggressive onset targeting"
        )
        self.onset_emphasis_spin.valueChanged.connect(self._on_onset_emphasis_changed)
        sub_layout.addRow("Onset:", self.onset_emphasis_spin)

        layout.addWidget(self.subtraction_group)

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
            mode = self._settings_manager.mode
            fade_ms = self._settings_manager.fade_ms
            attenuation_db = self._settings_manager.attenuation_db
            subtract_gain = self._settings_manager.subtract_gain
            onset_emphasis = self._settings_manager.onset_emphasis
        except Exception as e:
            Log.error(f"AudioNegatePanel: Failed to load settings: {e}")
            return

        # Block signals while updating
        self.mode_combo.blockSignals(True)
        self.fade_spin.blockSignals(True)
        self.attenuation_spin.blockSignals(True)
        self.subtract_gain_spin.blockSignals(True)
        self.onset_emphasis_spin.blockSignals(True)

        # Set mode
        for i in range(self.mode_combo.count()):
            if self.mode_combo.itemData(i) == mode:
                self.mode_combo.setCurrentIndex(i)
                break

        # Set fade
        self.fade_spin.setValue(fade_ms)

        # Set attenuation
        self.attenuation_spin.setValue(attenuation_db)

        # Set subtraction controls
        self.subtract_gain_spin.setValue(subtract_gain)
        self.onset_emphasis_spin.setValue(onset_emphasis)

        # Update description and visibility
        self._update_mode_description(mode)
        self._update_visibility(mode)

        # Unblock signals
        self.mode_combo.blockSignals(False)
        self.fade_spin.blockSignals(False)
        self.attenuation_spin.blockSignals(False)
        self.subtract_gain_spin.blockSignals(False)
        self.onset_emphasis_spin.blockSignals(False)

        # Force Qt to update visuals
        self.mode_combo.update()

        # Update status
        self.set_status_message("Settings loaded")

    def _update_mode_description(self, mode: str):
        """Update the mode description label."""
        if mode in NEGATE_MODES:
            info = NEGATE_MODES[mode]
            self.mode_desc_label.setText(info["description"])
        else:
            self.mode_desc_label.setText("")

    def _update_visibility(self, mode: str):
        """Show/hide controls based on negation mode."""
        # Attenuation group only visible in attenuate mode
        self.attenuation_group.setVisible(mode == "attenuate")
        # Subtraction group only visible in subtract mode
        self.subtraction_group.setVisible(mode == "subtract")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_mode_changed(self, index: int):
        """Handle mode selection change."""
        mode = self.mode_combo.itemData(index)
        if not mode:
            return

        self._update_mode_description(mode)
        self._update_visibility(mode)

        try:
            self._settings_manager.mode = mode
            name = NEGATE_MODES[mode]["name"]
            self.set_status_message(f"Mode set to {name}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_fade_changed(self, value: float):
        """Handle fade duration change."""
        try:
            self._settings_manager.fade_ms = value
            self.set_status_message(f"Fade set to {value:.1f} ms", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_attenuation_changed(self, value: float):
        """Handle attenuation change."""
        try:
            self._settings_manager.attenuation_db = value
            self.set_status_message(f"Attenuation set to {value:.1f} dB", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_subtract_gain_changed(self, value: float):
        """Handle subtract gain change."""
        try:
            self._settings_manager.subtract_gain = value
            self.set_status_message(f"Subtract gain set to {value:.1f}x", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_onset_emphasis_changed(self, value: float):
        """Handle onset emphasis change."""
        try:
            self._settings_manager.onset_emphasis = value
            self.set_status_message(f"Onset emphasis set to {value:.1f}x", error=False)
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
                    f"AudioNegatePanel: Skipping refresh during save for {self.block_id}"
                )
                return

            Log.debug(
                f"AudioNegatePanel: Block {self.block_id} updated externally, refreshing UI"
            )

            # Reload block data from database
            result = self.facade.describe_block(self.block_id)
            if result.success:
                self.block = result.data
                self._update_header()
            else:
                Log.warning(
                    f"AudioNegatePanel: Failed to reload block {self.block_id}"
                )
                return

            # Reload settings from database
            if hasattr(self, "_settings_manager") and self._settings_manager:
                self._settings_manager.reload_from_storage()
                Log.debug("AudioNegatePanel: Settings manager reloaded from database")
            else:
                Log.warning("AudioNegatePanel: Settings manager not available")
                return

            from PyQt6.QtCore import QTimer

            QTimer.singleShot(0, self.refresh)

    def _on_setting_changed(self, setting_name: str):
        """React to settings changes from this panel's settings manager."""
        relevant_settings = [
            "mode", "fade_ms", "attenuation_db",
            "subtract_gain", "onset_emphasis",
        ]
        if setting_name in relevant_settings:
            self.refresh()
