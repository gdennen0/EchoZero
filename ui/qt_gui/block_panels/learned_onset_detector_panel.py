"""
LearnedOnsetDetector block panel.

Minimal controls for PoC learned onset detection.
"""

from pathlib import Path
import os

from PyQt6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from src.application.settings.learned_onset_detector_settings import (
    LearnedOnsetDetectorSettingsManager,
)
from src.utils.message import Log
from src.utils.settings import app_settings
from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing


@register_block_panel("LearnedOnsetDetector")
class LearnedOnsetDetectorPanel(BlockPanelBase):
    """Panel for configuring the LearnedOnsetDetector block."""

    def __init__(self, block_id: str, facade, parent=None):
        super().__init__(block_id, facade, parent)
        self._settings_manager = LearnedOnsetDetectorSettingsManager(facade, block_id, parent=self)
        self._settings_manager.settings_changed.connect(self._on_setting_changed)
        if self.block:
            self.refresh()

    def create_content_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)

        info = QLabel(
            "PoC learned onset detector.\n"
            "If model_path is set and valid, a tiny CNN predicts frame-level onset probabilities.\n"
            "If model inference fails, optional spectral-flux fallback can still produce events."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;")
        layout.addWidget(info)

        model_group = QGroupBox("Model")
        model_form = QFormLayout(model_group)
        model_form.setSpacing(Spacing.SM)

        path_row = QVBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Optional .pth checkpoint path")
        self.model_path_edit.textChanged.connect(self._on_model_path_changed)
        path_row.addWidget(self.model_path_edit)

        browse_row = QHBoxLayout()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_model)
        browse_row.addWidget(browse_btn)
        browse_row.addStretch()
        path_row.addLayout(browse_row)
        model_form.addRow("Model Path:", path_row)

        self.model_status_label = QLabel("")
        self.model_status_label.setWordWrap(True)
        self.model_status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt;")
        model_form.addRow("", self.model_status_label)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda", "mps"])
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        model_form.addRow("Device:", self.device_combo)

        self.fallback_checkbox = QCheckBox("Fallback to spectral flux if model unavailable")
        self.fallback_checkbox.toggled.connect(self._on_fallback_changed)
        model_form.addRow("", self.fallback_checkbox)

        layout.addWidget(model_group)

        detect_group = QGroupBox("Detection")
        detect_form = QFormLayout(detect_group)
        detect_form.setSpacing(Spacing.SM)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        detect_form.addRow("Threshold:", self.threshold_spin)

        self.min_silence_spin = QDoubleSpinBox()
        self.min_silence_spin.setRange(0.0, 1.0)
        self.min_silence_spin.setSingleStep(0.005)
        self.min_silence_spin.setDecimals(3)
        self.min_silence_spin.setSuffix(" sec")
        self.min_silence_spin.valueChanged.connect(self._on_min_silence_changed)
        detect_form.addRow("Min Silence:", self.min_silence_spin)

        self.hop_length_spin = QSpinBox()
        self.hop_length_spin.setRange(64, 4096)
        self.hop_length_spin.setSingleStep(64)
        self.hop_length_spin.valueChanged.connect(self._on_hop_length_changed)
        detect_form.addRow("Hop Length:", self.hop_length_spin)

        self.n_mels_spin = QSpinBox()
        self.n_mels_spin.setRange(16, 512)
        self.n_mels_spin.setSingleStep(8)
        self.n_mels_spin.valueChanged.connect(self._on_n_mels_changed)
        detect_form.addRow("Mel Bins:", self.n_mels_spin)

        self.backtrack_checkbox = QCheckBox("Use backtrack for onset timing")
        self.backtrack_checkbox.toggled.connect(self._on_backtrack_changed)
        detect_form.addRow("", self.backtrack_checkbox)

        layout.addWidget(detect_group)
        layout.addStretch()
        return widget

    def refresh(self):
        if not hasattr(self, "_settings_manager") or not self._settings_manager:
            return
        if not self.block or not self._settings_manager.is_loaded():
            return

        try:
            model_path = self._settings_manager.model_path or ""
            device = self._settings_manager.device
            threshold = self._settings_manager.threshold
            min_silence = self._settings_manager.min_silence
            hop_length = self._settings_manager.hop_length
            n_mels = self._settings_manager.n_mels
            use_backtrack = self._settings_manager.use_backtrack
            fallback = self._settings_manager.fallback_to_spectral_flux
        except Exception as exc:
            Log.error(f"LearnedOnsetDetectorPanel: Failed loading settings: {exc}")
            return

        self.model_path_edit.blockSignals(True)
        self.device_combo.blockSignals(True)
        self.threshold_spin.blockSignals(True)
        self.min_silence_spin.blockSignals(True)
        self.hop_length_spin.blockSignals(True)
        self.n_mels_spin.blockSignals(True)
        self.backtrack_checkbox.blockSignals(True)
        self.fallback_checkbox.blockSignals(True)

        self.model_path_edit.setText(model_path)
        idx = self.device_combo.findText(device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        self.threshold_spin.setValue(threshold)
        self.min_silence_spin.setValue(min_silence)
        self.hop_length_spin.setValue(hop_length)
        self.n_mels_spin.setValue(n_mels)
        self.backtrack_checkbox.setChecked(use_backtrack)
        self.fallback_checkbox.setChecked(fallback)

        self.model_path_edit.blockSignals(False)
        self.device_combo.blockSignals(False)
        self.threshold_spin.blockSignals(False)
        self.min_silence_spin.blockSignals(False)
        self.hop_length_spin.blockSignals(False)
        self.n_mels_spin.blockSignals(False)
        self.backtrack_checkbox.blockSignals(False)
        self.fallback_checkbox.blockSignals(False)

        self._update_model_status(model_path)

    def _update_model_status(self, model_path: str):
        if not model_path:
            self.model_status_label.setText("No model set. Fallback mode will be used if enabled.")
            return
        if not os.path.exists(model_path):
            self.model_status_label.setText("Model path not found.")
            self.model_status_label.setStyleSheet(f"color: {Colors.STATUS_ERROR.name()}; font-size: 9pt;")
            return
        self.model_status_label.setStyleSheet(f"color: {Colors.STATUS_SUCCESS.name()}; font-size: 9pt;")
        self.model_status_label.setText(f"Model file found: {Path(model_path).name}")

    def _on_model_path_changed(self, text: str):
        try:
            self._settings_manager.model_path = text if text else None
            self._update_model_status(text)
        except ValueError as exc:
            self.set_status_message(str(exc), error=True)
            self.refresh()

    def _on_browse_model(self):
        current = self._settings_manager.model_path or ""
        start_dir = str(Path(current).parent) if current and os.path.exists(current) else app_settings.get_dialog_path("learned_onset_model")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select onset model checkpoint",
            start_dir,
            "PyTorch checkpoint (*.pth);;All Files (*)",
        )
        if file_path:
            app_settings.set_dialog_path("learned_onset_model", file_path)
            self.model_path_edit.setText(file_path)

    def _on_device_changed(self, value: str):
        try:
            self._settings_manager.device = value
        except ValueError as exc:
            self.set_status_message(str(exc), error=True)
            self.refresh()

    def _on_threshold_changed(self, value: float):
        try:
            self._settings_manager.threshold = value
        except ValueError as exc:
            self.set_status_message(str(exc), error=True)
            self.refresh()

    def _on_min_silence_changed(self, value: float):
        try:
            self._settings_manager.min_silence = value
        except ValueError as exc:
            self.set_status_message(str(exc), error=True)
            self.refresh()

    def _on_hop_length_changed(self, value: int):
        try:
            self._settings_manager.hop_length = value
        except ValueError as exc:
            self.set_status_message(str(exc), error=True)
            self.refresh()

    def _on_n_mels_changed(self, value: int):
        try:
            self._settings_manager.n_mels = value
        except ValueError as exc:
            self.set_status_message(str(exc), error=True)
            self.refresh()

    def _on_backtrack_changed(self, value: bool):
        try:
            self._settings_manager.use_backtrack = bool(value)
        except ValueError as exc:
            self.set_status_message(str(exc), error=True)
            self.refresh()

    def _on_fallback_changed(self, value: bool):
        try:
            self._settings_manager.fallback_to_spectral_flux = bool(value)
        except ValueError as exc:
            self.set_status_message(str(exc), error=True)
            self.refresh()

    def _on_setting_changed(self, setting_name: str):
        if setting_name in {
            "model_path",
            "device",
            "threshold",
            "min_silence",
            "hop_length",
            "n_mels",
            "use_backtrack",
            "fallback_to_spectral_flux",
        }:
            self.refresh()
