"""
LearnedOnsetTrainer block panel.

Minimal panel for configuring PoC onset model training.
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
)

from src.application.settings.learned_onset_trainer_settings import LearnedOnsetTrainerSettingsManager
from src.utils.datasets import get_managed_external_datasets_dir, resolve_dataset_path
from src.utils.message import Log
from src.utils.settings import app_settings
from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing


@register_block_panel("LearnedOnsetTrainer")
class LearnedOnsetTrainerPanel(BlockPanelBase):
    """Panel for LearnedOnsetTrainer settings."""

    def __init__(self, block_id: str, facade, parent=None):
        super().__init__(block_id, facade, parent)
        self._settings_manager = LearnedOnsetTrainerSettingsManager(facade, block_id, parent=self)
        self._settings_manager.settings_changed.connect(self._on_setting_changed)
        if self.block:
            self.refresh()

    def create_content_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)

        info = QLabel(
            "PoC onset trainer for IDMT-style datasets.\n"
            "Expected folder layout: dataset_root/audio + dataset_root/annotation_xml"
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;")
        layout.addWidget(info)

        data_group = QGroupBox("Dataset")
        data_form = QFormLayout(data_group)
        data_form.setSpacing(Spacing.SM)

        self.dataset_root_edit = QLineEdit()
        self.dataset_root_edit.setPlaceholderText("Path to extracted dataset root")
        self.dataset_root_edit.textChanged.connect(self._on_dataset_root_changed)
        data_row = QHBoxLayout()
        data_row.addWidget(self.dataset_root_edit)
        browse_dataset_btn = QPushButton("Browse...")
        browse_dataset_btn.clicked.connect(self._on_browse_dataset_root)
        data_row.addWidget(browse_dataset_btn)
        data_form.addRow("Dataset Root:", data_row)

        self.max_files_spin = QSpinBox()
        self.max_files_spin.setRange(0, 100000)
        self.max_files_spin.setToolTip("0 = use all annotation files")
        self.max_files_spin.valueChanged.connect(self._on_max_files_changed)
        data_form.addRow("Max Files:", self.max_files_spin)
        layout.addWidget(data_group)

        train_group = QGroupBox("Training")
        train_form = QFormLayout(train_group)
        train_form.setSpacing(Spacing.SM)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.valueChanged.connect(self._on_epochs_changed)
        train_form.addRow("Epochs:", self.epochs_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.valueChanged.connect(self._on_batch_size_changed)
        train_form.addRow("Batch Size:", self.batch_size_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(1e-7, 1.0)
        self.learning_rate_spin.setDecimals(7)
        self.learning_rate_spin.valueChanged.connect(self._on_learning_rate_changed)
        train_form.addRow("Learning Rate:", self.learning_rate_spin)

        self.validation_split_spin = QDoubleSpinBox()
        self.validation_split_spin.setRange(0.05, 0.5)
        self.validation_split_spin.setSingleStep(0.01)
        self.validation_split_spin.valueChanged.connect(self._on_validation_split_changed)
        train_form.addRow("Validation Split:", self.validation_split_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda", "mps"])
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        train_form.addRow("Device:", self.device_combo)

        layout.addWidget(train_group)

        feature_group = QGroupBox("Feature/Window")
        feature_form = QFormLayout(feature_group)
        feature_form.setSpacing(Spacing.SM)

        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 96000)
        self.sample_rate_spin.valueChanged.connect(self._on_sample_rate_changed)
        feature_form.addRow("Sample Rate:", self.sample_rate_spin)

        self.n_mels_spin = QSpinBox()
        self.n_mels_spin.setRange(16, 512)
        self.n_mels_spin.valueChanged.connect(self._on_n_mels_changed)
        feature_form.addRow("Mel Bins:", self.n_mels_spin)

        self.mel_hop_spin = QSpinBox()
        self.mel_hop_spin.setRange(32, 4096)
        self.mel_hop_spin.valueChanged.connect(self._on_mel_hop_changed)
        feature_form.addRow("Mel Hop Length:", self.mel_hop_spin)

        self.window_seconds_spin = QDoubleSpinBox()
        self.window_seconds_spin.setRange(0.1, 10.0)
        self.window_seconds_spin.setSingleStep(0.1)
        self.window_seconds_spin.valueChanged.connect(self._on_window_seconds_changed)
        feature_form.addRow("Window Seconds:", self.window_seconds_spin)

        self.positive_radius_ms_spin = QDoubleSpinBox()
        self.positive_radius_ms_spin.setRange(1.0, 200.0)
        self.positive_radius_ms_spin.valueChanged.connect(self._on_positive_radius_changed)
        feature_form.addRow("Positive Radius (ms):", self.positive_radius_ms_spin)

        self.negative_ratio_spin = QDoubleSpinBox()
        self.negative_ratio_spin.setRange(0.1, 10.0)
        self.negative_ratio_spin.setSingleStep(0.1)
        self.negative_ratio_spin.valueChanged.connect(self._on_negative_ratio_changed)
        feature_form.addRow("Negative Ratio:", self.negative_ratio_spin)

        layout.addWidget(feature_group)

        output_group = QGroupBox("Output")
        output_form = QFormLayout(output_group)
        output_form.setSpacing(Spacing.SM)

        self.output_model_path_edit = QLineEdit()
        self.output_model_path_edit.setPlaceholderText("Optional .pth output path (auto if empty)")
        self.output_model_path_edit.textChanged.connect(self._on_output_model_path_changed)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_model_path_edit)
        browse_out_btn = QPushButton("Browse...")
        browse_out_btn.clicked.connect(self._on_browse_output_path)
        output_row.addWidget(browse_out_btn)
        output_form.addRow("Model Path:", output_row)
        layout.addWidget(output_group)

        layout.addStretch()
        return widget

    def refresh(self):
        if not hasattr(self, "_settings_manager") or not self._settings_manager:
            return
        if not self.block or not self._settings_manager.is_loaded():
            return
        try:
            self.dataset_root_edit.blockSignals(True)
            self.max_files_spin.blockSignals(True)
            self.epochs_spin.blockSignals(True)
            self.batch_size_spin.blockSignals(True)
            self.learning_rate_spin.blockSignals(True)
            self.validation_split_spin.blockSignals(True)
            self.device_combo.blockSignals(True)
            self.sample_rate_spin.blockSignals(True)
            self.n_mels_spin.blockSignals(True)
            self.mel_hop_spin.blockSignals(True)
            self.window_seconds_spin.blockSignals(True)
            self.positive_radius_ms_spin.blockSignals(True)
            self.negative_ratio_spin.blockSignals(True)
            self.output_model_path_edit.blockSignals(True)

            self.dataset_root_edit.setText(resolve_dataset_path(self._settings_manager.dataset_root) or "")
            self.max_files_spin.setValue(self._settings_manager.max_files)
            self.epochs_spin.setValue(self._settings_manager.epochs)
            self.batch_size_spin.setValue(self._settings_manager.batch_size)
            self.learning_rate_spin.setValue(self._settings_manager.learning_rate)
            self.validation_split_spin.setValue(self._settings_manager.validation_split)

            dev_idx = self.device_combo.findText(self._settings_manager.device)
            if dev_idx >= 0:
                self.device_combo.setCurrentIndex(dev_idx)
            self.sample_rate_spin.setValue(self._settings_manager.sample_rate)
            self.n_mels_spin.setValue(self._settings_manager.n_mels)
            self.mel_hop_spin.setValue(self._settings_manager.mel_hop_length)
            self.window_seconds_spin.setValue(self._settings_manager.window_seconds)
            self.positive_radius_ms_spin.setValue(self._settings_manager.positive_radius_ms)
            self.negative_ratio_spin.setValue(self._settings_manager.negative_ratio)
            self.output_model_path_edit.setText(self._settings_manager.output_model_path or "")
        except Exception as exc:
            Log.error(f"LearnedOnsetTrainerPanel: Failed to refresh settings: {exc}")
        finally:
            self.dataset_root_edit.blockSignals(False)
            self.max_files_spin.blockSignals(False)
            self.epochs_spin.blockSignals(False)
            self.batch_size_spin.blockSignals(False)
            self.learning_rate_spin.blockSignals(False)
            self.validation_split_spin.blockSignals(False)
            self.device_combo.blockSignals(False)
            self.sample_rate_spin.blockSignals(False)
            self.n_mels_spin.blockSignals(False)
            self.mel_hop_spin.blockSignals(False)
            self.window_seconds_spin.blockSignals(False)
            self.positive_radius_ms_spin.blockSignals(False)
            self.negative_ratio_spin.blockSignals(False)
            self.output_model_path_edit.blockSignals(False)

    def _on_browse_dataset_root(self):
        start_dir = (
            resolve_dataset_path(self._settings_manager.dataset_root)
            or app_settings.get_dialog_path("learned_onset_dataset_root")
            or str(get_managed_external_datasets_dir())
        )
        directory = QFileDialog.getExistingDirectory(self, "Select onset dataset root", start_dir)
        if directory:
            app_settings.set_dialog_path("learned_onset_dataset_root", directory)
            self.dataset_root_edit.setText(directory)

    def _on_browse_output_path(self):
        start_path = self._settings_manager.output_model_path or app_settings.get_dialog_path("learned_onset_model_output")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save onset model",
            start_path,
            "PyTorch checkpoint (*.pth);;All Files (*)",
        )
        if file_path:
            app_settings.set_dialog_path("learned_onset_model_output", str(Path(file_path).parent))
            self.output_model_path_edit.setText(file_path)

    def _on_dataset_root_changed(self, value: str):
        try:
            self._settings_manager.dataset_root = resolve_dataset_path(value) or value
        except ValueError as exc:
            self.set_status_message(str(exc), error=True)

    def _on_max_files_changed(self, value: int):
        self._settings_manager.max_files = value

    def _on_epochs_changed(self, value: int):
        self._settings_manager.epochs = value

    def _on_batch_size_changed(self, value: int):
        self._settings_manager.batch_size = value

    def _on_learning_rate_changed(self, value: float):
        self._settings_manager.learning_rate = value

    def _on_validation_split_changed(self, value: float):
        self._settings_manager.validation_split = value

    def _on_device_changed(self, value: str):
        self._settings_manager.device = value

    def _on_sample_rate_changed(self, value: int):
        self._settings_manager.sample_rate = value

    def _on_n_mels_changed(self, value: int):
        self._settings_manager.n_mels = value

    def _on_mel_hop_changed(self, value: int):
        self._settings_manager.mel_hop_length = value

    def _on_window_seconds_changed(self, value: float):
        self._settings_manager.window_seconds = value

    def _on_positive_radius_changed(self, value: float):
        self._settings_manager.positive_radius_ms = value

    def _on_negative_ratio_changed(self, value: float):
        self._settings_manager.negative_ratio = value

    def _on_output_model_path_changed(self, value: str):
        self._settings_manager.output_model_path = value if value else None

    def _on_setting_changed(self, setting_name: str):
        if setting_name:
            self.refresh()
