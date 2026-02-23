"""
PyTorchAudioClassify block panel.

Provides UI for configuring PyTorch Audio Classify settings.
Specifically designed for models created by PyTorch Audio Trainer.
Can receive a model via the 'model' input port from a connected Trainer block,
or select a model from the models directory dropdown.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QGroupBox, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QTextEdit, QCheckBox, QProgressBar, QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from pathlib import Path
import os

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing
from src.application.settings.pytorch_audio_classify_settings import PyTorchAudioClassifySettingsManager
from src.application.services.model_download_service import (
    ModelDownloadService,
    ModelDownloadError,
    RemoteModel,
    create_model_download_service,
)
from src.utils.message import Log
from src.utils.paths import get_models_dir


class _ManifestWorker(QThread):
    """Background thread to fetch the model manifest."""

    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, service: ModelDownloadService, parent=None):
        super().__init__(parent)
        self._service = service

    def run(self):
        try:
            models = self._service.fetch_available_models()
            self.finished.emit(models)
        except ModelDownloadError as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            self.error.emit(f"Unexpected error: {exc}")


class _DownloadWorker(QThread):
    """Background thread to download a single model with progress."""

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, service: ModelDownloadService, model: RemoteModel, parent=None):
        super().__init__(parent)
        self._service = service
        self._model = model

    def run(self):
        try:
            path = self._service.download_model(
                self._model,
                progress_callback=lambda downloaded, total: self.progress.emit(downloaded, total),
            )
            self.finished.emit(path)
        except ModelDownloadError as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            self.error.emit(f"Download failed: {exc}")


@register_block_panel("PyTorchAudioClassify")
class PyTorchAudioClassifyPanel(BlockPanelBase):
    """Panel for PyTorchAudioClassify block configuration."""

    def __init__(self, block_id: str, facade, parent=None):
        # Call parent init first (sets up UI structure).
        # Note: parent.__init__ calls refresh() before _settings_manager exists,
        # so refresh() must be defensive.
        super().__init__(block_id, facade, parent)

        # Initialize settings manager AFTER parent init
        self._settings_manager = PyTorchAudioClassifySettingsManager(facade, block_id, parent=self)

        # Connect to settings changes for UI updates
        self._settings_manager.settings_changed.connect(self._on_setting_changed)
        self._settings_manager.settings_save_failed.connect(self._on_settings_save_failed)

        # Refresh UI now that settings manager is ready
        if self.block:
            self.refresh()

    def create_content_widget(self) -> QWidget:
        """Create PyTorchAudioClassify-specific UI."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)

        # -- Model Source Info --
        source_group = QGroupBox("Model Source")
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(Spacing.SM)

        self.model_source_label = QLabel(
            "No model selected.\n"
            "Connect a PyTorch Audio Trainer output, or select a model below."
        )
        self.model_source_label.setWordWrap(True)
        self.model_source_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt; padding: 4px;"
        )
        source_layout.addWidget(self.model_source_label)

        layout.addWidget(source_group)

        # -- Model Store --
        store_group = QGroupBox("Model Store")
        store_layout = QVBoxLayout(store_group)
        store_layout.setSpacing(Spacing.SM)

        store_header = QHBoxLayout()
        store_header.setSpacing(Spacing.SM)

        self.check_models_btn = QPushButton("Check for Available Models")
        self.check_models_btn.setToolTip(
            "Connect to the EchoZero model store and check for\n"
            "pre-trained models available for download."
        )
        self.check_models_btn.clicked.connect(self._on_check_models)
        store_header.addWidget(self.check_models_btn)
        store_header.addStretch()

        self.store_status_label = QLabel("")
        self.store_status_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt;"
        )
        store_header.addWidget(self.store_status_label)
        store_layout.addLayout(store_header)

        self.store_list_container = QWidget()
        self.store_list_layout = QVBoxLayout(self.store_list_container)
        self.store_list_layout.setContentsMargins(0, 0, 0, 0)
        self.store_list_layout.setSpacing(Spacing.SM)
        store_layout.addWidget(self.store_list_container)

        layout.addWidget(store_group)

        # -- Model Details (comprehensive info loaded from checkpoint) --
        details_group = QGroupBox("Model Details")
        details_layout = QVBoxLayout(details_group)
        details_layout.setSpacing(Spacing.SM)

        self.model_details_display = QTextEdit()
        self.model_details_display.setReadOnly(True)
        self.model_details_display.setStyleSheet(
            f"QTextEdit {{"
            f"  background-color: {Colors.BG_MEDIUM.name()};"
            f"  color: {Colors.TEXT_PRIMARY.name()};"
            f"  border: 1px solid {Colors.BORDER.name()};"
            f"  border-radius: 4px;"
            f"  padding: 8px;"
            f"  font-size: 10pt;"
            f"  font-family: -apple-system, system-ui, 'Segoe UI', monospace;"
            f"}}"
        )
        self.model_details_display.setMinimumHeight(120)
        self.model_details_display.setMaximumHeight(400)
        self.model_details_display.setPlaceholderText(
            "No model loaded. Select a model or connect a trainer block to see details."
        )
        details_layout.addWidget(self.model_details_display)

        layout.addWidget(details_group)

        # -- Model Selection --
        model_group = QGroupBox("Model Selection")
        model_form = QFormLayout(model_group)
        model_form.setSpacing(Spacing.SM)

        model_container = QVBoxLayout()

        combo_row = QHBoxLayout()
        combo_row.setSpacing(Spacing.SM)

        self.model_combo = QComboBox()
        self.model_combo.setToolTip(
            "Select a model from the models directory.\n\n"
            "Models are .pth files created by PyTorch Audio Trainer.\n"
            "If a Trainer block is connected to the model input port,\n"
            "the connected model takes priority."
        )
        self.model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        combo_row.addWidget(self.model_combo, 1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Rescan the models directory for available models")
        refresh_btn.setFixedWidth(70)
        refresh_btn.clicked.connect(self._refresh_model_list)
        combo_row.addWidget(refresh_btn)

        model_container.addLayout(combo_row)

        self.model_path_status_label = QLabel("")
        self.model_path_status_label.setWordWrap(True)
        self.model_path_status_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt;"
        )
        model_container.addWidget(self.model_path_status_label)

        model_form.addRow("Model:", model_container)
        layout.addWidget(model_group)

        # -- Processing Settings --
        processing_group = QGroupBox("Processing Settings")
        processing_layout = QFormLayout(processing_group)
        processing_layout.setSpacing(Spacing.SM)

        # Sample rate (0 = auto from model config)
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(0, 96000)
        self.sample_rate_spin.setSingleStep(1000)
        self.sample_rate_spin.setSpecialValueText("Auto (from model)")
        self.sample_rate_spin.setValue(0)
        self.sample_rate_spin.setToolTip(
            "Audio sample rate in Hz.\n"
            "Leave at 0 to use the sample rate from the model's training config."
        )
        self.sample_rate_spin.valueChanged.connect(self._on_sample_rate_changed)
        processing_layout.addRow("Sample Rate:", self.sample_rate_spin)

        # Batch size (0 = auto)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(0, 512)
        self.batch_size_spin.setSpecialValueText("Auto")
        self.batch_size_spin.setValue(0)
        self.batch_size_spin.setToolTip(
            "Batch size for prediction.\n"
            "Set to 0 for automatic batch sizing."
        )
        self.batch_size_spin.valueChanged.connect(self._on_batch_size_changed)
        processing_layout.addRow("Batch Size:", self.batch_size_spin)

        # Device
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda", "mps"])
        self.device_combo.setToolTip(
            "Device to use for inference.\n\n"
            "cpu: Always available\n"
            "cuda: NVIDIA GPU (requires CUDA)\n"
            "mps: Apple Silicon GPU (macOS)"
        )
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        processing_layout.addRow("Device:", self.device_combo)

        # Confidence threshold
        self.confidence_threshold_spin = QDoubleSpinBox()
        self.confidence_threshold_spin.setRange(0.0, 1.0)
        self.confidence_threshold_spin.setSingleStep(0.05)
        self.confidence_threshold_spin.setDecimals(2)
        self.confidence_threshold_spin.setSpecialValueText("Auto (from model)")
        self.confidence_threshold_spin.setValue(0.0)
        self.confidence_threshold_spin.setToolTip(
            "Confidence threshold for classification.\n\n"
            "In binary mode, events with confidence above this threshold\n"
            "are classified as the target class.\n\n"
            "Set to 0 to use the model's optimal threshold\n"
            "(tuned during training)."
        )
        self.confidence_threshold_spin.valueChanged.connect(self._on_confidence_threshold_changed)
        processing_layout.addRow("Confidence Threshold:", self.confidence_threshold_spin)

        # Multiclass multi-label (only applies to multiclass models)
        self.multiclass_multi_label_check = QCheckBox("Allow multiple classes per event")
        self.multiclass_multi_label_check.setToolTip(
            "When enabled (multiclass and positive_vs_other models), a single onset can produce events "
            "in multiple layers if more than one class exceeds the minimum confidence.\n\n"
            "Useful for samples with layered sounds (e.g. kick + hi-hat)."
        )
        self.multiclass_multi_label_check.stateChanged.connect(self._on_multiclass_multi_label_changed)
        processing_layout.addRow("", self.multiclass_multi_label_check)

        self.multiclass_threshold_spin = QDoubleSpinBox()
        self.multiclass_threshold_spin.setRange(0.0, 1.0)
        self.multiclass_threshold_spin.setSingleStep(0.05)
        self.multiclass_threshold_spin.setDecimals(2)
        self.multiclass_threshold_spin.setValue(0.4)
        self.multiclass_threshold_spin.setToolTip(
            "Minimum probability to include a class when 'Allow multiple classes' is enabled."
        )
        self.multiclass_threshold_spin.valueChanged.connect(self._on_multiclass_threshold_changed)
        processing_layout.addRow("Multi-label threshold:", self.multiclass_threshold_spin)

        # Create layer for "other" (binary/positive_vs_other models)
        self.create_other_layer_check = QCheckBox("Create layer for 'other'")
        self.create_other_layer_check.setToolTip(
            "When enabled, events classified as 'other' (below threshold in binary/positive-vs-other "
            "mode) get their own layer. When disabled, those events are dropped from the output."
        )
        self.create_other_layer_check.stateChanged.connect(self._on_create_other_layer_changed)
        processing_layout.addRow("", self.create_other_layer_check)

        layout.addWidget(processing_group)

        # -- Execution Summary --
        summary_group = QGroupBox("Last Execution Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.setSpacing(Spacing.SM)

        self.summary_label = QLabel(
            "No execution data available.\nRun the block to see results here."
        )
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt; padding: 8px;"
        )
        summary_layout.addWidget(self.summary_label)

        layout.addWidget(summary_group)

        layout.addStretch()
        return widget

    # =========================================================================
    # Refresh
    # =========================================================================

    def refresh(self):
        """Update UI with current block settings."""
        if not hasattr(self, "_settings_manager") or not self._settings_manager:
            return

        if not self.block or not self._settings_manager.is_loaded():
            return

        try:
            model_path = self._settings_manager.model_path
            sample_rate = self._settings_manager.sample_rate
            batch_size = self._settings_manager.batch_size
            device = self._settings_manager.device
            confidence_threshold = self._settings_manager.confidence_threshold
            multiclass_multi_label = self._settings_manager.multiclass_multi_label
            multiclass_confidence_threshold = self._settings_manager.multiclass_confidence_threshold
            create_other_layer = self._settings_manager.create_other_layer
        except Exception as e:
            Log.error(f"PyTorchAudioClassifyPanel: Failed to load settings: {e}")
            return

        # Block signals while updating
        self.model_combo.blockSignals(True)
        self.sample_rate_spin.blockSignals(True)
        self.batch_size_spin.blockSignals(True)
        self.device_combo.blockSignals(True)
        self.confidence_threshold_spin.blockSignals(True)
        if hasattr(self, "multiclass_multi_label_check"):
            self.multiclass_multi_label_check.blockSignals(True)
        if hasattr(self, "multiclass_threshold_spin"):
            self.multiclass_threshold_spin.blockSignals(True)
        if hasattr(self, "create_other_layer_check"):
            self.create_other_layer_check.blockSignals(True)

        # Populate model list and select current
        self._populate_model_combo(model_path)

        # Sample rate (None -> 0 = Auto)
        self.sample_rate_spin.setValue(sample_rate if sample_rate else 0)

        # Batch size (None -> 0 = Auto)
        self.batch_size_spin.setValue(batch_size if batch_size else 0)

        # Device
        device_index = self.device_combo.findText(device)
        if device_index >= 0:
            self.device_combo.setCurrentIndex(device_index)

        # Confidence threshold (None -> 0.0 = Auto)
        self.confidence_threshold_spin.setValue(confidence_threshold if confidence_threshold is not None else 0.0)

        # Multiclass multi-label
        if hasattr(self, "multiclass_multi_label_check"):
            self.multiclass_multi_label_check.setChecked(multiclass_multi_label)
        if hasattr(self, "multiclass_threshold_spin"):
            self.multiclass_threshold_spin.setValue(multiclass_confidence_threshold)
        if hasattr(self, "create_other_layer_check"):
            self.create_other_layer_check.setChecked(create_other_layer)

        # Unblock signals
        self.model_combo.blockSignals(False)
        self.sample_rate_spin.blockSignals(False)
        self.batch_size_spin.blockSignals(False)
        self.device_combo.blockSignals(False)
        self.confidence_threshold_spin.blockSignals(False)
        if hasattr(self, "multiclass_multi_label_check"):
            self.multiclass_multi_label_check.blockSignals(False)
        if hasattr(self, "multiclass_threshold_spin"):
            self.multiclass_threshold_spin.blockSignals(False)
        if hasattr(self, "create_other_layer_check"):
            self.create_other_layer_check.blockSignals(False)

        # Update derived displays
        self._update_model_source_info()
        self._update_model_path_validation()
        self._update_execution_summary()

        self.set_status_message("Settings loaded")

    # =========================================================================
    # Model Source Display
    # =========================================================================

    def _update_model_source_info(self):
        """Update the model source indicator (connected trainer vs. dropdown selection)."""
        if not hasattr(self, "model_source_label"):
            return

        model_connected = False
        if self.block and hasattr(self.facade, "connection_service"):
            try:
                connections = self.facade.connection_service.list_connections_by_block(self.block.id)
                model_connected = any(
                    c.target_block_id == self.block.id and c.target_input_name == "model"
                    for c in connections
                )
            except Exception:
                pass

        if model_connected:
            self.model_source_label.setText(
                "Model source: Connected trainer block (takes priority over selection)"
            )
            self.model_source_label.setStyleSheet(
                f"color: {Colors.STATUS_SUCCESS.name()}; font-size: 10pt; padding: 4px;"
            )
        else:
            model_path = self._get_selected_model_path()
            if model_path and os.path.exists(model_path):
                self.model_source_label.setText(
                    f"Model source: {Path(model_path).name}"
                )
                self.model_source_label.setStyleSheet(
                    f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt; padding: 4px;"
                )
            else:
                self.model_source_label.setText(
                    "No model selected.\n"
                    "Connect a PyTorch Audio Trainer output, or select a model above."
                )
                self.model_source_label.setStyleSheet(
                    f"color: {Colors.STATUS_WARNING.name()}; font-size: 10pt; padding: 4px;"
                )

    def _update_model_path_validation(self):
        """Validate model path and display comprehensive model metadata."""
        if not hasattr(self, "model_path_status_label"):
            return

        path = self._get_selected_model_path()

        if not path:
            self.model_path_status_label.setText("")
            self.model_details_display.setHtml("")
            return

        if not os.path.exists(path):
            self.model_path_status_label.setText("Path not found")
            self.model_path_status_label.setStyleSheet(
                f"color: {Colors.STATUS_ERROR.name()}; font-size: 9pt;"
            )
            self.model_details_display.setHtml("")
            return

        # Try to load model metadata (lazy import to avoid top-level torch dependency)
        try:
            import torch
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            if "classes" not in checkpoint or "config" not in checkpoint or "model_state_dict" not in checkpoint:
                self.model_path_status_label.setText("Not a PyTorch Audio Trainer model")
                self.model_path_status_label.setStyleSheet(
                    f"color: {Colors.STATUS_WARNING.name()}; font-size: 9pt;"
                )
                self.model_details_display.setPlainText(
                    "This model file does not appear to be from PyTorch Audio Trainer.\n"
                    "It must contain 'classes', 'config', and 'model_state_dict' keys."
                )
                return

            self.model_path_status_label.setText("Valid PyTorch Audio Trainer model")
            self.model_path_status_label.setStyleSheet(
                f"color: {Colors.STATUS_SUCCESS.name()}; font-size: 9pt;"
            )

            html = self._build_model_details_html(checkpoint, path)
            self.model_details_display.setHtml(html)

        except ImportError:
            self.model_path_status_label.setText("PyTorch not installed -- cannot inspect model")
            self.model_path_status_label.setStyleSheet(
                f"color: {Colors.STATUS_WARNING.name()}; font-size: 9pt;"
            )
            self.model_details_display.setHtml("")
        except Exception as e:
            self.model_path_status_label.setText(f"Error loading model: {str(e)[:80]}")
            self.model_path_status_label.setStyleSheet(
                f"color: {Colors.STATUS_ERROR.name()}; font-size: 9pt;"
            )
            self.model_details_display.setHtml("")

    def _build_model_details_html(self, checkpoint: dict, model_path: str) -> str:
        """
        Build a comprehensive HTML display of all model checkpoint details.

        Extracts and presents every useful field stored in the checkpoint:
        identity, architecture, classification mode, training config,
        dataset statistics, test metrics, and preprocessing parameters.
        """
        classes = checkpoint.get("classes", [])
        config = checkpoint.get("config", {})
        training_date = checkpoint.get("training_date", "Unknown")
        classification_mode = checkpoint.get("classification_mode", "multiclass")
        target_class = checkpoint.get("target_class")
        optimal_threshold = checkpoint.get("optimal_threshold")
        normalization = checkpoint.get("normalization")
        test_metrics = checkpoint.get("test_metrics")
        dataset_stats = checkpoint.get("dataset_stats")
        training_history = checkpoint.get("training_history")

        # Model parameter count
        param_count = None
        state_dict = checkpoint.get("model_state_dict", {})
        if state_dict:
            try:
                param_count = sum(v.numel() for v in state_dict.values())
            except Exception:
                pass

        # File size
        try:
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        except Exception:
            file_size_mb = None

        # Colors for HTML
        heading = Colors.ACCENT_BLUE.name()
        label_color = Colors.TEXT_SECONDARY.name()
        value_color = Colors.TEXT_PRIMARY.name()
        success = Colors.STATUS_SUCCESS.name()
        warning = Colors.STATUS_WARNING.name()

        def section(title: str) -> str:
            return (
                f'<p style="margin-top:10px; margin-bottom:4px;">'
                f'<b style="color:{heading}; font-size:10pt;">{title}</b></p>'
            )

        def row(label: str, value, fmt: str = None, color: str = None) -> str:
            if value is None:
                return ""
            c = color or value_color
            if fmt and isinstance(value, (int, float)):
                v = fmt.format(value)
            else:
                v = str(value)
            return (
                f'<tr>'
                f'<td style="color:{label_color}; padding-right:12px; white-space:nowrap;">{label}</td>'
                f'<td style="color:{c};"><b>{v}</b></td>'
                f'</tr>'
            )

        def table_start() -> str:
            return '<table cellspacing="0" cellpadding="2">'

        def table_end() -> str:
            return '</table>'

        parts = []

        # ── Identity ──
        model_type = config.get("model_type", config.get("model_architecture", "Unknown"))
        parts.append(section("Identity"))
        parts.append(table_start())
        parts.append(row("File", Path(model_path).name))
        if file_size_mb is not None:
            parts.append(row("Size", file_size_mb, fmt="{:.1f} MB"))
        if isinstance(training_date, str) and len(training_date) >= 10:
            parts.append(row("Trained", training_date[:19].replace("T", " ")))
        parts.append(row("Architecture", model_type.upper()))
        if param_count is not None:
            if param_count >= 1_000_000:
                parts.append(row("Parameters", param_count / 1_000_000, fmt="{:.2f}M"))
            else:
                parts.append(row("Parameters", f"{param_count:,}"))
        parts.append(table_end())

        # ── Classification Mode ──
        parts.append(section("Classification"))
        parts.append(table_start())
        mode_display = classification_mode.capitalize()
        if classification_mode == "binary" and target_class:
            mode_display = f"Binary (target: {target_class})"
        elif classification_mode == "positive_vs_other":
            mode_display = "Positive vs Other"
        parts.append(row("Mode", mode_display))
        parts.append(row("Classes", f"{len(classes)} -- {', '.join(classes)}"))
        if classification_mode == "binary" and optimal_threshold is not None:
            parts.append(row("Optimal Threshold", optimal_threshold, fmt="{:.4f}"))
        parts.append(table_end())

        # ── Training Configuration ──
        parts.append(section("Training Configuration"))
        parts.append(table_start())
        parts.append(row("Epochs", config.get("epochs")))
        parts.append(row("Batch Size", config.get("batch_size")))
        eff_batch = config.get("batch_size", 0) * config.get("gradient_accumulation_steps", 1)
        if config.get("gradient_accumulation_steps", 1) > 1:
            parts.append(row("Effective Batch", eff_batch))
        parts.append(row("Learning Rate", config.get("learning_rate"), fmt="{:.6f}"))
        parts.append(row("Optimizer", (config.get("optimizer") or "").upper() or None))
        parts.append(row("LR Scheduler", config.get("lr_scheduler")))
        parts.append(row("Warmup Epochs", config.get("warmup_epochs")))
        parts.append(row("Weight Decay", config.get("weight_decay"), fmt="{:.1e}"))
        parts.append(row("Dropout", config.get("dropout_rate"), fmt="{:.2f}"))
        parts.append(row("Label Smoothing", config.get("label_smoothing"), fmt="{:.2f}"))
        parts.append(row("Gradient Clip", config.get("gradient_clip_norm"), fmt="{:.1f}"))
        parts.append(row("Early Stopping", f"patience {config.get('early_stopping_patience')}" if config.get("early_stopping_patience") else None))
        parts.append(row("Seed", config.get("seed")))
        parts.append(table_end())

        # ── Advanced Techniques ──
        techniques = []
        if config.get("use_amp"):
            techniques.append("Mixed Precision (AMP)")
        if config.get("use_ema"):
            techniques.append(f"EMA (decay={config.get('ema_decay', 0.999)})")
        if config.get("use_swa"):
            techniques.append(f"SWA (from epoch {config.get('swa_start_epoch', '?')})")
        if config.get("use_class_weights"):
            techniques.append("Class Weights")
        if config.get("use_weighted_sampling"):
            techniques.append("Weighted Sampling")
        if techniques:
            parts.append(section("Advanced Techniques"))
            parts.append(
                f'<p style="color:{value_color}; margin:0; padding-left:4px;">'
                + " | ".join(techniques) + '</p>'
            )

        # ── Data Augmentation ──
        if config.get("use_augmentation"):
            aug_items = []
            if config.get("pitch_shift_range"):
                aug_items.append(f"Pitch +/-{config['pitch_shift_range']}st")
            if config.get("time_stretch_range"):
                aug_items.append(f"Time Stretch +/-{config['time_stretch_range']}")
            if config.get("noise_factor"):
                aug_items.append(f"Noise ({config['noise_factor']})")
            if config.get("volume_factor"):
                aug_items.append(f"Volume +/-{config['volume_factor']}")
            if config.get("frequency_mask"):
                aug_items.append(f"FreqMask ({config['frequency_mask']})")
            if config.get("time_mask"):
                aug_items.append(f"TimeMask ({config['time_mask']})")
            if config.get("use_mixup"):
                aug_items.append(f"Mixup (a={config.get('mixup_alpha', 0.2)})")
            if config.get("use_cutmix"):
                aug_items.append(f"CutMix (a={config.get('cutmix_alpha', 1.0)})")
            if config.get("polarity_inversion_prob"):
                aug_items.append("Polarity Inversion")
            if config.get("use_random_eq"):
                aug_items.append("Random EQ")
            if aug_items:
                parts.append(section("Augmentation"))
                parts.append(
                    f'<p style="color:{value_color}; margin:0; padding-left:4px;">'
                    + ", ".join(aug_items) + '</p>'
                )
        else:
            parts.append(section("Augmentation"))
            parts.append(
                f'<p style="color:{label_color}; margin:0; padding-left:4px;">'
                'Disabled</p>'
            )

        # ── Preprocessing / Spectrogram ──
        parts.append(section("Preprocessing"))
        parts.append(table_start())
        parts.append(row("Sample Rate", f"{config.get('sample_rate', 22050)} Hz"))
        max_len = config.get("max_length", 22050)
        sr = config.get("sample_rate", 22050)
        if sr > 0:
            parts.append(row("Max Length", f"{max_len} samples ({max_len / sr:.2f}s)"))
        parts.append(row("Mel Bins", config.get("n_mels", 128)))
        parts.append(row("Hop Length", config.get("hop_length", 512)))
        parts.append(row("FFT Size", config.get("n_fft", 2048)))
        parts.append(row("Fmax", f"{config.get('fmax', 8000)} Hz"))
        if normalization:
            parts.append(row("Normalization", "Per-dataset (mean/std saved)", color=success))
        else:
            parts.append(row("Normalization", "Per-sample"))
        parts.append(table_end())

        # ── Dataset Statistics ──
        if dataset_stats:
            parts.append(section("Dataset Statistics"))
            parts.append(table_start())
            if isinstance(dataset_stats, dict):
                parts.append(row("Total Samples", dataset_stats.get("total_samples")))
                parts.append(row("Training", dataset_stats.get("train_samples")))
                parts.append(row("Validation", dataset_stats.get("val_samples")))
                parts.append(row("Test", dataset_stats.get("test_samples")))

                # Balance strategy (from training)
                balance_strategy = dataset_stats.get("balance_strategy")
                if balance_strategy and balance_strategy != "none":
                    strategy_display = balance_strategy.replace("_", " ").title()
                    parts.append(row("Balance Strategy", strategy_display, color=success))
                    pre_balance = dataset_stats.get("pre_balance_distribution")
                    if pre_balance and isinstance(pre_balance, dict):
                        summary = ", ".join(
                            f"{k} {v}" for k, v in sorted(pre_balance.items())
                        )
                        parts.append(row("Pre-balance", summary))

                class_dist = dataset_stats.get("class_distribution")
                if class_dist and isinstance(class_dist, dict):
                    for cls_name, count in sorted(class_dist.items()):
                        parts.append(row(f"  {cls_name}", count))
                formats_used = dataset_stats.get("formats_used")
                if formats_used:
                    if isinstance(formats_used, (list, set)):
                        parts.append(row("Audio Formats", ", ".join(sorted(formats_used))))
                    else:
                        parts.append(row("Audio Formats", str(formats_used)))
            parts.append(table_end())

        # ── Test Metrics ──
        if test_metrics and isinstance(test_metrics, dict):
            parts.append(section("Test Metrics"))
            parts.append(table_start())
            parts.append(row("Accuracy", test_metrics.get("accuracy"), fmt="{:.4f}"))
            parts.append(row("Loss", test_metrics.get("loss"), fmt="{:.4f}"))
            # Binary-specific
            parts.append(row("ROC-AUC", test_metrics.get("roc_auc"), fmt="{:.4f}"))
            parts.append(row("PR-AUC", test_metrics.get("pr_auc"), fmt="{:.4f}"))
            # Per-class from classification report
            report = test_metrics.get("classification_report")
            if report and isinstance(report, dict):
                # Show per-class precision/recall/f1
                for cls_name in classes:
                    cls_metrics = report.get(cls_name)
                    if cls_metrics and isinstance(cls_metrics, dict):
                        p = cls_metrics.get("precision", 0)
                        r = cls_metrics.get("recall", 0)
                        f = cls_metrics.get("f1-score", 0)
                        parts.append(row(
                            f"  {cls_name}",
                            f"P={p:.3f}  R={r:.3f}  F1={f:.3f}"
                        ))
                # Macro averages
                macro = report.get("macro avg")
                if macro and isinstance(macro, dict):
                    p = macro.get("precision", 0)
                    r = macro.get("recall", 0)
                    f = macro.get("f1-score", 0)
                    parts.append(row("Macro Avg", f"P={p:.3f}  R={r:.3f}  F1={f:.3f}"))
            parts.append(table_end())

        # ── Training History (final epoch) ──
        if training_history and isinstance(training_history, dict):
            final_train_acc = training_history.get("final_train_accuracy")
            final_val_acc = training_history.get("final_val_accuracy")
            final_train_loss = training_history.get("final_train_loss")
            final_val_loss = training_history.get("final_val_loss")
            best_epoch = training_history.get("best_epoch")
            total_epochs = training_history.get("total_epochs_run")
            if any(v is not None for v in [final_train_acc, final_val_acc, best_epoch]):
                parts.append(section("Training History"))
                parts.append(table_start())
                parts.append(row("Epochs Run", total_epochs))
                parts.append(row("Best Epoch", best_epoch))
                parts.append(row("Final Train Acc", final_train_acc, fmt="{:.4f}"))
                parts.append(row("Final Val Acc", final_val_acc, fmt="{:.4f}"))
                parts.append(row("Final Train Loss", final_train_loss, fmt="{:.4f}"))
                parts.append(row("Final Val Loss", final_val_loss, fmt="{:.4f}"))
                parts.append(table_end())

        # ── Architecture-specific details ──
        arch_type = (model_type or "").lower()
        has_arch_detail = False
        arch_rows = []
        if "cnn" in arch_type:
            arch_rows.append(row("Conv Layers", config.get("num_conv_layers")))
            arch_rows.append(row("Base Channels", config.get("base_channels")))
            arch_rows.append(row("FC Hidden Size", config.get("fc_hidden_size")))
            arch_rows.append(row("SE Blocks", "Yes" if config.get("use_se_blocks") else "No"))
            has_arch_detail = True
        elif "resnet" in arch_type or "efficientnet" in arch_type:
            arch_rows.append(row("Pretrained", "Yes" if config.get("pretrained_backbone") else "No"))
            has_arch_detail = True
        elif "rnn" in arch_type or "lstm" in arch_type or "gru" in arch_type:
            arch_rows.append(row("RNN Type", config.get("rnn_type")))
            arch_rows.append(row("Hidden Size", config.get("rnn_hidden_size")))
            arch_rows.append(row("Layers", config.get("rnn_num_layers")))
            arch_rows.append(row("Bidirectional", "Yes" if config.get("rnn_bidirectional") else "No"))
            arch_rows.append(row("Attention", "Yes" if config.get("use_attention") else "No"))
            has_arch_detail = True
        elif "transformer" in arch_type:
            arch_rows.append(row("d_model", config.get("transformer_d_model")))
            arch_rows.append(row("Heads", config.get("transformer_nhead")))
            arch_rows.append(row("Layers", config.get("transformer_num_layers")))
            has_arch_detail = True

        if has_arch_detail and any(r for r in arch_rows):
            parts.append(section(f"Architecture Details ({model_type.upper()})"))
            parts.append(table_start())
            parts.extend(arch_rows)
            parts.append(table_end())

        return "\n".join(parts)

    def _update_execution_summary(self):
        """Update the execution summary display."""
        if not hasattr(self, "summary_label") or not self.block:
            return

        last_execution = (self.block.metadata or {}).get("last_execution")
        if not last_execution:
            self.summary_label.setText(
                "No execution data available.\nRun the block to see results here."
            )
            self.summary_label.setStyleSheet(
                f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt; padding: 8px;"
            )
            return

        lines = ["<b>Last Execution Summary</b>", ""]

        event_items_processed = last_execution.get("event_items_processed", 0)
        total_input = last_execution.get("total_events_input", 0)
        total_classified = last_execution.get("total_classified", 0)
        total_skipped = last_execution.get("total_skipped", 0)
        model_path = last_execution.get("model_path", "unknown")

        lines.append(f"Event Items Processed: <b>{event_items_processed}</b>")
        lines.append(f"Total Events Input: <b>{total_input}</b>")
        lines.append(f"Classified: <b>{total_classified}</b>")
        if total_skipped > 0:
            lines.append(f"Skipped: <b>{total_skipped}</b>")

        lines.append("")
        lines.append(f"Model: <b>{Path(model_path).name}</b>")

        # Classification distribution
        classification_counts = last_execution.get("classification_counts", {})
        if classification_counts:
            lines.append("")
            lines.append("<b>Classification Distribution:</b>")
            for cls_name, count in sorted(classification_counts.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {cls_name}: <b>{count}</b>")

        # Confidence statistics
        confidence_stats = last_execution.get("confidence_stats", {})
        if confidence_stats and any(v is not None for v in confidence_stats.values()):
            lines.append("")
            lines.append("<b>Confidence Statistics:</b>")
            if confidence_stats.get("min") is not None:
                lines.append(f"  Min: <b>{confidence_stats['min']:.3f}</b>")
            if confidence_stats.get("max") is not None:
                lines.append(f"  Max: <b>{confidence_stats['max']:.3f}</b>")
            if confidence_stats.get("avg") is not None:
                lines.append(f"  Avg: <b>{confidence_stats['avg']:.3f}</b>")

        self.summary_label.setText("\n".join(lines))
        self.summary_label.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 10pt; padding: 8px;"
        )

    # =========================================================================
    # Model Directory Scanning & Event Handlers
    # =========================================================================

    def _scan_models_dir(self) -> list:
        """
        Scan the models directory for .pth files.

        Returns a list of (display_name, full_path) tuples sorted by modification
        time (newest first). Skips the checkpoints subdirectory.
        """
        models_dir = get_models_dir()
        results = []

        if not models_dir.exists():
            return results

        for pth_file in models_dir.rglob("*.pth"):
            # Skip checkpoints directory (training intermediates, not final models)
            rel = pth_file.relative_to(models_dir)
            if rel.parts and rel.parts[0] == "checkpoints":
                continue

            # Display name: use parent folder name if the .pth is inside a subfolder,
            # otherwise just the filename
            if pth_file.parent != models_dir:
                display = f"{pth_file.parent.name} / {pth_file.name}"
            else:
                display = pth_file.name

            results.append((display, str(pth_file)))

        # Sort by modification time, newest first
        results.sort(key=lambda item: os.path.getmtime(item[1]), reverse=True)
        return results

    def _populate_model_combo(self, selected_path: str = None):
        """
        Populate the model combo box with available models.

        If selected_path matches one of the discovered models, that entry is
        selected.  Otherwise the first placeholder item is shown.
        """
        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        models = self._scan_models_dir()

        self.model_combo.addItem("-- Select a model --", "")

        selected_idx = 0
        for idx, (display, full_path) in enumerate(models, start=1):
            self.model_combo.addItem(display, full_path)
            if selected_path and os.path.normpath(full_path) == os.path.normpath(selected_path):
                selected_idx = idx

        if not models:
            models_dir = get_models_dir()
            self.model_combo.setItemText(0, f"No models found in {models_dir}")

        self.model_combo.setCurrentIndex(selected_idx)
        self.model_combo.blockSignals(False)

    def _get_selected_model_path(self) -> str:
        """Return the full path stored in the currently selected combo item, or empty string."""
        if not hasattr(self, "model_combo"):
            return ""
        return self.model_combo.currentData() or ""

    def _refresh_model_list(self):
        """Rescan models directory and repopulate the dropdown."""
        current_path = self._get_selected_model_path()
        self._populate_model_combo(current_path or self._settings_manager.model_path)
        self._update_model_source_info()
        self._update_model_path_validation()
        count = self.model_combo.count() - 1  # subtract placeholder
        self.set_status_message(f"Found {count} model(s)", error=False)

    def _on_model_combo_changed(self, index: int):
        """Handle model selection from dropdown."""
        path = self.model_combo.currentData()
        try:
            self._settings_manager.model_path = path if path else None
            self._update_model_path_validation()
            self._update_model_source_info()
            name = Path(path).name if path else "(none)"
            self.set_status_message(f"Model selected: {name}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_sample_rate_changed(self, value: int):
        """Handle sample rate change."""
        try:
            self._settings_manager.sample_rate = None if value == 0 else value
            label = "Auto (from model)" if value == 0 else f"{value} Hz"
            self.set_status_message(f"Sample rate set to {label}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_batch_size_changed(self, value: int):
        """Handle batch size change."""
        try:
            self._settings_manager.batch_size = None if value == 0 else value
            label = "Auto" if value == 0 else str(value)
            self.set_status_message(f"Batch size set to {label}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_device_changed(self, text: str):
        """Handle device change."""
        try:
            self._settings_manager.device = text
            self.set_status_message(f"Device set to {text}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_confidence_threshold_changed(self, value: float):
        """Handle confidence threshold change."""
        try:
            # 0.0 means auto (use model default)
            self._settings_manager.confidence_threshold = None if value == 0.0 else value
            label = "Auto (from model)" if value == 0.0 else f"{value:.2f}"
            self.set_status_message(f"Confidence threshold set to {label}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_multiclass_multi_label_changed(self, state):
        """Handle multiclass multi-label checkbox change."""
        try:
            enabled = state == Qt.CheckState.Checked.value
            self._settings_manager.multiclass_multi_label = enabled
            self.set_status_message(
                f"Multiple classes per event: {'on' if enabled else 'off'}", error=False
            )
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_create_other_layer_changed(self, state):
        """Handle create other layer checkbox change."""
        try:
            enabled = state == Qt.CheckState.Checked.value
            self._settings_manager.create_other_layer = enabled
            self.set_status_message(
                f"Create 'other' layer: {'on' if enabled else 'off'}", error=False
            )
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_multiclass_threshold_changed(self, value: float):
        """Handle multiclass confidence threshold change."""
        try:
            self._settings_manager.multiclass_confidence_threshold = value
            self.set_status_message(f"Multi-label threshold set to {value:.2f}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()

    def _on_setting_changed(self, setting_name: str):
        """React to settings changes from this panel's settings manager."""
        relevant = [
            "model_path",
            "sample_rate",
            "batch_size",
            "device",
            "confidence_threshold",
            "multiclass_multi_label",
            "multiclass_confidence_threshold",
            "create_other_layer",
        ]
        if setting_name in relevant:
            self.refresh()

    def _on_settings_save_failed(self, keys: str, error: str) -> None:
        """Display loud, user-facing save errors from settings manager."""
        self.set_status_message(f"Save failed ({keys}): {error}", error=True)

    # =========================================================================
    # Model Store
    # =========================================================================

    def _get_download_service(self) -> ModelDownloadService:
        """Lazy-create the download service."""
        if not hasattr(self, "_download_service") or self._download_service is None:
            self._download_service = create_model_download_service()
        return self._download_service

    def _on_check_models(self):
        """Fetch the remote model manifest in a background thread."""
        self.check_models_btn.setEnabled(False)
        self.store_status_label.setText("Checking...")
        self.store_status_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt;"
        )

        service = self._get_download_service()
        self._manifest_worker = _ManifestWorker(service, parent=self)
        self._manifest_worker.finished.connect(self._on_manifest_loaded)
        self._manifest_worker.error.connect(self._on_manifest_error)
        self._manifest_worker.start()

    def _on_manifest_loaded(self, models: list):
        """Populate the store list with available models."""
        self.check_models_btn.setEnabled(True)

        service = self._get_download_service()
        status = service.get_download_status(models)

        installed = sum(1 for v in status.values() if v)
        available = len(models) - installed
        parts = []
        if installed:
            parts.append(f"{installed} installed")
        if available:
            parts.append(f"{available} available")
        self.store_status_label.setText(", ".join(parts) if parts else "No models found")
        self.store_status_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt;"
        )

        self._store_models = models
        self._rebuild_store_list(models, status)

    def _on_manifest_error(self, error_msg: str):
        """Handle manifest fetch failure."""
        self.check_models_btn.setEnabled(True)
        self.store_status_label.setText(error_msg)
        self.store_status_label.setStyleSheet(
            f"color: {Colors.STATUS_ERROR.name()}; font-size: 9pt;"
        )

    def _rebuild_store_list(self, models: list, status: dict):
        """Clear and rebuild the model store list widgets."""
        while self.store_list_layout.count():
            child = self.store_list_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for model in models:
            row = self._create_store_row(model, installed=status.get(model.id, False))
            self.store_list_layout.addWidget(row)

    def _create_store_row(self, model: RemoteModel, installed: bool) -> QWidget:
        """Build a single model row for the store list."""
        row = QWidget()
        row.setStyleSheet(
            f"QWidget {{ background: {Colors.BG_MEDIUM.name()};"
            f" border: 1px solid {Colors.BORDER.name()};"
            f" border-radius: 4px; }}"
        )
        h = QHBoxLayout(row)
        h.setContentsMargins(8, 6, 8, 6)
        h.setSpacing(Spacing.SM)

        info = QVBoxLayout()
        info.setSpacing(2)

        name_label = QLabel(f"<b>{model.name}</b>  <small>v{model.version}</small>")
        name_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; border: none;")
        info.addWidget(name_label)

        meta_parts = []
        if model.architecture:
            meta_parts.append(model.architecture.upper())
        if model.classification_mode:
            meta_parts.append(model.classification_mode)
        if model.classes:
            meta_parts.append(f"{len(model.classes)} classes")
        if model.size_bytes:
            meta_parts.append(f"{model.size_mb:.1f} MB")
        meta_label = QLabel(" | ".join(meta_parts))
        meta_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; border: none;"
        )
        info.addWidget(meta_label)

        if model.description:
            desc_label = QLabel(model.description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet(
                f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt; border: none;"
            )
            info.addWidget(desc_label)

        h.addLayout(info, 1)

        action_layout = QVBoxLayout()
        action_layout.setSpacing(2)

        if installed:
            status_label = QLabel("Installed")
            status_label.setStyleSheet(
                f"color: {Colors.STATUS_SUCCESS.name()}; font-weight: bold;"
                f" font-size: 9pt; border: none;"
            )
            action_layout.addWidget(status_label, alignment=Qt.AlignmentFlag.AlignRight)
        else:
            dl_btn = QPushButton("Download")
            dl_btn.setFixedWidth(90)
            dl_btn.clicked.connect(lambda checked, m=model, b=dl_btn: self._on_download_model(m, b))
            action_layout.addWidget(dl_btn, alignment=Qt.AlignmentFlag.AlignRight)

        progress_bar = QProgressBar()
        progress_bar.setFixedHeight(14)
        progress_bar.setTextVisible(True)
        progress_bar.setFormat("%p%")
        progress_bar.setVisible(False)
        progress_bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid {Colors.BORDER.name()};"
            f" border-radius: 3px; background: {Colors.BG_DARK.name()};"
            f" text-align: center; color: {Colors.TEXT_PRIMARY.name()};"
            f" font-size: 8pt; }}"
            f" QProgressBar::chunk {{ background: {Colors.ACCENT_BLUE.name()};"
            f" border-radius: 2px; }}"
        )
        progress_bar.setObjectName(f"progress_{model.id}")
        action_layout.addWidget(progress_bar)

        h.addLayout(action_layout)
        return row

    def _on_download_model(self, model: RemoteModel, button: QPushButton):
        """Start downloading a model in the background."""
        button.setEnabled(False)
        button.setText("Downloading...")

        progress_bar = self.store_list_container.findChild(QProgressBar, f"progress_{model.id}")
        if progress_bar:
            progress_bar.setValue(0)
            progress_bar.setVisible(True)

        service = self._get_download_service()
        worker = _DownloadWorker(service, model, parent=self)

        def on_progress(downloaded: int, total: int):
            if progress_bar and total > 0:
                pct = int(downloaded * 100 / total)
                progress_bar.setValue(pct)

        def on_finished(path: str):
            if progress_bar:
                progress_bar.setValue(100)
                QTimer.singleShot(1000, lambda: progress_bar.setVisible(False))
            button.setText("Installed")
            button.setStyleSheet(f"color: {Colors.STATUS_SUCCESS.name()};")
            self.set_status_message(f"Downloaded {model.name}", error=False)
            self._refresh_model_list()

        def on_error(msg: str):
            if progress_bar:
                progress_bar.setVisible(False)
            button.setEnabled(True)
            button.setText("Retry")
            self.set_status_message(f"Download failed: {msg}", error=True)

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)

        if not hasattr(self, "_download_workers"):
            self._download_workers = []
        self._download_workers.append(worker)
        worker.start()

    # =========================================================================
    # External Updates
    # =========================================================================

    def refresh_for_undo(self):
        """Refresh panel after undo/redo operation."""
        if hasattr(self, "_settings_manager") and self._settings_manager:
            self._settings_manager.reload_from_storage()
        self.refresh()

    def _on_block_updated(self, event):
        """Handle block update event -- reload settings and refresh UI."""
        updated_block_id = event.data.get("id")
        if updated_block_id == self.block_id:
            if self._is_saving:
                Log.debug(
                    f"PyTorchAudioClassifyPanel: Skipping refresh during save for {self.block_id}"
                )
                return

            Log.debug(
                f"PyTorchAudioClassifyPanel: Block {self.block_id} updated externally, refreshing UI"
            )

            # Reload block data from database
            result = self.facade.describe_block(self.block_id)
            if result.success:
                self.block = result.data
                self._update_header()
            else:
                Log.warning(
                    f"PyTorchAudioClassifyPanel: Failed to reload block {self.block_id}"
                )
                return

            # Reload settings from database
            if hasattr(self, "_settings_manager") and self._settings_manager:
                self._settings_manager.reload_from_storage()
            else:
                return

            QTimer.singleShot(0, self.refresh)
