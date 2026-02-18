"""
PyTorchAudioTrainer block panel.

Provides comprehensive UI for configuring advanced PyTorch audio training settings.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QGroupBox, QLineEdit, QPushButton, QFileDialog, QSpinBox,
    QComboBox, QCheckBox, QDoubleSpinBox, QTextEdit, QTabWidget,
    QScrollArea, QFrame, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt
from pathlib import Path
import os
from datetime import datetime

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing
from src.application.api.application_facade import ApplicationFacade
from src.utils.datasets import get_managed_datasets_dir, resolve_dataset_path
from src.utils.message import Log
from src.utils.paths import get_models_dir


class DataDirEdit(QLineEdit):
    """Custom QLineEdit for data directory selection"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Select directory containing class folders...")

    def mouseDoubleClickEvent(self, event):
        """Open directory selection dialog on double-click"""
        self._browse_directory()

    def _browse_directory(self):
        """Open directory browser dialog"""
        current_path = (
            resolve_dataset_path(self.text())
            or str(get_managed_datasets_dir())
        )
        directory = QFileDialog.getExistingDirectory(
            self, "Select Training Data Directory",
            current_path,
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.setText(directory)
            self.textChanged.emit(directory)


@register_block_panel("PyTorchAudioTrainer")
class PyTorchAudioTrainerPanel(BlockPanelBase):
    """
    Panel for configuring PyTorchAudioTrainer block settings.

    Provides comprehensive controls for advanced audio classification training:
    - Training data directory selection
    - Multiple model architectures
    - Training hyperparameters and optimization
    - Cross-validation and data augmentation
    - Hyperparameter optimization
    """

    def __init__(self, block_id: str, facade: ApplicationFacade, parent=None):
        super().__init__(block_id, facade, parent)
        self.setWindowTitle("PyTorch Audio Trainer Settings")

        # Initialize empty settings for UI setup
        self.settings = {}
        self._class_counts = {}
        self._is_refreshing = False  # Guard to prevent auto-save during UI refresh

        # Load current settings after parent initialization
        self._load_current_settings()

        # Refresh UI with loaded settings
        self._refresh_ui_with_settings()
        self._update_status_from_last_training()

        # Connect signals (AFTER initial refresh to avoid saving defaults)
        self._connect_signals()

    def _refresh_ui_with_settings(self):
        """Refresh UI controls with loaded settings"""
        self._is_refreshing = True
        try:
            self._refresh_ui_with_settings_inner()
        finally:
            self._is_refreshing = False

    def _generate_unique_model_name(self) -> str:
        """Generate a unique model name from mode, architecture, and timestamp. Used when field is empty."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = self.settings.get("classification_mode", "multiclass")
        arch = (self.settings.get("model_type") or "cnn").lower().replace(" ", "_")
        if mode == "binary":
            names = self.settings.get("positive_classes") or []
            if not names and self.settings.get("target_class"):
                names = [self.settings.get("target_class")]
            tag = "_".join(names)[:60] if names else "positive"
            base = f"binary_{tag}_{arch}"
        elif mode == "positive_vs_other":
            names = self.settings.get("positive_classes") or []
            if not names and self.settings.get("target_class"):
                names = [self.settings.get("target_class")]
            tag = "_".join(names)[:60] if names else "positive"
            base = f"positive_vs_other_{tag}_{arch}"
        else:
            base = f"multiclass_{arch}"
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)[:80]
        if not safe:
            safe = "model"
        return f"{safe}_{timestamp}"

    def _refresh_ui_with_settings_inner(self):
        """Inner refresh logic, called while _is_refreshing is True."""
        # Data tab
        self.data_dir_edit.setText(resolve_dataset_path(self.settings.get("data_dir", "")) or "")
        model_name = (self.settings.get("model_name") or "").strip()
        if not model_name:
            model_name = self._generate_unique_model_name()
            self.settings["model_name"] = model_name
            result = self.facade.update_block_metadata(self.block_id, {"model_name": model_name})
            if result and result.success:
                Log.debug(f"PyTorchAudioTrainerPanel: auto-filled model name '{model_name}'")
        self.model_name_edit.setText(model_name)
        self.sample_rate_spin.setValue(self.settings.get("sample_rate", 22050))
        self.max_length_spin.setValue(self.settings.get("max_length", 22050))
        self.fmax_spin.setValue(self.settings.get("fmax", 8000))
        self.n_mels_spin.setValue(self.settings.get("n_mels", 128))
        self.hop_length_spin.setValue(self.settings.get("hop_length", 512))
        self.n_fft_spin.setValue(self.settings.get("n_fft", 2048))
        self.normalize_per_dataset_check.setChecked(self.settings.get("normalize_per_dataset", True))
        self.cv_check.setChecked(self.settings.get("use_cross_validation", False))
        self.cv_folds_spin.setValue(self.settings.get("cv_folds", 5))

        # Model tab
        self.model_type_combo.setCurrentText(self.settings.get("model_type", "cnn"))
        self.num_conv_layers_spin.setValue(self.settings.get("num_conv_layers", 4))
        self.base_channels_spin.setValue(self.settings.get("base_channels", 32))
        self.use_se_blocks_check.setChecked(self.settings.get("use_se_blocks", False))
        self.rnn_type_combo.setCurrentText(self.settings.get("rnn_type", "lstm"))
        self.rnn_hidden_size_spin.setValue(self.settings.get("rnn_hidden_size", 256))
        self.rnn_num_layers_spin.setValue(self.settings.get("rnn_num_layers", 2))
        self.rnn_bidirectional_check.setChecked(self.settings.get("rnn_bidirectional", True))
        self.use_attention_check.setChecked(self.settings.get("use_attention", False))
        self.transformer_d_model_spin.setValue(self.settings.get("transformer_d_model", 256))
        self.transformer_nhead_spin.setValue(self.settings.get("transformer_nhead", 8))
        self.transformer_num_layers_spin.setValue(self.settings.get("transformer_num_layers", 4))
        self.wav2vec2_model_edit.setText(self.settings.get("wav2vec2_model", "facebook/wav2vec2-base"))
        self.freeze_wav2vec2_check.setChecked(self.settings.get("freeze_wav2vec2", True))

        # Training tab
        self.epochs_spin.setValue(self.settings.get("epochs", 100))
        self.batch_size_spin.setValue(self.settings.get("batch_size", 32))
        self.num_workers_spin.setValue(self.settings.get("num_workers", 0))
        self.optimizer_combo.setCurrentText(self.settings.get("optimizer", "adam"))
        self.learning_rate_spin.setValue(self.settings.get("learning_rate", 0.001))
        self.momentum_spin.setValue(self.settings.get("momentum", 0.9))
        self.weight_decay_spin.setValue(self.settings.get("weight_decay", 1e-4))
        self.dropout_rate_spin.setValue(self.settings.get("dropout_rate", 0.4))
        self.lr_scheduler_combo.setCurrentText(self.settings.get("lr_scheduler", "cosine_restarts"))
        self.lr_step_size_spin.setValue(self.settings.get("lr_step_size", 30))
        self.lr_gamma_spin.setValue(self.settings.get("lr_gamma", 0.1))
        self.early_stopping_check.setChecked(self.settings.get("use_early_stopping", True))
        self.early_stopping_patience_spin.setValue(self.settings.get("early_stopping_patience", 15))
        current_device = self.settings.get("device", "auto")
        self.device_combo.setCurrentText(current_device)
        # Save to directory (legacy: output_model_path may be a file path; show parent as dir)
        out = self.settings.get("output_model_path", "") or ""
        if out and out.endswith(".pth"):
            out = str(Path(out).parent)
        self.save_to_dir_edit.setText(out)

        # Augmentation tab
        self.augmentation_check.setChecked(self.settings.get("use_augmentation", False))
        self.pitch_shift_spin.setValue(self.settings.get("pitch_shift_range", 2.0))
        self.time_stretch_spin.setValue(self.settings.get("time_stretch_range", 0.2))
        self.noise_factor_spin.setValue(self.settings.get("noise_factor", 0.01))
        self.volume_factor_spin.setValue(self.settings.get("volume_factor", 0.1))
        self.time_shift_spin.setValue(self.settings.get("time_shift_max", 0.1))
        self.frequency_mask_spin.setValue(self.settings.get("frequency_mask", 0))
        self.time_mask_spin.setValue(self.settings.get("time_mask", 0))
        self.polarity_inversion_spin.setValue(self.settings.get("polarity_inversion_prob", 0.0))
        self.use_random_eq_check.setChecked(self.settings.get("use_random_eq", False))
        self.use_mixup_check.setChecked(self.settings.get("use_mixup", False))
        self.mixup_alpha_spin.setValue(self.settings.get("mixup_alpha", 0.2))
        self.use_cutmix_check.setChecked(self.settings.get("use_cutmix", False))
        self.cutmix_alpha_spin.setValue(self.settings.get("cutmix_alpha", 1.0))

        # Advanced tab
        self.hyperopt_check.setChecked(self.settings.get("use_hyperopt", False))
        self.hyperopt_trials_spin.setValue(self.settings.get("hyperopt_trials", 50))
        self.hyperopt_timeout_spin.setValue(self.settings.get("hyperopt_timeout", 3600))
        self.ensemble_check.setChecked(self.settings.get("model_type") == "ensemble")
        self.meta_classifier_check.setChecked(self.settings.get("use_meta_classifier", False))
        
        # DOSE-inspired features
        self.use_onset_weighting_check.setChecked(self.settings.get("use_onset_weighting", False))
        self.onset_loss_weight_spin.setValue(self.settings.get("onset_loss_weight", 0.3))
        self.use_transient_emphasis_check.setChecked(self.settings.get("use_transient_emphasis", False))
        self.use_multi_scale_features_check.setChecked(self.settings.get("use_multi_scale_features", False))

        # Classification mode
        mode = self.settings.get("classification_mode", "multiclass")
        if mode == "binary":
            self.mode_combo.setCurrentIndex(1)
        elif mode == "positive_vs_other":
            self.mode_combo.setCurrentIndex(2)
        else:
            self.mode_combo.setCurrentIndex(0)
        self.negative_ratio_spin.setValue(self.settings.get("negative_ratio", 1.0))
        self.hard_negative_dir_edit.setText(self.settings.get("hard_negative_dir", "") or "")
        # Positive class filter
        pf_type = self.settings.get("positive_filter_type")
        idx = self.positive_filter_type_combo.findData(pf_type)
        if idx >= 0:
            self.positive_filter_type_combo.setCurrentIndex(idx)
        self.positive_filter_cutoff_spin.setValue(self.settings.get("positive_filter_cutoff_hz", 1000))
        self.positive_filter_cutoff_high_spin.setValue(
            self.settings.get("positive_filter_cutoff_high_hz", 4000)
        )
        self.positive_filter_order_spin.setValue(self.settings.get("positive_filter_order", 4))
        self._on_positive_filter_type_changed()
        self.auto_tune_check.setChecked(self.settings.get("auto_tune_threshold", True))
        threshold_metric = self.settings.get("threshold_metric", "f1")
        idx = self.threshold_metric_combo.findText(threshold_metric)
        if idx >= 0:
            self.threshold_metric_combo.setCurrentIndex(idx)
        # Positive classes (binary): checked state is restored when list is populated in _on_data_dir_changed
        self._saved_positive_classes = self.settings.get("positive_classes") or []
        if not self._saved_positive_classes and self.settings.get("target_class"):
            self._saved_positive_classes = [self.settings.get("target_class")]

        # Dataset balancing
        balance_strategy = self.settings.get("balance_strategy", "none")
        for i in range(self.balance_strategy_combo.count()):
            if self.balance_strategy_combo.itemData(i) == balance_strategy:
                self.balance_strategy_combo.setCurrentIndex(i)
                break
        balance_target = self.settings.get("balance_target_count")
        if balance_target:
            self.balance_target_spin.setValue(int(balance_target))
        self._on_balance_strategy_changed(self.balance_strategy_combo.currentIndex())

        # Update UI state based on settings
        self._on_model_type_changed(self.model_type_combo.currentText())
        self._on_optimizer_changed(self.optimizer_combo.currentText())
        self._on_scheduler_changed(self.lr_scheduler_combo.currentText())
        self._on_cv_toggled(self.cv_check.isChecked())
        self._on_augmentation_toggled(self.augmentation_check.isChecked())
        self._on_hyperopt_toggled(self.hyperopt_check.isChecked())
        self._on_mode_changed(self.mode_combo.currentIndex())
        self._apply_unsupported_feature_locks()

        self._update_model_output_path_display()

        # Explicitly populate the class list from the loaded data_dir.
        # Signals are not yet connected during init, so _on_data_dir_changed
        # won't fire from setText alone. This restores checked positive classes
        # via _saved_positive_classes (set above).
        data_dir = self.data_dir_edit.text().strip()
        if data_dir:
            self._on_data_dir_changed(data_dir)

    def create_content_widget(self) -> QWidget:
        """Create the comprehensive user interface with tabs"""
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(Spacing.MD)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)

        # Create tab widget for organizing options
        self.tab_widget = QTabWidget()

        # Data Tab
        self._setup_data_tab()

        # Model Tab
        self._setup_model_tab()

        # Training Tab
        self._setup_training_tab()

        # Augmentation Tab
        self._setup_augmentation_tab()

        # Advanced Tab
        self._setup_advanced_tab()

        layout.addWidget(self.tab_widget)

        # Status Section
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(120)
        self.status_text.setReadOnly(True)
        self.status_text.setPlainText("Ready to configure advanced training settings.")
        status_layout.addWidget(self.status_text)

        layout.addWidget(status_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.validate_btn = QPushButton("Validate Configuration")
        self.validate_btn.clicked.connect(self._validate_configuration)
        button_layout.addWidget(self.validate_btn)

        button_layout.addStretch()

        layout.addLayout(button_layout)
        layout.addStretch()

        return content_widget

    def _setup_data_tab(self):
        """Setup the data configuration tab"""
        data_widget = QWidget()
        data_layout = QVBoxLayout(data_widget)

        # Data Configuration Section
        data_group = QGroupBox("Training Data")
        group_layout = QFormLayout(data_group)

        # Data directory selection
        self.data_dir_edit = DataDirEdit()
        self.data_dir_edit.setToolTip(
            "Directory containing subfolders for each audio class.\n"
            "Each subfolder name becomes a class label.\n"
            "Example: data/kick/, data/snare/, data/hihat/\n\n"
            "Supports drag-and-drop and browse button."
        )

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.data_dir_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.data_dir_edit._browse_directory)
        dir_layout.addWidget(browse_btn)

        group_layout.addRow("Data Directory:", dir_layout)

        # Data directory info
        self.data_info_label = QLabel("Select a directory containing subfolders for each audio class")
        self.data_info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        group_layout.addRow("", self.data_info_label)

        # Save to: directory where the model folder will be created (subdir + .pth + MODEL_SUMMARY.txt)
        self.save_to_dir_edit = QLineEdit()
        self.save_to_dir_edit.setPlaceholderText(str(get_models_dir()))
        self.save_to_dir_edit.setToolTip(
            "Directory where the model will be saved.\n\n"
            "A subfolder (model name + timestamp) is created here containing the .pth file and MODEL_SUMMARY.txt.\n"
            "Leave empty to use the default EchoZero models directory."
        )
        save_to_layout = QHBoxLayout()
        save_to_layout.addWidget(self.save_to_dir_edit)
        save_to_browse_btn = QPushButton("Browse...")
        save_to_browse_btn.clicked.connect(self._browse_save_to_dir)
        save_to_layout.addWidget(save_to_browse_btn)
        group_layout.addRow("Save to:", save_to_layout)

        # Model name: used as the base for the output folder and filename
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("Auto-generated if empty")
        self.model_name_edit.setToolTip(
            "Name for the trained model. Used as the folder and file base (e.g. supersnare_20250101_120000).\n\n"
            "Leave empty to auto-generate from mode and class (e.g. binary_Snare_cnn_...)."
        )
        group_layout.addRow("Model name:", self.model_name_edit)

        # --- Classification Mode ---
        mode_group = QGroupBox("Classification Mode")
        mode_layout = QFormLayout(mode_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Multiclass",
            "Binary (Select classes to classify)",
            "Positive vs Other (select positive classes; rest = other)",
        ])
        self.mode_combo.setToolTip(
            "Multiclass: Train one model to distinguish all classes.\n\n"
            "Binary: Select which classes to classify as 'positive'.\n"
            "All other classes are treated as 'negative'. Check the classes\n"
            "you want the model to detect; leave others unchecked.\n\n"
            "Positive vs Other: Select multiple positive classes. The model\n"
            "outputs one label per positive class plus 'other'. Unselected\n"
            "classes are grouped as 'other'."
        )
        mode_layout.addRow("Mode:", self.mode_combo)

        # Classes to classify (binary and positive_vs_other) – checkable list; checked = positive
        self.positive_classes_list = QListWidget()
        self.positive_classes_list.setToolTip(
            "Check the classes you want as positive.\n"
            "Binary: unchecked = negative. Positive vs Other: unchecked = 'other'.\n"
            "Set the data directory first to see available classes."
        )
        self.positive_classes_list.setMaximumHeight(120)
        self.positive_classes_label = QLabel("Classes to classify (check = positive):")
        self.positive_classes_label.setToolTip(
            "Checked = positive. Binary: others = negative. Positive vs Other: others = 'other'."
        )
        mode_layout.addRow(self.positive_classes_label, self.positive_classes_list)

        # Negative ratio (binary only)
        self.negative_ratio_spin = QDoubleSpinBox()
        self.negative_ratio_spin.setRange(0.1, 10.0)
        self.negative_ratio_spin.setSingleStep(0.1)
        self.negative_ratio_spin.setValue(1.0)
        self.negative_ratio_spin.setToolTip(
            "Ratio of negative to positive samples.\n\n"
            "1.0 = equal numbers of positive and negative samples.\n"
            "2.0 = twice as many negatives as positives.\n"
            "0.5 = half as many negatives as positives.\n\n"
            "If there are more negatives than the ratio allows,\n"
            "they will be under-sampled. If fewer, they will be\n"
            "over-sampled by duplication."
        )
        self.negative_ratio_label = QLabel("Negative Ratio:")
        mode_layout.addRow(self.negative_ratio_label, self.negative_ratio_spin)

        # Hard negative directory (binary only)
        self.hard_negative_dir_edit = DataDirEdit()
        self.hard_negative_dir_edit.setPlaceholderText("Optional: directory of hard negative samples...")
        self.hard_negative_dir_edit.setToolTip(
            "Optional directory containing hard negative samples.\n\n"
            "Hard negatives are samples that are similar to your target\n"
            "class but should NOT be classified as positive.\n"
            "For example, if training a 'kick' detector, hard negatives\n"
            "might be bass toms or low-frequency synth hits.\n\n"
            "These are added to the negative pool to improve accuracy."
        )
        self.hard_neg_widget = QWidget()
        hard_neg_layout = QHBoxLayout(self.hard_neg_widget)
        hard_neg_layout.setContentsMargins(0, 0, 0, 0)
        hard_neg_layout.addWidget(self.hard_negative_dir_edit)
        hard_neg_browse_btn = QPushButton("Browse...")
        hard_neg_browse_btn.clicked.connect(self.hard_negative_dir_edit._browse_directory)
        hard_neg_layout.addWidget(hard_neg_browse_btn)
        self.hard_negative_label = QLabel("Hard Negatives:")
        mode_layout.addRow(self.hard_negative_label, self.hard_neg_widget)

        # Auto-tune threshold (binary only)
        self.threshold_widget = QWidget()
        threshold_layout = QHBoxLayout(self.threshold_widget)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        self.auto_tune_check = QCheckBox("Auto-tune threshold")
        self.auto_tune_check.setChecked(True)
        self.auto_tune_check.setToolTip(
            "Automatically find the optimal decision threshold\n"
            "on the validation set after training.\n\n"
            "The default 0.5 threshold is almost never optimal.\n"
            "Auto-tuning typically improves F1 score by 2-5%."
        )
        threshold_layout.addWidget(self.auto_tune_check)

        self.threshold_metric_combo = QComboBox()
        self.threshold_metric_combo.addItems(["f1", "precision", "recall", "youden"])
        self.threshold_metric_combo.setToolTip(
            "Metric to optimize when tuning the threshold.\n\n"
            "F1: Balance of precision and recall (recommended).\n"
            "Precision: Minimize false positives.\n"
            "Recall: Minimize missed detections.\n"
            "Youden: Maximize sensitivity + specificity."
        )
        threshold_layout.addWidget(QLabel("Optimize:"))
        threshold_layout.addWidget(self.threshold_metric_combo)
        self.threshold_tune_label = QLabel("Threshold:")
        mode_layout.addRow(self.threshold_tune_label, self.threshold_widget)

        # Positive class filter (binary only): apply lowpass/highpass/bandpass to all positive samples
        self.positive_filter_label = QLabel("Positive class filter:")
        self.positive_filter_container = QWidget()
        pf_layout = QFormLayout(self.positive_filter_container)
        pf_layout.setContentsMargins(0, 0, 0, 0)

        self.positive_filter_type_combo = QComboBox()
        self.positive_filter_type_combo.addItem("None", None)
        self.positive_filter_type_combo.addItem("Lowpass", "lowpass")
        self.positive_filter_type_combo.addItem("Highpass", "highpass")
        self.positive_filter_type_combo.addItem("Band", "bandpass")
        self.positive_filter_type_combo.setToolTip(
            "Apply a filter to all positive-class samples only.\n\n"
            "Use to focus training on a frequency band (e.g. lowpass for kick, highpass for hi-hat).\n"
            "Only positive samples are filtered; negatives are unchanged."
        )
        pf_layout.addRow("Type:", self.positive_filter_type_combo)

        self.positive_filter_cutoff_spin = QDoubleSpinBox()
        self.positive_filter_cutoff_spin.setRange(20, 24000)
        self.positive_filter_cutoff_spin.setValue(1000)
        self.positive_filter_cutoff_spin.setSuffix(" Hz")
        self.positive_filter_cutoff_spin.setToolTip(
            "Cutoff frequency for lowpass/highpass; low edge for band filter."
        )
        pf_layout.addRow("Cutoff (low):", self.positive_filter_cutoff_spin)

        self.positive_filter_cutoff_high_spin = QDoubleSpinBox()
        self.positive_filter_cutoff_high_spin.setRange(20, 24000)
        self.positive_filter_cutoff_high_spin.setValue(4000)
        self.positive_filter_cutoff_high_spin.setSuffix(" Hz")
        self.positive_filter_cutoff_high_spin.setToolTip("High edge for band filter only.")
        self.positive_filter_cutoff_high_label = QLabel("Cutoff (high):")
        pf_layout.addRow(self.positive_filter_cutoff_high_label, self.positive_filter_cutoff_high_spin)

        self.positive_filter_order_spin = QSpinBox()
        self.positive_filter_order_spin.setRange(1, 8)
        self.positive_filter_order_spin.setValue(4)
        self.positive_filter_order_spin.setToolTip("Butterworth filter order (higher = steeper rolloff).")
        pf_layout.addRow("Order:", self.positive_filter_order_spin)

        mode_layout.addRow(self.positive_filter_label, self.positive_filter_container)

        group_layout.addRow(mode_group)

        # Audio settings
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 96000)
        self.sample_rate_spin.setToolTip(
            "Target sample rate for audio preprocessing in Hz.\n\n"
            "• 22050 Hz: Default; max usable frequency (fmax) is 11025 Hz (Nyquist).\n"
            "• 44100 Hz: Use for snare/hihat; fmax can be set up to 22050 Hz (captures crack 12-16 kHz).\n\n"
            "fmax is always clamped to half the sample rate. All audio is resampled to this rate."
        )
        group_layout.addRow("Sample Rate (Hz):", self.sample_rate_spin)

        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(1000, 200000)
        self.max_length_spin.setToolTip(
            "Maximum audio length in samples (after resampling).\n\n"
            "Longer audio will be truncated, shorter audio will be padded.\n"
            "Use at least 4096 so n_fft=2048 is valid (avoids frequency resolution loss).\n"
            "For 22050 Hz: 11025 = 0.5 s, 22050 = 1 s. For 44100 Hz: 22050 = 0.5 s."
        )
        group_layout.addRow("Max Length (samples):", self.max_length_spin)

        # Spectrogram / frequency resolution (mel spectrogram used for training)
        spec_group = QGroupBox("Spectrogram (frequency resolution)")
        spec_layout = QFormLayout(spec_group)
        self.fmax_spin = QSpinBox()
        self.fmax_spin.setRange(1000, 24000)
        self.fmax_spin.setSingleStep(500)
        self.fmax_spin.setValue(8000)
        self.fmax_spin.setToolTip(
            "Maximum frequency (Hz) for the mel spectrogram. Clamped to half the sample rate (Nyquist).\n\n"
            "At 22050 Hz sr: fmax max = 11025 Hz. At 44100 Hz sr: fmax max = 22050 Hz.\n"
            "Snare/hihat: set Sample Rate to 44100 and fmax to 12000-16000 for crack/wires. Kick: 8000 is fine."
        )
        spec_layout.addRow("Max frequency (Hz):", self.fmax_spin)
        self.n_mels_spin = QSpinBox()
        self.n_mels_spin.setRange(64, 256)
        self.n_mels_spin.setValue(128)
        self.n_mels_spin.setToolTip(
            "Number of mel frequency bands. More bands = finer frequency resolution.\n"
            "128 is a good default; increase for tasks that need fine frequency distinction."
        )
        spec_layout.addRow("Mel bands (n_mels):", self.n_mels_spin)
        self.hop_length_spin = QSpinBox()
        self.hop_length_spin.setRange(256, 1024)
        self.hop_length_spin.setSingleStep(128)
        self.hop_length_spin.setValue(512)
        self.hop_length_spin.setToolTip(
            "Frame step in samples. Smaller = finer time resolution, larger spectrograms.\n"
            "512 is a common default for 22050 Hz."
        )
        spec_layout.addRow("Hop length (samples):", self.hop_length_spin)
        self.n_fft_spin = QSpinBox()
        self.n_fft_spin.setRange(512, 4096)
        self.n_fft_spin.setSingleStep(512)
        self.n_fft_spin.setValue(2048)
        self.n_fft_spin.setToolTip(
            "FFT window size. Larger = better frequency resolution, worse time resolution.\n"
            "2048 is a typical default."
        )
        spec_layout.addRow("FFT size (n_fft):", self.n_fft_spin)
        self.normalize_per_dataset_check = QCheckBox("Use per-dataset normalization (save mean/std)")
        self.normalize_per_dataset_check.setChecked(True)
        self.normalize_per_dataset_check.setToolTip(
            "When enabled, computes dataset-wide mean/std and saves them in the model.\n\n"
            "When disabled, each sample is normalized independently at runtime and no\n"
            "dataset mean/std is written to the trained model."
        )
        spec_layout.addRow("Normalization:", self.normalize_per_dataset_check)
        group_layout.addRow(spec_group)

        # Cross-validation
        cv_layout = QHBoxLayout()
        self.cv_check = QCheckBox("Enable Cross-Validation")
        self.cv_check.setToolTip(
            "Enable K-fold cross-validation for robust model evaluation.\n\n"
            "Splits data into K folds and trains K models, using each fold\n"
            "as validation set once. Provides more reliable performance estimates."
        )
        cv_layout.addWidget(self.cv_check)

        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(2, 10)
        self.cv_folds_spin.setToolTip(
            "Number of folds for cross-validation (K).\n\n"
            "• 5 folds (default): Good balance of reliability and compute time\n"
            "• 10 folds: More reliable but slower (requires more data)\n\n"
            "Training time = epochs × K folds"
        )
        cv_layout.addWidget(QLabel("Folds:"))
        cv_layout.addWidget(self.cv_folds_spin)

        group_layout.addRow("Validation:", cv_layout)

        data_layout.addWidget(data_group)

        # --- Dataset Balancing ---
        balance_group = QGroupBox("Dataset Balancing")
        balance_layout = QFormLayout(balance_group)

        # Per-selection tooltips for the strategy combo (order must match addItem)
        self._balance_strategy_tooltips = [
            "No Balancing: Use all samples as-is. No reduction or duplication.",
            "Undersample to Smallest Class: Cap every class at the size of the smallest class. Simple but discards data from larger classes.",
            "Undersample to Median Class: Cap every class at the median class size. Better data utilization than smallest.",
            "Undersample to Target Count: Reduce each class to a fixed target count (set below). Use when you want a specific per-class size.",
            "Oversample to Largest Class: Duplicate minority-class samples until all classes match the largest. Duplicates are flagged for extra augmentation.",
            "Oversample to Target Count: Duplicate minority-class samples up to a target count (set below). Use when you want a specific per-class size.",
            "Smart Undersample (Cluster-Based): Use k-means on audio features to pick the most diverse samples per class. Preserves variation while reducing size.",
            "Hybrid (Smart Under + Augmented Over): Smart undersample large classes and oversample small ones with augmentation. Recommended for imbalanced data.",
        ]

        self.balance_strategy_combo = QComboBox()
        self.balance_strategy_combo.addItem("No Balancing", "none")
        self.balance_strategy_combo.addItem("Undersample to Smallest Class", "undersample_min")
        self.balance_strategy_combo.addItem("Undersample to Median Class", "undersample_median")
        self.balance_strategy_combo.addItem("Undersample to Target Count", "undersample_target")
        self.balance_strategy_combo.addItem("Oversample to Largest Class", "oversample_max")
        self.balance_strategy_combo.addItem("Oversample to Target Count", "oversample_target")
        self.balance_strategy_combo.addItem("Smart Undersample (Cluster-Based)", "smart_undersample")
        self.balance_strategy_combo.addItem("Hybrid (Smart Under + Augmented Over)", "hybrid")
        self.balance_strategy_combo.setToolTip(self._balance_strategy_tooltips[0])
        self.balance_strategy_combo.currentIndexChanged.connect(self._on_balance_strategy_changed)
        balance_strategy_label = QLabel("Strategy:")
        balance_strategy_label.setToolTip("Choose how to balance class sizes before training. Affects which samples are used, not the raw files.")
        balance_layout.addRow(balance_strategy_label, self.balance_strategy_combo)

        # Target count (for target-based strategies)
        self.balance_target_spin = QSpinBox()
        self.balance_target_spin.setRange(10, 100000)
        self.balance_target_spin.setValue(500)
        self.balance_target_spin.setSingleStep(50)
        self.balance_target_spin.setToolTip(
            "Target number of samples per class.\n\n"
            "For undersample_target: classes above this count are reduced.\n"
            "For oversample_target: classes below this count are duplicated.\n"
            "For hybrid: all classes are brought to this count."
        )
        self.balance_target_spin.valueChanged.connect(lambda _: self._update_balance_preview())
        self.balance_target_label = QLabel("Target Count:")
        self.balance_target_label.setToolTip("Per-class sample count used by undersample_target, oversample_target, and hybrid strategies.")
        self.balance_target_label.setVisible(False)
        self.balance_target_spin.setVisible(False)
        balance_layout.addRow(self.balance_target_label, self.balance_target_spin)

        # Balance preview label
        self.balance_preview_label = QLabel("")
        self.balance_preview_label.setWordWrap(True)
        self.balance_preview_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px; padding: 4px;"
        )
        balance_layout.addRow("", self.balance_preview_label)

        data_layout.addWidget(balance_group)

        data_layout.addStretch()

        self.tab_widget.addTab(data_widget, "Data")

    def _setup_model_tab(self):
        """Setup the model architecture tab"""
        model_widget = QWidget()
        model_layout = QVBoxLayout(model_widget)

        # Model Architecture Section
        arch_group = QGroupBox("Model Architecture")
        arch_layout = QFormLayout(arch_group)

        # Model type selection
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["cnn", "rnn", "transformer", "wav2vec2", "ensemble"])
        self.model_type_combo.currentTextChanged.connect(self._on_model_type_changed)
        self.model_type_combo.setToolTip(
            "Model architecture to use:\n\n"
            "• CNN: Convolutional Neural Network (default, best for spectrograms)\n"
            "• RNN: Recurrent Neural Network (LSTM/GRU, good for sequences)\n"
            "• Transformer: Attention-based model (state-of-the-art)\n"
            "• Wav2Vec2: Pre-trained audio model (requires transformers library)\n"
            "• Ensemble: Combine multiple models (experimental)"
        )
        arch_layout.addRow("Model Type:", self.model_type_combo)

        # CNN-specific options
        self.cnn_group = QGroupBox("CNN Configuration")
        cnn_layout = QFormLayout(self.cnn_group)

        self.num_conv_layers_spin = QSpinBox()
        self.num_conv_layers_spin.setRange(1, 8)
        self.num_conv_layers_spin.setToolTip(
            "Number of convolutional layers in the CNN.\n\n"
            "• 2-4 layers: Simpler model, faster training, may underfit\n"
            "• 4-6 layers: Recommended (default: 4), good balance\n"
            "• 6-8 layers: Deeper model, slower, may overfit small datasets\n\n"
            "Each layer doubles the number of channels."
        )
        cnn_layout.addRow("Conv Layers:", self.num_conv_layers_spin)

        self.base_channels_spin = QSpinBox()
        self.base_channels_spin.setRange(8, 512)
        self.base_channels_spin.setToolTip(
            "Number of channels in the first convolutional layer.\n\n"
            "Subsequent layers double this (32 → 64 → 128 → 256).\n"
            "• 16-32: Smaller model, faster (default: 32)\n"
            "• 64-128: Larger model, more capacity\n"
            "• 256+: Very large, requires more data and GPU memory"
        )
        cnn_layout.addRow("Base Channels:", self.base_channels_spin)

        self.use_se_blocks_check = QCheckBox("Enable Squeeze-and-Excitation blocks")
        self.use_se_blocks_check.setChecked(False)
        self.use_se_blocks_check.setToolTip(
            "Add Squeeze-and-Excitation (SE) channel attention to the CNN.\n\n"
            "SE blocks reweight channel responses by global context (squeeze)\n"
            "and a small MLP (excitation). Can improve accuracy with a small\n"
            "parameter increase. Reference: Hu et al., CVPR 2018."
        )
        cnn_layout.addRow("", self.use_se_blocks_check)

        arch_layout.addRow(self.cnn_group)

        # RNN-specific options
        self.rnn_group = QGroupBox("RNN Configuration")
        rnn_layout = QFormLayout(self.rnn_group)

        self.rnn_type_combo = QComboBox()
        self.rnn_type_combo.addItems(["lstm", "gru", "rnn"])
        self.rnn_type_combo.setToolTip(
            "Type of RNN cell to use:\n\n"
            "• LSTM: Long Short-Term Memory (default, best for long sequences)\n"
            "• GRU: Gated Recurrent Unit (faster, similar performance)\n"
            "• RNN: Basic RNN (fastest but limited memory)"
        )
        rnn_layout.addRow("RNN Type:", self.rnn_type_combo)

        self.rnn_hidden_size_spin = QSpinBox()
        self.rnn_hidden_size_spin.setRange(32, 1024)
        self.rnn_hidden_size_spin.setToolTip(
            "Size of hidden state vector in each RNN layer.\n\n"
            "• 128-256: Smaller model, faster (default: 256)\n"
            "• 512: Larger model, more capacity\n"
            "• 1024: Very large, requires more data and memory"
        )
        rnn_layout.addRow("Hidden Size:", self.rnn_hidden_size_spin)

        self.rnn_num_layers_spin = QSpinBox()
        self.rnn_num_layers_spin.setRange(1, 4)
        self.rnn_num_layers_spin.setToolTip(
            "Number of RNN layers (stacked).\n\n"
            "• 1-2 layers: Simpler model, faster (default: 2)\n"
            "• 3-4 layers: Deeper model, more capacity but slower"
        )
        rnn_layout.addRow("Layers:", self.rnn_num_layers_spin)

        self.rnn_bidirectional_check = QCheckBox("Bidirectional")
        self.rnn_bidirectional_check.setToolTip(
            "Process sequence in both forward and backward directions.\n\n"
            "Doubles the hidden size but captures context from both directions.\n"
            "Recommended for audio classification (default: enabled)."
        )
        rnn_layout.addRow(self.rnn_bidirectional_check)

        self.use_attention_check = QCheckBox("Use Attention")
        self.use_attention_check.setToolTip(
            "Add attention mechanism to focus on important time steps.\n\n"
            "Can improve performance on long sequences by learning\n"
            "which parts of the audio are most relevant for classification."
        )
        rnn_layout.addRow(self.use_attention_check)

        arch_layout.addRow(self.rnn_group)

        # Transformer-specific options
        self.transformer_group = QGroupBox("Transformer Configuration")
        transformer_layout = QFormLayout(self.transformer_group)

        self.transformer_d_model_spin = QSpinBox()
        self.transformer_d_model_spin.setRange(64, 1024)
        self.transformer_d_model_spin.setToolTip(
            "Dimension of the model (d_model).\n\n"
            "Must be divisible by number of attention heads.\n"
            "• 256 (default): Standard size, good balance\n"
            "• 512: Larger model, more capacity\n"
            "• 1024: Very large, requires more data and memory"
        )
        transformer_layout.addRow("Model Dimension:", self.transformer_d_model_spin)

        self.transformer_nhead_spin = QSpinBox()
        self.transformer_nhead_spin.setRange(1, 16)
        self.transformer_nhead_spin.setToolTip(
            "Number of attention heads (parallel attention mechanisms).\n\n"
            "d_model must be divisible by nhead.\n"
            "• 8 (default): Standard configuration\n"
            "• 16: More heads, can capture more patterns"
        )
        transformer_layout.addRow("Attention Heads:", self.transformer_nhead_spin)

        self.transformer_num_layers_spin = QSpinBox()
        self.transformer_num_layers_spin.setRange(1, 12)
        self.transformer_num_layers_spin.setToolTip(
            "Number of transformer encoder layers.\n\n"
            "• 4 (default): Standard depth\n"
            "• 6-8: Deeper model, more capacity\n"
            "• 12: Very deep, requires more data and compute"
        )
        transformer_layout.addRow("Layers:", self.transformer_num_layers_spin)

        arch_layout.addRow(self.transformer_group)

        # Wav2Vec2-specific options
        self.wav2vec2_group = QGroupBox("Wav2Vec2 Configuration")
        wav2vec2_layout = QFormLayout(self.wav2vec2_group)

        self.wav2vec2_model_edit = QLineEdit()
        self.wav2vec2_model_edit.setToolTip(
            "Hugging Face model identifier for Wav2Vec2.\n\n"
            "Examples:\n"
            "• facebook/wav2vec2-base\n"
            "• facebook/wav2vec2-large\n\n"
            "Requires transformers library. Downloads automatically."
        )
        wav2vec2_layout.addRow("Model Name:", self.wav2vec2_model_edit)

        self.freeze_wav2vec2_check = QCheckBox("Freeze Pre-trained Weights")
        self.freeze_wav2vec2_check.setToolTip(
            "Keep pre-trained Wav2Vec2 weights frozen (default: enabled).\n\n"
            "Only train the classification head on top.\n"
            "Faster training, good for small datasets.\n"
            "Disable to fine-tune the entire model (requires more data)."
        )
        wav2vec2_layout.addRow(self.freeze_wav2vec2_check)

        arch_layout.addRow(self.wav2vec2_group)

        # Update visibility based on model type
        self._on_model_type_changed(self.model_type_combo.currentText())

        model_layout.addWidget(arch_group)
        model_layout.addStretch()

        self.tab_widget.addTab(model_widget, "Model")

    def _setup_training_tab(self):
        """Setup the training configuration tab"""
        training_widget = QWidget()
        training_layout = QVBoxLayout(training_widget)

        # Training Parameters Section
        train_group = QGroupBox("Training Parameters")
        train_form = QFormLayout(train_group)

        # Basic training settings
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setToolTip(
            "Number of training epochs (complete passes through the dataset).\n\n"
            "• 50-100: Quick experiments\n"
            "• 100-200: Standard training (default: 100)\n"
            "• 200+: Long training, use with early stopping\n\n"
            "Early stopping can stop training before max epochs."
        )
        train_form.addRow("Epochs:", self.epochs_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.setToolTip(
            "Number of samples per training batch.\n\n"
            "• 16-32: Smaller batches, more stable gradients (default: 32)\n"
            "• 64-128: Larger batches, faster training (needs more GPU memory)\n"
            "• 256+: Very large batches, may require gradient accumulation\n\n"
            "Larger batches need more GPU memory but train faster."
        )
        train_form.addRow("Batch Size:", self.batch_size_spin)

        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 8)
        self.num_workers_spin.setToolTip(
            "DataLoader worker processes (0 = load data in training thread).\n\n"
            "• 0: Safest; data loading runs in same thread as training (can feel slow)\n"
            "• 2-4: Prefetch batches in background; usually much faster training\n"
            "• 6-8: More workers; try if GPU/CPU is still idle between batches\n\n"
            "Note: On macOS, when running from a background UI thread, worker processes\n"
            "may be auto-disabled to avoid training hangs. If training stalls with\n"
            "augmentation enabled, set this to 0 and retry."
        )
        train_form.addRow("Data Workers:", self.num_workers_spin)

        # Optimizer selection
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["adam", "adamw", "sgd"])
        self.optimizer_combo.currentTextChanged.connect(self._on_optimizer_changed)
        self.optimizer_combo.setToolTip(
            "Optimization algorithm for training:\n\n"
            "• Adam: Adaptive learning rate (default, good for most cases)\n"
            "• AdamW: Adam with weight decay fix (often better than Adam)\n"
            "• SGD: Stochastic Gradient Descent (requires tuning learning rate)"
        )
        train_form.addRow("Optimizer:", self.optimizer_combo)

        # Learning rate
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(1e-7, 1.0)
        self.learning_rate_spin.setDecimals(7)
        self.learning_rate_spin.setToolTip(
            "Initial learning rate for optimizer.\n\n"
            "• 1e-4 to 1e-3: Common range for Adam/AdamW (default: 0.001)\n"
            "• 1e-3 to 1e-2: For SGD (with momentum)\n"
            "• 1e-5 to 1e-4: For fine-tuning pre-trained models\n\n"
            "Learning rate scheduler can adjust this during training."
        )
        train_form.addRow("Learning Rate:", self.learning_rate_spin)

        # Momentum (for SGD)
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_label = QLabel("Momentum:")
        self.momentum_spin.setToolTip(
            "Momentum coefficient for SGD optimizer (only used with SGD).\n\n"
            "• 0.9 (default): Standard momentum, smooths gradient updates\n"
            "• 0.95-0.99: Higher momentum, more stable but slower convergence\n\n"
            "Only applicable when optimizer is set to SGD."
        )
        train_form.addRow(self.momentum_label, self.momentum_spin)

        # Weight decay
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0, 1e-1)
        self.weight_decay_spin.setDecimals(7)
        self.weight_decay_spin.setToolTip(
            "L2 regularization strength (weight decay).\n\n"
            "Helps prevent overfitting by penalizing large weights.\n"
            "• 1e-5 to 1e-4: Light regularization (default: 1e-4)\n"
            "• 1e-4 to 1e-3: Medium regularization\n"
            "• 1e-3 to 1e-2: Strong regularization\n\n"
            "Higher values = stronger regularization = simpler model."
        )
        train_form.addRow("Weight Decay:", self.weight_decay_spin)

        # Dropout
        self.dropout_rate_spin = QDoubleSpinBox()
        self.dropout_rate_spin.setRange(0.0, 0.9)
        self.dropout_rate_spin.setToolTip(
            "Dropout probability (fraction of neurons randomly set to zero).\n\n"
            "Helps prevent overfitting by adding randomness during training.\n"
            "• 0.2-0.4: Light to medium dropout (default: 0.4)\n"
            "• 0.5-0.7: High dropout, use with overfitting\n\n"
            "Applied before final classification layer."
        )
        train_form.addRow("Dropout Rate:", self.dropout_rate_spin)

        training_layout.addWidget(train_group)

        # Learning Rate Scheduler Section
        scheduler_group = QGroupBox("Learning Rate Scheduler")
        scheduler_form = QFormLayout(scheduler_group)

        self.lr_scheduler_combo = QComboBox()
        self.lr_scheduler_combo.addItems(["none", "step", "cosine", "cosine_restarts", "plateau"])
        self.lr_scheduler_combo.currentTextChanged.connect(self._on_scheduler_changed)
        self.lr_scheduler_combo.setToolTip(
            "Learning rate scheduling strategy:\n\n"
            "• None: Keep learning rate constant\n"
            "• Step: Reduce LR by gamma every N epochs\n"
            "• Cosine: Cosine annealing (smooth decrease)\n"
            "• Cosine Restarts: Cosine annealing with warm restarts\n"
            "• Plateau: Reduce LR when validation loss plateaus (recommended)"
        )
        scheduler_form.addRow("Scheduler:", self.lr_scheduler_combo)

        # Step scheduler options
        self.lr_step_size_spin = QSpinBox()
        self.lr_step_size_spin.setRange(1, 1000)
        self.lr_step_label = QLabel("Step Size:")
        self.lr_step_size_spin.setToolTip(
            "Number of epochs between learning rate reductions (for Step scheduler).\n\n"
            "Learning rate is multiplied by gamma every step_size epochs.\n"
            "Default: 30 epochs"
        )
        scheduler_form.addRow(self.lr_step_label, self.lr_step_size_spin)

        # Gamma for step/cosine
        self.lr_gamma_spin = QDoubleSpinBox()
        self.lr_gamma_spin.setRange(0.1, 1.0)
        self.lr_gamma_label = QLabel("Gamma:")
        self.lr_gamma_spin.setToolTip(
            "Multiplier for learning rate reduction.\n\n"
            "Used with Step and Cosine schedulers.\n"
            "• 0.1 (default): Reduce LR to 10% of current value\n"
            "• 0.5: Reduce LR to 50% of current value\n\n"
            "Lower values = more aggressive reduction."
        )
        scheduler_form.addRow(self.lr_gamma_label, self.lr_gamma_spin)

        # Early stopping
        self.early_stopping_check = QCheckBox("Enable Early Stopping")
        self.early_stopping_check.setToolTip(
            "Stop training early if validation loss doesn't improve.\n\n"
            "Prevents overfitting and saves training time.\n"
            "Training stops if validation loss doesn't improve for 'Patience' epochs.\n"
            "Recommended: enabled (default)"
        )
        scheduler_form.addRow(self.early_stopping_check)

        self.early_stopping_patience_spin = QSpinBox()
        self.early_stopping_patience_spin.setRange(1, 200)
        self.early_stopping_patience_spin.setToolTip(
            "Number of epochs to wait before early stopping.\n\n"
            "Training stops if validation loss doesn't improve for this many epochs.\n"
            "• 10-15: Standard patience (default: 15)\n"
            "• 20-30: More patient, allow longer training\n\n"
            "Higher values = wait longer before stopping."
        )
        scheduler_form.addRow("Patience:", self.early_stopping_patience_spin)

        self._on_scheduler_changed(self.lr_scheduler_combo.currentText())

        training_layout.addWidget(scheduler_group)

        # Device Section
        device_group = QGroupBox("Device & Output")
        device_form = QFormLayout(device_group)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        self.device_combo.setToolTip(
            "Device to use for training:\n\n"
            "• Auto: Use best available device (CUDA, then MPS, else CPU)\n"
            "• CPU: Slower but works on any machine\n"
            "• CUDA: Much faster if NVIDIA GPU is available\n"
            "• MPS: Apple Silicon GPU acceleration on macOS"
        )
        device_form.addRow("Device:", self.device_combo)

        # Read-only: where the model will be saved (set in Data tab: Save to + Model name)
        self.model_output_path_label = QLabel("")
        self.model_output_path_label.setWordWrap(True)
        self.model_output_path_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px; padding: 4px 0;"
        )
        self.model_output_path_label.setToolTip(
            "Resolved save location. Configure \"Save to\" and \"Model name\" in the Data tab."
        )
        device_form.addRow("Will save to:", self.model_output_path_label)

        training_layout.addWidget(device_group)
        training_layout.addStretch()

        self.tab_widget.addTab(training_widget, "Training")

    def _setup_augmentation_tab(self):
        """Setup the data augmentation tab"""
        augment_widget = QWidget()
        augment_layout = QVBoxLayout(augment_widget)

        # Data Augmentation Section
        aug_group = QGroupBox("Data Augmentation")
        aug_form = QFormLayout(aug_group)

        self.augmentation_check = QCheckBox("Enable Data Augmentation")
        self.augmentation_check.toggled.connect(self._on_augmentation_toggled)
        self.augmentation_check.setToolTip(
            "Enable data augmentation to artificially increase dataset size.\n\n"
            "Applies random transformations during training to improve\n"
            "generalization and reduce overfitting.\n\n"
            "Recommended for small datasets."
        )
        aug_form.addRow(self.augmentation_check)

        # Audio augmentations
        audio_group = QGroupBox("Audio Transformations")
        audio_form = QFormLayout(audio_group)

        self.pitch_shift_spin = QDoubleSpinBox()
        self.pitch_shift_spin.setRange(0, 12)
        self.pitch_shift_spin.setToolTip(
            "Maximum pitch shift in semitones (random shift ±range).\n\n"
            "Shifts audio pitch up or down randomly.\n"
            "• 0: Disabled\n"
            "• 1-2: Light augmentation (default: 2)\n"
            "• 3-5: Moderate augmentation\n"
            "• 6+: Strong augmentation (may change class characteristics)"
        )
        audio_form.addRow("Pitch Shift Range (semitones):", self.pitch_shift_spin)

        self.time_stretch_spin = QDoubleSpinBox()
        self.time_stretch_spin.setRange(0, 1)
        self.time_stretch_spin.setToolTip(
            "Time stretch range (0.0 = no stretch, 1.0 = ±100% speed change).\n\n"
            "Stretches or compresses audio duration.\n"
            "• 0: Disabled\n"
            "• 0.1-0.2: Light augmentation (default: 0.2)\n"
            "• 0.3-0.5: Moderate augmentation\n\n"
            "Higher values = more aggressive time stretching."
        )
        audio_form.addRow("Time Stretch Range:", self.time_stretch_spin)

        self.noise_factor_spin = QDoubleSpinBox()
        self.noise_factor_spin.setRange(0, 1)
        self.noise_factor_spin.setToolTip(
            "Amount of random Gaussian noise to add (0.0-1.0).\n\n"
            "Adds noise to make model more robust to recording conditions.\n"
            "• 0: Disabled\n"
            "• 0.01-0.05: Light noise (default: 0.01)\n"
            "• 0.05-0.1: Moderate noise\n\n"
            "Higher values = more noise added."
        )
        audio_form.addRow("Noise Factor:", self.noise_factor_spin)

        self.volume_factor_spin = QDoubleSpinBox()
        self.volume_factor_spin.setRange(0, 1)
        self.volume_factor_spin.setToolTip(
            "Volume variation range (random gain ±range).\n\n"
            "Randomly adjusts audio volume/amplitude.\n"
            "• 0: Disabled\n"
            "• 0.1-0.2: Light variation (default: 0.1)\n"
            "• 0.3-0.5: Moderate variation\n\n"
            "Higher values = larger volume variations."
        )
        audio_form.addRow("Volume Factor:", self.volume_factor_spin)

        self.time_shift_spin = QDoubleSpinBox()
        self.time_shift_spin.setRange(0, 1)
        self.time_shift_spin.setToolTip(
            "Maximum time shift as fraction of audio length (0.0-1.0).\n\n"
            "Randomly shifts audio content within the sample.\n"
            "• 0: Disabled\n"
            "• 0.1-0.2: Light shifting (default: 0.1)\n"
            "• 0.3-0.5: Moderate shifting\n\n"
            "Helps model be invariant to temporal position."
        )
        audio_form.addRow("Time Shift Max:", self.time_shift_spin)

        aug_form.addRow(audio_group)

        # Spectrogram augmentations
        spec_group = QGroupBox("Spectrogram Augmentations")
        spec_form = QFormLayout(spec_group)

        self.frequency_mask_spin = QSpinBox()
        self.frequency_mask_spin.setRange(0, 50)
        self.frequency_mask_spin.setToolTip(
            "Number of frequency bins to mask (SpecAugment-style). 0 = off.\n\n"
            "Randomly masks out frequency bands in spectrogram.\n"
            "• 0: Disabled\n"
            "• 5-15: Light masking\n"
            "• 15-30: Moderate masking (typical SpecAugment range)\n\n"
            "Helps model focus on important frequency content."
        )
        spec_form.addRow("Frequency Mask:", self.frequency_mask_spin)

        self.time_mask_spin = QSpinBox()
        self.time_mask_spin.setRange(0, 50)
        self.time_mask_spin.setToolTip(
            "Number of time frames to mask (SpecAugment-style). 0 = off.\n\n"
            "Randomly masks out time segments in spectrogram.\n"
            "• 0: Disabled\n"
            "• 10-20: Light masking\n"
            "• 20-40: Moderate masking (typical SpecAugment range)\n\n"
            "Helps model be robust to missing temporal information."
        )
        spec_form.addRow("Time Mask:", self.time_mask_spin)

        aug_form.addRow(spec_group)

        # Polarity and Random EQ (optional)
        self.polarity_inversion_spin = QDoubleSpinBox()
        self.polarity_inversion_spin.setRange(0, 1)
        self.polarity_inversion_spin.setSingleStep(0.05)
        self.polarity_inversion_spin.setToolTip(
            "Probability of flipping audio polarity (phase inversion).\n\n"
            "0 = disabled. 0.5 = 50% chance per sample. Can help with invariance."
        )
        aug_form.addRow("Polarity Inversion Prob:", self.polarity_inversion_spin)

        self.use_random_eq_check = QCheckBox("Use Random EQ")
        self.use_random_eq_check.setToolTip(
            "Apply random EQ (lowpass/highpass/bandpass) to simulate different recording conditions."
        )
        aug_form.addRow(self.use_random_eq_check)

        # Batch-level augmentation (Mixup / CutMix)
        batch_aug_group = QGroupBox("Batch Augmentation")
        batch_aug_form = QFormLayout(batch_aug_group)

        self.use_mixup_check = QCheckBox("Use Mixup")
        self.use_mixup_check.setToolTip(
            "Blend pairs of samples and labels in each batch (Beta distribution). "
            "Use with multiclass or binary; soft labels are applied automatically."
        )
        batch_aug_form.addRow(self.use_mixup_check)

        self.mixup_alpha_spin = QDoubleSpinBox()
        self.mixup_alpha_spin.setRange(0, 2)
        self.mixup_alpha_spin.setSingleStep(0.1)
        self.mixup_alpha_spin.setToolTip("Beta(alpha, alpha) for Mixup. 0.2 = light, 0.4+ = stronger.")
        batch_aug_form.addRow("Mixup Alpha:", self.mixup_alpha_spin)

        self.use_cutmix_check = QCheckBox("Use CutMix")
        self.use_cutmix_check.setToolTip(
            "Cut a region from one spectrogram and paste onto another; blend labels by area. "
            "Typically use either Mixup or CutMix, not both."
        )
        batch_aug_form.addRow(self.use_cutmix_check)

        self.cutmix_alpha_spin = QDoubleSpinBox()
        self.cutmix_alpha_spin.setRange(0, 2)
        self.cutmix_alpha_spin.setSingleStep(0.1)
        self.cutmix_alpha_spin.setToolTip("Beta(alpha, alpha) for CutMix. 1.0 = common default.")
        batch_aug_form.addRow("CutMix Alpha:", self.cutmix_alpha_spin)

        aug_form.addRow(batch_aug_group)

        augment_layout.addWidget(aug_group)
        augment_layout.addStretch()

        self.tab_widget.addTab(augment_widget, "Augmentation")

    def _setup_advanced_tab(self):
        """Setup the advanced features tab"""
        advanced_widget = QWidget()
        advanced_layout = QVBoxLayout(advanced_widget)

        # Hyperparameter Optimization Section
        hyper_group = QGroupBox("Hyperparameter Optimization")
        hyper_form = QFormLayout(hyper_group)

        self.hyperopt_check = QCheckBox("Enable Hyperparameter Optimization")
        self.hyperopt_check.toggled.connect(self._on_hyperopt_toggled)
        self.hyperopt_check.setToolTip(
            "Enable automated hyperparameter optimization using Optuna.\n\n"
            "Automatically searches for optimal hyperparameters (learning rate,\n"
            "batch size, dropout, etc.) by training multiple model variants.\n\n"
            "Requires optuna library. Can take significant time."
        )
        hyper_form.addRow(self.hyperopt_check)

        self.hyperopt_trials_spin = QSpinBox()
        self.hyperopt_trials_spin.setRange(5, 1000)
        self.hyperopt_trials_spin.setToolTip(
            "Number of optimization trials to run.\n\n"
            "Each trial trains a model with different hyperparameters.\n"
            "• 20-50: Quick optimization (default: 50)\n"
            "• 100-200: Thorough search\n"
            "• 500+: Extensive search (very time-consuming)\n\n"
            "More trials = better results but longer optimization time."
        )
        hyper_form.addRow("Number of Trials:", self.hyperopt_trials_spin)

        self.hyperopt_timeout_spin = QSpinBox()
        self.hyperopt_timeout_spin.setRange(60, 86400)
        self.hyperopt_timeout_spin.setToolTip(
            "Maximum time for hyperparameter optimization in seconds.\n\n"
            "Optimization stops after this time even if trials remain.\n"
            "• 3600 (1 hour): Quick optimization (default)\n"
            "• 7200-14400 (2-4 hours): Longer optimization\n"
            "• 86400 (24 hours): Very long optimization\n\n"
            "Set based on available compute time."
        )
        hyper_form.addRow("Timeout (seconds):", self.hyperopt_timeout_spin)

        advanced_layout.addWidget(hyper_group)

        # Transfer Learning Section
        transfer_group = QGroupBox("Transfer Learning")
        transfer_form = QFormLayout(transfer_group)

        self.transfer_learning_check = QCheckBox("Use Pre-trained Model")
        self.transfer_learning_check.setChecked(False)  # Not yet implemented
        self.transfer_learning_check.setEnabled(False)  # Disable until implemented
        transfer_form.addRow(self.transfer_learning_check)

        self.pretrained_model_edit = QLineEdit()
        self.pretrained_model_edit.setText("")
        self.pretrained_model_edit.setPlaceholderText("Path to pre-trained model")
        self.pretrained_model_edit.setEnabled(False)
        transfer_form.addRow("Pre-trained Model:", self.pretrained_model_edit)

        advanced_layout.addWidget(transfer_group)

        # Ensemble Section
        ensemble_group = QGroupBox("Ensemble Methods")
        ensemble_form = QFormLayout(ensemble_group)

        self.ensemble_check = QCheckBox("Create Ensemble Model")
        ensemble_form.addRow(self.ensemble_check)

        self.meta_classifier_check = QCheckBox("Use Meta Classifier")
        ensemble_form.addRow("Meta Classifier:", self.meta_classifier_check)

        advanced_layout.addWidget(ensemble_group)

        # DOSE-Inspired Features Section (Drum One-Shot Extraction)
        dose_group = QGroupBox("DOSE-Inspired Features")
        dose_form = QFormLayout(dose_group)

        self.use_onset_weighting_check = QCheckBox("Enable Onset-Aware Loss")
        self.use_onset_weighting_check.setToolTip(
            "Enable onset-aware loss weighting inspired by DOSE paper.\n\n"
            "Emphasizes accurate prediction of initial transients,\n"
            "which are crucial for drum sound classification.\n"
            "Samples with strong onsets are weighted more heavily during training.\n\n"
            "Reference: DOSE - Drum One-Shot Extraction (https://arxiv.org/pdf/2504.18157)"
        )
        dose_form.addRow(self.use_onset_weighting_check)

        self.onset_loss_weight_spin = QDoubleSpinBox()
        self.onset_loss_weight_spin.setRange(0.0, 1.0)
        self.onset_loss_weight_spin.setSingleStep(0.1)
        self.onset_loss_weight_spin.setToolTip(
            "Weight for onset component in loss function (0.0-1.0).\n\n"
            "Controls how much to emphasize samples with strong onsets.\n"
            "• 0.0: No onset weighting (standard loss)\n"
            "• 0.1-0.3: Light onset weighting (default: 0.3)\n"
            "• 0.4-0.6: Moderate onset weighting\n"
            "• 0.7-1.0: Strong onset weighting\n\n"
            "Higher values = more emphasis on transient accuracy."
        )
        dose_form.addRow("Onset Loss Weight:", self.onset_loss_weight_spin)

        self.use_transient_emphasis_check = QCheckBox("Enable Transient Emphasis")
        self.use_transient_emphasis_check.setToolTip(
            "Emphasize transients in audio preprocessing (DOSE-inspired).\n\n"
            "Uses high-pass filtering and transient enhancement to emphasize\n"
            "attack transients, which are crucial for drum sound classification.\n\n"
            "Reference: DOSE - Drum One-Shot Extraction (https://arxiv.org/pdf/2504.18157)"
        )
        dose_form.addRow(self.use_transient_emphasis_check)

        self.use_multi_scale_features_check = QCheckBox("Enable Multi-Scale Features")
        self.use_multi_scale_features_check.setToolTip(
            "Use multi-scale spectral features (DOSE-inspired).\n\n"
            "Combines features at multiple scales to capture both fine-grained\n"
            "and coarse-grained spectral characteristics.\n\n"
            "Reference: DOSE - Drum One-Shot Extraction (https://arxiv.org/pdf/2504.18157)"
        )
        dose_form.addRow(self.use_multi_scale_features_check)

        advanced_layout.addWidget(dose_group)

        # Experimental Features Section
        exp_group = QGroupBox("Experimental Features")
        exp_form = QFormLayout(exp_group)

        self.mixed_precision_check = QCheckBox("Enable Mixed Precision Training")
        self.mixed_precision_check.setChecked(False)  # Not yet implemented
        self.mixed_precision_check.setEnabled(False)
        exp_form.addRow(self.mixed_precision_check)

        self.gradient_checkpointing_check = QCheckBox("Enable Gradient Checkpointing")
        self.gradient_checkpointing_check.setChecked(False)  # Not yet implemented
        self.gradient_checkpointing_check.setEnabled(False)
        exp_form.addRow(self.gradient_checkpointing_check)

        advanced_layout.addWidget(exp_group)

        advanced_layout.addStretch()

        self.tab_widget.addTab(advanced_widget, "Advanced")

    def _apply_unsupported_feature_locks(self):
        """Disable settings that are intentionally unsupported by this trainer."""
        self.cv_check.setChecked(False)
        self.cv_check.setEnabled(False)
        self.cv_folds_spin.setEnabled(False)
        self.cv_check.setToolTip(
            "Cross-validation is not implemented in PyTorchAudioTrainer yet."
        )
        self.cv_folds_spin.setToolTip(
            "Cross-validation is not implemented in PyTorchAudioTrainer yet."
        )

        self.hyperopt_check.setChecked(False)
        self.hyperopt_check.setEnabled(False)
        self.hyperopt_trials_spin.setEnabled(False)
        self.hyperopt_timeout_spin.setEnabled(False)
        self.hyperopt_check.setToolTip(
            "Hyperparameter optimization is not implemented in PyTorchAudioTrainer yet."
        )

    def refresh(self):
        """Reload settings and refresh UI when block is updated (e.g. after training)."""
        self._load_current_settings()
        self._refresh_ui_with_settings()
        self._update_status_from_last_training()

    def _connect_signals(self):
        """Connect widget signals to handlers and auto-save."""
        # --- UI logic handlers (show/hide, populate combos, etc.) ---
        self.data_dir_edit.textChanged.connect(self._on_data_dir_changed)
        self.model_name_edit.textChanged.connect(self._auto_save)
        self.model_name_edit.textChanged.connect(self._update_model_output_path_display)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.save_to_dir_edit.textChanged.connect(self._update_model_output_path_display)
        self.positive_classes_list.itemChanged.connect(self._on_positive_classes_changed)
        self.auto_tune_check.toggled.connect(
            lambda checked: self.threshold_metric_combo.setEnabled(checked)
        )
        self.cv_check.toggled.connect(self._on_cv_toggled)
        self.augmentation_check.toggled.connect(self._on_augmentation_toggled)
        self.hyperopt_check.toggled.connect(self._on_hyperopt_toggled)

        # --- Auto-save on any setting change ---
        # Data tab
        self.data_dir_edit.textChanged.connect(self._auto_save)
        self.sample_rate_spin.valueChanged.connect(self._auto_save)
        self.max_length_spin.valueChanged.connect(self._auto_save)
        self.fmax_spin.valueChanged.connect(self._auto_save)
        self.n_mels_spin.valueChanged.connect(self._auto_save)
        self.hop_length_spin.valueChanged.connect(self._auto_save)
        self.n_fft_spin.valueChanged.connect(self._auto_save)
        self.normalize_per_dataset_check.toggled.connect(self._auto_save)
        self.cv_check.toggled.connect(self._auto_save)
        self.cv_folds_spin.valueChanged.connect(self._auto_save)
        self.mode_combo.currentIndexChanged.connect(self._auto_save)
        self.positive_classes_list.itemChanged.connect(self._auto_save)
        self.negative_ratio_spin.valueChanged.connect(self._auto_save)
        self.hard_negative_dir_edit.textChanged.connect(self._auto_save)
        self.positive_filter_type_combo.currentIndexChanged.connect(self._on_positive_filter_type_changed)
        self.positive_filter_type_combo.currentIndexChanged.connect(self._auto_save)
        self.positive_filter_cutoff_spin.valueChanged.connect(self._auto_save)
        self.positive_filter_cutoff_high_spin.valueChanged.connect(self._auto_save)
        self.positive_filter_order_spin.valueChanged.connect(self._auto_save)
        self.auto_tune_check.toggled.connect(self._auto_save)
        self.threshold_metric_combo.currentTextChanged.connect(self._auto_save)

        # Model tab
        self.model_type_combo.currentTextChanged.connect(self._auto_save)
        self.num_conv_layers_spin.valueChanged.connect(self._auto_save)
        self.base_channels_spin.valueChanged.connect(self._auto_save)
        self.use_se_blocks_check.toggled.connect(self._auto_save)
        self.rnn_type_combo.currentTextChanged.connect(self._auto_save)
        self.rnn_hidden_size_spin.valueChanged.connect(self._auto_save)
        self.rnn_num_layers_spin.valueChanged.connect(self._auto_save)
        self.rnn_bidirectional_check.toggled.connect(self._auto_save)
        self.use_attention_check.toggled.connect(self._auto_save)
        self.transformer_d_model_spin.valueChanged.connect(self._auto_save)
        self.transformer_nhead_spin.valueChanged.connect(self._auto_save)
        self.transformer_num_layers_spin.valueChanged.connect(self._auto_save)
        self.wav2vec2_model_edit.textChanged.connect(self._auto_save)
        self.freeze_wav2vec2_check.toggled.connect(self._auto_save)

        # Training tab
        self.epochs_spin.valueChanged.connect(self._auto_save)
        self.batch_size_spin.valueChanged.connect(self._auto_save)
        self.num_workers_spin.valueChanged.connect(self._auto_save)
        self.optimizer_combo.currentTextChanged.connect(self._auto_save)
        self.learning_rate_spin.valueChanged.connect(self._auto_save)
        self.momentum_spin.valueChanged.connect(self._auto_save)
        self.weight_decay_spin.valueChanged.connect(self._auto_save)
        self.dropout_rate_spin.valueChanged.connect(self._auto_save)
        self.lr_scheduler_combo.currentTextChanged.connect(self._auto_save)
        self.lr_step_size_spin.valueChanged.connect(self._auto_save)
        self.lr_gamma_spin.valueChanged.connect(self._auto_save)
        self.early_stopping_check.toggled.connect(self._auto_save)
        self.early_stopping_patience_spin.valueChanged.connect(self._auto_save)
        self.device_combo.currentTextChanged.connect(self._auto_save)
        self.save_to_dir_edit.textChanged.connect(self._auto_save)

        # Augmentation tab
        self.augmentation_check.toggled.connect(self._auto_save)
        self.pitch_shift_spin.valueChanged.connect(self._auto_save)
        self.time_stretch_spin.valueChanged.connect(self._auto_save)
        self.noise_factor_spin.valueChanged.connect(self._auto_save)
        self.volume_factor_spin.valueChanged.connect(self._auto_save)
        self.time_shift_spin.valueChanged.connect(self._auto_save)
        self.frequency_mask_spin.valueChanged.connect(self._auto_save)
        self.time_mask_spin.valueChanged.connect(self._auto_save)
        self.polarity_inversion_spin.valueChanged.connect(self._auto_save)
        self.use_random_eq_check.toggled.connect(self._auto_save)
        self.use_mixup_check.toggled.connect(self._auto_save)
        self.mixup_alpha_spin.valueChanged.connect(self._auto_save)
        self.use_cutmix_check.toggled.connect(self._auto_save)
        self.cutmix_alpha_spin.valueChanged.connect(self._auto_save)

        # Advanced tab
        self.hyperopt_check.toggled.connect(self._auto_save)
        self.hyperopt_trials_spin.valueChanged.connect(self._auto_save)
        self.hyperopt_timeout_spin.valueChanged.connect(self._auto_save)
        self.ensemble_check.toggled.connect(self._auto_save)
        self.meta_classifier_check.toggled.connect(self._auto_save)
        self.use_onset_weighting_check.toggled.connect(self._auto_save)
        self.onset_loss_weight_spin.valueChanged.connect(self._auto_save)
        self.use_transient_emphasis_check.toggled.connect(self._auto_save)
        self.use_multi_scale_features_check.toggled.connect(self._auto_save)

    def _load_current_settings(self):
        """Load current block settings from block metadata"""
        try:
            result = self.facade.get_block_metadata(self.block_id)
            if result and result.success:
                self.settings = result.data or {}
            else:
                self.settings = {}
        except Exception as e:
            Log.warning(f"Failed to load settings for block {self.block_id}: {e}")
            self.settings = {}

    def _update_status_from_last_training(self):
        """Show plain-language model coach feedback from the last training run."""
        last_training = self.settings.get("last_training") or {}
        if not isinstance(last_training, dict):
            return

        coach_feedback = last_training.get("coach_feedback") or {}
        excluded_count = int(last_training.get("excluded_bad_file_count") or 0)
        excluded_files = last_training.get("excluded_bad_files") or []

        has_coach = isinstance(coach_feedback, dict) and bool(coach_feedback)
        if not has_coach and excluded_count <= 0:
            return

        lines = []
        status_color = "black"

        if has_coach:
            verdict = str(coach_feedback.get("verdict") or "unknown").strip().lower()
            score = coach_feedback.get("score")
            summary = str(coach_feedback.get("summary") or "").strip()
            trend = coach_feedback.get("trend") or {}
            trend_msg = str(trend.get("message") or "").strip()

            lines.append("Model Coach:")
            lines.append(f"Verdict: {verdict.replace('_', ' ').title()}")
            if score is not None:
                lines.append(f"Score: {score}/100")
            if summary:
                lines.append(summary)
            if trend_msg:
                lines.append(trend_msg)

            findings = coach_feedback.get("findings") or []
            if isinstance(findings, list):
                for finding in findings[:3]:
                    if isinstance(finding, str) and finding.strip():
                        lines.append(f"- {finding.strip()}")

            actions = coach_feedback.get("next_actions") or []
            if isinstance(actions, list) and actions:
                lines.append("Recommended next steps:")
                for action in actions[:3]:
                    if isinstance(action, str) and action.strip():
                        lines.append(f"- {action.strip()}")

            if verdict == "good":
                status_color = "green"
            elif verdict == "unreliable":
                status_color = "red"
            else:
                status_color = "warning"

        if excluded_count > 0:
            if lines:
                lines.append("")
            lines.append("Dataset integrity check:")
            lines.append(f"Excluded unreadable/invalid files: {excluded_count}")
            max_lines = 12
            shown = excluded_files[:max_lines] if isinstance(excluded_files, list) else []
            for item in shown:
                if isinstance(item, dict):
                    path = str(item.get("path") or "").strip()
                    reason = str(item.get("reason") or "").strip()
                    name = Path(path).name if path else "unknown_file"
                    lines.append(f"- {name}: {reason}")

            remaining = excluded_count - len(shown)
            if remaining > 0:
                lines.append(f"... and {remaining} more")
            if status_color == "black":
                status_color = "warning"

        self._update_status("\n".join(lines), status_color)

    def _auto_save(self):
        """Auto-save settings whenever a control changes."""
        if self._is_refreshing:
            return
        self._save_settings()

    def _save_settings(self):
        """Save comprehensive settings to the block."""
        mode_index = self.mode_combo.currentIndex()
        is_binary = mode_index == 1
        is_positive_vs_other = mode_index == 2
        if is_binary:
            classification_mode = "binary"
        elif is_positive_vs_other:
            classification_mode = "positive_vs_other"
        else:
            classification_mode = "multiclass"
        pos_list = self._get_positive_classes_list() if (is_binary or is_positive_vs_other) else None

        # Collect all settings from UI
        settings = {
            # Data settings
            "data_dir": self.data_dir_edit.text().strip(),
            "sample_rate": self.sample_rate_spin.value(),
            "max_length": self.max_length_spin.value(),
            "fmax": self.fmax_spin.value(),
            "n_mels": self.n_mels_spin.value(),
            "hop_length": self.hop_length_spin.value(),
            "n_fft": self.n_fft_spin.value(),
            "normalize_per_dataset": self.normalize_per_dataset_check.isChecked(),
            "use_cross_validation": self.cv_check.isChecked(),
            "cv_folds": self.cv_folds_spin.value(),

            # Classification mode
            "classification_mode": classification_mode,
            "positive_classes": pos_list,
            "target_class": (pos_list[0] if pos_list else None) if (is_binary or is_positive_vs_other) else None,
            "negative_ratio": self.negative_ratio_spin.value(),
            "hard_negative_dir": self.hard_negative_dir_edit.text().strip() or None,
            "positive_filter_type": self.positive_filter_type_combo.currentData(),
            "positive_filter_cutoff_hz": self.positive_filter_cutoff_spin.value(),
            "positive_filter_cutoff_high_hz": self.positive_filter_cutoff_high_spin.value(),
            "positive_filter_order": self.positive_filter_order_spin.value(),
            "auto_tune_threshold": self.auto_tune_check.isChecked(),
            "threshold_metric": self.threshold_metric_combo.currentText(),

            # Model settings
            "model_type": self.model_type_combo.currentText(),
            "num_conv_layers": self.num_conv_layers_spin.value(),
            "base_channels": self.base_channels_spin.value(),
            "use_se_blocks": self.use_se_blocks_check.isChecked(),
            "rnn_type": self.rnn_type_combo.currentText(),
            "rnn_hidden_size": self.rnn_hidden_size_spin.value(),
            "rnn_num_layers": self.rnn_num_layers_spin.value(),
            "rnn_bidirectional": self.rnn_bidirectional_check.isChecked(),
            "use_attention": self.use_attention_check.isChecked(),
            "transformer_d_model": self.transformer_d_model_spin.value(),
            "transformer_nhead": self.transformer_nhead_spin.value(),
            "transformer_num_layers": self.transformer_num_layers_spin.value(),
            "wav2vec2_model": self.wav2vec2_model_edit.text().strip(),
            "freeze_wav2vec2": self.freeze_wav2vec2_check.isChecked(),

            # Training settings
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "num_workers": self.num_workers_spin.value(),
            "optimizer": self.optimizer_combo.currentText(),
            "learning_rate": self.learning_rate_spin.value(),
            "momentum": self.momentum_spin.value(),
            "weight_decay": self.weight_decay_spin.value(),
            "dropout_rate": self.dropout_rate_spin.value(),
            "lr_scheduler": self.lr_scheduler_combo.currentText(),
            "lr_step_size": self.lr_step_size_spin.value(),
            "lr_gamma": self.lr_gamma_spin.value(),
            "use_early_stopping": self.early_stopping_check.isChecked(),
            "early_stopping_patience": self.early_stopping_patience_spin.value(),
            "device": self.device_combo.currentText(),
            "output_model_path": self.save_to_dir_edit.text().strip() or None,

            # Augmentation settings
            "use_augmentation": self.augmentation_check.isChecked(),
            "pitch_shift_range": self.pitch_shift_spin.value(),
            "time_stretch_range": self.time_stretch_spin.value(),
            "noise_factor": self.noise_factor_spin.value(),
            "volume_factor": self.volume_factor_spin.value(),
            "time_shift_max": self.time_shift_spin.value(),
            "frequency_mask": self.frequency_mask_spin.value(),
            "time_mask": self.time_mask_spin.value(),
            "polarity_inversion_prob": self.polarity_inversion_spin.value(),
            "use_random_eq": self.use_random_eq_check.isChecked(),
            "use_mixup": self.use_mixup_check.isChecked(),
            "mixup_alpha": self.mixup_alpha_spin.value(),
            "use_cutmix": self.use_cutmix_check.isChecked(),
            "cutmix_alpha": self.cutmix_alpha_spin.value(),

            # Advanced settings
            "use_hyperopt": self.hyperopt_check.isChecked(),
            "hyperopt_trials": self.hyperopt_trials_spin.value(),
            "hyperopt_timeout": self.hyperopt_timeout_spin.value(),
            
            # DOSE-inspired features
            "use_onset_weighting": self.use_onset_weighting_check.isChecked(),
            "onset_loss_weight": self.onset_loss_weight_spin.value(),
            "use_transient_emphasis": self.use_transient_emphasis_check.isChecked(),
            "use_multi_scale_features": self.use_multi_scale_features_check.isChecked(),

            # Dataset balancing
            "balance_strategy": self.balance_strategy_combo.currentData() or "none",
            "balance_target_count": self.balance_target_spin.value() if self.balance_target_spin.isVisible() else None,

            # Model name
            "model_name": self.model_name_edit.text().strip() or None,
        }

        # Unsupported features are always forced off.
        settings["use_cross_validation"] = False
        settings["use_hyperopt"] = False

        # Keep explicit clears for nullable keys so merge-based metadata updates can reset old values.
        nullable_keys = {
            "positive_filter_type",
            "hard_negative_dir",
            "output_model_path",
            "balance_target_count",
            "model_name",
            "target_class",
        }
        filtered = {}
        for k, v in settings.items():
            if k in nullable_keys and v is None:
                filtered[k] = None
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            if v is None:
                continue
            filtered[k] = v
        settings = filtered

        # Save settings to block metadata
        result = self.facade.update_block_metadata(self.block_id, settings)
        success = result and result.success

        if not success:
            Log.warning(f"PyTorchAudioTrainerPanel: Failed to auto-save settings for {self.block_id}")

    def _validate_configuration(self):
        """Validate the current configuration"""
        # Get current settings from UI
        temp_settings = {
            "data_dir": self.data_dir_edit.text().strip(),
        }

        # Create temporary block for validation (match this panel's block type)
        from src.features.blocks.domain import Block
        temp_block = Block(
            id=self.block_id,
            project_id="temp",
            name="temp",
            type="PyTorchAudioTrainer",
            metadata=temp_settings
        )

        # Validate using the correct processor for this block type
        from src.application.blocks.pytorch_audio_trainer_block import PyTorchAudioTrainerBlockProcessor
        processor = PyTorchAudioTrainerBlockProcessor()
        errors = processor.validate_configuration(temp_block)

        if errors:
            error_text = "\n\n".join(errors)
            self._update_status(f"❌ Configuration Issues:\n\n{error_text}", "red")
        else:
            self._update_status("✅ Configuration is valid and ready for training!", "green")

    def _browse_save_to_dir(self):
        """Browse for save-to directory (model folder will be created inside it)."""
        current = self.save_to_dir_edit.text().strip() or str(get_models_dir())
        if current and not Path(current).is_dir():
            current = str(Path(current).parent) if Path(current).parent.exists() else str(get_models_dir())
        directory = QFileDialog.getExistingDirectory(self, "Save Model To Directory", current)
        if directory:
            self.save_to_dir_edit.setText(directory)

    def _on_data_dir_changed(self, path):
        """Handle data directory change -- update info label and populate target class combo"""
        resolved = resolve_dataset_path(path)
        if resolved and resolved != path and not self._is_refreshing:
            self.data_dir_edit.blockSignals(True)
            self.data_dir_edit.setText(resolved)
            self.data_dir_edit.blockSignals(False)
            path = resolved

        if path and os.path.exists(path):
            try:
                data_path = Path(path)
                class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

                if class_dirs:
                    audio_exts = {
                        ".wav", ".flac", ".ogg", ".aiff", ".aif",
                        ".mp3", ".m4a", ".aac", ".wma", ".alac", ".opus", ".mp4",
                    }

                    def _count_audio(d):
                        return sum(
                            1 for f in d.rglob("*")
                            if f.is_file() and f.suffix.lower() in audio_exts
                        )

                    # Store class counts for binary mode info display
                    self._class_counts = {d.name: _count_audio(d) for d in class_dirs}

                    # Populate checkable list of classes (binary mode: check = positive)
                    saved = getattr(self, "_saved_positive_classes", []) or []
                    self.positive_classes_list.blockSignals(True)
                    self.positive_classes_list.clear()
                    for d in class_dirs:
                        item = QListWidgetItem(d.name)
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        item.setCheckState(
                            Qt.CheckState.Checked if d.name in saved else Qt.CheckState.Unchecked
                        )
                        self.positive_classes_list.addItem(item)
                    self.positive_classes_list.blockSignals(False)

                    # Update info label based on mode
                    self._update_data_info_label()
                    self._update_balance_preview()
                else:
                    self._class_counts = {}
                    self.positive_classes_list.clear()
                    self.data_info_label.setText("No class subdirectories found. Create folders for each audio class.")
                    self.data_info_label.setStyleSheet(f"color: {Colors.ACCENT_YELLOW.name()}; font-size: 11px;")
            except Exception as e:
                self.data_info_label.setText(f"Error reading directory: {e}")
                self.data_info_label.setStyleSheet(f"color: {Colors.ACCENT_RED.name()}; font-size: 11px;")
        else:
            self._class_counts = {}
            self.positive_classes_list.clear()
            self.data_info_label.setText("Select a directory containing subfolders for each audio class")
            self.data_info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")

    def _on_positive_classes_changed(self):
        """Update data info and model output path when selection changes."""
        self._update_data_info_label()
        self._update_model_output_path_display()

    def _update_model_output_path_display(self):
        """Update the read-only 'Will save to' label."""
        if not hasattr(self, "model_output_path_label"):
            return
        last_training = self.settings.get("last_training") or {}
        if isinstance(last_training, dict) and last_training.get("model_path"):
            path = last_training["model_path"]
            self.model_output_path_label.setText(path)
            self.model_output_path_label.setToolTip("Path where the model was last saved.")
            return
        base = self.save_to_dir_edit.text().strip() if hasattr(self, "save_to_dir_edit") else ""
        if not base:
            base = str(get_models_dir())
        name = (self.model_name_edit.text().strip() if hasattr(self, "model_name_edit") else "") or "model"
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:60]
        if not safe:
            safe = "model"
        idx = self.mode_combo.currentIndex()
        mode = "binary" if idx == 1 else ("positive_vs_other" if idx == 2 else "multiclass")
        if not self.model_name_edit.text().strip() and hasattr(self, "model_name_edit"):
            arch = self.model_type_combo.currentText() if hasattr(self, "model_type_combo") else "cnn"
            if mode == "binary":
                pos_list = self._get_positive_classes_list()
                tag = "_".join(pos_list)[:60] if pos_list else "unknown"
                safe = f"binary_{tag}_{arch}_YYYYMMDD_HHMMSS"
            elif mode == "positive_vs_other":
                pos_list = self._get_positive_classes_list()
                tag = "_".join(pos_list)[:60] if pos_list else "unknown"
                safe = f"positive_vs_other_{tag}_{arch}_YYYYMMDD_HHMMSS"
            else:
                safe = f"multiclass_{arch}_YYYYMMDD_HHMMSS"
        self.model_output_path_label.setText(f"{base} / {safe} /")
        self.model_output_path_label.setToolTip(
            "Model folder (containing .pth and MODEL_SUMMARY.txt) will be created here. Set in Data tab."
        )

    def _update_data_info_label(self):
        """Update the data info label based on current mode and class counts."""
        if not hasattr(self, '_class_counts') or not self._class_counts:
            return

        mode_index = self.mode_combo.currentIndex()
        is_binary = mode_index == 1
        is_positive_vs_other = mode_index == 2
        total_files = sum(self._class_counts.values())

        if is_binary:
            pos_list = self._get_positive_classes_list()
            if pos_list:
                pos_count = sum(self._class_counts.get(p, 0) for p in pos_list)
                neg_count = total_files - pos_count
                pos_label = " + ".join(f"'{p}'" for p in pos_list)
                info_text = (
                    f"Binary: {pos_label} ({pos_count} positive) "
                    f"vs all others ({neg_count} negative)\n"
                )
                other_classes = [
                    f"  {name} ({count} files)"
                    for name, count in sorted(self._class_counts.items())
                    if name not in pos_list
                ]
                if other_classes:
                    info_text += "Negative classes:\n" + "\n".join(other_classes)
            else:
                info_text = f"Check one or more classes to classify (positive); the rest will be negative."
            self.data_info_label.setText(info_text)
            self.data_info_label.setStyleSheet(f"color: {Colors.ACCENT_GREEN.name()}; font-size: 11px;")
        elif is_positive_vs_other:
            pos_list = self._get_positive_classes_list()
            if pos_list:
                pos_count = sum(self._class_counts.get(p, 0) for p in pos_list)
                other_count = total_files - pos_count
                pos_label = ", ".join(f"'{p}'" for p in pos_list)
                info_text = (
                    f"Positive vs Other: {pos_label} ({pos_count} samples) "
                    f"vs other ({other_count} samples)\n"
                )
                other_classes = [
                    f"  {name} ({count} files)"
                    for name, count in sorted(self._class_counts.items())
                    if name not in pos_list
                ]
                if other_classes:
                    info_text += "Other (grouped):\n" + "\n".join(other_classes)
            else:
                info_text = "Check one or more classes as positive; the rest will be grouped as 'other'."
            self.data_info_label.setText(info_text)
            self.data_info_label.setStyleSheet(f"color: {Colors.ACCENT_GREEN.name()}; font-size: 11px;")
        else:
            class_info = [
                f"{name} ({count} files)"
                for name, count in sorted(self._class_counts.items())
            ]
            info_text = f"Found {len(self._class_counts)} classes with {total_files} total audio files:\n"
            info_text += "\n".join(f"  {cls}" for cls in class_info)
            self.data_info_label.setText(info_text)
            self.data_info_label.setStyleSheet(f"color: {Colors.ACCENT_GREEN.name()}; font-size: 11px;")

    def _get_positive_classes_list(self):
        """Return list of class names that are checked (positive) in binary or positive_vs_other mode."""
        if not hasattr(self, "positive_classes_list"):
            return []
        result = []
        for i in range(self.positive_classes_list.count()):
            item = self.positive_classes_list.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                text = (item.text() or "").strip()
                if text:
                    result.append(text)
        return result

    def _on_positive_filter_type_changed(self):
        """Show cutoff (high) row only when filter type is Band (bandpass)."""
        is_band = self.positive_filter_type_combo.currentData() == "bandpass"
        self.positive_filter_cutoff_high_label.setVisible(is_band)
        self.positive_filter_cutoff_high_spin.setVisible(is_band)

    def _on_mode_changed(self, index):
        """Show/hide mode-specific controls. 0=Multiclass, 1=Binary, 2=Positive vs Other."""
        is_binary = index == 1
        is_positive_vs_other = index == 2
        show_positive_list = is_binary or is_positive_vs_other
        self.positive_classes_label.setVisible(show_positive_list)
        self.positive_classes_list.setVisible(show_positive_list)
        self.negative_ratio_label.setVisible(is_binary)
        self.negative_ratio_spin.setVisible(is_binary)
        self.hard_negative_label.setVisible(is_binary)
        self.hard_neg_widget.setVisible(is_binary)
        self.threshold_tune_label.setVisible(is_binary)
        self.threshold_widget.setVisible(is_binary)
        self.positive_filter_label.setVisible(is_binary)
        self.positive_filter_container.setVisible(is_binary)
        if is_binary:
            self._on_positive_filter_type_changed()
        self._update_data_info_label()
        self._update_model_output_path_display()

    def _on_balance_strategy_changed(self, index):
        """Show/hide target count, set selection tooltip, and update balance preview."""
        strategy = self.balance_strategy_combo.currentData()
        needs_target = strategy in ("undersample_target", "oversample_target", "hybrid")
        self.balance_target_label.setVisible(needs_target)
        self.balance_target_spin.setVisible(needs_target)
        if 0 <= index < len(getattr(self, "_balance_strategy_tooltips", [])):
            self.balance_strategy_combo.setToolTip(self._balance_strategy_tooltips[index])
        self._update_balance_preview()

    def _update_balance_preview(self):
        """Show a before/after preview of the selected balancing strategy."""
        if not hasattr(self, "balance_preview_label"):
            return
        if not hasattr(self, "_class_counts") or not self._class_counts:
            self.balance_preview_label.setText("")
            return

        strategy = self.balance_strategy_combo.currentData()
        if strategy == "none":
            self.balance_preview_label.setText("All samples used as-is (no balancing)")
            return

        target_count = self.balance_target_spin.value() if strategy in (
            "undersample_target", "oversample_target", "hybrid"
        ) else None

        try:
            from src.application.blocks.training.balancing import preview_balance

            # Build a fake sample list for preview. Binary: 0/1; positive_vs_other: 0..K + other; multiclass: per folder.
            mode_index = self.mode_combo.currentIndex()
            is_binary = mode_index == 1
            is_positive_vs_other = mode_index == 2
            pos_list = self._get_positive_classes_list() if (is_binary or is_positive_vs_other) else []
            positive_set = set(pos_list) if pos_list else None

            fake_samples = []
            class_names = {}
            if is_binary and positive_set is not None:
                neg_count = sum(c for name, c in self._class_counts.items() if name not in positive_set)
                pos_count = sum(c for name, c in self._class_counts.items() if name in positive_set)
                for _ in range(neg_count):
                    fake_samples.append((None, 0))
                for _ in range(pos_count):
                    fake_samples.append((None, 1))
                class_names = {0: "negative", 1: "positive"}
            elif is_positive_vs_other and positive_set is not None:
                # Labels 0..K-1 = positive classes in order, K = other
                ordered_pos = [c for c in pos_list if c in self._class_counts]
                class_names = {i: name for i, name in enumerate(ordered_pos)}
                class_names[len(ordered_pos)] = "other"
                other_count = sum(c for name, c in self._class_counts.items() if name not in positive_set)
                for idx, name in enumerate(ordered_pos):
                    for _ in range(self._class_counts.get(name, 0)):
                        fake_samples.append((None, idx))
                for _ in range(other_count):
                    fake_samples.append((None, len(ordered_pos)))
            else:
                for idx, (cls_name, count) in enumerate(sorted(self._class_counts.items())):
                    class_names[idx] = cls_name
                    for _ in range(count):
                        fake_samples.append((None, idx))

            preview = preview_balance(
                samples=fake_samples,
                strategy=strategy,
                target_count=target_count,
                class_names=class_names,
            )

            lines = []
            lines.append(f"Strategy: {self.balance_strategy_combo.currentText()}")
            if preview.get("target_per_class") is not None:
                lines.append(f"Target per class: {preview['target_per_class']}")
            lines.append("")

            # Before/After table
            before = preview["before"]
            after = preview["after"]
            changes = preview["changes"]

            for cls_name in sorted(before.keys()):
                b = before[cls_name]
                a = after[cls_name]
                delta = changes[cls_name]
                if delta > 0:
                    change_str = f" (+{delta} oversampled)"
                elif delta < 0:
                    change_str = f" ({delta} removed)"
                else:
                    change_str = " (unchanged)"
                lines.append(f"  {cls_name}: {b} -> {a}{change_str}")

            lines.append("")
            lines.append(
                f"Total: {preview['total_before']} -> {preview['total_after']}"
            )

            self.balance_preview_label.setText("\n".join(lines))
            self.balance_preview_label.setStyleSheet(
                f"color: {Colors.ACCENT_GREEN.name()}; font-size: 11px; padding: 4px;"
            )
        except Exception as e:
            self.balance_preview_label.setText(f"Preview error: {e}")
            self.balance_preview_label.setStyleSheet(
                f"color: {Colors.STATUS_WARNING.name()}; font-size: 11px; padding: 4px;"
            )

    def _on_model_type_changed(self, model_type):
        """Handle model type change to show/hide relevant options"""
        # Hide all model-specific groups
        self.cnn_group.setVisible(model_type == "cnn")
        self.rnn_group.setVisible(model_type in ["rnn", "lstm", "gru"])
        self.transformer_group.setVisible(model_type == "transformer")
        self.wav2vec2_group.setVisible(model_type == "wav2vec2")
        self._update_model_output_path_display()

    def _on_optimizer_changed(self, optimizer):
        """Handle optimizer change to show/hide momentum"""
        show_momentum = optimizer == "sgd"
        self.momentum_label.setVisible(show_momentum)
        self.momentum_spin.setVisible(show_momentum)

    def _on_scheduler_changed(self, scheduler):
        """Handle scheduler change to show/hide relevant options"""
        show_step = scheduler == "step"
        show_gamma = scheduler in ["step", "cosine"]
        self.lr_step_label.setVisible(show_step)
        self.lr_step_size_spin.setVisible(show_step)
        self.lr_gamma_label.setVisible(show_gamma)
        self.lr_gamma_spin.setVisible(show_gamma)

    def _on_cv_toggled(self, enabled):
        """Handle cross-validation toggle"""
        self.cv_folds_spin.setEnabled(enabled)

    def _on_augmentation_toggled(self, enabled):
        """Handle augmentation toggle"""
        self.pitch_shift_spin.setEnabled(enabled)
        self.time_stretch_spin.setEnabled(enabled)
        self.noise_factor_spin.setEnabled(enabled)
        self.volume_factor_spin.setEnabled(enabled)
        self.time_shift_spin.setEnabled(enabled)
        self.frequency_mask_spin.setEnabled(enabled)
        self.time_mask_spin.setEnabled(enabled)
        self.polarity_inversion_spin.setEnabled(enabled)
        self.use_random_eq_check.setEnabled(enabled)
        self.use_mixup_check.setEnabled(enabled)
        self.mixup_alpha_spin.setEnabled(enabled)
        self.use_cutmix_check.setEnabled(enabled)
        self.cutmix_alpha_spin.setEnabled(enabled)

    def _on_hyperopt_toggled(self, enabled):
        """Handle hyperparameter optimization toggle"""
        self.hyperopt_trials_spin.setEnabled(enabled)
        self.hyperopt_timeout_spin.setEnabled(enabled)

    def _update_status(self, message, color="black"):
        """Update status display"""
        self.status_text.setPlainText(message)
        if color == "green":
            color_code = Colors.ACCENT_GREEN.name()
        elif color == "red":
            color_code = Colors.ACCENT_RED.name()
        elif color == "warning":
            color_code = Colors.STATUS_WARNING.name()
        else:
            color_code = Colors.TEXT_PRIMARY.name()

        self.status_text.setStyleSheet(f"color: {color_code};")
