"""
PyTorchDrumTrainer block panel.

Provides UI for configuring PyTorch drum training settings.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QGroupBox, QLineEdit, QPushButton, QFileDialog, QSpinBox,
    QComboBox, QCheckBox, QDoubleSpinBox, QTextEdit
)
from PyQt6.QtCore import Qt
from pathlib import Path
import os

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing
from src.utils.message import Log


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
        current_path = self.text() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(
            self, "Select Training Data Directory",
            current_path,
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.setText(directory)
            self.textChanged.emit(directory)


@register_block_panel("PyTorchDrumTrainer")
class PyTorchDrumTrainerPanel(BlockPanelBase):
    """
    Panel for configuring PyTorchDrumTrainer block settings.

    Provides intuitive controls for:
    - Training data directory selection
    - Training hyperparameters
    - Cross-validation options
    - Data augmentation settings
    """

    def __init__(self, block_id: str, parent=None):
        super().__init__(block_id, parent)
        self.setWindowTitle("PyTorch Drum Trainer Settings")

        # Initialize settings managers
        self._load_current_settings()

        # Setup UI
        self._setup_ui()

        # Connect signals
        self._connect_signals()

    def _setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)

        # Data Configuration Section
        data_group = QGroupBox("Training Data")
        data_layout = QFormLayout(data_group)

        # Data directory selection
        self.data_dir_edit = DataDirEdit()
        self.data_dir_edit.setText(self.settings.get("data_dir", ""))

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.data_dir_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.data_dir_edit._browse_directory)
        dir_layout.addWidget(browse_btn)

        data_layout.addRow("Data Directory:", dir_layout)

        # Data directory info
        self.data_info_label = QLabel("Select a directory containing subfolders for each drum class")
        self.data_info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        data_layout.addRow("", self.data_info_label)

        layout.addWidget(data_group)

        # Training Configuration Section
        training_group = QGroupBox("Training Configuration")
        training_layout = QFormLayout(training_group)

        # Basic settings
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(self.settings.get("epochs", 100))
        training_layout.addRow("Epochs:", self.epochs_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(self.settings.get("batch_size", 32))
        training_layout.addRow("Batch Size:", self.batch_size_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.00001, 1.0)
        self.learning_rate_spin.setDecimals(5)
        self.learning_rate_spin.setValue(self.settings.get("learning_rate", 0.001))
        training_layout.addRow("Learning Rate:", self.learning_rate_spin)

        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        current_device = self.settings.get("device", "cpu")
        if current_device == "cuda" and not self._cuda_available():
            current_device = "cpu"
            Log.warning("CUDA requested but not available, defaulting to CPU")
        self.device_combo.setCurrentText(current_device)
        training_layout.addRow("Device:", self.device_combo)

        # Output path
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setText(self.settings.get("output_model_path", ""))
        self.output_path_edit.setPlaceholderText("Auto-generated if empty")

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_path_edit)

        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(self._browse_output_path)
        output_layout.addWidget(output_browse_btn)

        training_layout.addRow("Output Path:", output_layout)

        layout.addWidget(training_group)

        # Advanced Options Section
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QFormLayout(advanced_group)

        # Early stopping
        self.early_stopping_check = QCheckBox("Enable Early Stopping")
        self.early_stopping_check.setChecked(True)  # Enabled by default
        advanced_layout.addRow(self.early_stopping_check)

        self.early_stopping_patience_spin = QSpinBox()
        self.early_stopping_patience_spin.setRange(1, 100)
        self.early_stopping_patience_spin.setValue(self.settings.get("early_stopping_patience", 15))
        advanced_layout.addRow("Early Stopping Patience:", self.early_stopping_patience_spin)

        # Cross-validation
        self.cv_check = QCheckBox("Enable Cross-Validation")
        self.cv_check.setChecked(self.settings.get("use_cross_validation", False))
        self.cv_check.toggled.connect(self._on_cv_toggled)
        advanced_layout.addRow(self.cv_check)

        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(2, 10)
        self.cv_folds_spin.setValue(self.settings.get("cv_folds", 5))
        self.cv_folds_spin.setEnabled(self.cv_check.isChecked())
        advanced_layout.addRow("CV Folds:", self.cv_folds_spin)

        # Data augmentation
        self.augmentation_check = QCheckBox("Enable Data Augmentation")
        self.augmentation_check.setChecked(self.settings.get("use_augmentation", False))
        self.augmentation_check.toggled.connect(self._on_augmentation_toggled)
        advanced_layout.addRow(self.augmentation_check)

        self.pitch_shift_spin = QDoubleSpinBox()
        self.pitch_shift_spin.setRange(0, 12)
        self.pitch_shift_spin.setValue(self.settings.get("augment_pitch_shift", 2))
        self.pitch_shift_spin.setEnabled(self.augmentation_check.isChecked())
        advanced_layout.addRow("Max Pitch Shift (semitones):", self.pitch_shift_spin)

        self.time_stretch_spin = QDoubleSpinBox()
        self.time_stretch_spin.setRange(0, 1)
        self.time_stretch_spin.setDecimals(2)
        self.time_stretch_spin.setValue(self.settings.get("augment_time_stretch", 0.1))
        self.time_stretch_spin.setEnabled(self.augmentation_check.isChecked())
        advanced_layout.addRow("Max Time Stretch:", self.time_stretch_spin)

        layout.addWidget(advanced_group)

        # Status Section
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        self.status_text.setPlainText("Ready to configure training settings.")
        status_layout.addWidget(self.status_text)

        layout.addWidget(status_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.validate_btn = QPushButton("Validate Configuration")
        self.validate_btn.clicked.connect(self._validate_configuration)
        button_layout.addWidget(self.validate_btn)

        button_layout.addStretch()

        self.save_btn = QPushButton("Save Settings")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._save_settings)
        button_layout.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def _connect_signals(self):
        """Connect widget signals to handlers"""
        self.data_dir_edit.textChanged.connect(self._on_data_dir_changed)

    def _load_current_settings(self):
        """Load current block settings"""
        self.settings = self.facade.get_block_settings(self.block_id) or {}

    def _save_settings(self):
        """Save settings to the block"""
        # Collect all settings from UI
        settings = {
            "data_dir": self.data_dir_edit.text().strip(),
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "learning_rate": self.learning_rate_spin.value(),
            "device": self.device_combo.currentText(),
            "output_model_path": self.output_path_edit.text().strip() or None,
            "early_stopping_patience": self.early_stopping_patience_spin.value(),
            "use_cross_validation": self.cv_check.isChecked(),
            "cv_folds": self.cv_folds_spin.value(),
            "use_augmentation": self.augmentation_check.isChecked(),
            "augment_pitch_shift": self.pitch_shift_spin.value(),
            "augment_time_stretch": self.time_stretch_spin.value(),
        }

        # Remove empty values
        settings = {k: v for k, v in settings.items() if v is not None and v != ""}

        # Save settings
        success = self.facade.update_block_settings(self.block_id, settings)

        if success:
            self._update_status("✅ Settings saved successfully!", "green")
            self.accept()
        else:
            self._update_status("❌ Failed to save settings!", "red")

    def _validate_configuration(self):
        """Validate the current configuration"""
        # Get current settings from UI
        temp_settings = {
            "data_dir": self.data_dir_edit.text().strip(),
        }

        # Create temporary block for validation
        from src.features.blocks.domain import Block
        temp_block = Block(
            id=self.block_id,
            project_id="temp",
            name="temp",
            type="PyTorchDrumTrainer",
            metadata=temp_settings
        )

        # Validate
        from src.application.blocks.pytorch_drum_trainer_block import PyTorchDrumTrainerBlockProcessor
        processor = PyTorchDrumTrainerBlockProcessor()
        errors = processor.validate_configuration(temp_block)

        if errors:
            error_text = "\n\n".join(errors)
            self._update_status(f"❌ Configuration Issues:\n\n{error_text}", "red")
        else:
            self._update_status("✅ Configuration is valid and ready for training!", "green")

    def _browse_output_path(self):
        """Browse for output model path"""
        current_path = self.output_path_edit.text() or str(Path.home() / "models")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Trained Model",
            current_path,
            "PyTorch Models (*.pth);;All Files (*)"
        )
        if file_path:
            self.output_path_edit.setText(file_path)

    def _on_data_dir_changed(self, path):
        """Handle data directory change"""
        if path and os.path.exists(path):
            try:
                # Count classes and files
                data_path = Path(path)
                class_dirs = [d for d in data_path.iterdir() if d.is_dir()]

                if class_dirs:
                    audio_exts = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.aiff", "*.aif")
                    def _count_audio(d):
                        return sum(len(list(d.rglob(ext))) for ext in audio_exts)
                    total_files = sum(_count_audio(d) for d in class_dirs)
                    class_info = [f"{d.name} ({_count_audio(d)} files)"
                                for d in class_dirs]

                    info_text = f"Found {len(class_dirs)} classes with {total_files} total audio files:\n"
                    info_text += "\n".join(f"• {cls}" for cls in class_info)
                    self.data_info_label.setText(info_text)
                    self.data_info_label.setStyleSheet(f"color: {Colors.SUCCESS}; font-size: 11px;")
                else:
                    self.data_info_label.setText("No class subdirectories found. Create folders for each drum type.")
                    self.data_info_label.setStyleSheet(f"color: {Colors.WARNING}; font-size: 11px;")
            except Exception as e:
                self.data_info_label.setText(f"Error reading directory: {e}")
                self.data_info_label.setStyleSheet(f"color: {Colors.ERROR}; font-size: 11px;")
        else:
            self.data_info_label.setText("Select a directory containing subfolders for each drum class")
            self.data_info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")

    def _on_cv_toggled(self, enabled):
        """Handle cross-validation toggle"""
        self.cv_folds_spin.setEnabled(enabled)

    def _on_augmentation_toggled(self, enabled):
        """Handle augmentation toggle"""
        self.pitch_shift_spin.setEnabled(enabled)
        self.time_stretch_spin.setEnabled(enabled)

    def _cuda_available(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _update_status(self, message, color="black"):
        """Update status display"""
        self.status_text.setPlainText(message)
        if color == "green":
            color_code = Colors.SUCCESS
        elif color == "red":
            color_code = Colors.ERROR
        else:
            color_code = Colors.TEXT_PRIMARY

        self.status_text.setStyleSheet(f"color: {color_code};")

