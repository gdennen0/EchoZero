"""
Separator Cloud block panel.

Provides UI for configuring cloud separator settings:
- AWS credentials (Access Key ID, Secret Access Key)
- AWS configuration (S3 Bucket, Batch Queue, Job Definition)
- Model selection (same as regular Separator)
- Processing options
"""
from typing import Any
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QLabel,
    QVBoxLayout, QGroupBox, QSpinBox, QLineEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Spacing, Colors
from src.utils.message import Log

# Import Demucs models info
from src.application.blocks.separator_block import DEMUCS_MODELS


@register_block_panel("SeparatorCloud")
class SeparatorCloudPanel(BlockPanelBase):
    """Panel for SeparatorCloud block configuration"""
    
    def __init__(self, block_id: str, facade, parent=None):
        super().__init__(block_id, facade, parent)
        if self.block:
            self.refresh()
    
    def create_content_widget(self) -> QWidget:
        """Create SeparatorCloud-specific UI"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)
        
        # AWS Credentials group
        credentials_group = QGroupBox("AWS Credentials")
        credentials_layout = QFormLayout(credentials_group)
        credentials_layout.setSpacing(Spacing.SM)
        
        # Access Key ID
        self.access_key_input = QLineEdit()
        self.access_key_input.setPlaceholderText("AKIA...")
        self.access_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
        self.access_key_input.textChanged.connect(self._on_access_key_changed)
        credentials_layout.addRow("Access Key ID:", self.access_key_input)
        
        # Secret Access Key
        self.secret_key_input = QLineEdit()
        self.secret_key_input.setPlaceholderText("Enter secret key...")
        self.secret_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.secret_key_input.textChanged.connect(self._on_secret_key_changed)
        credentials_layout.addRow("Secret Access Key:", self.secret_key_input)
        
        # Region
        self.region_input = QLineEdit()
        self.region_input.setPlaceholderText("us-east-1")
        self.region_input.setText("us-east-1")
        self.region_input.textChanged.connect(self._on_region_changed)
        credentials_layout.addRow("AWS Region:", self.region_input)
        
        layout.addWidget(credentials_group)
        
        # AWS Configuration group
        config_group = QGroupBox("AWS Configuration")
        config_layout = QFormLayout(config_group)
        config_layout.setSpacing(Spacing.SM)
        
        # S3 Bucket
        self.bucket_input = QLineEdit()
        self.bucket_input.setPlaceholderText("echozero-cloud-storage")
        self.bucket_input.textChanged.connect(self._on_bucket_changed)
        config_layout.addRow("S3 Bucket:", self.bucket_input)
        
        # Batch Queue
        self.queue_input = QLineEdit()
        self.queue_input.setPlaceholderText("echozero-batch-queue")
        self.queue_input.textChanged.connect(self._on_queue_changed)
        config_layout.addRow("Batch Queue:", self.queue_input)
        
        # Job Definition
        self.job_def_input = QLineEdit()
        self.job_def_input.setPlaceholderText("echozero-demucs")
        self.job_def_input.textChanged.connect(self._on_job_def_changed)
        config_layout.addRow("Job Definition:", self.job_def_input)
        
        layout.addWidget(config_group)
        
        # Model settings group (same as regular Separator)
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
        
        # Two-stems mode
        self.two_stems_combo = QComboBox()
        self.two_stems_combo.addItem("All stems (4-way separation)", None)
        self.two_stems_combo.addItem("Vocals + No Vocals", "vocals")
        self.two_stems_combo.addItem("Drums + No Drums", "drums")
        self.two_stems_combo.addItem("Bass + No Bass", "bass")
        self.two_stems_combo.addItem("Other + No Other", "other")
        self.two_stems_combo.currentIndexChanged.connect(self._on_two_stems_changed)
        processing_layout.addRow("Separation Mode:", self.two_stems_combo)
        
        layout.addWidget(processing_group)
        
        # Info label
        info_label = QLabel(
            "This block runs Demucs on AWS Batch.\n"
            "Make sure your AWS Batch job definition is set up correctly."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px; padding: 8px;")
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        return widget
    
    def refresh(self):
        """Update UI with current block settings"""
        if not self.block:
            return
        
        # Block signals during refresh to prevent change handlers from firing
        # This prevents infinite loops when refresh is triggered by BlockUpdated events
        self.access_key_input.blockSignals(True)
        self.secret_key_input.blockSignals(True)
        self.region_input.blockSignals(True)
        self.bucket_input.blockSignals(True)
        self.queue_input.blockSignals(True)
        self.job_def_input.blockSignals(True)
        self.model_combo.blockSignals(True)
        self.two_stems_combo.blockSignals(True)
        
        try:
            # Load settings from block metadata
            metadata = self.block.metadata
            
            # AWS Credentials
            self.access_key_input.setText(metadata.get("aws_access_key_id", ""))
            self.secret_key_input.setText(metadata.get("aws_secret_access_key", ""))
            self.region_input.setText(metadata.get("aws_region", "us-east-1"))
            
            # AWS Configuration
            self.bucket_input.setText(metadata.get("aws_s3_bucket", ""))
            self.queue_input.setText(metadata.get("aws_batch_queue", ""))
            self.job_def_input.setText(metadata.get("aws_batch_job_def", ""))
            
            # Model
            model = metadata.get("model", "htdemucs")
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == model:
                    self.model_combo.setCurrentIndex(i)
                    break
            self._update_model_info(model)
            
            # Two-stems
            two_stems = metadata.get("two_stems")
            if two_stems is None:
                self.two_stems_combo.setCurrentIndex(0)
            else:
                for i in range(1, self.two_stems_combo.count()):
                    if self.two_stems_combo.itemData(i) == two_stems:
                        self.two_stems_combo.setCurrentIndex(i)
                        break
        finally:
            # Unblock signals after refresh
            self.access_key_input.blockSignals(False)
            self.secret_key_input.blockSignals(False)
            self.region_input.blockSignals(False)
            self.bucket_input.blockSignals(False)
            self.queue_input.blockSignals(False)
            self.job_def_input.blockSignals(False)
            self.model_combo.blockSignals(False)
            self.two_stems_combo.blockSignals(False)
    
    def _update_model_info(self, model_name: str):
        """Update model information label"""
        if model_name in DEMUCS_MODELS:
            info = DEMUCS_MODELS[model_name]
            self.model_info_label.setText(
                f"{info['description']} - {info['stems']} stems"
            )
        else:
            self.model_info_label.setText("")
    
    def _save_setting(self, key: str, value: Any):
        """Save a setting to block metadata"""
        if not self.block:
            return
        
        # Use base class method for undoable metadata updates
        self.set_block_metadata_key(key, value, success_message=f"Saved {key}")
    
    def _on_access_key_changed(self, text: str):
        """Handle access key change"""
        self._save_setting("aws_access_key_id", text)
    
    def _on_secret_key_changed(self, text: str):
        """Handle secret key change"""
        self._save_setting("aws_secret_access_key", text)
    
    def _on_region_changed(self, text: str):
        """Handle region change"""
        self._save_setting("aws_region", text)
    
    def _on_bucket_changed(self, text: str):
        """Handle bucket change"""
        self._save_setting("aws_s3_bucket", text)
    
    def _on_queue_changed(self, text: str):
        """Handle queue change"""
        self._save_setting("aws_batch_queue", text)
    
    def _on_job_def_changed(self, text: str):
        """Handle job definition change"""
        self._save_setting("aws_batch_job_def", text)
    
    def _on_model_changed(self, index: int):
        """Handle model change"""
        model = self.model_combo.itemData(index)
        if model:
            self._update_model_info(model)
            self._save_setting("model", model)
    
    def _on_two_stems_changed(self, index: int):
        """Handle two-stems change"""
        two_stems = self.two_stems_combo.itemData(index)
        self._save_setting("two_stems", two_stems)
