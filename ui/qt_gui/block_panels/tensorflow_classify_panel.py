"""
TensorFlowClassify block panel.

Provides UI for configuring TensorFlow classification settings with drag-and-drop support.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QGroupBox, QLineEdit, QPushButton, QFileDialog, QSpinBox, QTextEdit,
    QCheckBox, QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from pathlib import Path
import os
import json

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing
from src.application.settings.tensorflow_classify_settings import TensorFlowClassifySettingsManager
from src.application.blocks.builtin_models import get_builtin_model_path, list_builtin_models, get_builtin_model_info
from src.utils.paths import get_models_dir
from src.utils.message import Log
from src.utils.settings import app_settings


class ModelPathEdit(QLineEdit):
    """Custom QLineEdit with drag-and-drop support for model files"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept drag events with file URLs"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Handle dropped files"""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            # Validate path
            if os.path.exists(path):
                self.setText(path)
                # Emit textChanged signal to trigger validation
                self.textChanged.emit(path)
            else:
                Log.warning(f"TensorFlowClassifyPanel: Dropped path does not exist: {path}")


@register_block_panel("TensorFlowClassify")
class TensorFlowClassifyPanel(BlockPanelBase):
    """Panel for TensorFlowClassify block configuration"""
    
    def __init__(self, block_id: str, facade, parent=None):
        super().__init__(block_id, facade, parent)
        
        # Initialize settings manager AFTER parent init
        self._settings_manager = TensorFlowClassifySettingsManager(facade, block_id, parent=self)
        
        # Connect to settings changes for UI updates
        self._settings_manager.settings_changed.connect(self._on_setting_changed)
        
        # Refresh UI now that settings manager is ready
        if self.block:
            self.refresh()
    
    def create_content_widget(self) -> QWidget:
        """Create TensorFlowClassify-specific UI"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)
        
        # Model info group
        model_group = QGroupBox("Model Information")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(Spacing.SM)
        
        info_label = QLabel(
            "TensorFlowClassify uses TensorFlow/Keras models to classify events.\n"
            "Supports .h5, .keras files and SavedModel directories.\n"
            "Built-in models are available and auto-download when needed."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11pt;")
        model_layout.addWidget(info_label)
        
        # Built-in model info display
        self.builtin_model_info_label = QLabel("")
        self.builtin_model_info_label.setWordWrap(True)
        self.builtin_model_info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt; padding: 4px;")
        model_layout.addWidget(self.builtin_model_info_label)
        
        # Model format display
        self.model_format_label = QLabel("")
        self.model_format_label.setWordWrap(True)
        self.model_format_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt; padding: 4px;")
        model_layout.addWidget(self.model_format_label)
        
        layout.addWidget(model_group)
        
        # Model settings group
        settings_group = QGroupBox("Model Settings")
        settings_layout = QFormLayout(settings_group)
        settings_layout.setSpacing(Spacing.SM)
        
        # Model path with drag-and-drop
        path_layout = QVBoxLayout()
        path_input_layout = QVBoxLayout()
        
        self.model_path_edit = ModelPathEdit()
        self.model_path_edit.setPlaceholderText("Drag and drop model file/directory here or enter path")
        self.model_path_edit.textChanged.connect(self._on_model_path_changed)
        path_input_layout.addWidget(self.model_path_edit)
        
        # Model path validation status
        self.model_path_status_label = QLabel("")
        self.model_path_status_label.setWordWrap(True)
        self.model_path_status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt;")
        path_input_layout.addWidget(self.model_path_status_label)
        
        path_layout.addLayout(path_input_layout)
        
        # Buttons for model path
        button_layout = QHBoxLayout()
        button_layout.setSpacing(Spacing.SM)
        
        use_builtin_btn = QPushButton("Use Built-in Model")
        use_builtin_btn.clicked.connect(self._on_use_builtin_model)
        use_builtin_btn.setToolTip("Use the built-in Drum Audio Classifier model (auto-downloads if needed)")
        button_layout.addWidget(use_builtin_btn)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_model)
        button_layout.addWidget(browse_btn)
        
        path_layout.addLayout(button_layout)
        
        settings_layout.addRow("Model Path:", path_layout)
        
        layout.addWidget(settings_group)
        
        # Processing settings group
        processing_group = QGroupBox("Processing Settings")
        processing_layout = QFormLayout(processing_group)
        processing_layout.setSpacing(Spacing.SM)
        
        # Sample rate
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 96000)
        self.sample_rate_spin.setSingleStep(1000)
        self.sample_rate_spin.setValue(22050)
        self.sample_rate_spin.setSuffix(" Hz")
        self.sample_rate_spin.valueChanged.connect(self._on_sample_rate_changed)
        self.sample_rate_spin.setToolTip(
            "Audio sample rate for processing.\n"
            "Must match the sample rate used when training the model.\n"
            "Default: 22050 Hz (common for audio analysis)."
        )
        processing_layout.addRow("Sample Rate:", self.sample_rate_spin)
        
        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1024)
        self.batch_size_spin.setSpecialValueText("Auto")
        self.batch_size_spin.setValue(0)  # 0 = None = Auto
        self.batch_size_spin.valueChanged.connect(self._on_batch_size_changed)
        self.batch_size_spin.setToolTip(
            "Batch size for prediction.\n"
            "Set to 0 (Auto) to let the system determine the optimal batch size."
        )
        processing_layout.addRow("Batch Size:", self.batch_size_spin)
        
        layout.addWidget(processing_group)
        
        # Preprocessing configuration (optional)
        preprocessing_group = QGroupBox("Preprocessing Configuration (Optional)")
        preprocessing_layout = QVBoxLayout(preprocessing_group)
        preprocessing_layout.setSpacing(Spacing.SM)
        
        info_label = QLabel(
            "Advanced: Configure custom preprocessing as JSON.\n"
            "Leave empty to use default mel spectrogram preprocessing."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;")
        preprocessing_layout.addWidget(info_label)
        
        self.preprocessing_edit = QTextEdit()
        self.preprocessing_edit.setPlaceholderText('{"type": "mel_spectrogram", "n_mels": 128, "hop_length": 512}')
        self.preprocessing_edit.setMaximumHeight(100)
        self.preprocessing_edit.textChanged.connect(self._on_preprocessing_changed)
        preprocessing_layout.addWidget(self.preprocessing_edit)
        
        layout.addWidget(preprocessing_group)
        
        # Execution Summary section
        summary_group = QGroupBox("Last Execution Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.setSpacing(Spacing.SM)
        
        self.summary_label = QLabel("No execution data available.\nRun the block to see results here.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt; padding: 8px;")
        summary_layout.addWidget(self.summary_label)
        
        layout.addWidget(summary_group)
        
        layout.addStretch()
        
        return widget
    
    def refresh(self):
        """Update UI with current settings from settings manager"""
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return
        
        if not self.block or not self._settings_manager.is_loaded():
            return
        
        # Load settings from settings manager
        try:
            model_path = self._settings_manager.model_path
            preprocessing_config = self._settings_manager.preprocessing_config
            sample_rate = self._settings_manager.sample_rate
            batch_size = self._settings_manager.batch_size
        except Exception as e:
            Log.error(f"TensorFlowClassifyPanel: Failed to load settings: {e}")
            return
        
        # Block signals while updating
        self.model_path_edit.blockSignals(True)
        self.sample_rate_spin.blockSignals(True)
        self.batch_size_spin.blockSignals(True)
        self.preprocessing_edit.blockSignals(True)
        
        # Set model path
        self.model_path_edit.setText(model_path or "")
        
        # Set sample rate
        self.sample_rate_spin.setValue(sample_rate)
        
        # Set batch size (0 = None = Auto)
        self.batch_size_spin.setValue(batch_size if batch_size is not None else 0)
        
        # Set preprocessing config
        if preprocessing_config:
            self.preprocessing_edit.setPlainText(json.dumps(preprocessing_config, indent=2))
        else:
            self.preprocessing_edit.setPlainText("")
        
        # Filter widget will handle its own state loading
        
        # Unblock signals
        self.model_path_edit.blockSignals(False)
        self.sample_rate_spin.blockSignals(False)
        self.batch_size_spin.blockSignals(False)
        self.preprocessing_edit.blockSignals(False)
        
        # Update model format and validation
        self._update_model_format()
        self._update_model_path_validation()
        self._update_builtin_model_info()
        
        # Update execution summary
        self._update_execution_summary()
        
        # Update status
        self.set_status_message("Settings loaded")
    
    def _on_model_path_changed(self, text: str):
        """Handle model path change"""
        try:
            self._settings_manager.model_path = text if text else None
            self._update_model_format()
            self._update_model_path_validation()
            self.set_status_message(f"Model path: {Path(text).name if text else '(none)'}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_use_builtin_model(self):
        """Set model path to built-in model (downloads if needed)"""
        try:
            # Use drum_audio_classifier as the built-in model
            model_id = "drum_audio_classifier"
            self.set_status_message("Getting built-in model...", error=False)
            
            builtin_path = get_builtin_model_path(model_id)
            
            if builtin_path:
                # Update via settings manager
                self._settings_manager.model_path = builtin_path
                self.set_status_message(f"Using built-in model: {Path(builtin_path).name}", error=False)
                # Refresh to update validation
                self.refresh()
            else:
                self.set_status_message(
                    f"Failed to get built-in model '{model_id}'. "
                    "Check your internet connection and try again.",
                    error=True
                )
        except Exception as e:
            Log.error(f"TensorFlowClassifyPanel: Error setting built-in model: {e}")
            self.set_status_message(f"Failed to set built-in model: {str(e)}", error=True)
    
    def _on_browse_model(self):
        """Open file/directory dialog to select model"""
        current_path = self._settings_manager.model_path or ""
        start_dir = str(Path(current_path).parent) if current_path else app_settings.get_dialog_path("tensorflow_model")
        
        # Try directory first (SavedModel format)
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Model Directory or File",
            start_dir
        )
        
        if dir_path:
            app_settings.set_dialog_path("tensorflow_model", dir_path)
            try:
                self._settings_manager.model_path = dir_path
                self.set_status_message(f"Model path set: {Path(dir_path).name}", error=False)
                self._update_model_format()
                self._update_model_path_validation()
            except ValueError as e:
                self.set_status_message(str(e), error=True)
                self.refresh()
    
    def _on_sample_rate_changed(self, value: int):
        """Handle sample rate change"""
        try:
            self._settings_manager.sample_rate = value
            self.set_status_message(f"Sample rate set to {value} Hz", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_batch_size_changed(self, value: int):
        """Handle batch size change"""
        try:
            # 0 = None = Auto
            self._settings_manager.batch_size = None if value == 0 else value
            self.set_status_message(f"Batch size set to {'Auto' if value == 0 else value}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_preprocessing_changed(self):
        """Handle preprocessing config change"""
        try:
            text = self.preprocessing_edit.toPlainText().strip()
            if text:
                # Validate JSON
                config = json.loads(text)
                self._settings_manager.preprocessing_config = config
                self.set_status_message("Preprocessing config updated", error=False)
            else:
                self._settings_manager.preprocessing_config = None
                self.set_status_message("Preprocessing config cleared", error=False)
        except json.JSONDecodeError as e:
            # Don't show error for incomplete JSON while typing
            pass
        except ValueError as e:
            self.set_status_message(str(e), error=True)
    
    def _update_model_format(self):
        """Update model format display"""
        if not hasattr(self, 'model_format_label'):
            return
        
        try:
            model_path = self._settings_manager.model_path
            if not model_path:
                self.model_format_label.setText("")
                return
            
            path_obj = Path(model_path)
            
            # Detect format
            if path_obj.is_dir() and (path_obj / "saved_model.pb").exists():
                # Check if .h5 version exists (preferred)
                h5_path = path_obj.parent / f"{path_obj.name}.h5"
                if h5_path.exists():
                    format_text = (
                        "<b>Format:</b> TensorFlow SavedModel (directory)<br/>"
                        f"<span style='color: {Colors.STATUS_SUCCESS.name()};'>✓ .h5 version available (preferred)</span>"
                    )
                else:
                    format_text = (
                        "<b>Format:</b> TensorFlow SavedModel (directory)<br/>"
                        f"<span style='color: {Colors.STATUS_WARNING.name()};'>⚠ Consider converting to .h5 for better compatibility</span>"
                    )
            elif path_obj.is_file():
                ext = path_obj.suffix.lower()
                if ext in [".h5", ".keras"]:
                    format_text = f"<b>Format:</b> TensorFlow/Keras ({ext})<br/><span style='color: {Colors.STATUS_SUCCESS.name()};'>✓ Preferred format</span>"
                else:
                    format_text = f"<b>Format:</b> Unknown ({ext})"
            else:
                format_text = "<b>Format:</b> Unknown"
            
            self.model_format_label.setText(format_text)
        except Exception as e:
            Log.warning(f"TensorFlowClassifyPanel: Error updating model format: {e}")
    
    def _update_builtin_model_info(self):
        """Update built-in model information display"""
        if not hasattr(self, 'builtin_model_info_label'):
            return
        
        try:
            model_path = self._settings_manager.model_path
            if not model_path:
                self.builtin_model_info_label.setText("")
                return
            
            # Check if this is a built-in model path
            models_dir = get_models_dir()
            if str(models_dir) in model_path:
                # Check which built-in model this might be
                for model_id in list_builtin_models(framework="tensorflow"):
                    model_info = get_builtin_model_info(model_id)
                    if model_info:
                        local_dir = models_dir / model_info["local_dir"]
                        h5_file = models_dir / f"{model_id}.h5"
                        
                        # Check if this matches SavedModel directory or .h5 file
                        if str(local_dir) == model_path or str(h5_file) == model_path:
                            # This is a built-in model
                            info_text = (
                                f"<b>Built-in Model:</b> {model_info['name']}<br/>"
                                f"{model_info['description']}"
                            )
                            
                            # Add .h5 availability info
                            if h5_file.exists():
                                info_text += f"<br/><span style='color: {Colors.STATUS_SUCCESS.name()};'>✓ .h5 format available (preferred)</span>"
                            elif local_dir.exists() and (local_dir / "saved_model.pb").exists():
                                info_text += f"<br/><span style='color: {Colors.STATUS_WARNING.name()};'>⚠ Using SavedModel (consider converting to .h5)</span>"
                            
                            self.builtin_model_info_label.setText(info_text)
                            return
            
            # Not a built-in model
            self.builtin_model_info_label.setText("")
        except Exception as e:
            Log.warning(f"TensorFlowClassifyPanel: Error updating built-in model info: {e}")
            self.builtin_model_info_label.setText("")
    
    def _update_model_path_validation(self):
        """Update model path validation status display"""
        if not hasattr(self, 'model_path_status_label'):
            return
        
        try:
            model_path = self._settings_manager.model_path
            if not model_path:
                self.model_path_status_label.setText("⚠ Model path required")
                self.model_path_status_label.setStyleSheet(
                    f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt;"
                )
                return
            
            # Check if path exists
            if os.path.exists(model_path):
                path_obj = Path(model_path)
                if path_obj.is_dir():
                    # Check for SavedModel
                    if (path_obj / "saved_model.pb").exists():
                        status_text = f"✓ Valid SavedModel directory: {path_obj.name}"
                        status_color = Colors.STATUS_SUCCESS.name()
                    else:
                        status_text = f"⚠ Directory exists but no saved_model.pb found"
                        status_color = Colors.STATUS_WARNING.name()
                else:
                    # Check file extension
                    ext = path_obj.suffix.lower()
                    if ext in [".h5", ".keras"]:
                        status_text = f"✓ Valid TensorFlow model file: {path_obj.name}"
                        status_color = Colors.STATUS_SUCCESS.name()
                    else:
                        status_text = f"⚠ File exists but format may not be supported: {ext}"
                        status_color = Colors.STATUS_WARNING.name()
            else:
                status_text = f"✗ Path not found: {Path(model_path).name}"
                status_color = Colors.STATUS_ERROR.name()
            
            self.model_path_status_label.setText(status_text)
            self.model_path_status_label.setToolTip(f"Model path: {model_path}")
            self.model_path_status_label.setStyleSheet(
                f"color: {status_color}; font-size: 9pt;"
            )
        except Exception as e:
            Log.warning(f"TensorFlowClassifyPanel: Error validating model path: {e}")
            self.model_path_status_label.setText("")
    
    def _update_execution_summary(self):
        """Update the execution summary display"""
        if not hasattr(self, 'summary_label') or not self.block:
            return
        
        try:
            last_execution = self.block.metadata.get("last_execution")
            if not last_execution:
                self.summary_label.setText("No execution data available.\nRun the block to see results here.")
                self.summary_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt; padding: 8px;")
                return
            
            # Build summary text
            summary_lines = []
            summary_lines.append(f"<b>Last Execution Summary</b>")
            
            event_items_processed = last_execution.get("event_items_processed", 0)
            total_events_input = last_execution.get("total_events_input", 0)
            total_events_output = last_execution.get("total_events_output", 0)
            total_classified = last_execution.get("total_classified", 0)
            total_skipped = last_execution.get("total_skipped", 0)
            model_path = last_execution.get("model_path", "unknown")
            
            summary_lines.append(f"")
            summary_lines.append(f"Event Items Processed: <b>{event_items_processed}</b>")
            summary_lines.append(f"Total Events Input: <b>{total_events_input}</b>")
            summary_lines.append(f"Total Events Output: <b>{total_events_output}</b>")
            summary_lines.append(f"Classified: <b>{total_classified}</b>")
            if total_skipped > 0:
                summary_lines.append(f"Skipped: <b>{total_skipped}</b>")
            
            summary_lines.append(f"")
            summary_lines.append(f"Model: <b>{Path(model_path).name}</b>")
            
            # Show classification counts
            classification_counts = last_execution.get("classification_counts", {})
            if classification_counts:
                summary_lines.append(f"")
                summary_lines.append(f"<b>Classification Distribution:</b>")
                for classification, count in sorted(classification_counts.items(), key=lambda x: x[1], reverse=True):
                    summary_lines.append(f"  • {classification}: <b>{count}</b>")
            
            # Show confidence statistics
            confidence_stats = last_execution.get("confidence_stats", {})
            if confidence_stats and any(v is not None for v in confidence_stats.values()):
                summary_lines.append(f"")
                summary_lines.append(f"<b>Confidence Statistics:</b>")
                if confidence_stats.get("min") is not None:
                    summary_lines.append(f"  • Min: <b>{confidence_stats['min']:.3f}</b>")
                if confidence_stats.get("max") is not None:
                    summary_lines.append(f"  • Max: <b>{confidence_stats['max']:.3f}</b>")
                if confidence_stats.get("avg") is not None:
                    summary_lines.append(f"  • Avg: <b>{confidence_stats['avg']:.3f}</b>")
            
            summary_text = "\n".join(summary_lines)
            self.summary_label.setText(summary_text)
            self.summary_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 10pt; padding: 8px;")
            
        except Exception as e:
            Log.error(f"TensorFlowClassifyPanel: Error updating execution summary: {e}")
            self.summary_label.setText("Error loading execution summary.")
            self.summary_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt; padding: 8px;")
    
    def _on_filter_selection_changed(self, port_name: str):
        """Handle filter selection change - update status message"""
        if hasattr(self, 'filter_widget') and self.filter_widget:
            selected_ids = self.filter_widget.get_selected_item_ids()
            self.set_status_message(
                f"Filter selection saved: {len(selected_ids)} EventDataItem(s) selected",
                error=False
            )
    
    
    def _on_setting_changed(self, setting_name: str):
        """React to settings changes"""
        if setting_name in ['model_path', 'sample_rate', 'batch_size', 'preprocessing_config']:
            self.refresh()
    
    def refresh_for_undo(self):
        """Refresh panel after undo/redo operation"""
        if hasattr(self, '_settings_manager') and self._settings_manager:
            self._settings_manager.reload_from_storage()
        self.refresh()
    
    def _on_block_updated(self, event):
        """Handle block update event"""
        updated_block_id = event.data.get('id')
        if updated_block_id == self.block_id:
            if self._is_saving:
                Log.debug(f"TensorFlowClassifyPanel: Skipping refresh during save for {self.block_id}")
                return
            
            Log.debug(f"TensorFlowClassifyPanel: Block {self.block_id} updated externally, refreshing UI")
            
            result = self.facade.describe_block(self.block_id)
            if result.success:
                self.block = result.data
                self._update_header()
            else:
                return
            
            if hasattr(self, '_settings_manager') and self._settings_manager:
                self._settings_manager.reload_from_storage()
            
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, self.refresh)

