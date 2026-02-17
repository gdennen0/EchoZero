"""
Setlist Action Configuration Dialog

Dialog for configuring action parameters (e.g., threshold values, file paths).
Auto-detects input type from action metadata and provides appropriate input widget.
"""
from typing import Dict, Any, Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QFileDialog, QDialogButtonBox, QMessageBox
)
from PyQt6.QtCore import Qt
from pathlib import Path

from src.application.api.application_facade import ApplicationFacade
from ui.qt_gui.design_system import Colors, Spacing, Typography
from src.utils.message import Log
from src.utils.settings import app_settings


class SetlistActionConfigDialog(QDialog):
    """
    Dialog for configuring action parameters.
    
    Auto-detects input type from action metadata and provides
    appropriate input widget (text, number, file, choice, etc.).
    """
    
    def __init__(
        self,
        facade: ApplicationFacade,
        block_id: str,
        action_name: str,
        action_data: Dict[str, Any],
        current_args: Optional[Dict[str, Any]] = None,
        parent=None
    ):
        super().__init__(parent)
        self.facade = facade
        self.block_id = block_id
        self.action_name = action_name
        self.action_data = action_data
        self.current_args = current_args or {}
        self.action_args: Dict[str, Any] = {}
        
        self._setup_ui()
        self._load_current_values()
    
    def _setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle(f"Configure Action: {self.action_name}")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.MD)
        
        # Description
        description = self.action_data.get("description", "")
        if description:
            desc_label = QLabel(description)
            desc_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)
        
        # Form for action parameters
        form = QFormLayout()
        form.setSpacing(Spacing.SM)
        
        # For now, we'll use a simple approach:
        # Call the action handler with no args to see what it needs
        # Then provide appropriate input based on the response
        
        # Try to get input requirements from action
        # This is a simplified version - in practice, we'd need to inspect
        # the action handler's signature or metadata
        
        # For MVP, we'll provide a generic text input
        # Advanced: parse action handler signature or use action metadata
        
        self.input_widgets: Dict[str, Any] = {}
        
        # Common action parameters we know about
        if "file" in self.action_name.lower() or "path" in self.action_name.lower():
            # File/path input with dynamic value support
            file_input = QLineEdit()
            file_input.setPlaceholderText("Enter file path, or use {song_audio_path}, {song_name}, etc.")
            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(lambda: self._browse_file(file_input))
            
            file_layout = QHBoxLayout()
            file_layout.addWidget(file_input)
            file_layout.addWidget(browse_btn)
            
            # Add help text for dynamic values
            help_label = QLabel(
                "Available placeholders: {song_audio_path}, {song_name}, {song_full_name}, {song_index}, {setlist_id}"
            )
            help_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10px;")
            help_label.setWordWrap(True)
            
            form.addRow("File Path:", file_layout)
            form.addRow("", help_label)  # Empty label for spacing
            self.input_widgets["file_path"] = file_input
        
        elif "directory" in self.action_name.lower() or "dir" in self.action_name.lower():
            # Directory input
            dir_input = QLineEdit()
            dir_input.setPlaceholderText("Enter directory path or click Browse...")
            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(lambda: self._browse_directory(dir_input))
            
            dir_layout = QHBoxLayout()
            dir_layout.addWidget(dir_input)
            dir_layout.addWidget(browse_btn)
            
            form.addRow("Directory:", dir_layout)
            self.input_widgets["directory"] = dir_input
        
        elif "threshold" in self.action_name.lower() or "sensitivity" in self.action_name.lower():
            # Number input (0.0-1.0)
            number_input = QDoubleSpinBox()
            number_input.setRange(0.0, 1.0)
            number_input.setSingleStep(0.05)
            number_input.setDecimals(2)
            form.addRow("Value:", number_input)
            self.input_widgets["value"] = number_input
        
        elif "format" in self.action_name.lower():
            # Choice input
            format_combo = QComboBox()
            format_combo.addItems(["wav", "mp3", "flac", "ogg"])
            form.addRow("Format:", format_combo)
            self.input_widgets["fmt"] = format_combo
        
        else:
            # Generic text input
            text_input = QLineEdit()
            text_input.setPlaceholderText("Enter value...")
            form.addRow("Value:", text_input)
            self.input_widgets["value"] = text_input
        
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _load_current_values(self):
        """Load current action argument values into input widgets"""
        for key, widget in self.input_widgets.items():
            if key in self.current_args:
                value = self.current_args[key]
                if isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findText(str(value))
                    if index >= 0:
                        widget.setCurrentIndex(index)
            elif key == "file_path" and isinstance(widget, QLineEdit):
                # Default to placeholder for file_path if not set
                widget.setText("{song_audio_path}")
    
    def _browse_file(self, line_edit: QLineEdit):
        """Browse for file"""
        start_dir = app_settings.get_dialog_path("action_file")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", start_dir,
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)"
        )
        if file_path:
            app_settings.set_dialog_path("action_file", file_path)
            line_edit.setText(file_path)
    
    def _browse_directory(self, line_edit: QLineEdit):
        """Browse for directory"""
        start_dir = app_settings.get_dialog_path("action_dir")
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", start_dir,
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            app_settings.set_dialog_path("action_dir", directory)
            line_edit.setText(directory)
    
    def get_action_args(self) -> Dict[str, Any]:
        """Get configured action arguments"""
        args = {}
        for key, widget in self.input_widgets.items():
            if isinstance(widget, QLineEdit):
                value = widget.text().strip()
                if value:
                    args[key] = value
            elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                args[key] = widget.value()
            elif isinstance(widget, QComboBox):
                args[key] = widget.currentText()
        return args
    
    def _is_placeholder(self, value: str) -> bool:
        """Check if a value is a placeholder (e.g., {song_audio_path})"""
        if not isinstance(value, str):
            return False
        return value.strip().startswith("{") and value.strip().endswith("}")
    
    def accept(self):
        """Validate and accept"""
        # Basic validation
        args = self.get_action_args()
        
        # For file paths, validate they exist (unless it's a placeholder)
        if "file_path" in args:
            file_path = args["file_path"]
            # Skip validation for placeholders - they'll be resolved during setlist processing
            if not self._is_placeholder(file_path):
                path = Path(file_path)
                if not path.is_file():
                    QMessageBox.warning(
                        self, "Invalid File",
                        f"File not found: {file_path}\n\n"
                        "Tip: Use placeholders like {{song_audio_path}} for dynamic values that will be resolved during setlist processing."
                    )
                    return
        
        if "directory" in args:
            dir_path = args["directory"]
            # Skip validation for placeholders
            if not self._is_placeholder(dir_path):
                path = Path(dir_path)
                if not path.is_dir():
                    QMessageBox.warning(
                        self, "Invalid Directory",
                        f"Directory not found: {dir_path}\n\n"
                        "Tip: Use placeholders like {{song_name}} for dynamic values that will be resolved during setlist processing."
                    )
                    return
        
        super().accept()

