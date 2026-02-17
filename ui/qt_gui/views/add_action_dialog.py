"""
Add Action Dialog

Dialog window for adding actions to an action set with structured parameter configuration.
Shows Blocks | Actions | Params in a column layout with information display areas.
"""
from typing import Dict, Any, List, Optional, Tuple
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QDoubleSpinBox,
    QSpinBox, QTextEdit, QGroupBox, QDialogButtonBox, QMessageBox,
    QFileDialog, QScrollArea, QWidget, QSplitter, QFrame, QAbstractItemView
)
from PyQt6.QtCore import Qt
from pathlib import Path

from src.application.api.application_facade import ApplicationFacade
from src.features.projects.domain import ActionItem
from ui.qt_gui.design_system import Colors, Spacing, Typography, ThemeAwareMixin, border_radius
from ui.qt_gui.style_factory import StyleFactory
from src.utils.message import Log
from src.utils.settings import app_settings


class AddActionDialog(ThemeAwareMixin, QDialog):
    """
    Dialog for adding an action to an action set.
    
    Features:
    - Column layout: Blocks | Actions | Params
    - Information display areas
    - Structured parameter fields with validation
    """
    
    def __init__(
        self,
        facade: ApplicationFacade,
        project_id: str,
        actions_by_block: Dict[str, List[Dict[str, Any]]],
        parent=None
    ):
        super().__init__(parent)
        self.facade = facade
        self.project_id = project_id
        self.actions_by_block = actions_by_block
        
        self.selected_block_id: Optional[str] = None
        self.selected_block_name: Optional[str] = None
        self.selected_action_name: Optional[str] = None
        self.selected_action_data: Optional[Dict[str, Any]] = None
        self.action_args: Dict[str, Any] = {}
        self.input_widgets: Dict[str, Any] = {}
        
        self.setWindowTitle("Add Action")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the dialog UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)
        
        # Header with title and action set controls
        header_layout = QHBoxLayout()
        
        title = QLabel("Add Action")
        title.setFont(Typography.heading_font())
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 14px; font-weight: bold;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Load action set dropdown
        load_label = QLabel("Load Set:")
        load_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        header_layout.addWidget(load_label)
        
        self.load_set_combo = QComboBox()
        self.load_set_combo.setStyleSheet(StyleFactory.combo("detailed"))
        self.load_set_combo.setMinimumWidth(200)
        self.load_set_combo.setEditable(False)
        self._configure_combo_stretch(self.load_set_combo)
        self.load_set_combo.currentIndexChanged.connect(self._on_load_set_selected)
        header_layout.addWidget(self.load_set_combo)
        
        # Save action set button
        self.save_set_btn = QPushButton("Save Set")
        self.save_set_btn.setStyleSheet(StyleFactory.button())
        self.save_set_btn.clicked.connect(self._on_save_set)
        header_layout.addWidget(self.save_set_btn)
        
        main_layout.addLayout(header_layout)
        
        # Populate load set combo
        self._populate_load_set_combo()
        
        # Main content area with resizable splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(StyleFactory.splitter())
        
        # Left column: Block selection
        block_group = self._create_block_section()
        splitter.addWidget(block_group)
        
        # Middle column: Action selection
        action_group = self._create_action_section()
        splitter.addWidget(action_group)
        
        # Right column: Parameter configuration
        params_group = self._create_params_section()
        splitter.addWidget(params_group)
        
        # Set initial sizes (proportional)
        splitter.setSizes([200, 200, 300])
        
        main_layout.addWidget(splitter, 1)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(StyleFactory.button())
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("Add Action")
        ok_btn.setStyleSheet(StyleFactory.button("primary"))
        ok_btn.clicked.connect(self._validate_and_accept)
        button_layout.addWidget(ok_btn)
        
        main_layout.addLayout(button_layout)
        
        # Initialize
        self._populate_blocks()
        self._init_theme_aware()
    
    def _populate_load_set_combo(self):
        """Populate the load set combo box"""
        self.load_set_combo.clear()
        self.load_set_combo.addItem("-- Select Action Set --", None)
        
        # List available action sets
        result = self.facade.list_action_sets(self.project_id)
        if result.success:
            action_sets = result.data or []
            for action_set in action_sets:
                display_text = f"{action_set.name} ({len(action_set.actions)} actions)"
                self.load_set_combo.addItem(display_text, action_set.id)
    
    def _on_load_set_selected(self, index: int):
        """Handle action set selection from dropdown"""
        if index <= 0:  # First item is "-- Select Action Set --"
            return
        
        action_set_id = self.load_set_combo.itemData(index)
        if not action_set_id:
            return
        
        # Load the action set
        load_result = self.facade.load_action_set(action_set_id)
        if load_result.success:
            loaded_set = load_result.data
            if loaded_set.actions:
                # Use first action from the set
                first_action = loaded_set.actions[0]
                # Pre-populate dialog with this action
                self._populate_from_action(first_action)
                Log.info(f"AddActionDialog: Loaded action set '{loaded_set.name}'")
    
    def _create_block_section(self) -> QWidget:
        """Create block selection section"""
        container = QWidget()
        container.setStyleSheet(StyleFactory.container())
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        
        # Title
        title = QLabel("Block")
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 11px; font-weight: bold;")
        layout.addWidget(title)
        
        # Block combo
        self.block_combo = QComboBox()
        self.block_combo.setStyleSheet(StyleFactory.combo())
        self._configure_combo_stretch(self.block_combo)
        self.block_combo.currentIndexChanged.connect(self._on_block_changed)
        layout.addWidget(self.block_combo)
        
        # Block info display
        self.block_info = QTextEdit()
        self.block_info.setReadOnly(True)
        self.block_info.setMaximumHeight(100)
        self.block_info.setStyleSheet(StyleFactory.text_edit())
        self.block_info.setPlaceholderText("Select a block to see information...")
        layout.addWidget(self.block_info)
        
        layout.addStretch()
        return container
    
    def _create_action_section(self) -> QWidget:
        """Create action selection section"""
        container = QWidget()
        container.setStyleSheet(StyleFactory.container())
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        
        # Title
        title = QLabel("Action")
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 11px; font-weight: bold;")
        layout.addWidget(title)
        
        # Action combo
        self.action_combo = QComboBox()
        self.action_combo.setStyleSheet(StyleFactory.combo())
        self._configure_combo_stretch(self.action_combo)
        self.action_combo.setEnabled(False)
        self.action_combo.currentIndexChanged.connect(self._on_action_changed)
        layout.addWidget(self.action_combo)
        
        # Action info display
        self.action_info = QTextEdit()
        self.action_info.setReadOnly(True)
        self.action_info.setMaximumHeight(150)
        self.action_info.setStyleSheet(StyleFactory.text_edit())
        self.action_info.setPlaceholderText("Select a block first, then choose an action...")
        layout.addWidget(self.action_info)
        
        layout.addStretch()
        return container
    
    def _create_params_section(self) -> QWidget:
        """Create parameter configuration section"""
        container = QWidget()
        container.setStyleSheet(StyleFactory.container())
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        
        # Title
        title = QLabel("Parameters")
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 11px; font-weight: bold;")
        layout.addWidget(title)
        
        # Parameters form (will be populated when action is selected)
        self.params_form = QFormLayout()
        self.params_form.setSpacing(4)
        self.params_form.setContentsMargins(0, 0, 0, 0)
        
        # Container widget for form
        form_widget = QWidget()
        form_widget.setLayout(self.params_form)
        
        # Wrap in scroll area
        params_scroll = QScrollArea()
        params_scroll.setWidget(form_widget)
        params_scroll.setWidgetResizable(True)
        params_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        params_scroll.setMinimumHeight(200)
        layout.addWidget(params_scroll, 1)
        
        # Help text
        self.params_help = QLabel("")
        self.params_help.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9px; padding: 2px;")
        self.params_help.setWordWrap(True)
        layout.addWidget(self.params_help)
        
        layout.addStretch()
        return container
    
    def _populate_blocks(self):
        """Populate block combo box"""
        self.block_combo.clear()
        
        # Add project actions first
        if "project" in self.actions_by_block:
            project_data = self.actions_by_block["project"]
            if isinstance(project_data, dict) and "actions" in project_data:
                display_name = "Project (Project Actions)"
                self.block_combo.addItem(display_name, "project")
        
        # Then add all blocks
        for block_id, block_data in self.actions_by_block.items():
            if block_id == "project":
                continue  # Already added
            if isinstance(block_data, dict) and "actions" in block_data:
                block_name = block_data.get("block_name", f"Block {block_id[:8]}...")
                block_type = block_data.get("block_type", "Unknown")
            else:
                block_name = f"Block {block_id[:8]}..."
                block_type = "Unknown"
            
            display_name = f"{block_name} ({block_type})"
            self.block_combo.addItem(display_name, block_id)
        
        if self.block_combo.count() > 0:
            self.block_combo.setCurrentIndex(0)
    
    def _on_block_changed(self, index: int):
        """Handle block selection change"""
        if index < 0:
            return
        
        block_id = self.block_combo.itemData(index)
        if not block_id:
            return
        
        self.selected_block_id = block_id
        
        # Get block data
        is_project = (block_id == "project")
        block_data = self.actions_by_block.get(block_id, {})
        if isinstance(block_data, dict) and "actions" in block_data:
            self.selected_block_name = block_data.get("block_name", "Project" if is_project else f"Block {block_id[:8]}...")
            block_type = block_data.get("block_type", "Project" if is_project else "Unknown")
            actions = block_data.get("actions", [])
        else:
            self.selected_block_name = "Project" if is_project else f"Block {block_id[:8]}..."
            block_type = "Project" if is_project else "Unknown"
            actions = block_data if isinstance(block_data, list) else []
        
        # Update block info
        info_text = f"<b>Block:</b> {self.selected_block_name}<br/>"
        info_text += f"<b>Type:</b> {block_type}<br/>"
        info_text += f"<b>Available Actions:</b> {len(actions)}"
        self.block_info.setHtml(info_text)
        
        # Populate actions
        self.action_combo.clear()
        self.action_combo.setEnabled(len(actions) > 0)
        
        for action in actions:
            action_name = action.get("name", "Unknown")
            self.action_combo.addItem(action_name, action)
        
        if self.action_combo.count() > 0:
            self.action_combo.setCurrentIndex(0)
        else:
            self.action_info.setHtml("<i>No actions available for this block.</i>")
            self._clear_params_form()
    
    def _on_action_changed(self, index: int):
        """Handle action selection change"""
        if index < 0:
            return
        
        action_data = self.action_combo.itemData(index)
        if not action_data:
            return
        
        self.selected_action_data = action_data
        self.selected_action_name = action_data.get("name", "")
        action_desc = action_data.get("description", "")
        
        # Update action info
        info_text = f"<b>Action:</b> {self.selected_action_name}<br/><br/>"
        if action_desc:
            info_text += f"<b>Description:</b><br/>{action_desc}"
        else:
            info_text += "<i>No description available.</i>"
        self.action_info.setHtml(info_text)
        
        # Build parameter form
        self._build_params_form(action_data)
    
    def _build_params_form(self, action_data: Dict[str, Any]):
        """Build parameter form based on action"""
        # Clear existing form
        self._clear_params_form()
        
        action_name = action_data.get("name", "").lower()
        action_desc = action_data.get("description", "")
        
        # Determine parameter types based on action name and description
        # This is a structured approach with guardrails
        
        if "file" in action_name or "audio" in action_name:
            # File path parameter
            file_input = QLineEdit()
            file_input.setPlaceholderText("File path or {song_audio_path}...")
            file_input.setStyleSheet(StyleFactory.input())
            file_input.setEnabled(True)
            file_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            file_input.setCursor(Qt.CursorShape.IBeamCursor)
            browse_btn = QPushButton("Browse")
            browse_btn.setStyleSheet(StyleFactory.button("small"))
            browse_btn.clicked.connect(lambda: self._browse_file(file_input))
            
            file_layout = QHBoxLayout()
            file_layout.setSpacing(4)
            file_layout.addWidget(file_input, 1)
            file_layout.addWidget(browse_btn)
            
            self.params_form.addRow("File:", file_layout)
            self.input_widgets["file_path"] = file_input
            
            # Help text for placeholders
            help_text = (
                "<b>Available Placeholders:</b><br/>"
                "• {song_audio_path} - Full path to current song<br/>"
                "• {song_name} - Song name without extension<br/>"
                "• {song_full_name} - Song name with extension<br/>"
                "• {song_index} - Order index in setlist<br/>"
                "• {setlist_id} - Setlist identifier"
            )
            self.params_help.setText(help_text)
            
        elif action_name == "save_as":
            # save_as requires save_directory and optional name
            dir_input = QLineEdit()
            dir_input.setPlaceholderText("Directory path (use {song_name}, {song_audio_path}, etc.)...")
            dir_input.setStyleSheet(StyleFactory.input())
            dir_input.setEnabled(True)
            dir_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            dir_input.setCursor(Qt.CursorShape.IBeamCursor)
            browse_btn = QPushButton("Browse")
            browse_btn.setStyleSheet(StyleFactory.button("small"))
            browse_btn.clicked.connect(lambda: self._browse_directory(dir_input))
            
            dir_layout = QHBoxLayout()
            dir_layout.setSpacing(4)
            dir_layout.addWidget(dir_input, 1)
            dir_layout.addWidget(browse_btn)
            
            self.params_form.addRow("Save Directory:", dir_layout)
            self.input_widgets["save_directory"] = dir_input
            
            # Optional name parameter
            name_input = QLineEdit()
            name_input.setPlaceholderText("Project name (optional, use {song_name}, etc.)...")
            name_input.setStyleSheet(StyleFactory.input())
            self.params_form.addRow("Project Name:", name_input)
            self.input_widgets["name"] = name_input
            
            help_text = (
                "<b>Available Placeholders:</b><br/>"
                "• {song_audio_path} - Full path to current song<br/>"
                "• {song_name} - Song name without extension<br/>"
                "• {song_full_name} - Song name with extension<br/>"
                "• {song_index} - Order index in setlist<br/>"
                "• {setlist_id} - Setlist identifier"
            )
            self.params_help.setText(help_text)
        
        elif "directory" in action_name or "dir" in action_name or "folder" in action_name:
            # Directory parameter
            dir_input = QLineEdit()
            dir_input.setPlaceholderText("Directory path (use {song_name}, {song_audio_path}, etc.)...")
            dir_input.setStyleSheet(StyleFactory.input())
            dir_input.setEnabled(True)
            dir_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            dir_input.setCursor(Qt.CursorShape.IBeamCursor)
            browse_btn = QPushButton("Browse")
            browse_btn.setStyleSheet(StyleFactory.button("small"))
            browse_btn.clicked.connect(lambda: self._browse_directory(dir_input))
            
            dir_layout = QHBoxLayout()
            dir_layout.setSpacing(4)
            dir_layout.addWidget(dir_input, 1)
            dir_layout.addWidget(browse_btn)
            
            self.params_form.addRow("Directory:", dir_layout)
            self.input_widgets["directory"] = dir_input
            
            help_text = (
                "<b>Tip:</b> Use placeholders like {song_name} for dynamic directory names."
            )
            self.params_help.setText(help_text)
            
        elif "threshold" in action_name or "sensitivity" in action_name:
            # Numeric parameter (0.0-1.0)
            threshold_input = QDoubleSpinBox()
            threshold_input.setRange(0.0, 1.0)
            threshold_input.setSingleStep(0.05)
            threshold_input.setDecimals(3)
            threshold_input.setValue(0.5)
            threshold_input.setStyleSheet(StyleFactory.input())
            
            self.params_form.addRow("Threshold:", threshold_input)
            self.input_widgets["value"] = threshold_input
            
            help_text = (
                "<b>Range:</b> 0.0 to 1.0<br/>"
                "<b>Default:</b> 0.5<br/>"
                "Lower values = more sensitive, Higher values = less sensitive"
            )
            self.params_help.setText(help_text)
            
        elif "length" in action_name or "duration" in action_name:
            # Numeric parameter (seconds)
            length_input = QDoubleSpinBox()
            length_input.setRange(0.0, 3600.0)
            length_input.setSingleStep(0.1)
            length_input.setDecimals(2)
            length_input.setSuffix("s")
            length_input.setStyleSheet(StyleFactory.input())
            
            self.params_form.addRow("Length:", length_input)
            self.input_widgets["value"] = length_input
            
            help_text = "<b>Range:</b> 0.0 to 3600.0 seconds"
            self.params_help.setText(help_text)
            
        elif "format" in action_name:
            # Format selection
            format_combo = QComboBox()
            format_combo.addItems(["wav", "mp3", "flac", "ogg", "m4a"])
            format_combo.setStyleSheet(StyleFactory.combo())
            self._configure_combo_stretch(format_combo)
            
            self.params_form.addRow("Format:", format_combo)
            self.input_widgets["fmt"] = format_combo
            
            help_text = "<b>Supported Formats:</b> WAV, MP3, FLAC, OGG, M4A"
            self.params_help.setText(help_text)
            
        elif "model" in action_name:
            # Model selection
            model_combo = QComboBox()
            model_combo.addItems(["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"])
            model_combo.setStyleSheet(StyleFactory.combo())
            self._configure_combo_stretch(model_combo)
            
            self.params_form.addRow("Model:", model_combo)
            self.input_widgets["model"] = model_combo
            
            help_text = "<b>Available Models:</b> Select the separation model to use"
            self.params_help.setText(help_text)
            
        elif "device" in action_name:
            # Device selection
            device_combo = QComboBox()
            device_combo.addItems(["cpu", "cuda", "mps"])
            device_combo.setStyleSheet(StyleFactory.combo())
            self._configure_combo_stretch(device_combo)
            
            self.params_form.addRow("Device:", device_combo)
            self.input_widgets["device"] = device_combo
            
            help_text = "<b>Device Options:</b> CPU, CUDA (GPU), or MPS (Apple Silicon)"
            self.params_help.setText(help_text)
            
        else:
            # Generic text parameter
            text_input = QLineEdit()
            text_input.setPlaceholderText("Value...")
            text_input.setStyleSheet(StyleFactory.input())
            text_input.setEnabled(True)
            text_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            text_input.setCursor(Qt.CursorShape.IBeamCursor)
            
            self.params_form.addRow("Value:", text_input)
            self.input_widgets["value"] = text_input
            
            help_text = "<b>Enter the parameter value.</b>"
            self.params_help.setText(help_text)
    
    def _clear_params_form(self):
        """Clear the parameters form"""
        while self.params_form.rowCount() > 0:
            self.params_form.removeRow(0)
        self.input_widgets.clear()
        self.params_help.setText("")
    
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
    
    def _validate_and_accept(self):
        """Validate inputs and accept dialog"""
        if not self.selected_block_id or not self.selected_action_name:
            QMessageBox.warning(
                self, "Incomplete Selection",
                "Please select both a block and an action."
            )
            return
        
        # Validate parameters
        args = self._get_action_args()
        
        # Validate file paths (if not placeholders)
        if "file_path" in args:
            file_path = args["file_path"]
            if not self._is_placeholder(file_path):
                path = Path(file_path)
                if not path.is_file():
                    QMessageBox.warning(
                        self, "Invalid File",
                        f"File not found: {file_path}\n\n"
                        "Tip: Use placeholders like {song_audio_path} for dynamic values."
                    )
                    return
        
        # Validate save_as parameters
        if self.selected_action_name == "save_as":
            if "save_directory" in args:
                dir_path = args["save_directory"]
                if not self._is_placeholder(dir_path) and dir_path.strip():
                    path = Path(dir_path)
                    # Don't require directory to exist (it will be created)
                    # Just validate it's a valid path format
                    if not dir_path.strip():
                        QMessageBox.warning(
                            self, "Invalid Directory",
                            "Save directory cannot be empty."
                        )
                        return
        
        if "directory" in args:
            dir_path = args["directory"]
            if not self._is_placeholder(dir_path):
                path = Path(dir_path)
                if not path.is_dir():
                    QMessageBox.warning(
                        self, "Invalid Directory",
                        f"Directory not found: {dir_path}\n\n"
                        "Tip: Use placeholders like {song_name} for dynamic values."
                    )
                    return
        
        self.action_args = args
        self.accept()
    
    def _get_action_args(self) -> Dict[str, Any]:
        """Get configured action arguments"""
        args = {}
        for key, widget in self.input_widgets.items():
            if isinstance(widget, QLineEdit):
                value = widget.text().strip()
                if value:
                    args[key] = value
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                args[key] = widget.value()
            elif isinstance(widget, QComboBox):
                args[key] = widget.currentText()
        return args
    
    def _is_placeholder(self, value: str) -> bool:
        """Check if a value is a placeholder"""
        if not isinstance(value, str):
            return False
        return value.strip().startswith("{") and value.strip().endswith("}")
    
    def get_action_item(self) -> Optional[ActionItem]:
        """Get the configured action item"""
        if not self.selected_block_id or not self.selected_action_name:
            return None
        
        # Determine action_type based on whether block_id is set
        action_type = "project" if not self.selected_block_id or self.selected_block_id == "project" else "block"
        
        return ActionItem(
            action_type=action_type,
            block_id=self.selected_block_id if action_type == "block" else None,
            block_name=self.selected_block_name or "",
            action_name=self.selected_action_name,
            action_description=self.selected_action_data.get("description", "") if self.selected_action_data else "",
            action_args=self.action_args
        )
    
    def _on_save_set(self):
        """Save current action set"""
        from PyQt6.QtWidgets import QInputDialog
        from src.features.projects.domain import ActionSet
        
        # Get name from user
        name, ok = QInputDialog.getText(
            self, "Save Action Set", "Action Set Name:",
            text="Untitled"
        )
        
        if not ok or not name.strip():
            return
        
        # Create action set from current selection
        if not self.selected_block_id or not self.selected_action_name:
            QMessageBox.warning(
                self, "Incomplete Action",
                "Please select a block and action before saving."
            )
            return
        
        action_item = self.get_action_item()
        if not action_item:
            return
        
        action_set = ActionSet(
            id="",
            name=name.strip(),
            description="",
            actions=[action_item],
            project_id=self.project_id
        )
        
        result = self.facade.save_action_set(action_set)
        if result.success:
            QMessageBox.information(self, "Saved", f"Action set '{name}' saved successfully.")
            Log.info(f"AddActionDialog: Saved action set '{name}'")
            # Refresh the load combo
            self._populate_load_set_combo()
        else:
            error_msg = result.message
            if result.errors:
                error_msg += f"\n\n{result.errors[0]}"
            QMessageBox.warning(self, "Failed to Save", error_msg)
    
    
    def _populate_from_action(self, action: ActionItem):
        """Populate dialog fields from an action item"""
        # Find and select block (or "project" for project actions)
        block_id_to_match = "project" if action.action_type == "project" else action.block_id
        for i in range(self.block_combo.count()):
            if self.block_combo.itemData(i) == block_id_to_match:
                self.block_combo.setCurrentIndex(i)
                # Wait for actions to populate
                from PyQt6.QtCore import QTimer
                def select_action():
                    for j in range(self.action_combo.count()):
                        act_data = self.action_combo.itemData(j)
                        if act_data and act_data.get("name") == action.action_name:
                            self.action_combo.setCurrentIndex(j)
                            # Populate parameters
                            QTimer.singleShot(100, lambda: self._populate_params_from_action(action))
                            break
                QTimer.singleShot(200, select_action)
                break
    
    def _populate_params_from_action(self, action: ActionItem):
        """Populate parameter widgets from action"""
        for key, widget in self.input_widgets.items():
            if key in action.action_args:
                value = action.action_args[key]
                if isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(float(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findText(str(value))
                    if index >= 0:
                        widget.setCurrentIndex(index)
    
    def _configure_combo_stretch(self, combo: QComboBox):
        """Configure a combo box so dropdown items stretch to fill the width"""
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        view = combo.view()
        if view:
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            view.setTextElideMode(Qt.TextElideMode.ElideRight)
    
    # Style helpers - sharp, minimal, clean