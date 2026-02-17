"""
Execution Report Dialog

Shows detailed results after executing all blocks in a project.
Displays success/failure status, executed blocks, errors, and execution mode.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QGroupBox, QScrollArea, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.utils.message import Log
from ui.qt_gui.core.colors import Colors


class ExecutionReportDialog(QDialog):
    """
    Dialog showing execution results.
    
    Displays:
    - Overall success/failure
    - Number of blocks executed vs failed
    - Execution mode (fail-fast vs partial)
    - Detailed error messages for failed blocks
    """
    
    def __init__(self, execution_result: dict, parent=None):
        """
        Initialize execution report dialog.
        
        Args:
            execution_result: Dictionary with execution results
            parent: Parent widget
        """
        super().__init__(parent)
        self.execution_result = execution_result
        
        self.setWindowTitle("Execution Report")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup dialog UI"""
        layout = QVBoxLayout()
        layout.setSpacing(16)
        
        # Title
        title = QLabel("Project Execution Report")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Summary section
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout()
        
        success = self.execution_result.get("success", False)
        executed_count = len(self.execution_result.get("executed_blocks", []))
        failed_count = len(self.execution_result.get("failed_blocks", []))
        total_count = self.execution_result.get("total_blocks", executed_count + failed_count)
        
        # Status
        status_label = QLabel()
        if success:
            status_label.setText(f"✓ Success: All {executed_count} block(s) executed successfully")
            status_label.setStyleSheet(f"color: {Colors.SUCCESS.name()}; font-size: 14px; font-weight: bold;")
        else:
            status_label.setText(f"✗ Partial Success: {executed_count}/{total_count} block(s) executed, {failed_count} failed")
            status_label.setStyleSheet(f"color: {Colors.ACCENT_RED.name()}; font-size: 14px; font-weight: bold;")
        summary_layout.addWidget(status_label)
        
        # Execution mode info
        stop_on_error = self.execution_result.get("stop_on_error", True)
        mode_label = QLabel()
        if stop_on_error:
            mode_text = "Execution Mode: Fail-Fast (stopped on first error)"
        else:
            mode_text = "Execution Mode: Partial Execution (continued despite errors)"
        mode_label.setText(mode_text)
        mode_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 12px;")
        summary_layout.addWidget(mode_label)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Executed blocks section
        if executed_count > 0:
            executed_group = QGroupBox(f"Successfully Executed ({executed_count})")
            executed_layout = QVBoxLayout()
            
            executed_list = self.execution_result.get("executed_blocks", [])
            error_details = self.execution_result.get("error_details", {})
            
            for block_id in executed_list:
                # Get block name from error_details if available (contains all block info)
                block_info = error_details.get(block_id, {})
                block_name = block_info.get("block_name", block_id)
                
                block_label = QLabel(f"  ✓ {block_name}")
                block_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()};")
                executed_layout.addWidget(block_label)
            
            executed_group.setLayout(executed_layout)
            layout.addWidget(executed_group)
        
        # Failed blocks section
        if failed_count > 0:
            failed_group = QGroupBox(f"Failed Blocks ({failed_count})")
            failed_layout = QVBoxLayout()
            
            errors = self.execution_result.get("errors", {})
            error_details = self.execution_result.get("error_details", {})
            
            for block_id in self.execution_result.get("failed_blocks", []):
                details = error_details.get(block_id, {})
                block_name = details.get("block_name", block_id)
                error_msg = errors.get(block_id, "Unknown error")
                error_type = details.get("error_type", "Error")
                
                # Block name
                block_label = QLabel(f"✗ {block_name}")
                block_label.setStyleSheet(f"color: {Colors.ACCENT_RED.name()}; font-weight: bold;")
                failed_layout.addWidget(block_label)
                
                # Error type
                type_label = QLabel(f"   Type: {error_type}")
                type_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
                failed_layout.addWidget(type_label)
                
                # Error message
                error_label = QLabel(f"   {error_msg}")
                error_label.setWordWrap(True)
                error_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 11px;")
                failed_layout.addWidget(error_label)
                
                # Filter error details
                if error_type == "FilterError":
                    port_name = details.get("port_name", "unknown")
                    remedy = details.get("remedy_action", "")
                    if remedy:
                        remedy_label = QLabel(f"   Remedy: {remedy}")
                        remedy_label.setStyleSheet(f"color: {Colors.ACCENT_CYAN.name()}; font-size: 11px;")
                        failed_layout.addWidget(remedy_label)
                
                failed_layout.addSpacing(8)
            
            failed_group.setLayout(failed_layout)
            layout.addWidget(failed_group)
        
        layout.addStretch()
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setMinimumWidth(100)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

