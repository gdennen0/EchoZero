"""
Disconnect Dialog

Dialog for disconnecting blocks.
Used from Actions panel and context menu.
"""
from typing import Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QPushButton, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer

from src.application.api.application_facade import ApplicationFacade
from ui.qt_gui.connection.connection_helper import ConnectionHelper
from ui.qt_gui.design_system import Colors, Spacing, Typography


class DisconnectDialog(QDialog):
    """
    Dialog for disconnecting blocks.
    
    Features:
    - Block selection (or pre-filled if opened from a specific block)
    - Port selection (shows only connected ports)
    - Connection list (shows all connections for selected block/port)
    """
    
    def __init__(
        self, 
        facade: ApplicationFacade, 
        block_id: Optional[str] = None,
        port_name: Optional[str] = None,
        parent=None
    ):
        super().__init__(parent)
        self.facade = facade
        self.helper = ConnectionHelper(facade)
        
        # Pre-selected block/port (when opened from a specific block)
        self.initial_block_id = block_id
        self.initial_port_name = port_name
        
        self._setup_ui()
        self._load_blocks()
        
        # Pre-select if block provided
        if block_id:
            self._preselect_block(block_id, port_name)
    
    def _setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle("Disconnect Blocks")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        
        # Instructions
        instructions = QLabel(
            "Select a block and port to disconnect, or choose a specific connection."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        layout.addWidget(instructions)
        
        # Block selection
        block_group = QGroupBox("Block")
        block_layout = QFormLayout(block_group)
        
        self.block_combo = QComboBox()
        self.block_combo.currentIndexChanged.connect(self._on_block_changed)
        block_layout.addRow("Block:", self.block_combo)
        
        layout.addWidget(block_group)
        
        # Port selection
        port_group = QGroupBox("Port")
        port_layout = QFormLayout(port_group)
        
        self.port_combo = QComboBox()
        self.port_combo.currentIndexChanged.connect(self._on_port_changed)
        port_layout.addRow("Input Port:", self.port_combo)
        
        layout.addWidget(port_group)
        
        # Connection selection (alternative to port)
        connection_group = QGroupBox("Or Select Connection")
        connection_layout = QVBoxLayout(connection_group)
        
        self.connection_combo = QComboBox()
        self.connection_combo.currentIndexChanged.connect(self._on_connection_changed)
        connection_layout.addWidget(self.connection_combo)
        
        layout.addWidget(connection_group)
        
        # Status
        self.status_label = QLabel("Select a block to see connections")
        self.status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setDefault(True)
        self.disconnect_btn.clicked.connect(self._on_disconnect)
        self.disconnect_btn.setEnabled(False)
        button_layout.addWidget(self.disconnect_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def _load_blocks(self):
        """Load all blocks into the combo box"""
        self.block_combo.clear()
        blocks = self.helper.get_all_blocks()
        
        for block in blocks:
            self.block_combo.addItem(f"{block['name']} ({block['type']})", block['id'])
        
        if not blocks:
            self.block_combo.addItem("No blocks available", None)
            self.block_combo.setEnabled(False)
    
    def _preselect_block(self, block_id: str, port_name: Optional[str] = None):
        """Pre-select a block and optionally a port"""
        # Find block in combo
        for i in range(self.block_combo.count()):
            if self.block_combo.itemData(i) == block_id:
                self.block_combo.setCurrentIndex(i)
                break
        
        # If port specified, select it after ports load
        if port_name:
            QTimer.singleShot(100, lambda: self._select_port(port_name))
    
    def _select_port(self, port_name: str):
        """Select a specific port in the combo"""
        for i in range(self.port_combo.count()):
            if self.port_combo.itemData(i) == port_name:
                self.port_combo.setCurrentIndex(i)
                break
    
    def _on_block_changed(self):
        """Handle block selection change"""
        block_id = self.block_combo.currentData()
        if not block_id:
            self.port_combo.clear()
            self.connection_combo.clear()
            self._update_status("No block selected")
            return
        
        # Load connected input ports for this block
        self._load_ports(block_id)
        
        # Load connections for this block
        self._load_connections(block_id)
    
    def _load_ports(self, block_id: str):
        """Load connected input ports for a block"""
        self.port_combo.clear()
        
        inputs = self.helper.get_block_inputs(block_id)
        connected_inputs = [p for p in inputs if p.is_connected]
        
        if not connected_inputs:
            self.port_combo.addItem("No connected ports", None)
            self.port_combo.setEnabled(False)
            return
        
        self.port_combo.setEnabled(True)
        for port in connected_inputs:
            self.port_combo.addItem(
                f"{port.port_name} ({port.port_type})",
                port.port_name
            )
    
    def _load_connections(self, block_id: str):
        """Load all connections for a block"""
        self.connection_combo.clear()
        
        connections = self.helper.get_connections_for_block(block_id)
        
        if not connections:
            self.connection_combo.addItem("No connections", None)
            self.connection_combo.setEnabled(False)
            return
        
        self.connection_combo.setEnabled(True)
        for conn in connections:
            label = (
                f"{conn['source_block_name']}.{conn['source_port_name']} → "
                f"{conn['target_block_name']}.{conn['target_port_name']}"
            )
            self.connection_combo.addItem(label, conn['id'])
    
    def _on_port_changed(self):
        """Handle port selection change"""
        self._update_disconnect_button()
    
    def _on_connection_changed(self):
        """Handle connection selection change"""
        self._update_disconnect_button()
    
    def _update_disconnect_button(self):
        """Update disconnect button state"""
        has_port = self.port_combo.currentData() is not None
        has_connection = self.connection_combo.currentData() is not None
        
        can_disconnect = has_port or has_connection
        self.disconnect_btn.setEnabled(can_disconnect)
        
        if can_disconnect:
            if has_connection:
                self._update_status("Ready to disconnect selected connection", error=False)
            else:
                self._update_status("Ready to disconnect selected port", error=False)
        else:
            self._update_status("Select a port or connection to disconnect", error=False)
    
    def _update_status(self, message: str, error: bool = False):
        """Update status label"""
        self.status_label.setText(message)
        if error:
            self.status_label.setStyleSheet(f"color: {Colors.ACCENT_RED.name()};")
        else:
            self.status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
    
    def _on_disconnect(self):
        """Disconnect the selected port or connection"""
        connection_id = self.connection_combo.currentData()
        
        if connection_id:
            # Disconnect by connection ID
            result = self.helper.disconnect_connection(connection_id)
            if result.success:
                self.accept()
            else:
                QMessageBox.warning(
                    self,
                    "Disconnect Failed",
                    result.message or "Unknown error"
                )
        else:
            # Disconnect by port
            block_id = self.block_combo.currentData()
            port_name = self.port_combo.currentData()
            
            if not block_id or not port_name:
                QMessageBox.warning(self, "Invalid Selection", "Please select a port or connection")
                return
            
            result = self.helper.disconnect_by_port(block_id, port_name)
            if result.success:
                self.accept()
            else:
                QMessageBox.warning(
                    self,
                    "Disconnect Failed",
                    result.message or "Unknown error"
                )
    
    @classmethod
    def show_dialog(
        cls,
        facade: ApplicationFacade,
        block_id: Optional[str] = None,
        port_name: Optional[str] = None,
        parent=None
    ) -> bool:
        """
        Show the disconnect dialog.
        
        Args:
            facade: Application facade
            block_id: Optional pre-selected block
            port_name: Optional pre-selected port
            parent: Parent widget
            
        Returns:
            True if disconnection was successful, False otherwise
        """
        dialog = cls(facade, block_id, port_name, parent)
        return dialog.exec() == QDialog.DialogCode.Accepted

