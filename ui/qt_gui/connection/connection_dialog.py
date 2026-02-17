"""
Connection Dialog

Dialog for creating connections between blocks manually.
Used from Actions panel and context menu.
"""
from typing import Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QPushButton, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt

from src.application.api.application_facade import ApplicationFacade
from ui.qt_gui.connection.connection_helper import ConnectionHelper, PortInfo
from ui.qt_gui.design_system import Colors, Spacing, Typography


class ConnectionDialog(QDialog):
    """
    Dialog for creating connections between blocks.
    
    Features:
    - Source block/port selection (or pre-filled if opened from a block)
    - Target block dropdown (filtered to compatible blocks)
    - Target port dropdown (populated after block selection)
    - Compatibility indicator
    """
    
    def __init__(
        self, 
        facade: ApplicationFacade, 
        source_block_id: Optional[str] = None,
        source_port_name: Optional[str] = None,
        parent=None
    ):
        super().__init__(parent)
        self.facade = facade
        self.helper = ConnectionHelper(facade)
        
        # Pre-selected source (when opened from a specific block)
        self.initial_source_block_id = source_block_id
        self.initial_source_port_name = source_port_name
        
        self._setup_ui()
        self._load_blocks()
        
        # Pre-select if source provided
        if source_block_id:
            self._preselect_source(source_block_id, source_port_name)
    
    def _setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle("Create Connection")
        self.setMinimumWidth(400)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)
        
        # Source group
        source_group = QGroupBox("Source (Output)")
        source_layout = QFormLayout(source_group)
        
        self.source_block_combo = QComboBox()
        self.source_block_combo.currentIndexChanged.connect(self._on_source_block_changed)
        source_layout.addRow("Block:", self.source_block_combo)
        
        self.source_port_combo = QComboBox()
        self.source_port_combo.currentIndexChanged.connect(self._on_source_port_changed)
        source_layout.addRow("Output Port:", self.source_port_combo)
        
        layout.addWidget(source_group)
        
        # Arrow indicator
        arrow_label = QLabel("-->")
        arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 18px;")
        layout.addWidget(arrow_label)
        
        # Target group
        target_group = QGroupBox("Target (Input)")
        target_layout = QFormLayout(target_group)
        
        self.target_block_combo = QComboBox()
        self.target_block_combo.currentIndexChanged.connect(self._on_target_block_changed)
        target_layout.addRow("Block:", self.target_block_combo)
        
        self.target_port_combo = QComboBox()
        target_layout.addRow("Input Port:", self.target_port_combo)
        
        layout.addWidget(target_group)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setDefault(True)
        self.connect_btn.clicked.connect(self._on_connect)
        self.connect_btn.setEnabled(False)
        button_layout.addWidget(self.connect_btn)
        
        layout.addLayout(button_layout)
    
    def _load_blocks(self):
        """Load blocks into source combo"""
        blocks = self.helper.get_all_blocks()
        
        self.source_block_combo.clear()
        self.source_block_combo.addItem("-- Select Block --", None)
        
        for block in blocks:
            self.source_block_combo.addItem(
                f"{block['name']} ({block['type']})",
                block['id']
            )
    
    def _preselect_source(self, block_id: str, port_name: Optional[str]):
        """Pre-select source block and optionally port"""
        # Find and select the block
        for i in range(self.source_block_combo.count()):
            if self.source_block_combo.itemData(i) == block_id:
                self.source_block_combo.setCurrentIndex(i)
                break
        
        # Select port if specified
        if port_name:
            for i in range(self.source_port_combo.count()):
                if self.source_port_combo.itemData(i) == port_name:
                    self.source_port_combo.setCurrentIndex(i)
                    break
    
    def _on_source_block_changed(self, index: int):
        """Handle source block selection"""
        block_id = self.source_block_combo.currentData()
        
        self.source_port_combo.clear()
        self.target_block_combo.clear()
        self.target_port_combo.clear()
        self._update_status("")
        
        if not block_id:
            return
        
        # Load output ports for selected block
        outputs = self.helper.get_block_outputs(block_id)
        
        if not outputs:
            self.source_port_combo.addItem("-- No outputs --", None)
            return
        
        self.source_port_combo.addItem("-- Select Port --", None)
        for port in outputs:
            self.source_port_combo.addItem(
                f"{port.port_name} ({port.port_type})",
                port.port_name
            )
    
    def _on_source_port_changed(self, index: int):
        """Handle source port selection - update target blocks"""
        source_block_id = self.source_block_combo.currentData()
        source_port_name = self.source_port_combo.currentData()
        
        self.target_block_combo.clear()
        self.target_port_combo.clear()
        self._update_status("")
        
        if not source_block_id or not source_port_name:
            return
        
        # Get compatible target blocks
        compatible = self.helper.get_compatible_targets(source_block_id, source_port_name)
        
        if not compatible:
            self.target_block_combo.addItem("-- No compatible targets --", None)
            self._update_status("No blocks with compatible input ports", error=True)
            return
        
        self.target_block_combo.addItem("-- Select Block --", None)
        for block_info, ports in compatible:
            port_count = len(ports)
            self.target_block_combo.addItem(
                f"{block_info['name']} ({port_count} compatible port{'s' if port_count > 1 else ''})",
                block_info['id']
            )
    
    def _on_target_block_changed(self, index: int):
        """Handle target block selection - update target ports"""
        source_block_id = self.source_block_combo.currentData()
        source_port_name = self.source_port_combo.currentData()
        target_block_id = self.target_block_combo.currentData()
        
        self.target_port_combo.clear()
        self._update_connect_button()
        
        if not target_block_id or not source_block_id or not source_port_name:
            return
        
        # Get compatible input ports for this target block
        compatible = self.helper.get_compatible_targets(source_block_id, source_port_name)
        
        target_ports = []
        for block_info, ports in compatible:
            if block_info['id'] == target_block_id:
                target_ports = ports
                break
        
        if not target_ports:
            self.target_port_combo.addItem("-- No compatible ports --", None)
            return
        
        if len(target_ports) == 1:
            # Auto-select single port
            port = target_ports[0]
            self.target_port_combo.addItem(
                f"{port.port_name} ({port.port_type})",
                port.port_name
            )
            self.target_port_combo.setCurrentIndex(0)
        else:
            self.target_port_combo.addItem("-- Select Port --", None)
            for port in target_ports:
                self.target_port_combo.addItem(
                    f"{port.port_name} ({port.port_type})",
                    port.port_name
                )
        
        self._update_connect_button()
    
    def _update_connect_button(self):
        """Enable/disable connect button based on selections"""
        has_source = bool(self.source_block_combo.currentData())
        has_source_port = bool(self.source_port_combo.currentData())
        has_target = bool(self.target_block_combo.currentData())
        has_target_port = bool(self.target_port_combo.currentData())
        
        can_connect = has_source and has_source_port and has_target and has_target_port
        self.connect_btn.setEnabled(can_connect)
        
        if can_connect:
            self._update_status("Ready to connect", error=False)
    
    def _update_status(self, message: str, error: bool = False):
        """Update status label"""
        self.status_label.setText(message)
        if error:
            self.status_label.setStyleSheet(f"color: {Colors.ACCENT_RED.name()};")
        else:
            self.status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
    
    def _on_connect(self):
        """Create the connection"""
        source_block_id = self.source_block_combo.currentData()
        source_port_name = self.source_port_combo.currentData()
        target_block_id = self.target_block_combo.currentData()
        target_port_name = self.target_port_combo.currentData()
        
        # Final compatibility check
        is_compatible, reason = self.helper.check_compatibility(
            source_block_id, source_port_name,
            target_block_id, target_port_name
        )
        
        if not is_compatible:
            QMessageBox.warning(self, "Cannot Connect", reason)
            return
        
        # Create connection
        result = self.helper.create_connection(
            source_block_id, source_port_name,
            target_block_id, target_port_name
        )
        
        if result.success:
            self.accept()
        else:
            QMessageBox.warning(
                self, 
                "Connection Failed", 
                result.message or "Unknown error"
            )
    
    @classmethod
    def show_dialog(
        cls,
        facade: ApplicationFacade,
        source_block_id: Optional[str] = None,
        source_port_name: Optional[str] = None,
        parent=None
    ) -> bool:
        """
        Show the connection dialog.
        
        Args:
            facade: Application facade
            source_block_id: Optional pre-selected source block
            source_port_name: Optional pre-selected source port
            parent: Parent widget
            
        Returns:
            True if connection was created, False otherwise
        """
        dialog = cls(facade, source_block_id, source_port_name, parent)
        return dialog.exec() == QDialog.DialogCode.Accepted


