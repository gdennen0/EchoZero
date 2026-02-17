"""
Input Filter Dialog

Standalone window for previewing and filtering input data items for a block's input ports.
Can be opened from quick actions, right-click menu, or block panels.
"""
from typing import Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.application.api.application_facade import ApplicationFacade
from src.features.blocks.domain import Block
from ui.qt_gui.block_panels.components.data_filter_widget import InputFilterWidget
from ui.qt_gui.design_system import Colors, Spacing, Typography, border_radius
from src.utils.message import Log


class InputFilterDialog(QDialog):
    """
    Standalone dialog for managing input data filtering on all input ports of a block.
    """
    
    def __init__(self, block_id: str, facade: ApplicationFacade, parent=None):
        super().__init__(parent)
        self.block_id = block_id
        self.facade = facade
        self._filter_widgets: dict = {}  # port_key -> InputFilterWidget
        
        # Get block info
        block_result = self.facade.describe_block(block_id)
        if not block_result.success or not block_result.data:
            Log.error(f"InputFilterDialog: Block {block_id} not found")
            self.reject()
            return
        
        self.block: Block = block_result.data
        
        self._setup_ui()
        self._subscribe_to_events()
    
    def _setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle(f"Input Filtering: {self.block.name}")
        self.setMinimumSize(600, 500)
        self.resize(700, 600)
        
        # Apply dark theme
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(Spacing.MD)
        main_layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        
        # Header
        header_label = QLabel(f"Filter Input Data Items for: {self.block.name}")
        header_label.setFont(Typography.heading_font())
        header_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600; padding-bottom: {Spacing.SM}px;")
        main_layout.addWidget(header_label)
        
        # Description
        desc_label = QLabel(
            "Preview and filter input data items for each input port. "
            "Unchecked items will be skipped during block execution."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding-bottom: {Spacing.MD}px;")
        main_layout.addWidget(desc_label)
        
        # Input ports scroll area
        input_ports = self.block.get_inputs()
        if input_ports and len(input_ports) > 0:
            input_scroll = QScrollArea()
            input_scroll.setWidgetResizable(True)
            input_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            input_scroll.setStyleSheet(f"background-color: transparent; border: none;")
            
            input_container = QWidget()
            input_layout = QVBoxLayout(input_container)
            input_layout.setSpacing(Spacing.MD)
            input_layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
            
            for port_name in input_ports.keys():
                filter_widget = InputFilterWidget(
                    block=self.block,
                    port_name=port_name,
                    facade=self.facade,
                    data_filter_manager=self.facade.data_filter_manager,
                    data_state_service=self.facade.data_state_service,
                    parent=self
                )
                input_layout.addWidget(filter_widget)
                self._filter_widgets[f"input_{port_name}"] = filter_widget
            
            input_layout.addStretch()
            input_scroll.setWidget(input_container)
            main_layout.addWidget(input_scroll)
        else:
            # No input ports - show message
            no_ports_label = QLabel("This block has no input ports.")
            no_ports_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding: 20px;")
            no_ports_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            main_layout.addWidget(no_ports_label)
        
        main_layout.addStretch()
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.BG_LIGHT.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                padding: 8px 24px;
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_BLUE.name()};
            }}
        """)
        button_layout.addWidget(close_button)
        
        main_layout.addLayout(button_layout)
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events for auto-refresh"""
        # InputFilterWidget handles its own event subscriptions
        pass
    
    def closeEvent(self, event):
        """Clean up on close"""
        # InputFilterWidget handles its own cleanup
        super().closeEvent(event)


# Backward compatibility alias
DataFilterDialog = InputFilterDialog
