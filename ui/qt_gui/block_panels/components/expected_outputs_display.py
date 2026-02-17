"""
Expected Outputs Display Widget

Simple read-only widget that displays what a block expects to output.
Uses processor.get_expected_outputs() as the single source of truth.
"""
from typing import List, Dict
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox
from PyQt6.QtCore import Qt

from src.features.blocks.domain import Block
from src.application.api.application_facade import ApplicationFacade
from ui.qt_gui.design_system import Colors, Spacing, border_radius
from src.utils.message import Log


class ExpectedOutputsDisplay(QWidget):
    """
    Simple read-only widget that displays expected outputs for a block.
    
    Shows what a block will output based on processor.get_expected_outputs(block).
    No filtering, no interaction - just display.
    """
    
    def __init__(
        self,
        block: Block,
        facade: ApplicationFacade,
        parent=None
    ):
        super().__init__(parent)
        
        self.block = block
        self.facade = facade
        
        self._setup_ui()
        self._update_display()
    
    def _setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.SM)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header label
        header_label = QLabel("Expected Outputs")
        header_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600; font-size: 11pt;")
        layout.addWidget(header_label)
        
        # Container for port displays
        self.ports_container = QWidget()
        self.ports_layout = QVBoxLayout(self.ports_container)
        self.ports_layout.setSpacing(Spacing.SM)
        self.ports_layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(self.ports_container)
        layout.addStretch()
    
    def _update_display(self):
        """Update display with current expected outputs"""
        try:
            # Clear existing displays
            while self.ports_layout.count():
                item = self.ports_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Get processor and expected outputs using ExpectedOutputsService
            processor = self.facade.execution_engine.get_processor(self.block)
            if not processor:
                self._show_no_processor_message()
                return
            
            # Use ExpectedOutputsService if available, otherwise fall back to processor
            expected_outputs_service = getattr(self.facade, 'expected_outputs_service', None)
            if expected_outputs_service:
                all_expected = expected_outputs_service.calculate_expected_outputs(
                    self.block,
                    processor,
                    facade=self.facade
                )
            else:
                # Fallback: use processor directly
                all_expected = processor.get_expected_outputs(self.block)
            
            if not all_expected:
                self._show_no_outputs_message()
                return
            
            # Display each output port
            has_empty_outputs = False
            for port_name in sorted(all_expected.keys()):
                expected_names = all_expected.get(port_name, [])
                if expected_names == []:
                    # Empty list means all filters are off
                    has_empty_outputs = True
                    self._create_port_display(port_name, [], is_empty_filter=True)
                elif expected_names:
                    self._create_port_display(port_name, expected_names)
            
            if self.ports_layout.count() == 0:
                self._show_no_outputs_message()
                
        except Exception as e:
            Log.error(f"ExpectedOutputsDisplay: Error updating display: {e}")
            import traceback
            traceback.print_exc()
            self._show_error(str(e))
    
    
    def _create_port_display(self, port_name: str, expected_names: List[str], is_empty_filter: bool = False):
        """Create display for a single output port"""
        # Group box for this port
        group = QGroupBox(f"{port_name} port")
        group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: {Colors.TEXT_SECONDARY.name()};
            }}
        """)
        
        port_layout = QVBoxLayout(group)
        port_layout.setSpacing(Spacing.XS)
        port_layout.setContentsMargins(8, 8, 8, 8)
        
        # Display expected output names
        if expected_names:
            # Extract item names for display (e.g., "vocals" from "audio:vocals")
            display_names = []
            for output_name in sorted(expected_names):
                if ':' in output_name:
                    _, item_name = output_name.split(':', 1)
                    display_name = item_name.replace('_', ' ').title()
                else:
                    display_name = output_name
                display_names.append(display_name)
            
            names_text = ", ".join(display_names)
            names_label = QLabel(names_text)
            names_label.setWordWrap(True)
            names_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; padding: 4px;")
            port_layout.addWidget(names_label)
        elif is_empty_filter:
            # Empty list explicitly set means all filters are off
            no_outputs_label = QLabel("No outputs (all filters off)")
            no_outputs_label.setWordWrap(True)
            no_outputs_label.setStyleSheet(f"color: {Colors.ACCENT_ORANGE.name()}; padding: 4px; font-style: italic;")
            port_layout.addWidget(no_outputs_label)
        else:
            # No outputs defined (different from empty filter)
            no_outputs_label = QLabel("No expected outputs defined")
            no_outputs_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding: 4px;")
            port_layout.addWidget(no_outputs_label)
        
        self.ports_layout.addWidget(group)
    
    def _show_no_processor_message(self):
        """Show message when processor not available"""
        label = QLabel("Processor not available for this block type.")
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding: 8px;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ports_layout.addWidget(label)
    
    def _show_no_outputs_message(self):
        """Show message when no expected outputs"""
        label = QLabel("No expected outputs defined for this block.")
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding: 8px;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ports_layout.addWidget(label)
    
    def _show_error(self, error_msg: str):
        """Show error message"""
        label = QLabel(f"Error: {error_msg}")
        label.setStyleSheet(f"color: {Colors.ACCENT_RED.name()};")
        self.ports_layout.addWidget(label)
    
    def refresh(self):
        """Refresh the display with current block data"""
        # Reload block from facade
        block_result = self.facade.describe_block(self.block.id)
        if block_result.success and block_result.data:
            self.block = block_result.data
        self._update_display()

