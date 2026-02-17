"""
Properties Panel

Displays and edits properties of selected items (blocks, events, etc.)
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea,
    QFormLayout, QLineEdit, QPushButton, QGroupBox
)
from PyQt6.QtCore import Qt

from src.application.api.application_facade import ApplicationFacade
from ui.qt_gui.design_system import Colors, ThemeAwareMixin, border_radius
from ui.qt_gui.style_factory import StyleFactory


class PropertiesPanel(ThemeAwareMixin, QWidget):
    """
    Properties panel for inspecting and editing selected items.
    
    Displays different content based on selection type:
    - Block: Block metadata, parameters
    - Event: Event properties (time, duration, classification)
    - Connection: Source/target ports
    """
    
    def __init__(self, facade: ApplicationFacade):
        super().__init__()
        self.facade = facade
        self.current_item = None
        
        self._setup_ui()
        self._subscribe_to_events()
        self._init_theme_aware()
    
    def _setup_ui(self):
        """Setup the properties panel UI"""
        # Set background color on the panel itself to prevent flash during refresh
        self.setStyleSheet(f"background-color: {Colors.BG_DARK.name()};")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("Properties")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # Scroll area for properties
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"background-color: {Colors.BG_DARK.name()}; border: none;")
        
        self.properties_widget = QWidget()
        self.properties_widget.setStyleSheet(f"background-color: {Colors.BG_DARK.name()};")
        self.properties_layout = QVBoxLayout(self.properties_widget)
        self.properties_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll.setWidget(self.properties_widget)
        layout.addWidget(scroll)
        
        # Initial empty state
        self._show_empty_state()
    
    def _show_empty_state(self):
        """Show message when nothing is selected"""
        self._clear_properties()
        
        label = QLabel("No item selected")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        self.properties_layout.addWidget(label)
    
    def _clear_properties(self):
        """Clear all property widgets"""
        while self.properties_layout.count():
            item = self.properties_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def show_block_properties(self, block_id: str):
        """
        Display properties for a block.
        
        Args:
            block_id: Block ID to display
        """
        self._clear_properties()
        
        result = self.facade.describe_block(block_id)
        if not result.success or not result.data:
            return
        
        # Block entity - clean, typed access
        block = result.data
        
        # Block info group
        info_group = QGroupBox("Block Info")
        info_group.setStyleSheet(StyleFactory.group_box())
        info_layout = QFormLayout()
        
        info_layout.addRow("Name:", QLabel(block.name))
        info_layout.addRow("Type:", QLabel(block.type))
        info_layout.addRow("ID:", QLabel(block.id))
        
        info_group.setLayout(info_layout)
        self.properties_layout.addWidget(info_group)
        
        # Position group (from ui_state, not metadata)
        position_result = self.facade.get_ui_state("block_position", block_id)
        if position_result.success and position_result.data:
            position_group = QGroupBox("Position")
            position_group.setStyleSheet(StyleFactory.group_box())
            position_layout = QFormLayout()
            
            pos = position_result.data
            x = pos.get("x", 0)
            y = pos.get("y", 0)
            
            position_layout.addRow("X:", QLabel(f"{x:.1f}"))
            position_layout.addRow("Y:", QLabel(f"{y:.1f}"))
            
            position_group.setLayout(position_layout)
            self.properties_layout.addWidget(position_group)
        
        # Metadata group (filter out UI-specific keys - Phase B fix)
        if block.metadata:
            # Filter out UI-specific keys that shouldn't be shown to users
            UI_KEYS_TO_FILTER = {
                'ui_position', 'x', 'y', 'ui_zoom', 'ui_viewport', 
                'ui_panel_open', 'ui_selected'
            }
            
            # Filter metadata
            display_metadata = {
                key: value for key, value in block.metadata.items()
                if key not in UI_KEYS_TO_FILTER and not key.startswith('ui_')
            }
            
            if display_metadata:  # Only show if there's domain metadata
                metadata_group = QGroupBox("Parameters")
                metadata_group.setStyleSheet(StyleFactory.group_box())
                metadata_layout = QFormLayout()
                
                for key, value in display_metadata.items():
                    metadata_layout.addRow(f"{key}:", QLabel(str(value)))
                
                metadata_group.setLayout(metadata_layout)
                self.properties_layout.addWidget(metadata_group)
        
        # Ports group
        ports_group = QGroupBox("Ports")
        ports_group.setStyleSheet(StyleFactory.group_box())
        ports_layout = QVBoxLayout()
        
        input_ports = block.get_inputs()
        if input_ports:
            ports_layout.addWidget(QLabel("Inputs:"))
            for port_name, port in input_ports.items():
                type_name = port.port_type.name
                ports_layout.addWidget(QLabel(f"  • {port_name} ({type_name})"))
        
        output_ports = block.get_outputs()
        if output_ports:
            ports_layout.addWidget(QLabel("Outputs:"))
            for port_name, port in output_ports.items():
                type_name = port.port_type.name
                ports_layout.addWidget(QLabel(f"  • {port_name} ({type_name})"))
        
        bidirectional_ports = block.get_bidirectional()
        if bidirectional_ports:
            ports_layout.addWidget(QLabel("Bidirectional:"))
            for port_name, port in bidirectional_ports.items():
                type_name = port.port_type.name
                ports_layout.addWidget(QLabel(f"  • {port_name} ({type_name})"))
        
        ports_group.setLayout(ports_layout)
        self.properties_layout.addWidget(ports_group)
        
        self.current_item = ("block", block_id)
    
    def show_event_properties(self, event_data):
        """
        Display properties for an event.
        
        Args:
            event_data: Event object data
        """
        self._clear_properties()
        
        # Event info group
        info_group = QGroupBox("Event Info")
        info_group.setStyleSheet(StyleFactory.group_box())
        info_layout = QFormLayout()
        
        info_layout.addRow("Time:", QLabel(f"{event_data.get('time', 0):.3f} s"))
        info_layout.addRow("Duration:", QLabel(f"{event_data.get('duration', 0):.3f} s"))
        
        if event_data.get('classification'):
            info_layout.addRow("Class:", QLabel(event_data['classification']))
        
        info_group.setLayout(info_layout)
        self.properties_layout.addWidget(info_group)
        
        # Metadata group
        if event_data.get('metadata'):
            metadata_group = QGroupBox("Metadata")
            metadata_group.setStyleSheet(StyleFactory.group_box())
            metadata_layout = QFormLayout()
            
            for key, value in event_data['metadata'].items():
                metadata_layout.addRow(f"{key}:", QLabel(str(value)))
            
            metadata_group.setLayout(metadata_layout)
            self.properties_layout.addWidget(metadata_group)
        
        self.current_item = ("event", event_data)
    
    def clear_selection(self):
        """Clear the properties panel"""
        self._show_empty_state()
        self.current_item = None
    
    def _subscribe_to_events(self):
        """Subscribe to domain events for auto-refresh (Phase B fix)"""
        # Subscribe to block update events to auto-refresh display
        self.facade.event_bus.subscribe('BlockUpdated', self._on_block_updated)
        # Subscribe to UI state changes to refresh position display
        self.facade.event_bus.subscribe('UIStateChanged', self._on_ui_state_changed)
    
    def _on_block_updated(self, event):
        """
        Handle block updated event - refresh if currently displayed.
        
        This ensures properties panel always shows current data (Phase B fix).
        """
        if not self.current_item:
            return
        
        item_type, item_id = self.current_item
        
        # Check if the updated block is currently displayed
        if item_type == 'block':
            updated_block_id = event.data.get('id')
            if updated_block_id == item_id:
                # Refresh the display with fresh data from database
                self.show_block_properties(item_id)
    
    def _on_ui_state_changed(self, event):
        """
        Handle UI state changed event - refresh position if currently displayed block's position changed.
        
        This ensures Properties panel shows current position after user drags a block.
        """
        if not self.current_item:
            return
        
        state_type = event.data.get('state_type')
        entity_id = event.data.get('entity_id')
        
        # If this is a position change for the currently displayed block, refresh
        if state_type == 'block_position' and entity_id:
            item_type, item_id = self.current_item
            if item_type == 'block' and item_id == entity_id:
                # Refresh the display with fresh data from database
                self.show_block_properties(item_id)

