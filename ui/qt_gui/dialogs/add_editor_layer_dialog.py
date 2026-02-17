"""
Add Editor Layer Dialog

Dialog for selecting an Editor layer to add to synced layers.
Shows available Editor layers (excluding already synced ones).
Each item has a + button to add it.
"""

from typing import List, Dict, Any, Optional, Callable
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from ui.qt_gui.design_system import Colors, Spacing, Typography, ThemeAwareMixin, border_radius
from src.utils.message import Log


class AddEditorLayerDialog(ThemeAwareMixin, QDialog):
    """
    Dialog for adding Editor layers to synced layers.
    
    Shows available Editor layers (not already synced) with + button per item.
    """
    
    layer_added = pyqtSignal(str, str)  # Emits (layer_id, block_id) when layer is added
    
    def __init__(
        self,
        available_layers: List[Dict[str, Any]],  # List of layer dicts from controller.get_available_editor_layers()
        parent=None
    ):
        super().__init__(parent)
        self.available_layers = available_layers
        self.selected_layer: Optional[Dict[str, Any]] = None
        
        self.setWindowTitle("Add Editor Layer")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self.setModal(True)
        
        self._setup_ui()
        self._init_theme_aware()
    
    def _setup_ui(self):
        """Setup dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        
        # Header
        header_label = QLabel("Select an Editor layer to sync:")
        header_label.setFont(Typography.heading_font())
        header_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600; padding-bottom: {Spacing.SM}px;")
        layout.addWidget(header_label)
        
        # Description
        desc_label = QLabel("Available layers that are not yet synced:")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding-bottom: {Spacing.MD}px;")
        layout.addWidget(desc_label)
        
        # Scroll area with layer list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"background-color: transparent; border: none;")
        
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(Spacing.SM)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add layer rows
        if self.available_layers:
            grouped = self._group_layers_in_order(self.available_layers)
            for group_label, group_layers in grouped:
                header = QLabel(group_label)
                header.setStyleSheet(
                    f"color: {Colors.TEXT_SECONDARY.name()}; font-weight: 600; padding: {Spacing.SM}px;"
                )
                container_layout.addWidget(header)
                for layer_data in group_layers:
                    row = self._create_layer_row(layer_data)
                    container_layout.addWidget(row)
        else:
            # No layers available
            empty_label = QLabel("No available Editor layers.")
            empty_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding: {Spacing.MD}px;")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(empty_label)
        
        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.setMinimumWidth(100)
        close_button.clicked.connect(self.accept)
        close_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.BG_LIGHT.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: {Colors.HOVER.name()};
            }}
        """)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
    
    def _create_layer_row(self, layer_data: Dict[str, Any]) -> QWidget:
        """Create a row widget for a layer"""
        row = QFrame()
        row.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: {Spacing.SM}px;
            }}
        """)
        
        row_layout = QHBoxLayout(row)
        row_layout.setSpacing(Spacing.MD)
        row_layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        
        # Layer name
        layer_id = layer_data.get('name', '')
        layer_label = QLabel(layer_id)
        layer_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()};")
        row_layout.addWidget(layer_label, 1)
        
        # + button
        add_button = QPushButton("+")
        add_button.setFixedWidth(30)
        add_button.setFixedHeight(30)
        add_button.clicked.connect(lambda: self._on_add_layer(layer_data))
        add_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT_BLUE.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: none;
                border-radius: {border_radius(4)};
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_BLUE.lighter(110).name()};
            }}
            QPushButton:pressed {{
                background-color: {Colors.ACCENT_BLUE.darker(110).name()};
            }}
        """)
        row_layout.addWidget(add_button)
        
        return row
    
    def _on_add_layer(self, layer_data: Dict[str, Any]):
        """Handle + button click"""
        layer_id = layer_data.get('name', '')
        block_id = layer_data.get('block_id', '')
        
        if layer_id and block_id:
            self.selected_layer = layer_data
            self.layer_added.emit(layer_id, block_id)
            self.accept()  # Close dialog after adding

    def _group_layers_in_order(self, layers: List[Dict[str, Any]]):
        """Group layers by group_id while preserving original order."""
        grouped_entries = []
        group_index = {}
        group_name_counts = {}
        for layer in layers:
            group_id = layer.get('group_id')
            if not group_id:
                raise ValueError(f"Layer missing group_id: {layer}")
            group_name = layer.get('group_name') or group_id
            group_key = group_id
            if group_key not in group_index:
                group_index[group_key] = len(grouped_entries)
                grouped_entries.append({
                    "group_key": group_key,
                    "group_name": group_name,
                    "layers": [],
                })
                group_name_counts[group_name] = group_name_counts.get(group_name, 0) + 1
            grouped_entries[group_index[group_key]]["layers"].append(layer)

        grouped = []
        for entry in grouped_entries:
            label = entry["group_name"]
            if group_name_counts.get(entry["group_name"], 0) > 1:
                suffix = entry["group_key"]
                if isinstance(suffix, str):
                    suffix = suffix[:8]
                label = f"{label} ({suffix})"
            grouped.append((label, entry["layers"]))
        return grouped
