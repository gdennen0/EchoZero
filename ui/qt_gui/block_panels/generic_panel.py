"""
Generic block panel for blocks without custom UI.

Provides basic panel with metadata display and common actions.
This serves as a placeholder until block-specific panels are implemented.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit
from PyQt6.QtCore import Qt

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing, border_radius
import json


class GenericBlockPanel(BlockPanelBase):
    """Generic panel for blocks without specific UI"""
    
    def create_content_widget(self) -> QWidget:
        """Create generic content showing block metadata"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)
        
        # Info message
        info_label = QLabel(
            "This block doesn't have a custom configuration panel yet.\n"
            "Use the Actions panel to configure settings."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding: {Spacing.MD}px;")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        # Metadata display
        metadata_label = QLabel("Block Metadata:")
        metadata_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600;")
        layout.addWidget(metadata_label)
        
        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setMaximumHeight(200)
        self.metadata_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.BG_LIGHT.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: {Spacing.SM}px;
                color: {Colors.TEXT_SECONDARY.name()};
                font-family: monospace;
                font-size: 11px;
            }}
        """)
        layout.addWidget(self.metadata_text)
        
        layout.addStretch()
        
        return widget
    
    def refresh(self):
        """Update metadata display"""
        if not self.block:
            return
        
        # Display block metadata as formatted JSON
        try:
            metadata_str = json.dumps(self.block.metadata, indent=2)
            self.metadata_text.setPlainText(metadata_str)
        except Exception as e:
            self.metadata_text.setPlainText(f"Error displaying metadata: {e}")
        
        self.set_status_message("Settings loaded")


# Register stub panels for blocks without custom implementations yet
@register_block_panel("TranscribeLib")
class TranscribeLibPanel(GenericBlockPanel):
    """TranscribeLib panel - generic until custom UI needed"""
    pass


@register_block_panel("Editor")
class EditorPanel(GenericBlockPanel):
    """Editor panel - generic until custom UI needed"""
    pass


@register_block_panel("EditorV2")
class EditorV2Panel(GenericBlockPanel):
    """EditorV2 panel - generic until custom UI needed"""
    pass


@register_block_panel("CommandSequencer")
class CommandSequencerPanel(GenericBlockPanel):
    """CommandSequencer panel - generic until custom UI needed"""
    pass


@register_block_panel("LoadMultiple")
class LoadMultiplePanel(GenericBlockPanel):
    """LoadMultiple panel - generic until custom UI needed"""
    pass

