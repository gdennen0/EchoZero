"""
Block Status Widget

Reusable status indicator widget for blocks that automatically subscribes to
block changes and updates the displayed status.
"""
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.features.blocks.domain import BlockStatus
from src.application.events.events import BlockChanged
from ui.qt_gui.design_system import Colors, Typography
from src.utils.message import Log


class BlockStatusWidget(QWidget):
    """
    Reusable widget for displaying block status.
    
    Auto-subscribes to BlockChanged events and updates automatically.
    Can be used in panels, node editor, and properties panel.
    """
    
    def __init__(self, block_id: str, facade, parent=None, compact: bool = False):
        """
        Initialize block status widget.
        
        Args:
            block_id: Block identifier
            facade: ApplicationFacade instance
            parent: Parent widget
            compact: If True, use compact display (dot only), otherwise show dot + text
        """
        super().__init__(parent)
        self.block_id = block_id
        self.facade = facade
        self.compact = compact
        self._current_status: BlockStatus = None
        
        self._setup_ui()
        self._subscribe_to_events()
        self._update_status()
    
    def _setup_ui(self):
        """Setup the widget UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Status indicator (dot)
        self.status_dot = QLabel("●")
        status_font = QFont()
        status_font.setPointSize(10)
        self.status_dot.setFont(status_font)
        self.status_dot.setStyleSheet(f"color: {Colors.ACCENT_BLUE.name()};")
        layout.addWidget(self.status_dot)
        
        # Status text (only if not compact)
        if not self.compact:
            self.status_text = QLabel("Ready")
            status_text_font = Typography.default_font()
            status_text_font.setPointSize(11)
            self.status_text.setFont(status_text_font)
            self.status_text.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
            layout.addWidget(self.status_text)
        else:
            self.status_text = None
        
        layout.addStretch()
    
    def _subscribe_to_events(self):
        """Subscribe to block change events"""
        if not self.facade or not hasattr(self.facade, 'event_bus') or not self.facade.event_bus:
            return
        
        self.facade.event_bus.subscribe("BlockChanged", self._on_block_changed)
    
    def _on_block_changed(self, event: BlockChanged):
        """Handle block changed event"""
        if not event.data:
            return
        
        # Check if this event is for our block
        event_block_id = event.data.get('block_id') or event.data.get('id')
        if event_block_id == self.block_id:
            self._update_status()
    
    def _update_status(self):
        """Update the displayed status"""
        if not self.facade or not hasattr(self.facade, 'block_status_service') or not self.facade.block_status_service:
            # No status service - show default
            self._set_status_default()
            return
        
        if not self.block_id:
            self._set_status_default()
            return
        
        try:
            status = self.facade.block_status_service.get_block_status(self.block_id, self.facade)
            self._current_status = status
            
            # Update dot color
            color = status.color
            self.status_dot.setStyleSheet(f"color: {color};")
            
            # Update text and tooltip
            display_name = status.display_name
            tooltip = display_name
            if status.message:
                tooltip = f"{display_name}: {status.message}"
            
            if self.status_text:
                self.status_text.setText(display_name)
                self.status_text.setToolTip(tooltip)
            else:
                self.status_dot.setToolTip(tooltip)
        except Exception as e:
            Log.warning(f"BlockStatusWidget: Failed to update status for block {self.block_id}: {e}")
            self._set_status_default()
    
    def _set_status_default(self):
        """Set default status display"""
        self.status_dot.setStyleSheet(f"color: {Colors.ACCENT_BLUE.name()};")
        if self.status_text:
            self.status_text.setText("Ready")
            self.status_text.setToolTip("Ready")
        else:
            self.status_dot.setToolTip("Ready")
