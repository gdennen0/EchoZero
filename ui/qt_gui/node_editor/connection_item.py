"""
Connection Item V2 - Clean connection visualization

Smooth bezier curve between block ports with type-based coloring.
Modern visual effects with glow and smooth rendering.
"""
from PyQt6.QtWidgets import QGraphicsPathItem, QMenu
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QPen, QPainterPath, QColor

from ui.qt_gui.design_system import Colors, Sizes


class ConnectionItem(QGraphicsPathItem):
    """
    Visual representation of a connection between two blocks.
    
    Features:
    - Smooth bezier curve
    - Color matches port data type (audio=blue, event=orange)
    - Hover and selection states with glow
    - Auto-updates when blocks move
    - Special handling for bidirectional (manipulator) ports
    """
    
    def __init__(self, source_block, source_port: str, target_block, target_port: str, connection_data,
                 is_bidirectional: bool = False):
        super().__init__()
        
        self.source_block = source_block
        self.source_port = source_port
        self.target_block = target_block
        self.target_port = target_port
        self.connection_data = connection_data
        self.is_bidirectional = is_bidirectional
        
        # Determine connection color from port type
        self._base_color = self._get_connection_color()
        
        # Check if this is a manipulator connection (bidirectional command port)
        self._is_manipulator = self._check_is_manipulator()
        
        # Visual state
        self._hovered = False
        
        # Setup
        self.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(-1)  # Draw behind blocks
        
        # Initial draw
        self.update_position()
    
    def _get_connection_color(self) -> QColor:
        """Get connection color based on the port type"""
        try:
            # Check bidirectional ports first
            bidirectional_ports = self.source_block.block.get_bidirectional()
            if self.is_bidirectional or self.source_port in bidirectional_ports:
                port = bidirectional_ports.get(self.source_port)
                if port:
                    type_name = port.port_type.name
                    return Colors.get_port_color(type_name)
            
            # Try to get port type from source block outputs
            output_ports = self.source_block.block.get_outputs()
            if output_ports:
                port = output_ports.get(self.source_port)
                if port:
                    type_name = port.port_type.name
                    return Colors.get_port_color(type_name)
        except Exception:
            pass
        return Colors.CONNECTION_NORMAL
    
    def _check_is_manipulator(self) -> bool:
        """Check if this connection is for a manipulator (bidirectional command) port"""
        # Bidirectional connections are always manipulator-style
        if self.is_bidirectional:
            return True
        
        try:
            # Check bidirectional ports
            bidirectional_ports = self.source_block.block.get_bidirectional()
            if bidirectional_ports:
                port = bidirectional_ports.get(self.source_port)
                port_type = port.port_type if port else None
                if port_type:
                    type_name = port_type.name if hasattr(port_type, 'name') else str(port_type)
                    return Colors.is_manipulator_port(type_name)
            
            # Check output ports
            output_ports = self.source_block.block.get_outputs()
            if output_ports:
                port = output_ports.get(self.source_port)
                if port:
                    type_name = port.port_type.name
                    return Colors.is_manipulator_port(type_name)
        except Exception:
            pass
        return False
    
    def update_position(self):
        """Recalculate path based on current block positions"""
        # Get port positions in scene coordinates
        if self.is_bidirectional or self.source_port in getattr(self.source_block.block, 'bidirectional', {}):
            start_pos = self.source_block.get_port_position(self.source_port, is_output=False, is_bidirectional=True)
        else:
            start_pos = self.source_block.get_port_position(self.source_port, is_output=True)
        
        if self.is_bidirectional or self.target_port in getattr(self.target_block.block, 'bidirectional', {}):
            end_pos = self.target_block.get_port_position(self.target_port, is_output=False, is_bidirectional=True)
        else:
            end_pos = self.target_block.get_port_position(self.target_port, is_output=False)
        
        # Use Qt's built-in bezier curve for all connections
        path = QPainterPath(start_pos)
        ctrl_offset = max(abs(end_pos.x() - start_pos.x()) * 0.5, 50)
        
        if self.is_bidirectional:
            # Vertical bezier for bottom ports connecting to bottom ports
            ctrl1 = QPointF(start_pos.x(), start_pos.y() + ctrl_offset)
            ctrl2 = QPointF(end_pos.x(), end_pos.y() + ctrl_offset)
        else:
            # Horizontal bezier for regular connections
            ctrl1 = QPointF(start_pos.x() + ctrl_offset, start_pos.y())
            ctrl2 = QPointF(end_pos.x() - ctrl_offset, end_pos.y())
        
        path.cubicTo(ctrl1, ctrl2, end_pos)
        self.setPath(path)
        self._update_pen()
    
    def _update_pen(self):
        """Update pen based on state"""
        if self.isSelected():
            color = Colors.CONNECTION_SELECTED
            width = Sizes.CONNECTION_WIDTH_SELECTED
        elif self._hovered:
            color = self._base_color.lighter(140)
            width = Sizes.CONNECTION_WIDTH_HOVER
        else:
            color = self._base_color
            width = Sizes.CONNECTION_WIDTH
        
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        
        # Use dashed line style for manipulator (bidirectional) connections
        if self._is_manipulator:
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setDashPattern([8, 4])  # 8 pixels dash, 4 pixels gap
        
        self.setPen(pen)
    
    def paint(self, painter: QPainter, option, widget):
        """Draw the connection with modern visual effects"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw glow/outline for selected or hovered connections
        if self.isSelected() or self._hovered:
            glow_path = QPainterPath(self.path())
            glow_color = self._base_color if self._hovered else Colors.CONNECTION_SELECTED
            glow_pen = QPen(QColor(glow_color.red(), glow_color.green(), glow_color.blue(), 100), 
                           (Sizes.CONNECTION_WIDTH + 2) * 2)
            glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            glow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(glow_pen)
            painter.drawPath(glow_path)
        
        # Draw main connection line
        super().paint(painter, option, widget)
    
    def hoverEnterEvent(self, event):
        self._hovered = True
        self._update_pen()
        self.update()  # Trigger repaint for glow effect
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        self._hovered = False
        self._update_pen()
        self.update()  # Trigger repaint to remove glow
        super().hoverLeaveEvent(event)
    
    def itemChange(self, change, value):
        """Handle selection changes"""
        if change == QGraphicsPathItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._update_pen()
            self.update()  # Trigger repaint for selection glow
        return super().itemChange(change, value)
    
    def contextMenuEvent(self, event):
        """Handle right-click context menu"""
        menu = QMenu()
        
        delete_action = menu.addAction("Delete Connection")
        delete_action.triggered.connect(self._delete_connection)
        
        menu.exec(event.screenPos())
        event.accept()
    
    def _delete_connection(self):
        """Delete this connection"""
        if not self.connection_data or not hasattr(self.connection_data, 'id'):
            return
        
        # Get the scene to access facade and undo_stack
        scene = self.scene()
        if not scene or not hasattr(scene, 'facade'):
            return
        
        from src.application.commands import DeleteConnectionCommand
        
        # Use command bus for undoable deletion
        cmd = DeleteConnectionCommand(scene.facade, self.connection_data.id)
        scene.facade.command_bus.execute(cmd)
        
        # Scene will refresh via event bus

