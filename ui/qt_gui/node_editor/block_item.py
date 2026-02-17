"""
Block Item - Minimal, flat block visualization

Features:
- Sharp corners, flat fills, no shadows or gradients
- Clear title bar with block type color
- Simple solid port circles with clean labels
- Double-click to open panel
- Right-click context menu with quick actions
"""
from typing import Optional, List, Dict, TYPE_CHECKING
from PyQt6.QtWidgets import (
    QGraphicsItem, QMenu, QApplication,
    QInputDialog, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath,
    QFont, QFontMetrics
)

from src.features.blocks.domain import Block
from src.shared.domain.data_state import DataState
from src.application.api.application_facade import ApplicationFacade
from src.application.commands import DeleteBlockCommand, RenameBlockCommand, MoveBlockCommand, DuplicateBlockCommand
from ui.qt_gui.design_system import Colors, Sizes, Typography, Spacing, border_radius
from ui.qt_gui.node_editor.status_dot import StatusDotRenderer
from src.utils.message import Log
from src.utils.settings import app_settings

if TYPE_CHECKING:
    from PyQt6.QtGui import QUndoStack
    from src.features.blocks.domain import BlockStatus


class BlockItemSignals(QObject):
    """Signals for BlockItem (QGraphicsItem can't have signals)"""
    position_changed = pyqtSignal(str, QPointF)  # block_id, position
    selected = pyqtSignal(str)  # block_id
    double_clicked = pyqtSignal(str)  # block_id - for opening panel
    port_clicked = pyqtSignal(str, str, bool, bool, QPointF)  # block_id, port_name, is_output, is_bidirectional, scene_pos


class BlockItem(QGraphicsItem):
    """
    Visual representation of a Block in the node editor.
    
    Minimal design:
    - Flat rectangle with block type color header bar
    - Clear block name, no text shadows
    - Simple solid port circles with clean labels
    - Subtle hover/selection border changes
    """
    
    def __init__(self, block: Block, facade: ApplicationFacade, undo_stack: Optional["QUndoStack"] = None):
        super().__init__()
        self.block = block
        self.facade = facade
        self.undo_stack = undo_stack
        self.signals = BlockItemSignals()
        
        
        # Visual state
        self._hovered = False
        self._color = Colors.get_block_color(block.type)
        
        # Calculate dynamic height based on ports
        self._calculate_dimensions()
        
        # Setup item flags
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        
        # Connections attached to this block
        self.connections: List[QGraphicsItem] = []
        
        # Double-click detection
        self._click_timer = None
        self._click_count = 0
        
        # Drag tracking for undo support
        self._drag_start_pos: Optional[QPointF] = None
        self._is_dragging = False
        
        # Data state tracking
        self._data_state: Optional[DataState] = None
        
        # Block status tracking (for blocks that use BlockStatus instead of DataState)
        self._block_status: Optional["BlockStatus"] = None
        
        # Subscribe to events for data state updates
        self._subscribe_to_events()
        
        # Update status on initialization (after scene is set up)
        # Use QTimer with delay to ensure this happens after the item is fully added to scene
        # and facade/services are fully initialized
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, lambda: self._safe_update_status_if_needed())
    
    def _calculate_dimensions(self):
        """Calculate block dimensions based on number of ports"""
        input_ports = self.block.get_inputs()
        output_ports = self.block.get_outputs()
        bidirectional_ports = self.block.get_bidirectional()
        
        num_inputs = len(input_ports)
        num_outputs = len(output_ports)
        num_bidirectional = len(bidirectional_ports)
        # ShowManager blocks render bidirectional ports on the right side
        bidirectional_on_right = (self.block.type == "ShowManager")
        
        # Calculate max ports: inputs + bidirectional (if on left), outputs + bidirectional (if on right)
        num_ports_left = num_inputs + (num_bidirectional if not bidirectional_on_right else 0)
        num_ports_right = num_outputs + (num_bidirectional if bidirectional_on_right else 0)
        max_ports = max(num_ports_left, num_ports_right, 1)
        
        # Height = header + port zones + padding
        port_zone_height = max_ports * Sizes.PORT_ZONE_HEIGHT
        self._height = max(
            Sizes.BLOCK_MIN_HEIGHT,
            Sizes.BLOCK_HEADER_HEIGHT + port_zone_height + Spacing.SM * 2
        )
        
        self._width = Sizes.BLOCK_WIDTH
    
    def boundingRect(self) -> QRectF:
        """Define the bounding box"""
        # Extra padding for port labels that extend outside
        padding = 4
        
        return QRectF(
            -self._width / 2 - padding,
            -self._height / 2 - padding,
            self._width + padding * 2,
            self._height + padding * 2
        )
    
    def shape(self) -> QPainterPath:
        """Define the clickable shape (tighter than bounding rect)"""
        path = QPainterPath()
        path.addRect(
            -self._width / 2,
            -self._height / 2,
            self._width,
            self._height
        )
        return path
    
    def paint(self, painter: QPainter, option, widget):
        """Draw the block with clean, minimal visuals"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        
        rect = QRectF(
            -self._width / 2,
            -self._height / 2,
            self._width,
            self._height
        )
        
        is_selected = self.isSelected()
        is_hovered = self._hovered
        
        # -- Body: flat fill, no shadows, no gradients --
        body_path = QPainterPath()
        body_path.addRect(rect)
        
        if is_selected:
            body_color = Colors.BG_MEDIUM.lighter(112)
        elif is_hovered:
            body_color = Colors.BG_MEDIUM.lighter(106)
        else:
            body_color = Colors.BG_MEDIUM
        
        painter.fillPath(body_path, QBrush(body_color))
        
        # -- Border: single clean line --
        if is_selected:
            border_color = Colors.ACCENT_YELLOW
            border_width = 1.5
        elif is_hovered:
            border_color = self._color.lighter(130)
            border_width = 1.0
        else:
            border_color = QColor(255, 255, 255, 18)
            border_width = 1.0
        
        painter.setPen(QPen(border_color, border_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        
        # -- Header: flat color bar, full width --
        header_rect = QRectF(
            rect.left(),
            rect.top(),
            rect.width(),
            Sizes.BLOCK_HEADER_HEIGHT
        )
        painter.fillRect(header_rect, QBrush(self._color))
        
        # -- Block name: clear, prominent --
        name_font = Typography.default_font()
        name_font.setWeight(QFont.Weight.DemiBold)
        name_font.setPixelSize(12)
        painter.setFont(name_font)
        
        name_rect = header_rect.adjusted(Spacing.SM, 0, -Spacing.SM, 0)
        fm = QFontMetrics(name_font)
        elided_name = fm.elidedText(self.block.name, Qt.TextElideMode.ElideRight, int(name_rect.width()))
        
        text_color = QColor(255, 255, 255) if self._color.lightness() < 150 else QColor(20, 20, 25)
        painter.setPen(QPen(text_color))
        painter.drawText(
            name_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            elided_name
        )
        
        # -- Status dot (top-right of header) --
        if self._block_status:
            StatusDotRenderer.draw_status_dot(painter, header_rect, self._block_status)
        
        # -- Ports --
        self._draw_ports(painter, rect)
    
    
    def _draw_ports(self, painter: QPainter, rect: QRectF):
        """Draw input, output, and bidirectional ports -- minimal design"""
        port_start_y = rect.top() + Sizes.BLOCK_HEADER_HEIGHT + Spacing.SM
        
        port_font = Typography.default_font()
        port_font.setPixelSize(10)
        painter.setFont(port_font)
        
        port_radius = Sizes.PORT_RADIUS
        
        # Input ports (left side)
        input_ports_dict = self.block.get_inputs()
        left_ports = [(name, port.port_type) for name, port in input_ports_dict.items()]
        
        # Bidirectional ports
        bidirectional_ports_dict = self.block.get_bidirectional()
        bidirectional_ports = [(name, port.port_type) for name, port in bidirectional_ports_dict.items()]
        bidirectional_on_right = (self.block.type == "ShowManager")
        
        # -- Bidirectional on left --
        if bidirectional_ports and not bidirectional_on_right:
            for i, (port_name, port_type) in enumerate(bidirectional_ports):
                y = port_start_y + (i + 0.5) * Sizes.PORT_ZONE_HEIGHT
                port_center = QPointF(rect.left(), y)
                painter.setBrush(QBrush(Colors.PORT_MANIPULATOR))
                painter.setPen(QPen(Qt.PenStyle.NoPen))
                painter.drawEllipse(port_center, port_radius, port_radius)
        
        # -- Input ports (after bidirectional if on left) --
        bidir_count_on_left = len(bidirectional_ports) if not bidirectional_on_right else 0
        if left_ports:
            for i, (port_name, port_type) in enumerate(left_ports):
                y = port_start_y + ((bidir_count_on_left + i) + 0.5) * Sizes.PORT_ZONE_HEIGHT
                port_center = QPointF(rect.left(), y)
                
                type_name = port_type.name if hasattr(port_type, 'name') else str(port_type)
                port_color = Colors.get_port_color(type_name)
                
                # Single solid circle
                painter.setBrush(QBrush(port_color))
                painter.setPen(QPen(Qt.PenStyle.NoPen))
                painter.drawEllipse(port_center, port_radius, port_radius)
                
                # Port label
                label_x = rect.left() + Sizes.PORT_LABEL_OFFSET
                label_rect = QRectF(label_x, y - 8, 70, 16)
                painter.setPen(QPen(Colors.TEXT_SECONDARY))
                painter.drawText(
                    label_rect,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    port_name
                )
        
        # Output ports (right side)
        output_ports_dict = self.block.get_outputs()
        right_ports = [(name, port.port_type) for name, port in output_ports_dict.items()]
        
        # -- Bidirectional on right --
        if bidirectional_ports and bidirectional_on_right:
            for i, (port_name, port_type) in enumerate(bidirectional_ports):
                y = port_start_y + (i + 0.5) * Sizes.PORT_ZONE_HEIGHT
                port_center = QPointF(rect.right(), y)
                painter.setBrush(QBrush(Colors.PORT_MANIPULATOR))
                painter.setPen(QPen(Qt.PenStyle.NoPen))
                painter.drawEllipse(port_center, port_radius, port_radius)
        
        # -- Output ports (after bidirectional if on right) --
        bidir_count_on_right = len(bidirectional_ports) if bidirectional_on_right else 0
        if right_ports:
            for i, (port_name, port_type) in enumerate(right_ports):
                y = port_start_y + ((bidir_count_on_right + i) + 0.5) * Sizes.PORT_ZONE_HEIGHT
                port_center = QPointF(rect.right(), y)
                
                type_name = port_type.name if hasattr(port_type, 'name') else str(port_type)
                port_color = Colors.get_port_color(type_name)
                
                # Single solid circle
                painter.setBrush(QBrush(port_color))
                painter.setPen(QPen(Qt.PenStyle.NoPen))
                painter.drawEllipse(port_center, port_radius, port_radius)
                
                # Port label
                label_x = rect.right() - Sizes.PORT_LABEL_OFFSET - 70
                label_rect = QRectF(label_x, y - 8, 70, 16)
                painter.setPen(QPen(Colors.TEXT_SECONDARY))
                painter.drawText(
                    label_rect,
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    port_name
                )
    
    def _draw_bidirectional_ports(self, painter: QPainter, rect: QRectF):
        """Bidirectional ports are now drawn in _draw_ports() method - this is kept for compatibility"""
        pass
    
    def get_port_position(self, port_name: str, is_output: bool = False, is_bidirectional: bool = False) -> QPointF:
        """Get the scene position of a port
        
        Args:
            port_name: Name of the port
            is_output: True if this is an output port (for bidirectional, returns right side)
            is_bidirectional: True if this is a bidirectional port
        
        Returns:
            Scene position of the port
        """
        rect = QRectF(
            -self._width / 2,
            -self._height / 2,
            self._width,
            self._height
        )
        
        port_start_y = rect.top() + Sizes.BLOCK_HEADER_HEIGHT + Spacing.SM
        
        # Check if this is a bidirectional port
        bidirectional_ports = self.block.get_bidirectional()
        if is_bidirectional or port_name in bidirectional_ports:
            # Determine which side bidirectional ports are on
            bidirectional_on_right = (self.block.type == "ShowManager")
            
            # Get port index
            bidirectional_list = list(bidirectional_ports.keys())
            port_index = bidirectional_list.index(port_name)
            
            if bidirectional_on_right:
                # Right side: bidirectional ports come FIRST (position 0, 1, 2...)
                y = port_start_y + (port_index + 0.5) * Sizes.PORT_ZONE_HEIGHT
                local_pos = QPointF(rect.right(), y)
            else:
                # Left side: bidirectional ports come FIRST (position 0, 1, 2...)
                y = port_start_y + (port_index + 0.5) * Sizes.PORT_ZONE_HEIGHT
                local_pos = QPointF(rect.left(), y)
        else:
            # Regular input/output ports
            bidirectional_on_right = (self.block.type == "ShowManager")
            num_bidirectional = len(bidirectional_ports)
            output_ports = self.block.get_outputs()
            input_ports = self.block.get_inputs()
            
            if is_output and port_name in output_ports:
                # Right side: output ports come AFTER bidirectional ports (if on right)
                num_bidirectional_on_right = num_bidirectional if bidirectional_on_right else 0
                port_index = num_bidirectional_on_right + list(output_ports.keys()).index(port_name)
                y = port_start_y + (port_index + 0.5) * Sizes.PORT_ZONE_HEIGHT
                local_pos = QPointF(rect.right(), y)
            elif not is_output and port_name in input_ports:
                # Left side: input ports come AFTER bidirectional ports (if on left)
                num_bidirectional_on_left = num_bidirectional if not bidirectional_on_right else 0
                port_index = num_bidirectional_on_left + list(input_ports.keys()).index(port_name)
                y = port_start_y + (port_index + 0.5) * Sizes.PORT_ZONE_HEIGHT
                local_pos = QPointF(rect.left(), y)
            else:
                local_pos = QPointF(0, 0)
        
        return self.mapToScene(local_pos)
    
    def _is_valid(self) -> bool:
        """Check if this BlockItem is still valid (not deleted)"""
        try:
            # Try to access scene() - will raise RuntimeError if C++ object is deleted
            self.scene()
            return True
        except RuntimeError:
            # C++ object has been deleted
            return False
    
    def _update_status(self):
        """Update status for this block (DataState or BlockStatus depending on block type)"""
        # Check if item is still valid (might have been deleted)
        if not self._is_valid():
            return
        
        # Always use BlockStatusService (unified system for all blocks)
        self._update_block_status()
    
    def _update_block_status(self):
        """
        Update block status using BlockStatusService (unified system for all blocks).
        
        Uses BlockStatusService as the centralized status source, same as BlockStatusDot
        widget used in panels. Both panels and nodes now use the same status service
        for consistency. Subscribes to StatusChanged events for immediate updates.
        """
        if not self._is_valid():
            return
        
        if not self.facade or not hasattr(self.facade, 'block_status_service') or not self.facade.block_status_service:
            old_status = self._block_status
            self._block_status = None
            if old_status is not None:
                self._force_repaint()
            return
        
        try:
            from src.features.blocks.domain import BlockStatus
            
            
            # Force recalculation to ensure we get the latest status
            # This bypasses cache and ensures StatusChanged events are published if status changed
            status = self.facade.block_status_service.get_block_status(self.block.id, self.facade, force_recalculate=True)
            
            
            if status:
                old_status = self._block_status
                new_status = status
                
                # Always update status and repaint, even if status didn't change
                # This ensures UI is in sync with actual status
                self._block_status = new_status
                
                # Update tooltip with diagnostic information
                self._update_status_tooltip(new_status)
                
                self._force_repaint()
                
                if old_status != new_status:
                    Log.debug(
                        f"BlockItem: Updated block status for '{self.block.name}' ({self.block.id}): "
                        f"{old_status.level.name if old_status else None} -> {new_status.level.name} (color: {new_status.color})"
                    )
                else:
                    Log.debug(
                        f"BlockItem: Block status unchanged for '{self.block.name}' ({self.block.id}): {new_status.level.name}"
                    )
            else:
                # No status available - clear indicator
                old_status = self._block_status
                self._block_status = None
                if old_status is not None:
                    # Status was cleared - repaint
                    self._force_repaint()
                    Log.debug(
                        f"BlockItem: Cleared block status for '{self.block.name}' ({self.block.id})"
                    )
        except Exception as e:
            Log.warning(f"BlockItem: Failed to update block status for '{self.block.name}': {e}", exc_info=True)
            old_status = self._block_status
            self._block_status = None
            if old_status is not None:
                self._force_repaint()
    
    def _update_data_state(self):
        """Update data state for this block"""
        # Check if item is still valid (might have been deleted)
        if not self._is_valid():
            return
        
        if not self.facade or not hasattr(self.facade, 'data_state_service') or not self.facade.data_state_service:
            old_state = self._data_state
            self._data_state = None
            if old_state is not None:
                self._force_repaint()
            return
        
        try:
            result = self.facade.get_block_data_state(self.block.id)
            if result.success and result.data:
                old_state = self._data_state
                new_state = result.data
                
                # Always update state and repaint, even if state didn't change
                # This ensures UI is in sync with actual data
                self._data_state = new_state
                self._force_repaint()
                
                if old_state != new_state:
                    Log.debug(
                        f"BlockItem: Updated data state for '{self.block.name}' ({self.block.id}): "
                        f"{old_state} -> {new_state} (color: {new_state.color})"
                    )
                else:
                    Log.debug(
                        f"BlockItem: Data state unchanged for '{self.block.name}' ({self.block.id}): {new_state}"
                    )
            else:
                # No state available - clear indicator
                old_state = self._data_state
                self._data_state = None
                if old_state is not None or result.message:
                    # State was cleared or there was an error - repaint
                    self._force_repaint()
                    Log.debug(
                        f"BlockItem: Cleared data state for '{self.block.name}' ({self.block.id}): "
                        f"{result.message if hasattr(result, 'message') else 'No data'}"
                    )
        except Exception as e:
            Log.warning(f"BlockItem: Failed to update data state for '{self.block.name}': {e}", exc_info=True)
            old_state = self._data_state
            self._data_state = None
            if old_state is not None:
                self._force_repaint()
    
    def _update_status_if_needed(self):
        """Conditionally update status - only if item is visible and in scene"""
        # Check if item is still valid (might have been deleted)
        if not self._is_valid():
            return
        
        # Only update if item is visible (performance optimization)
        try:
            if not self.isVisible():
                return
        except RuntimeError:
            # C++ object has been deleted
            return
        
        self._update_status()
    
    def _safe_update_status_if_needed(self):
        """Safely update status - handles case where object may have been deleted"""
        try:
            self._update_status_if_needed()
        except RuntimeError:
            # C++ object has been deleted, ignore
            pass
    
    def _update_data_state_if_needed(self):
        """Conditionally update data state - only if item is visible and in scene"""
        # Check if item is still valid (might have been deleted)
        if not self._is_valid():
            return
        
        # Only update if item is visible (performance optimization)
        try:
            if not self.isVisible():
                return
        except RuntimeError:
            # C++ object has been deleted
            return
        
        self._update_data_state()
    
    def _safe_update_data_state_if_needed(self):
        """Safely update data state - handles case where object may have been deleted"""
        try:
            self._update_data_state_if_needed()
        except RuntimeError:
            # C++ object has been deleted, ignore
            pass
    
    def _update_status_tooltip(self, status: "BlockStatus"):
        """Update tooltip with diagnostic information about status evaluation"""
        # Check if object is still valid before updating
        if not self._is_valid():
            return
        
        if not self.facade or not hasattr(self.facade, 'block_status_service') or not self.facade.block_status_service:
            # Fallback to simple tooltip
            tooltip = f"{status.display_name}"
            if status.message:
                tooltip = f"{status.display_name}: {status.message}"
            # Check again before setting tooltip
            if self._is_valid():
                try:
                    self.setToolTip(tooltip)
                except (RuntimeError, AttributeError):
                    # Object deleted during tooltip update - ignore silently
                    return
            return
        
        try:
            diagnostics = self.facade.block_status_service.get_block_status_diagnostics(
                self.block.id,
                self.facade
            )
            
            if diagnostics.get("error"):
                if self._is_valid():
                    try:
                        self.setToolTip(f"{status.display_name}\n\nError: {diagnostics['error']}")
                    except (RuntimeError, AttributeError):
                        # Object deleted during tooltip update - ignore silently
                        return
                return
            
            # Build detailed tooltip
            lines = [f"Status: {status.display_name}"]
            
            if status.message:
                lines.append(f"Message: {status.message}")
            
            lines.append("")  # Blank line
            
            # Show evaluation for each level
            for level_info in diagnostics.get("levels", []):
                level = level_info["level"]
                is_active = level_info["is_active"]
                reason = level_info.get("reason", "Unknown")
                
                # Mark active level
                marker = "→ " if is_active else "  "
                lines.append(f"{marker}{level['display_name']} (Priority {level['priority']})")
                
                if level_info["condition_count"] > 0:
                    lines.append(f"    {reason}")
                else:
                    lines.append(f"    {reason}")
                
                # Show condition details if there are failures
                if not level_info["all_conditions_pass"] and level_info.get("failed_conditions"):
                    for failed in level_info["failed_conditions"][:3]:  # Limit to 3
                        lines.append(f"      ✗ {failed}")
                
                # Show actionable guidance for active level with failures
                if is_active and not level_info["all_conditions_pass"] and level_info.get("actionable_guidance"):
                    lines.append("")
                    lines.append("How to fix:")
                    for step in level_info["actionable_guidance"]:
                        lines.append(f"  {step}")
                
                lines.append("")  # Blank line between levels
            
            # Check again before setting tooltip (object might have been deleted during tooltip building)
            if self._is_valid():
                try:
                    self.setToolTip("\n".join(lines))
                except (RuntimeError, AttributeError):
                    # Object deleted during tooltip update - ignore silently
                    return
            
        except (RuntimeError, AttributeError) as e:
            # Object was deleted - ignore silently
            if "deleted" in str(e).lower() or "wrapped C/C++ object" in str(e):
                return
            Log.debug(f"BlockItem: Error building diagnostic tooltip: {e}")
            # Fallback to simple tooltip
            if not self._is_valid():
                return
            try:
                tooltip = f"{status.display_name}"
                if status.message:
                    tooltip = f"{status.display_name}: {status.message}"
                if self._is_valid():
                    self.setToolTip(tooltip)
            except (RuntimeError, AttributeError):
                # Object deleted during tooltip update - ignore
                return
        except Exception as e:
            Log.debug(f"BlockItem: Error building diagnostic tooltip: {e}")
            # Fallback to simple tooltip
            if not self._is_valid():
                return
            try:
                tooltip = f"{status.display_name}"
                if status.message:
                    tooltip = f"{status.display_name}: {status.message}"
                if self._is_valid():
                    self.setToolTip(tooltip)
            except (RuntimeError, AttributeError):
                # Object deleted during tooltip update - ignore
                return
    
    def _force_repaint(self):
        """Force a repaint of this item and its scene"""
        # Check if item is still valid before repainting
        if not self._is_valid():
            return
        
        try:
            # Update this item's bounding rect
            self.update(self.boundingRect())
            
            # Also update scene for more reliable visibility
            scene = self.scene()
            if scene:
                # Update just the area around this item for better performance
                item_rect = self.sceneBoundingRect()
                scene.update(item_rect)
        except RuntimeError:
            # C++ object has been deleted, ignore
            pass
    
    def _subscribe_to_events(self):
        """Subscribe to events for status updates"""
        if not self.facade or not self.facade.event_bus:
            return
        
        # Primary: Subscribe to StatusChanged events for immediate status updates
        from src.application.events import StatusChanged
        self.facade.event_bus.subscribe("StatusChanged", self._on_status_changed)
        
        # Fallback: Subscribe to BlockChanged for status updates (during migration)
        # This event is emitted when:
        # - A block executes (downstream blocks become stale)
        # - Block data changes (status needs refresh)
        self.facade.event_bus.subscribe("BlockChanged", self._on_block_status_changed)
        
        # Subscribe to BlockUpdated for metadata changes (errors, filters, etc.)
        # This ensures status updates when block metadata changes
        self.facade.event_bus.subscribe("BlockUpdated", self._on_block_updated)
        
        # Also subscribe to connection changes (affects status calculation)
        self.facade.event_bus.subscribe("ConnectionCreated", self._on_connection_changed)
        self.facade.event_bus.subscribe("ConnectionRemoved", self._on_connection_changed)
    
    def _unsubscribe_from_events(self):
        """Unsubscribe from events when item is removed"""
        if not self.facade or not self.facade.event_bus:
            return
        
        try:
            self.facade.event_bus.unsubscribe("StatusChanged", self._on_status_changed)
            self.facade.event_bus.unsubscribe("BlockChanged", self._on_block_status_changed)
            self.facade.event_bus.unsubscribe("BlockUpdated", self._on_block_updated)
            self.facade.event_bus.unsubscribe("ConnectionCreated", self._on_connection_changed)
            self.facade.event_bus.unsubscribe("ConnectionRemoved", self._on_connection_changed)
        except Exception as e:
            Log.debug(f"BlockItem: Error unsubscribing from events: {e}")
    
    def _on_status_changed(self, event):
        """
        Handle StatusChanged event - immediate update from single source of truth.
        
        This event is published by BlockStatusService when status actually changes.
        Provides immediate, accurate status updates.
        """
        # Check if object is still valid before processing
        if not self._is_valid():
            return
        
        try:
            if not hasattr(event, 'data') or not event.data:
                return
            
            block_id = event.data.get('block_id')
            if block_id != self.block.id:
                return
            
            # Check again after accessing self.block
            if not self._is_valid():
                return
            
            # Status object is in event data
            status_dict = event.data.get('status')
            if status_dict:
                # Reconstruct status from dict
                from src.features.blocks.domain import BlockStatus, BlockStatusLevel
                level_dict = status_dict.get('level', {})
                level = BlockStatusLevel(
                    priority=level_dict.get('priority', 0),
                    name=level_dict.get('name', 'ready'),
                    display_name=level_dict.get('display_name', 'Ready'),
                    color=level_dict.get('color', Colors.STATUS_SUCCESS.name()),
                    conditions=[]
                )
                status = BlockStatus(
                    level=level,
                    message=status_dict.get('message')
                )
                
                # Check again before updating (object might have been deleted)
                if not self._is_valid():
                    return
                
                # Update status immediately
                old_status = self._block_status
                self._block_status = status
                
                # Update tooltip with diagnostic information
                self._update_status_tooltip(status)
                
                # Check again before repainting
                if self._is_valid():
                    self._force_repaint()
                
                Log.debug(
                    f"BlockItem: StatusChanged event for '{self.block.name}' ({self.block.id}): "
                    f"{old_status.level.name if old_status else None} -> {status.level.name}"
                )
        except Exception as e:
            Log.warning(f"BlockItem: Error handling StatusChanged event: {e}", exc_info=True)
            # Fallback: recalculate status
            self._safe_update_status_if_needed()
    
    def _on_block_status_changed(self, event):
        """
        Handle block status changed event - fallback during migration.
        
        This event is emitted when a block's data state may have changed,
        requiring UI components to refresh their status indicators.
        Used as fallback during migration.
        """
        try:
            # Robust event data access
            if not hasattr(event, 'data') or not event.data:
                Log.debug(f"BlockItem: BlockChanged event has no data for '{self.block.name}'")
                return
            
            block_id = event.data.get('block_id')
            if not block_id:
                Log.debug(f"BlockItem: BlockChanged event missing block_id for '{self.block.name}'")
                return
            
            if block_id == self.block.id:
                # This block's status changed - update status indicator
                # Use longer delay to ensure data is fully persisted before checking status
                from PyQt6.QtCore import QTimer
                Log.debug(f"BlockItem: Scheduling status update for block '{self.block.name}' ({self.block.id})")
                QTimer.singleShot(150, lambda: self._safe_update_status_if_needed())
            else:
                # Even if not for this block, check if we should update
                # (e.g., if an upstream block changed, our status might be stale)
                # But only do this occasionally to avoid excessive updates
                from PyQt6.QtCore import QTimer
                # Use longer delay for non-direct updates
                # Use a safe lambda that checks if object is still valid
                QTimer.singleShot(300, lambda: self._safe_update_status_if_needed())
        except Exception as e:
            Log.warning(f"BlockItem: Error handling BlockChanged event: {e}", exc_info=True)
            # On error, still try to update status (safely)
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(200, lambda: self._safe_update_status_if_needed())
    
    def _on_block_updated(self, event):
        """
        Handle block updated event - refresh block entity and update status.
        
        This ensures the BlockItem's block entity is current when metadata changes
        (e.g., errors, filters), and triggers a status update.
        """
        try:
            updated_block_id = event.data.get('id') if hasattr(event, 'data') and event.data else None
            if updated_block_id == self.block.id:
                # Reload block entity to get latest metadata (errors, filters, etc.)
                result = self.facade.describe_block(self.block.id)
                if result.success and result.data:
                    self.block = result.data
                    Log.debug(f"BlockItem: Refreshed block entity for '{self.block.name}' ({self.block.id})")
                
                # Update status indicator (may have changed due to metadata updates)
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self._safe_update_status_if_needed())
        except Exception as e:
            Log.warning(f"BlockItem: Error handling BlockUpdated event: {e}", exc_info=True)
    
    def _on_connection_changed(self, event):
        """Handle connection change event - affects status calculation"""
        # Connections changed - update state (may affect this block if it's involved)
        # Check if this block is involved in the connection change
        try:
            if not hasattr(event, 'data') or not event.data:
                return
            
            source_block_id = event.data.get('source_block_id')
            target_block_id = event.data.get('target_block_id')
            if source_block_id == self.block.id or target_block_id == self.block.id:
                # Use QTimer with delay to ensure connection is fully persisted
                # Match BlockChanged delay for consistency
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(150, lambda: self._safe_update_status_if_needed())
                Log.debug(f"BlockItem: Connection changed for '{self.block.name}', scheduling status update")
        except Exception as e:
            Log.warning(f"BlockItem: Error handling connection change event: {e}", exc_info=True)
    
    def add_connection(self, connection_item):
        """Register a connection attached to this block"""
        if connection_item not in self.connections:
            self.connections.append(connection_item)
    
    def remove_connection(self, connection_item):
        """Unregister a connection"""
        if connection_item in self.connections:
            self.connections.remove(connection_item)
    
    def itemChange(self, change, value):
        """Handle item changes"""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Update all connected lines
            for conn in self.connections:
                conn.update_position()
            
            # Emit signal
            self.signals.position_changed.emit(self.block.id, self.pos())
        
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            if value:  # Selected
                self.signals.selected.emit(self.block.id)
        
        elif change == QGraphicsItem.GraphicsItemChange.ItemSceneHasChanged:
            # Item was removed from scene - cleanup event subscriptions
            if value is None:  # Scene is None when removed
                self._unsubscribe_from_events()
        
        return super().itemChange(change, value)
    
    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def port_at_position(self, local_pos: QPointF):
        """
        Check if a position is over a port.
        
        Returns:
            (port_name, is_output, is_bidirectional) if over a port, None otherwise
        """
        rect = QRectF(
            -self._width / 2,
            -self._height / 2,
            self._width,
            self._height
        )
        port_start_y = rect.top() + Sizes.BLOCK_HEADER_HEIGHT + Spacing.SM
        hit_radius = Sizes.PORT_RADIUS + 8  # Generous hit area
        
        # Determine which side bidirectional ports are on
        bidirectional_on_right = (self.block.type == "ShowManager")
        
        # Check bidirectional ports on left side FIRST (if not ShowManager) - matches rendering order
        num_bidirectional_on_left = 0
        bidirectional_ports = self.block.get_bidirectional()
        if bidirectional_ports and not bidirectional_on_right:
            for i, port_name in enumerate(bidirectional_ports.keys()):
                y = port_start_y + (i + 0.5) * Sizes.PORT_ZONE_HEIGHT
                port_center = QPointF(rect.left(), y)
                
                dx = local_pos.x() - port_center.x()
                dy = local_pos.y() - port_center.y()
                if (dx * dx + dy * dy) <= hit_radius * hit_radius:
                    return (port_name, False, True)  # is_bidirectional=True, treat as input when on left
            num_bidirectional_on_left = len(bidirectional_ports)
        
        # Check input ports AFTER bidirectional on left (if any) - matches rendering order
        input_ports = self.block.get_inputs()
        if input_ports:
            for i, port_name in enumerate(input_ports.keys()):
                y = port_start_y + (num_bidirectional_on_left + i + 0.5) * Sizes.PORT_ZONE_HEIGHT
                port_center = QPointF(rect.left(), y)
                
                dx = local_pos.x() - port_center.x()
                dy = local_pos.y() - port_center.y()
                if (dx * dx + dy * dy) <= hit_radius * hit_radius:
                    return (port_name, False, False)
        
        # Check bidirectional ports on right side FIRST (if ShowManager) - matches rendering order
        num_bidirectional_on_right = 0
        if bidirectional_ports and bidirectional_on_right:
            for i, port_name in enumerate(bidirectional_ports.keys()):
                y = port_start_y + (i + 0.5) * Sizes.PORT_ZONE_HEIGHT
                port_center = QPointF(rect.right(), y)
                
                dx = local_pos.x() - port_center.x()
                dy = local_pos.y() - port_center.y()
                if (dx * dx + dy * dy) <= hit_radius * hit_radius:
                    return (port_name, True, True)  # is_bidirectional=True, treat as output when on right
            num_bidirectional_on_right = len(bidirectional_ports)
        
        # Check output ports AFTER bidirectional on right (if any) - matches rendering order
        output_ports = self.block.get_outputs()
        if output_ports:
            for i, port_name in enumerate(output_ports.keys()):
                y = port_start_y + (num_bidirectional_on_right + i + 0.5) * Sizes.PORT_ZONE_HEIGHT
                port_center = QPointF(rect.right(), y)
                
                dx = local_pos.x() - port_center.x()
                dy = local_pos.y() - port_center.y()
                if (dx * dx + dy * dy) <= hit_radius * hit_radius:
                    return (port_name, True, False)
        
        return None
    
    def mousePressEvent(self, event):
        """Handle mouse press - check for port click to start connection"""
        if event.button() == Qt.MouseButton.LeftButton:
            port_info = self.port_at_position(event.pos())
            if port_info:
                port_name, is_output, is_bidirectional = port_info
                scene_pos = self.get_port_position(port_name, is_output, is_bidirectional)
                self.signals.port_clicked.emit(
                    self.block.id, port_name, is_output, is_bidirectional, scene_pos
                )
                event.accept()
                return
            
            # Store drag start position for undo support
            self._drag_start_pos = self.pos()
            self._is_dragging = True
        
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release - create undo command if position changed"""
        if event.button() == Qt.MouseButton.LeftButton and self._is_dragging:
            self._is_dragging = False
            
            # Check if position actually changed
            if self._drag_start_pos is not None:
                current_pos = self.pos()
                
                # Only create command if position changed significantly
                dx = abs(current_pos.x() - self._drag_start_pos.x())
                dy = abs(current_pos.y() - self._drag_start_pos.y())
                
                if dx > 1 or dy > 1:
                    # Create undoable move command with known old position
                    cmd = MoveBlockCommand(
                        self.facade,
                        self.block.id,
                        current_pos.x(),
                        current_pos.y(),
                        old_x=self._drag_start_pos.x(),
                        old_y=self._drag_start_pos.y()
                    )
                    self.facade.command_bus.execute(cmd)
                    Log.debug(f"BlockItem: Move command created for {self.block.name}")
            
            self._drag_start_pos = None
        
        super().mouseReleaseEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        """Handle double-click to open panel"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.signals.double_clicked.emit(self.block.id)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)
    
    def contextMenuEvent(self, event):
        """Show context menu with quick actions"""
        self._show_context_menu(event.screenPos())
        event.accept()
    
    def _show_context_menu(self, screen_pos: QPointF):
        """Display the context menu with quick actions"""
        menu = QMenu()
        menu.setMinimumWidth(Sizes.CONTEXT_MENU_MIN_WIDTH)
        
        # Apply dark theme styling
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(6)};
                padding: 4px;
            }}
            QMenu::item {{
                padding: 8px 16px 8px 12px;
                border-radius: {border_radius(4)};
                margin: 2px 4px;
            }}
            QMenu::item:selected {{
                background-color: {Colors.ACCENT_BLUE.name()};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {Colors.BORDER.name()};
                margin: 4px 8px;
            }}
            QMenu::item:disabled {{
                color: {Colors.TEXT_DISABLED.name()};
            }}
        """)
        
        # Try to get quick actions from the system
        try:
            from src.application.blocks.quick_actions import get_quick_actions, ActionCategory
            actions = get_quick_actions(self.block.type)
            
            # Group actions by category
            execute_actions = [a for a in actions if a.category == ActionCategory.EXECUTE]
            config_actions = [a for a in actions if a.category == ActionCategory.CONFIGURE]
            file_actions = [a for a in actions if a.category == ActionCategory.FILE]
            view_actions = [a for a in actions if a.category == ActionCategory.VIEW]
            edit_actions = [a for a in actions if a.category == ActionCategory.EDIT]
            
            # Add execute actions first
            for action in execute_actions:
                if action.primary:
                    menu_action = menu.addAction(f"  {action.name}")
                else:
                    menu_action = menu.addAction(action.name)
                menu_action.setData(("quick_action", action))
            
            if execute_actions:
                menu.addSeparator()
            
            # Add config and file actions
            for action in config_actions + file_actions:
                menu_action = menu.addAction(action.name)
                menu_action.setData(("quick_action", action))
            
            if config_actions or file_actions:
                menu.addSeparator()
            
            # Add view actions
            for action in view_actions:
                menu_action = menu.addAction(action.name)
                menu_action.setData(("quick_action", action))
            
            if view_actions:
                menu.addSeparator()
            
            # Add edit actions (rename, delete)
            for action in edit_actions:
                menu_action = menu.addAction(action.name)
                menu_action.setData(("quick_action", action))
                if action.dangerous:
                    menu_action.setIcon(self._create_danger_icon())
            
            # Add duplicate action (always available, not from quick_actions)
            if edit_actions:
                menu.addSeparator()
            duplicate_action = menu.addAction("Duplicate Block")
            duplicate_action.setData(("duplicate", None))
                    
        except ImportError:
            # Fallback if quick_actions not available
            self._add_fallback_menu_items(menu)
        
        # Show menu and handle selection
        from PyQt6.QtCore import QPoint
        selected_action = menu.exec(QPoint(int(screen_pos.x()), int(screen_pos.y())))
        if selected_action:
            self._handle_menu_action(selected_action)
    
    def _create_danger_icon(self):
        """Create a simple warning icon for dangerous actions"""
        from PyQt6.QtGui import QIcon, QPixmap
        # Return empty icon - styling handles this
        return QIcon()
    
    def _add_fallback_menu_items(self, menu: QMenu):
        """Add basic menu items when quick_actions not available"""
        menu.addAction("Execute Block").setData(("execute", None))
        menu.addSeparator()
        menu.addAction("Open Panel").setData(("open_panel", None))
        menu.addAction("Rename").setData(("rename", None))
        menu.addAction("Duplicate Block").setData(("duplicate", None))
        menu.addSeparator()
        menu.addAction("Delete").setData(("delete", None))
    
    def _handle_menu_action(self, menu_action):
        """Handle menu action selection"""
        data = menu_action.data()
        if not data:
            return
        
        action_type, action_data = data
        
        if action_type == "quick_action":
            self._execute_quick_action(action_data)
        elif action_type == "execute":
            self._execute_block()
        elif action_type == "open_panel":
            self.signals.double_clicked.emit(self.block.id)
        elif action_type == "rename":
            self._rename_block()
        elif action_type == "duplicate":
            self._duplicate_block()
        elif action_type == "delete":
            self._delete_block()
    
    def _execute_quick_action(self, action):
        """Execute a quick action, handling input requirements"""
        # Run Execute in background thread so UI stays responsive (e.g. PyTorch trainer)
        from src.application.blocks.quick_actions import is_execute_block_action
        if is_execute_block_action(action):
            self._execute_block()
            return

        try:
            result = action.handler(self.facade, self.block.id)
            
            if isinstance(result, dict):
                # Handle special return values
                if result.get("needs_input"):
                    self._handle_input_request(action, result)
                elif result.get("needs_confirmation"):
                    self._handle_confirmation(action, result)
                elif result.get("open_panel"):
                    self.signals.double_clicked.emit(self.block.id)
                elif result.get("open_filter_dialog"):
                    self._open_filter_dialog()
                elif hasattr(result, 'success') and not result.success:
                    Log.error(f"Action failed: {result.message if hasattr(result, 'message') else result}")
                    
        except Exception as e:
            Log.error(f"Quick action error: {e}")
            QMessageBox.warning(None, "Action Error", str(e))
    
    def _open_filter_dialog(self):
        """Open the data filter dialog for this block"""
        from ui.qt_gui.dialogs.data_filter_dialog import DataFilterDialog
        dialog = DataFilterDialog(self.block.id, self.facade, parent=None)
        dialog.exec()
    
    def _handle_input_request(self, action, request: Dict):
        """Handle action that needs user input"""
        input_type = request.get("input_type")
        title = request.get("title", action.name)
        
        if input_type == "text":
            text, ok = QInputDialog.getText(None, title, request.get("prompt", "Enter value:"))
            if ok and text:
                action.handler(self.facade, self.block.id, new_name=text)
                
        elif input_type == "file":
            start_dir = app_settings.get_dialog_path("action_file")
            file_path, _ = QFileDialog.getOpenFileName(
                None, title, start_dir,
                request.get("file_filter", "All Files (*)")
            )
            if file_path:
                app_settings.set_dialog_path("action_file", file_path)
                action.handler(self.facade, self.block.id, file_path=file_path)
                
        elif input_type == "directory":
            start_dir = app_settings.get_dialog_path("action_dir")
            directory = QFileDialog.getExistingDirectory(None, title, start_dir)
            if directory:
                app_settings.set_dialog_path("action_dir", directory)
                action.handler(self.facade, self.block.id, directory=directory)
                
        elif input_type == "choice":
            choices = request.get("choices", [])
            labels = request.get("labels", choices)
            default = request.get("default", choices[0] if choices else "")
            default_idx = choices.index(default) if default in choices else 0
            
            choice, ok = QInputDialog.getItem(
                None, title, "Select:", labels, default_idx, False
            )
            if ok:
                # Get actual value from choices (in case labels differ)
                idx = labels.index(choice) if choice in labels else 0
                value = choices[idx]
                # Determine which kwarg to pass based on action
                action.handler(self.facade, self.block.id, **{self._get_action_kwarg(action, value): value})
                
        elif input_type == "number":
            value, ok = QInputDialog.getDouble(
                None, title, "Value:",
                request.get("default", 0.5),
                request.get("min", 0.0),
                request.get("max", 1.0),
                2  # decimals
            )
            if ok:
                action.handler(self.facade, self.block.id, value=value)
    
    def _get_action_kwarg(self, action, value):
        """Determine the keyword argument name for an action"""
        # Map action names to their expected kwargs
        name_lower = action.name.lower()
        if "model" in name_lower:
            return "model"
        elif "device" in name_lower:
            return "device"
        elif "stem" in name_lower:
            return "stem"
        elif "format" in name_lower:
            return "fmt"
        elif "style" in name_lower:
            return "style"
        return "value"
    
    def _handle_confirmation(self, action, request: Dict):
        """Handle action that needs confirmation"""
        reply = QMessageBox.question(
            None, action.name,
            request.get("message", f"Are you sure you want to {action.name.lower()}?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            action.handler(self.facade, self.block.id, confirmed=True)
            # Emit signal to refresh if this was a delete
            if action.dangerous:
                try:
                    scene = self.scene()
                    if scene and hasattr(scene, 'refresh'):
                        scene.refresh()
                except RuntimeError:
                    # C++ object has been deleted, ignore
                    pass
    
    def _execute_block(self):
        """Execute this block synchronously on the main thread (for testing; UI blocks until done)."""
        main_window = None
        if self.scene() and self.scene().views():
            main_window = self.scene().views()[0].window()
        if main_window and hasattr(main_window, "_on_execute_single_block"):
            main_window._on_execute_single_block(self.block.id)
            return
        # Fallback: run synchronously (e.g. when not embedded in main window)
        try:
            result = self.facade.execute_block(self.block.id)
            if result.success:
                Log.info("Block execution completed.")
            else:
                Log.error(f"Execution failed: {result.message or 'Unknown error'}")
        except Exception as e:
            Log.error(f"Execution failed: {e}")
    
    def _rename_block(self):
        """Rename this block"""
        new_name, ok = QInputDialog.getText(
            None, "Rename Block", "New name:", text=self.block.name
        )
        if ok and new_name and new_name != self.block.name:
            # Use CommandBus for undoable command
            cmd = RenameBlockCommand(self.facade, self.block.id, new_name)
            self.facade.command_bus.execute(cmd)
            self.block.name = new_name
            self.update()
    
    def _duplicate_block(self):
        """Duplicate this block with settings and filters (but not connections)"""
        # Use CommandBus for undoable command
        cmd = DuplicateBlockCommand(self.facade, self.block.id)
        self.facade.command_bus.execute(cmd)
        try:
            scene = self.scene()
            if scene and hasattr(scene, 'refresh'):
                scene.refresh()
        except RuntimeError:
            # C++ object has been deleted, ignore
            pass
    
    def _delete_block(self):
        """Delete this block"""
        # Use CommandBus for undoable command
        cmd = DeleteBlockCommand(self.facade, self.block.id)
        self.facade.command_bus.execute(cmd)
        try:
            scene = self.scene()
            if scene and hasattr(scene, 'refresh'):
                scene.refresh()
        except RuntimeError:
            # C++ object has been deleted (block was removed), ignore
            pass
