"""
Playhead

The playhead indicator that shows current playback position.
Updates at 60 FPS for smooth animation (handled by PlaybackController).
"""

from PyQt6.QtWidgets import QGraphicsItem, QGraphicsLineItem, QGraphicsScene
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPen, QBrush, QPainter, QPainterPath

from ..constants import (
    PLAYHEAD_WIDTH, PLAYHEAD_HEAD_SIZE,
    RULER_HEIGHT, TRACK_HEIGHT, TRACK_SPACING
)
from ..core.style import TimelineStyle


class PlayheadItem(QGraphicsItem):
    """
    Visual playhead indicator.
    
    Features:
    - Vertical line spanning all tracks
    - Triangular head at top
    - Drag to seek (when enabled)
    - Smooth 60 FPS position updates
    """
    
    def __init__(self, scene_height: float = 500):
        super().__init__()
        
        self._position_seconds = 0.0
        self._pixels_per_second = 100.0
        self._scene_height = scene_height
        self._dragging = False
        self._color = TimelineStyle.PLAYHEAD_COLOR
        
        # Enable mouse interactions for seeking
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        
        # Draw on top of everything
        self.setZValue(1000)
        
        self._update_position()
    
    @property
    def position_seconds(self) -> float:
        """Get current position in seconds"""
        return self._position_seconds
    
    def set_position(self, seconds: float):
        """
        Set playhead position.
        
        The playhead is updated at 60 FPS by PlaybackController during playback,
        providing smooth movement without additional animation overhead.
        
        Args:
            seconds: Position in seconds
        """
        if seconds != self._position_seconds:
            self._position_seconds = max(0, seconds)
            self._update_position()
    
    def set_pixels_per_second(self, pps: float):
        """Update zoom level"""
        self._pixels_per_second = pps
        self._update_position()
    
    def set_scene_height(self, height: float):
        """Update height to span"""
        self._scene_height = height
        self.prepareGeometryChange()
    
    def _update_position(self):
        """
        Update x position based on time.
        
        CRITICAL: When playhead moves, only update the playhead item itself,
        NOT the entire foreground. This prevents redrawing all waveforms at 60 FPS.
        """
        x = self._position_seconds * self._pixels_per_second
        
        # Store old position for invalidation
        old_x = self.pos().x()
        
        # Update position - Qt will automatically invalidate the item
        self.setPos(x, 0)
        
        # CRITICAL FIX: Only update the playhead item's bounding rects (old and new)
        # Don't let this trigger full scene foreground redraws that would redraw waveforms
        # The playhead is drawn in the item layer, not foreground, so waveforms won't be affected
        scene = self.scene()
        if scene and abs(old_x - x) > 0.01:  # Only if position actually changed
            # Get bounding rects in scene coordinates
            old_rect = self.boundingRect()
            old_rect.translate(old_x, 0)
            new_rect = self.boundingRect()
            new_rect.translate(x, 0)
            
            # Union of old and new rects (the area that actually needs updating)
            update_rect = old_rect.united(new_rect)
            
            # Invalidate only this specific region for the item layer
            # This will redraw the playhead but NOT trigger drawForeground()
            scene.invalidate(update_rect, QGraphicsScene.SceneLayer.ItemLayer)
    
    def boundingRect(self) -> QRectF:
        """Get bounding rectangle"""
        # Include the head and a small margin
        return QRectF(
            -PLAYHEAD_HEAD_SIZE,
            -PLAYHEAD_HEAD_SIZE,
            PLAYHEAD_HEAD_SIZE * 2 + PLAYHEAD_WIDTH,
            self._scene_height + PLAYHEAD_HEAD_SIZE * 2
        )
    
    def paint(self, painter: QPainter, option, widget=None):
        """Paint the playhead"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Highlight when dragging
        if self._dragging:
            color = self._color.lighter(130)
        else:
            color = self._color
        
        # Draw vertical line
        pen = QPen(color, PLAYHEAD_WIDTH)
        painter.setPen(pen)
        painter.drawLine(0, 0, 0, int(self._scene_height))
        
        # Draw triangular head at top
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        
        path = QPainterPath()
        path.moveTo(-PLAYHEAD_HEAD_SIZE, -PLAYHEAD_HEAD_SIZE)
        path.lineTo(PLAYHEAD_HEAD_SIZE, -PLAYHEAD_HEAD_SIZE)
        path.lineTo(0, 0)
        path.closeSubpath()
        
        painter.drawPath(path)
    
    def hoverEnterEvent(self, event):
        """Show pointer cursor when hovering"""
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Reset cursor"""
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)
    
    def mousePressEvent(self, event):
        """Start drag for seeking"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.update()
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Drag to seek"""
        if self._dragging:
            # Calculate new position from scene coordinates
            scene_pos = event.scenePos()
            new_time = max(0, scene_pos.x() / self._pixels_per_second)
            
            # Snap to grid if available
            scene = self.scene()
            if scene and hasattr(scene, 'grid_system') and scene.grid_system.snap_enabled:
                # Try new SnapCalculator first
                if hasattr(scene, '_snap_calculator') and hasattr(scene, '_unit_preference'):
                    new_time = scene._snap_calculator.snap_time(
                        new_time,
                        self._pixels_per_second,
                        scene._unit_preference
                    )
                else:
                    # Fallback to legacy
                    new_time = scene.grid_system.snap_time(new_time, self._pixels_per_second)
            
            self._position_seconds = new_time
            self._update_position()
            
            # Emit seek signal
            if scene and hasattr(scene, 'playhead_seeked'):
                scene.playhead_seeked.emit(new_time)
            
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """End drag"""
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self.update()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class PlayheadHandle(QGraphicsItem):
    """
    Draggable handle in the ruler area for seeking.
    
    Appears in the time ruler and can be dragged to seek.
    """
    
    def __init__(self):
        super().__init__()
        
        self._position_seconds = 0.0
        self._pixels_per_second = 100.0
        self._color = TimelineStyle.PLAYHEAD_COLOR
        self._dragging = False
        
        self.setAcceptHoverEvents(True)
        self.setZValue(1001)  # Above playhead line
    
    def set_position(self, seconds: float):
        """Set handle position"""
        self._position_seconds = max(0, seconds)
        x = self._position_seconds * self._pixels_per_second
        self.setPos(x, RULER_HEIGHT / 2)
    
    def set_pixels_per_second(self, pps: float):
        """Update zoom level"""
        self._pixels_per_second = pps
        self.set_position(self._position_seconds)
    
    def boundingRect(self) -> QRectF:
        """Get bounding rectangle"""
        size = PLAYHEAD_HEAD_SIZE
        return QRectF(-size, -size/2, size * 2, size)
    
    def paint(self, painter: QPainter, option, widget=None):
        """Paint handle as downward pointing triangle"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self._dragging:
            color = self._color.lighter(130)
        else:
            color = self._color
        
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(120), 1))
        
        size = PLAYHEAD_HEAD_SIZE
        
        path = QPainterPath()
        path.moveTo(-size, -size/2)
        path.lineTo(size, -size/2)
        path.lineTo(0, size/2)
        path.closeSubpath()
        
        painter.drawPath(path)
    
    def hoverEnterEvent(self, event):
        """Show pointer cursor"""
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Reset cursor"""
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)
    
    def mousePressEvent(self, event):
        """Start drag"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.update()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Drag to seek"""
        if self._dragging:
            scene_pos = event.scenePos()
            new_time = max(0, scene_pos.x() / self._pixels_per_second)
            
            # Snap
            scene = self.scene()
            if scene and hasattr(scene, 'grid_system') and scene.grid_system.snap_enabled:
                # Try new SnapCalculator first
                if hasattr(scene, '_snap_calculator') and hasattr(scene, '_unit_preference'):
                    new_time = scene._snap_calculator.snap_time(
                        new_time,
                        self._pixels_per_second,
                        scene._unit_preference
                    )
                else:
                    # Fallback to legacy
                    new_time = scene.grid_system.snap_time(new_time, self._pixels_per_second)
            
            self._position_seconds = new_time
            self.set_position(new_time)
            
            # Emit seek
            if scene and hasattr(scene, 'playhead_seeked'):
                scene.playhead_seeked.emit(new_time)
            
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """End drag"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.update()
            event.accept()


