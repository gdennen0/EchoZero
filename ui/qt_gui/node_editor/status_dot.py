"""
Status Dot - Centralized status indicator rendering

Provides consistent status dot/LED rendering across all UI components.
Single source of truth for status visualization.
"""
from typing import Union, Optional, TYPE_CHECKING
from PyQt6.QtCore import QRectF, QPointF, Qt
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen
from ui.qt_gui.design_system import Colors

from src.shared.domain.data_state import DataState

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatus


class StatusDotRenderer:
    """
    Centralized renderer for status indicator dots.
    
    Ensures consistent appearance and behavior across all UI components.
    Single source of truth for status visualization.
    """
    
    # Constants for dot appearance (consistent across all uses)
    DOT_SIZE = 6
    DOT_PADDING = 4
    
    @staticmethod
    def get_dot_size() -> float:
        """Get standard dot size"""
        return StatusDotRenderer.DOT_SIZE
    
    @staticmethod
    def get_dot_color(state: Union[DataState, "BlockStatus", None]) -> QColor:
        """
        Get color for status dot based on data state or block status.
        
        Args:
            state: DataState enum value or BlockStatus instance
            
        Returns:
            QColor for the status dot
        """
        if not state:
            return Colors.STATUS_INACTIVE  # Gray for unknown/missing state
        
        # Handle BlockStatus (has color property)
        if hasattr(state, 'color') and not isinstance(state, DataState):
            return QColor(state.color)
        
        # Handle DataState (enum with color property)
        return QColor(state.color)
    
    @staticmethod
    def draw_status_dot(
        painter: QPainter,
        header_rect: QRectF,
        state: Union[DataState, "BlockStatus", None],
        dot_size: Optional[float] = None
    ) -> None:
        """
        Draw status dot in top-right corner of header rect.
        
        Centralized rendering ensures consistent appearance everywhere.
        
        Args:
            painter: QPainter to draw with
            header_rect: Header rectangle (dot positioned relative to this)
            state: DataState enum value or BlockStatus instance (determines color)
            dot_size: Optional custom dot size (defaults to standard size)
        """
        if not state:
            return
        
        if dot_size is None:
            dot_size = StatusDotRenderer.DOT_SIZE
        
        # Position: top-right corner with padding
        dot_x = header_rect.right() - dot_size - StatusDotRenderer.DOT_PADDING
        dot_y = header_rect.top() + StatusDotRenderer.DOT_PADDING
        dot_center = QPointF(dot_x + dot_size / 2, dot_y + dot_size / 2)
        
        # Get color from state (single source of truth)
        state_color = StatusDotRenderer.get_dot_color(state)
        
        # Draw solid dot with antialiasing
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QBrush(state_color))
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.drawEllipse(dot_center, dot_size / 2, dot_size / 2)
    
    @staticmethod
    def get_status_dot_rect(header_rect: QRectF, dot_size: float = None) -> QRectF:
        """
        Get bounding rectangle for status dot.
        
        Useful for hit testing or layout calculations.
        
        Args:
            header_rect: Header rectangle
            dot_size: Optional custom dot size
            
        Returns:
            QRectF bounding the status dot
        """
        if dot_size is None:
            dot_size = StatusDotRenderer.DOT_SIZE
        
        dot_x = header_rect.right() - dot_size - StatusDotRenderer.DOT_PADDING
        dot_y = header_rect.top() + StatusDotRenderer.DOT_PADDING
        
        return QRectF(dot_x, dot_y, dot_size, dot_size)


