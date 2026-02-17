"""
Event Items
============

Graphics items representing events on the timeline.
Supports block events (with duration) and marker events (instant).
Includes editing capabilities: move, resize, delete.

Uses LayerManager for coordinate mapping (single source of truth).
Delegates drag operations to MovementController.
"""

from typing import Optional, Dict, Any, TYPE_CHECKING, Tuple
from PyQt6.QtWidgets import (
    QGraphicsRectItem, QGraphicsItem, QMenu, QApplication, QStyleOptionGraphicsItem
)
from PyQt6.QtCore import Qt, QRectF, QPointF, QRect, QTimer
from math import cos, sin, pi
import numpy as np
from PyQt6.QtGui import (
    QColor, QPen, QBrush, QPainter, QPainterPath, QPolygonF, QPixmap,
    QCursor, QFont, QTransform
)
from PyQt6.QtWidgets import QGraphicsDropShadowEffect

# Local imports
from ..core.style import TimelineStyle as Typography
from ..logging import TimelineLog as Log
from ..types import EditHandle
from ..constants import (
    TRACK_HEIGHT, TRACK_SPACING, EVENT_HEIGHT, MARKER_WIDTH,
    RESIZE_HANDLE_WIDTH,
    MIN_RESIZE_HANDLE_WIDTH, MIN_MOVE_AREA_WIDTH, RESIZE_HANDLE_PERCENT
)
from ..core.style import TimelineStyle
from ..settings.storage import get_timeline_settings_manager

if TYPE_CHECKING:
    from .layer_manager import LayerManager
    from .movement_controller import MovementController
    from ..core.scene import TimelineScene


class BaseEventItem(QGraphicsRectItem):
    """
    Base class for timeline event items.
    
    Uses layer_id (not layer_index) for layer assignment.
    Classification is semantic metadata - it does NOT determine layer.
    
    Provides common functionality:
    - Selection highlighting
    - Hover effects
    - Context menu
    - Position/time conversion
    """
    
    def __init__(
        self,
        event_id: str,
        start_time: float,
        duration: float,
        classification: str,
        layer_id: str,
        layer_manager: 'LayerManager',
        pixels_per_second: float,
        audio_id: Optional[str] = None,
        audio_name: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None,
        editable: bool = True,
        render_as_marker: bool = False
    ):
        super().__init__()
        
        # Core identity
        self.event_id = event_id
        self.start_time = start_time
        self.duration = duration
        self.classification = classification  # Semantic type (immutable during moves)
        self.layer_id = layer_id              # Which layer (changes during moves)
        
        # Audio source (for clip events with waveforms)
        self.audio_id = audio_id
        self.audio_name = audio_name
        
        # User/application data (pass-through)
        self.user_data = user_data or {}
        
        # Visual rendering property (purely visual, not data-level distinction)
        self.render_as_marker = render_as_marker
        
        # Layer manager reference (for coordinate mapping)
        self._layer_manager = layer_manager
        
        # Display state
        self._pixels_per_second = pixels_per_second
        self._editable = editable
        
        # Visual state
        self._base_color = self._get_layer_color()
        self._selected = False
        self._hovered = False
        
        # Edit state
        self._edit_handle = EditHandle.NONE
        
        # Enable interactions
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        # Enable position change notifications for native snapping (POC-verified)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        if editable:
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        
        # Native snapping control (POC-verified infrastructure)
        # When True, itemChange() applies snapping. When False, MovementController handles it.
        # Default False for compatibility - can be enabled incrementally.
        self._use_native_snapping = False
        self._snap_enabled = False
        self._snap_interval = 0.1  # Default snap interval in seconds
        
        # Tooltip
        self._update_tooltip()
        
        # Connect to settings changes if available
        settings_mgr = get_timeline_settings_manager()
        if settings_mgr:
            settings_mgr.settings_changed.connect(self._on_settings_changed)
        
        # Initial geometry (styling applied in subclasses)
        self._update_geometry()
    
    # ===================================================================
    # Settings & Configuration
    # ===================================================================
    
    def _on_settings_changed(self, setting_name: str):
        """Handle settings changes - update if event styling changed."""
        # Update if appearance settings changed
        if setting_name in ('show_event_labels', 'show_event_duration_labels', 'highlight_current_event'):
            self.update()  # Repaint to show/hide labels or update highlighting
        # Block events listen to block_event_* settings
        # Marker events listen to marker_event_* settings
        # This is handled in subclasses
    
    def _get_layer_color(self) -> QColor:
        """Get color based on layer index."""
        layer = self._layer_manager.get_layer(self.layer_id)
        if layer:
            if layer.color:
                return QColor(layer.color)
            return TimelineStyle.get_layer_color(layer.index)
        return TimelineStyle.get_layer_color(0)
    
    # ===================================================================
    # Geometry & Rendering
    # ===================================================================
    
    def shape(self) -> QPainterPath:
        """
        Return exact shape for hit testing (matches visual appearance).
        
        PYTQT BEST PRACTICE: Override shape() to match visual appearance (rounded rectangle)
        rather than default rectangular shape. This provides better UX for hover/click detection.
        
        Note: This is implemented in BlockEventItem subclasses which have border radius.
        Base class returns default rectangular shape.
        """
        # Default implementation for base class (rectangular)
        # BlockEventItem will override this with rounded rectangle
        path = QPainterPath()
        path.addRect(self.rect())
        return path
    
    def _update_tooltip(self):
        """Update tooltip text."""
        tooltip = f"{self.classification}" if self.classification else "Event"
        if self.duration > 0:
            tooltip += f"\nDuration: {self.duration:.3f}s"
        tooltip += f"\nTime: {self.start_time:.3f}s"
        
        # Show layer name
        layer = self._layer_manager.get_layer(self.layer_id)
        if layer:
            tooltip += f"\nLayer: {layer.name}"
        
        # Show audio source if present
        if self.audio_name:
            tooltip += f"\nAudio: {self.audio_name}"
        
        # Show user data (skip internal keys)
        if self.user_data:
            for key, value in list(self.user_data.items())[:3]:
                if not key.startswith('_'):
                    tooltip += f"\n{key}: {value}"
        
        self.setToolTip(tooltip)
    
    # ===================================================================
    # Geometry Calculation & Updates
    # ===================================================================
    
    def _get_event_height(self) -> float:
        """Get event height from settings or use default (for block events)."""
        settings_mgr = get_timeline_settings_manager()
        if settings_mgr:
            return float(settings_mgr.block_event_height)
        return float(EVENT_HEIGHT)
    
    def _apply_qt_styling(self, prefix: str = "block_event"):
        """Apply Qt graphics item styling from settings.
        
        Args:
            prefix: Settings prefix - "block_event" or "marker_event"
        """
        settings_mgr = get_timeline_settings_manager()
        if not settings_mgr:
            return
        
        # Get settings based on prefix
        opacity = getattr(settings_mgr, f"{prefix}_opacity", 1.0)
        z_value = getattr(settings_mgr, f"{prefix}_z_value", 0.0)
        rotation = getattr(settings_mgr, f"{prefix}_rotation", 0.0)
        scale = getattr(settings_mgr, f"{prefix}_scale", 1.0)
        shadow_enabled = getattr(settings_mgr, f"{prefix}_drop_shadow_enabled", False)
        shadow_blur = getattr(settings_mgr, f"{prefix}_drop_shadow_blur_radius", 5.0)
        shadow_offset_x = getattr(settings_mgr, f"{prefix}_drop_shadow_offset_x", 2.0)
        shadow_offset_y = getattr(settings_mgr, f"{prefix}_drop_shadow_offset_y", 2.0)
        shadow_color_str = getattr(settings_mgr, f"{prefix}_drop_shadow_color", "#000000")
        shadow_opacity = getattr(settings_mgr, f"{prefix}_drop_shadow_opacity", 0.5)
        
        # Opacity
        self.setOpacity(opacity)
        
        # Z-value (stacking order)
        self.setZValue(z_value)
        
        # Transform (rotation and scale) - set origin to center of item
        rect = self.rect()
        self.setTransformOriginPoint(rect.center())
        
        # Create transform with rotation and scale
        transform = QTransform()
        transform.rotate(rotation)
        transform.scale(scale, scale)
        self.setTransform(transform)
        
        # Drop shadow effect
        if shadow_enabled:
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(shadow_blur)
            shadow.setOffset(shadow_offset_x, shadow_offset_y)
            shadow_color = QColor(shadow_color_str)
            shadow_color.setAlphaF(shadow_opacity)
            shadow.setColor(shadow_color)
            self.setGraphicsEffect(shadow)
        else:
            # Remove effect if disabled
            self.setGraphicsEffect(None)
    
    def _update_geometry(self):
        """Update position and size based on time and layer."""
        # DEFENSIVE: Ensure _pixels_per_second is valid before using it
        # This can be None during lazy updates if item wasn't fully initialized
        if self._pixels_per_second is None:
            # Try to get from scene if available
            scene = self.scene()
            if scene and hasattr(scene, '_pixels_per_second'):
                self._pixels_per_second = scene._pixels_per_second
            else:
                # Fallback to default - can't update geometry without valid PPS
                from ..constants import DEFAULT_PIXELS_PER_SECOND
                self._pixels_per_second = DEFAULT_PIXELS_PER_SECOND
        
        # Position is directly from time * pixels_per_second
        # Snapping happens BEFORE this (in snap_time() or itemChange())
        # Do NOT round here - grid lines are at float positions
        x = self.start_time * self._pixels_per_second
        
        # Apply visual alignment offset (subclasses can override _get_alignment_offset())
        x += self._get_alignment_offset()
        
        # Get Y position from layer manager (single source of truth)
        y_base = self._layer_manager.get_layer_y_position(self.layer_id)
        layer_height = self._layer_manager.get_layer_height(self.layer_id)
        event_height = self._get_event_height()
        y = y_base + (layer_height - event_height) / 2
        
        width = self._calculate_width()
        
        # Update geometry - always call prepareGeometryChange() and setRect/setPos
        # Attempted optimization to skip when unchanged was reverted because:
        # 1. self.rect() and self.pos() create objects on every call
        # 2. If _update_geometry() called frequently, this accumulates memory
        # 3. Qt's prepareGeometryChange() is fast enough for the benefit it provides
        self.prepareGeometryChange()
        self.setRect(0, 0, width, event_height)
        self.setPos(x, y)
        
        # PERFORMANCE: Invalidate waveform path cache when geometry changes
        # (position/duration/height changed - path needs rebuild)
        self._cached_waveform_path = None
        self._cached_path_key = None
        
        # Update color in case layer changed
        self._base_color = self._get_layer_color()
    
    def _get_marker_width(self) -> float:
        """Get marker width from settings or use default."""
        settings_mgr = get_timeline_settings_manager()
        if settings_mgr:
            return float(settings_mgr.marker_event_width)
        return float(MARKER_WIDTH)
    
    def _calculate_width(self) -> float:
        """Calculate item width - override in subclasses."""
        marker_width = self._get_marker_width()
        if self.duration > 0:
            return max(self.duration * self._pixels_per_second, marker_width)
        return marker_width
    
    def _get_alignment_offset(self) -> float:
        """Get X offset for visual alignment with grid lines.
        
        Subclasses override this to handle different alignment needs:
        - BlockEventItem: offset by -border_width/2 so left fill edge aligns
        - MarkerEventItem: offset by -width/2 so center aligns
        
        Returns:
            X offset in pixels (negative to move left)
        """
        # Default: no offset
        return 0.0
    
    def set_pixels_per_second(self, pps: float):
        """Update zoom level and recalculate geometry."""
        # DEFENSIVE: Validate input
        if pps is None or pps <= 0:
            from ..logging import TimelineLog as Log
            Log.warning(f"BaseEventItem: Invalid pixels_per_second value: {pps}, using default")
            from ..constants import DEFAULT_PIXELS_PER_SECOND
            pps = DEFAULT_PIXELS_PER_SECOND
        
        self._pixels_per_second = pps
        self._update_geometry()
    
    def set_native_snapping(self, enabled: bool, snap_enabled: bool = True, snap_interval: float = 0.1):
        """
        Configure native snapping behavior (POC-verified).
        
        Args:
            enabled: Whether to use native itemChange() snapping
            snap_enabled: Whether snapping is actually active
            snap_interval: Grid interval in seconds
        """
        self._use_native_snapping = enabled
        self._snap_enabled = snap_enabled
        self._snap_interval = snap_interval
    
    def sync_snap_settings_from_scene(self):
        """
        Sync snap settings from scene's grid system.
        
        Call this before starting a drag to ensure snap settings are current.
        """
        scene = self.scene()
        if scene and hasattr(scene, 'grid_system'):
            grid_system = scene.grid_system
            self._snap_enabled = grid_system.snap_enabled
            
            # Get current snap interval
            if hasattr(grid_system, 'get_snap_interval'):
                self._snap_interval = grid_system.get_snap_interval(self._pixels_per_second)
            elif hasattr(grid_system, 'get_major_minor_intervals'):
                _, minor = grid_system.get_major_minor_intervals(self._pixels_per_second)
                self._snap_interval = minor
    
    # ===================================================================
    # Time/Coordinate Conversion
    # ===================================================================
    
    def time_to_x(self, time: float) -> float:
        """Convert time to x coordinate."""
        return time * self._pixels_per_second
    
    def x_to_time(self, x: float) -> float:
        """Convert x coordinate to time."""
        return x / self._pixels_per_second
    
    # ===================================================================
    # Movement Controller
    # ===================================================================
    
    def _get_movement_controller(self) -> Optional['MovementController']:
        """Get the movement controller from scene."""
        scene = self.scene()
        if scene and hasattr(scene, 'movement_controller'):
            return scene.movement_controller
        return None
    
    # ===================================================================
    # Selection & Interaction
    # ===================================================================
    
    def _get_selected_items(self):
        """Get all selected event items.
        
        Returns only items that are actually selected in the scene.
        If no items are selected but self is selected, returns [self].
        If self is not selected and nothing else is, returns empty list.
        
        NOTE: This should be called AFTER mousePressEvent has updated selection.
        """
        scene = self.scene()
        if not scene:
            # No scene - return self only if we're selected
            result = [self] if self.isSelected() else []
            return result
        
        # Get all selected event items from scene
        all_selected = scene.selectedItems()
        selected = [item for item in all_selected 
                    if isinstance(item, BaseEventItem)]
        
        # Safety check: if somehow we're selected but not in the list
        # (should not happen, but defensive coding)
        if self.isSelected() and self not in selected:
            selected.append(self)
        
        return selected
    
    def _create_marker_path(self, shape: str, cx: float, cy: float, size: float) -> QPainterPath:
        """Create a QPainterPath for the specified marker shape.
        
        Shared by BlockEventItem (when render_as_marker=True) and MarkerEventItem.
        """
        path = QPainterPath()
        
        if shape == "diamond":
            path.moveTo(cx, cy - size)
            path.lineTo(cx + size, cy)
            path.lineTo(cx, cy + size)
            path.lineTo(cx - size, cy)
            path.closeSubpath()
        
        elif shape == "circle":
            path.addEllipse(cx - size, cy - size, size * 2, size * 2)
        
        elif shape == "square":
            path.addRect(cx - size, cy - size, size * 2, size * 2)
        
        elif shape == "triangle_up":
            path.moveTo(cx, cy - size)
            path.lineTo(cx + size, cy + size)
            path.lineTo(cx - size, cy + size)
            path.closeSubpath()
        
        elif shape == "triangle_down":
            path.moveTo(cx, cy + size)
            path.lineTo(cx + size, cy - size)
            path.lineTo(cx - size, cy - size)
            path.closeSubpath()
        
        elif shape == "triangle_left":
            path.moveTo(cx - size, cy)
            path.lineTo(cx + size, cy - size)
            path.lineTo(cx + size, cy + size)
            path.closeSubpath()
        
        elif shape == "triangle_right":
            path.moveTo(cx + size, cy)
            path.lineTo(cx - size, cy - size)
            path.lineTo(cx - size, cy + size)
            path.closeSubpath()
        
        elif shape == "arrow_up":
            # Arrow pointing up
            arrow_head_size = size * 0.6
            path.moveTo(cx, cy - size)
            path.lineTo(cx + arrow_head_size, cy - size * 0.3)
            path.lineTo(cx + size * 0.3, cy - size * 0.3)
            path.lineTo(cx + size * 0.3, cy + size)
            path.lineTo(cx - size * 0.3, cy + size)
            path.lineTo(cx - size * 0.3, cy - size * 0.3)
            path.lineTo(cx - arrow_head_size, cy - size * 0.3)
            path.closeSubpath()
        
        elif shape == "arrow_down":
            # Arrow pointing down
            arrow_head_size = size * 0.6
            path.moveTo(cx, cy + size)
            path.lineTo(cx + arrow_head_size, cy + size * 0.3)
            path.lineTo(cx + size * 0.3, cy + size * 0.3)
            path.lineTo(cx + size * 0.3, cy - size)
            path.lineTo(cx - size * 0.3, cy - size)
            path.lineTo(cx - size * 0.3, cy + size * 0.3)
            path.lineTo(cx - arrow_head_size, cy + size * 0.3)
            path.closeSubpath()
        
        elif shape == "arrow_left":
            # Arrow pointing left
            arrow_head_size = size * 0.6
            path.moveTo(cx - size, cy)
            path.lineTo(cx - size * 0.3, cy - arrow_head_size)
            path.lineTo(cx - size * 0.3, cy - size * 0.3)
            path.lineTo(cx + size, cy - size * 0.3)
            path.lineTo(cx + size, cy + size * 0.3)
            path.lineTo(cx - size * 0.3, cy + size * 0.3)
            path.lineTo(cx - size * 0.3, cy + arrow_head_size)
            path.closeSubpath()
        
        elif shape == "arrow_right":
            # Arrow pointing right
            arrow_head_size = size * 0.6
            path.moveTo(cx + size, cy)
            path.lineTo(cx + size * 0.3, cy - arrow_head_size)
            path.lineTo(cx + size * 0.3, cy - size * 0.3)
            path.lineTo(cx - size, cy - size * 0.3)
            path.lineTo(cx - size, cy + size * 0.3)
            path.lineTo(cx + size * 0.3, cy + size * 0.3)
            path.lineTo(cx + size * 0.3, cy + arrow_head_size)
            path.closeSubpath()
        
        elif shape == "star":
            # 5-pointed star
            outer_radius = size
            inner_radius = size * 0.4
            points = 5
            for i in range(points * 2):
                angle = (i * pi) / points - pi / 2
                radius = outer_radius if i % 2 == 0 else inner_radius
                x = cx + radius * cos(angle)
                y = cy + radius * sin(angle)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            path.closeSubpath()
        
        elif shape == "cross":
            # X shape
            thickness = size * 0.2
            path.moveTo(cx - size, cy - thickness)
            path.lineTo(cx - thickness, cy - thickness)
            path.lineTo(cx - thickness, cy - size)
            path.lineTo(cx + thickness, cy - size)
            path.lineTo(cx + thickness, cy - thickness)
            path.lineTo(cx + size, cy - thickness)
            path.lineTo(cx + size, cy + thickness)
            path.lineTo(cx + thickness, cy + thickness)
            path.lineTo(cx + thickness, cy + size)
            path.lineTo(cx - thickness, cy + size)
            path.lineTo(cx - thickness, cy + thickness)
            path.lineTo(cx - size, cy + thickness)
            path.closeSubpath()
        
        return path
    
    # ===================================================================
    # Qt Item Change Events
    # ===================================================================
    
    def itemChange(self, change, value):
        """
        Handle item state changes.
        
        Includes native Qt snapping support (POC-verified):
        - When _use_native_snapping is True, ItemPositionChange triggers snapping
        - This gives smoother, more responsive dragging than controller-based snapping
        - Layer constraints are always applied for Y position
        """
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            self._selected = bool(value)
            self.update()
        
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Native position snapping - uses base interval for finer snapping
            new_pos = value
            
            # === X AXIS: Time snapping ===
            if self._use_native_snapping and self._snap_enabled and self._pixels_per_second > 0:
                # Use base minor interval directly for finer snapping precision
                # (Grid renderer uses adaptive density for visual spacing, but snapping uses
                # the base interval for finer control)
                scene = self.scene()
                if scene and hasattr(scene, 'grid_system'):
                    _, minor_interval = scene.grid_system.get_major_minor_intervals(
                        self._pixels_per_second
                    )
                    
                    # Use base minor interval directly (no adaptive density skip)
                    if minor_interval > 0:
                        # Convert position to time, snap to base interval, convert back
                        time = new_pos.x() / self._pixels_per_second
                        line_idx = round(time / minor_interval)
                        snapped_time = line_idx * minor_interval
                        snapped_time = max(0, snapped_time)
                        new_x = snapped_time * self._pixels_per_second
                        self.start_time = snapped_time
                    else:
                        new_x = max(0, new_pos.x())
                        self.start_time = new_x / self._pixels_per_second
                else:
                    # Fallback to simple snapping
                    new_x = max(0, new_pos.x())
                    self.start_time = new_x / self._pixels_per_second
            else:
                # No native snapping - use position as-is (MovementController handles snapping)
                new_x = new_pos.x()
            
            # === Y AXIS: Layer constraint (always applied) ===
            # Constrain Y to valid layer positions using LayerManager
            new_y = new_pos.y()
            
            if self._layer_manager:
                # Get layer from Y position
                layer = self._layer_manager.get_layer_from_y(new_y)
                if layer:
                    # Snap to layer's Y position
                    constrained_y = self._layer_manager.get_layer_y_position(layer.id)
                    # Add padding for visual centering
                    layer_height = layer.height if hasattr(layer, 'height') else TRACK_HEIGHT
                    event_height = self._get_event_height()
                    padding = (layer_height - event_height) / 2
                    constrained_y += padding
                    new_y = constrained_y
                    
                    # Update layer assignment if changed
                    if self.layer_id != layer.id:
                        self.layer_id = layer.id
                        self._base_color = self._get_layer_color()
            
            return QPointF(new_x, new_y)
        
        return super().itemChange(change, value)
    
    def mousePressEvent(self, event):
        """Handle mouse press - intercept right-click to preserve multi-selection."""
        if event.button() == Qt.MouseButton.RightButton:
            # Right-click: preserve existing selection for context menu
            # If this item is part of selection, keep it; otherwise select just this item
            if not self.isSelected():
                scene = self.scene()
                if scene:
                    scene.clearSelection()
                self.setSelected(True)
            # Accept event to prevent default selection-clearing behavior
            event.accept()
            return
        
        # Let subclasses handle left-click and other buttons
        super().mousePressEvent(event)
    
    def contextMenuEvent(self, event):
        """Show context menu for selected events (supports batch operations)."""
        if not self._editable:
            return
        
        # Selection is already handled in mousePressEvent (right-click)
        # Get all selected items for batch operations
        selected_items = self._get_selected_items()
        selected_count = len(selected_items)
        selected_ids = [getattr(item, "event_id", None) for item in selected_items if getattr(item, "event_id", None)]
        
        menu = QMenu()
        
        # Delete action - reflects batch operation
        if selected_count > 1:
            delete_action = menu.addAction(f"Delete {selected_count} Events")
        else:
            delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda checked=False, ids=selected_ids: self._request_batch_delete(ids))
        
        menu.addSeparator()
        
        # Display Mode submenu (for render_as_marker)
        scene = self.scene()
        if scene and hasattr(scene, '_event_update_callback') and scene._event_update_callback:
            display_mode_menu = menu.addMenu("Display Mode")
            
            # Get current render_as_marker state from first selected item's metadata
            current_render_as_marker = False
            if selected_items:
                first_item = selected_items[0]
                # Check both the attribute and user_data metadata
                current_render_as_marker = getattr(first_item, 'render_as_marker', False)
                if not current_render_as_marker and hasattr(first_item, 'user_data'):
                    current_render_as_marker = first_item.user_data.get('render_as_marker', False)
            
            # Show as Marker action
            show_as_marker_action = display_mode_menu.addAction("Show as Marker")
            show_as_marker_action.setCheckable(True)
            show_as_marker_action.setChecked(current_render_as_marker)
            show_as_marker_action.triggered.connect(
                lambda checked=False, ids=selected_ids: self._request_batch_update_display_mode(ids, True)
            )
            
            # Show as Clip action
            show_as_clip_action = display_mode_menu.addAction("Show as Clip")
            show_as_clip_action.setCheckable(True)
            show_as_clip_action.setChecked(not current_render_as_marker)
            show_as_clip_action.triggered.connect(
                lambda checked=False, ids=selected_ids: self._request_batch_update_display_mode(ids, False)
            )
        
        menu.addSeparator()
        
        # Properties action (disabled - properties shown in EventInspector panel)
        if selected_count > 1:
            props_action = menu.addAction(f"Properties ({selected_count} Events)...")
        else:
            props_action = menu.addAction("Properties...")
        props_action.setEnabled(False)  # Properties handled by EventInspector panel, not context menu
        
        menu.exec(event.screenPos())
    
    def _request_batch_delete(self, event_ids: list):
        """Request deletion of multiple events through scene.
        
        Uses:
        1. CommandBus macro for single undo step
        2. Batch deletion API for performance (single signal, suspended updates)
        """
        scene = self.scene()
        if not scene:
            return
        
        count = len(event_ids)
        if count == 0:
            return
        
        # Get command_bus from scene (explicit dependency)
        command_bus = getattr(scene, '_command_bus', None)
        if not command_bus:
            Log.warning("BaseEventItem: Cannot delete events - CommandBus not set on scene")
            return
        
        # Use macro for batch operations (single undo step)
        if count > 1:
            command_bus.begin_macro(f"Delete {count} Events")
        
        try:
            # Always use batch API (works for single events too - unified pathway)
            if hasattr(scene, 'request_events_delete_batch'):
                scene.request_events_delete_batch(event_ids)
            else:
                # Fallback only if batch API not available (shouldn't happen)
                for event_id in event_ids:
                    if hasattr(scene, 'request_event_delete'):
                        scene.request_event_delete(event_id)
        finally:
            if count > 1:
                command_bus.end_macro()
    
    def _request_delete(self):
        """Request deletion through scene (single event - legacy method)."""
        scene = self.scene()
        if scene and hasattr(scene, 'request_event_delete'):
            scene.request_event_delete(self.event_id)
    
    def _request_batch_update_display_mode(self, event_ids: list, render_as_marker: bool):
        """Request batch update of render_as_marker for selected events.
        
        Args:
            event_ids: List of event IDs to update
            render_as_marker: New value for render_as_marker
        """
        scene = self.scene()
        if not scene:
            return
        
        # Get event update callback from scene
        callback = getattr(scene, '_event_update_callback', None)
        if not callback:
            Log.warning("BaseEventItem: Cannot update display mode - event update callback not set on scene")
            return

        command_bus = getattr(scene, '_command_bus', None)
        use_macro = command_bus is not None and len(event_ids) > 1
        if use_macro:
            mode_label = "Marker" if render_as_marker else "Clip"
            command_bus.begin_macro(f"Set {len(event_ids)} Events to {mode_label}")
        
        try:
            # Update each event
            for event_id in event_ids:
                if not event_id:
                    continue
                
                # Get current metadata from event item
                event_item = scene._event_items.get(event_id) if hasattr(scene, '_event_items') else None
                if event_item:
                    current_metadata = getattr(event_item, 'user_data', {}).copy()
                else:
                    current_metadata = {}
                
                # Update render_as_marker
                current_metadata['render_as_marker'] = render_as_marker
                
                # Call update callback
                try:
                    success = callback(event_id, current_metadata)
                    if not success:
                        Log.warning(f"BaseEventItem: Failed to update display mode for event {event_id}")
                except Exception as e:
                    Log.error(f"BaseEventItem: Error updating display mode for event {event_id}: {e}")
                    import traceback
                    traceback.print_exc()
        finally:
            if use_macro:
                command_bus.end_macro()


class BlockEventItem(BaseEventItem):
    """
    Event item with duration (rendered as a bar/block).
    
    Features:
    - Drag to move (time and layer)
    - Resize handles on left/right edges
    - Multi-select support
    - Waveform preview for clip events
    """
    
    def __init__(self, *args, **kwargs):
        # Extract render_as_marker from kwargs before passing to super
        render_as_marker = kwargs.pop('render_as_marker', False)
        super().__init__(*args, render_as_marker=render_as_marker, **kwargs)
        self._min_duration = 0.01  # Minimum 10ms
        
        # Simple waveform caching per-item
        self._cached_waveform_path: Optional[QPainterPath] = None
        self._cached_path_key: Optional[Tuple] = None
        self._cached_waveform_data: Optional[Tuple] = None
        self._cached_waveform_audio_key: Optional[str] = None
        
        # Drag state - during drag, don't show waveform (prevents stretching)
        self._is_dragging = False
        
        # Apply Qt styling based on render mode
        if self.render_as_marker:
            self._apply_qt_styling("marker_event")
        else:
            self._apply_qt_styling("block_event")
    
    def _on_settings_changed(self, setting_name: str):
        """Handle settings changes - update if event styling changed."""
        # Call parent first
        super()._on_settings_changed(setting_name)
        
        # Handle marker vs block event styling
        if self.render_as_marker:
            if setting_name.startswith('marker_event_'):
                if setting_name in ('marker_event_width', 'marker_event_shape'):
                    # Width or shape change requires geometry update
                    self._update_geometry()
                elif setting_name in ('marker_event_opacity', 'marker_event_z_value', 'marker_event_rotation', 'marker_event_scale',
                                     'marker_event_drop_shadow_enabled', 'marker_event_drop_shadow_blur_radius',
                                     'marker_event_drop_shadow_offset_x', 'marker_event_drop_shadow_offset_y',
                                     'marker_event_drop_shadow_color', 'marker_event_drop_shadow_opacity'):
                    # Qt styling changes
                    self._apply_qt_styling("marker_event")
                else:
                    # Other styling changes just need repaint
                    self.update()
        else:
            # Block event styling changes handled by parent
            pass
    
    def on_drag_start(self):
        """Called when drag/resize operation starts. Hides waveform during drag."""
        self._is_dragging = True
        self.update()  # Repaint without waveform
    
    def on_drag_end(self):
        """Called when drag/resize operation ends. Redraws waveform at new position.
        
        Clips are windows into source audio - position determines what audio slice to show.
        Waveform slice is computed from start_time and start_time + duration.
        """
        self._is_dragging = False
        
        # Invalidate cache to force reload at new position
        self._invalidate_waveform_cache()
        
        self.update()  # Repaint with new waveform
    
    # ===================================================================
    # Geometry Updates
    # ===================================================================
    
    def _update_geometry(self):
        """Update position and size based on time and layer."""
        # Call parent to handle standard geometry updates
        super()._update_geometry()
    
    # ===================================================================
    # Settings & Configuration
    # ===================================================================
    
    def _on_settings_changed(self, setting_name: str):
        """Handle settings changes - update if block event styling changed."""
        if setting_name.startswith('block_event_'):
            if setting_name == 'block_event_height':
                # Height change requires geometry update
                self._update_geometry()
            elif setting_name in ('block_event_opacity', 'block_event_z_value', 'block_event_rotation', 'block_event_scale',
                                 'block_event_drop_shadow_enabled', 'block_event_drop_shadow_blur_radius',
                                 'block_event_drop_shadow_offset_x', 'block_event_drop_shadow_offset_y',
                                 'block_event_drop_shadow_color', 'block_event_drop_shadow_opacity'):
                # Qt styling changes
                self._apply_qt_styling("block_event")
            else:
                # Other styling changes (border, font, etc.) just need repaint
                self.update()
        elif setting_name in ('show_event_labels', 'show_event_duration_labels', 'highlight_current_event'):
            # Appearance settings changed - repaint to update labels/highlighting
            self.update()
        elif setting_name == 'waveform_opacity':
            # Waveform opacity changed - repaint to update waveform rendering
            self.update()
        elif setting_name == 'show_waveforms_in_timeline':
            # Waveform visibility setting changed - trigger scene update for track-level redraw
            scene = self.scene()
            if scene:
                scene.invalidate(scene.sceneRect())
    
    # ===================================================================
    # Waveform Support
    # ===================================================================
    
    def _is_clip_event(self) -> bool:
        """Check if this event is a clip event (has audio source)."""
        return bool(self.audio_id or self.audio_name)
    
    def _invalidate_waveform_cache(self):
        """Clear cached waveform data so it reloads on next paint."""
        self._cached_waveform_data = None
        self._cached_waveform_audio_key = None
        self._cached_waveform_path = None
        self._cached_path_key = None
    
    def _should_show_waveform(self) -> bool:
        """
        Simple check: show waveform if global setting is enabled and event has audio.
        """
        from ..settings.storage import get_timeline_settings_manager
        settings_mgr = get_timeline_settings_manager()
        if not settings_mgr or not settings_mgr.show_waveforms_in_timeline:
            return False
        
        # Must be clip event with audio source
        return self._is_clip_event()
    
    # ===================================================================
    # Qt Item Change Events
    # ===================================================================
    
    def itemChange(self, change, value):
        """Handle item state changes."""
        result = super().itemChange(change, value)
        
        if change == QGraphicsItem.GraphicsItemChange.ItemSceneHasChanged:
            # Item was removed from scene - clear caches to free memory
            if value is None:
                self._cached_waveform_path = None
                self._cached_path_key = None
                self._cached_waveform_data = None
                self._cached_waveform_audio_key = None
                self._cached_waveform_data = None
                self._cached_waveform_audio_key = None
        
        return result
    
    def _cancel_pending_waveform_load(self):
        """Cancel any pending waveform load (no-op in simple mode)."""
        pass  # No async loading in simple mode
    
    # ===================================================================
    # Geometry & Positioning
    # ===================================================================
    
    def _calculate_width(self) -> float:
        """Calculate width based on duration or marker mode."""
        if self.render_as_marker:
            # Markers have configurable width from settings
            settings_mgr = get_timeline_settings_manager()
            if settings_mgr:
                return float(settings_mgr.marker_event_width)
            return float(MARKER_WIDTH)
        else:
            # Block/clip events use duration-based width
            marker_width = self._get_marker_width()
            return max(self.duration * self._pixels_per_second, marker_width)
    
    def _get_alignment_offset(self) -> float:
        """Get alignment offset - different for markers vs blocks."""
        if self.render_as_marker:
            # Markers are centered on grid lines
            width = self._calculate_width()
            return -width / 2.0
        else:
            # Blocks align left edge with grid lines
            # Qt draws borders centered on rect edges, so a 1px border means
            # the fill starts 0.5px to the right of the rect position.
            # Offset by -border_width/2 so the fill's left edge aligns with grid.
            border_width = self._get_border_width()
            return -border_width / 2.0
    
    def shape(self) -> QPainterPath:
        """
        Return exact shape for hit testing (matches visual appearance).
        
        PYTQT BEST PRACTICE: Override shape() to return shape that matches
        the visual appearance, rather than default rectangular shape. This provides better
        UX for hover/click detection - hit testing matches what the user sees.
        """
        rect = self.rect()
        
        # If rendering as marker, return diamond shape for hit testing
        if self.render_as_marker:
            path = QPainterPath()
            cx = rect.width() / 2
            cy = rect.height() / 2
            size = min(rect.width(), rect.height()) / 2
            path = self._create_marker_path("diamond", cx, cy, size)
            return path
        
        # Otherwise return rounded rectangle for clip/block events
        path = QPainterPath()
        settings_mgr = get_timeline_settings_manager()
        from ui.qt_gui.design_system import is_sharp_corners
        br = 0 if is_sharp_corners() else (settings_mgr.block_event_border_radius if settings_mgr else 3)
        path.addRoundedRect(rect, br, br)
        return path
    
    def _get_alignment_offset(self) -> float:
        """Offset for left edge alignment with grid lines.
        
        Qt draws borders centered on rect edges, so a 1px border means
        the fill starts 0.5px to the right of the rect position.
        Offset by -border_width/2 so the fill's left edge aligns with grid.
        """
        settings_mgr = get_timeline_settings_manager()
        border_width = settings_mgr.block_event_border_width if settings_mgr else 1
        return -border_width / 2.0
    
    # ===================================================================
    # Hover & Cursor Management
    # ===================================================================
    
    def hoverEnterEvent(self, event):
        """Handle hover enter."""
        self._hovered = True
        self._update_cursor(event.pos())
        # Only update visual state - no computation/loading on hover
        self.update()
        super().hoverEnterEvent(event)
    
    def hoverMoveEvent(self, event):
        """Update cursor based on position."""
        if self._editable:
            self._update_cursor(event.pos())
        super().hoverMoveEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Handle hover leave."""
        self._hovered = False
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        # Waveform loading is now automatic for visible events (handled in drawBackground)
        # No need to cancel on hover leave - waveforms persist for visible events
        self.update()
        super().hoverLeaveEvent(event)
    
    def _get_resize_handle_width(self) -> float:
        """Calculate adaptive resize handle width based on event size."""
        rect = self.rect()
        width = rect.width()
        
        handle_width = max(
            min(RESIZE_HANDLE_WIDTH, width * RESIZE_HANDLE_PERCENT),
            MIN_RESIZE_HANDLE_WIDTH
        )
        
        if handle_width * 2 + MIN_MOVE_AREA_WIDTH > width:
            handle_width = max((width - MIN_MOVE_AREA_WIDTH) / 2, MIN_RESIZE_HANDLE_WIDTH)
        
        return handle_width
    
    def _update_cursor(self, pos: QPointF):
        """Update cursor based on hover position."""
        if not self._editable:
            return
        
        rect = self.rect()
        handle_width = self._get_resize_handle_width()
        
        if pos.x() <= handle_width:
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
        elif pos.x() >= rect.width() - handle_width:
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
    
    # ===================================================================
    # Mouse Events
    # ===================================================================
    
    def mousePressEvent(self, event):
        """Handle mouse press - start drag operation."""
        if not self._editable or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        
        pos = event.pos()
        rect = self.rect()
        handle_width = self._get_resize_handle_width()
        
        # Alt modifier forces move mode
        force_move = event.modifiers() & Qt.KeyboardModifier.AltModifier
        
        # Determine edit handle
        if force_move:
            self._edit_handle = EditHandle.MOVE
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        elif pos.x() <= handle_width:
            self._edit_handle = EditHandle.RESIZE_LEFT
        elif pos.x() >= rect.width() - handle_width:
            self._edit_handle = EditHandle.RESIZE_RIGHT
        else:
            self._edit_handle = EditHandle.MOVE
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        
        # Handle selection
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.setSelected(True)
        else:
            if not self.isSelected():
                scene = self.scene()
                if scene:
                    scene.clearSelection()
                self.setSelected(True)
        
        # Start drag via controller
        controller = self._get_movement_controller()
        if controller:
            if self._edit_handle == EditHandle.MOVE:
                controller.begin_move(self, event.scenePos(), self._get_selected_items())
            elif self._edit_handle in (EditHandle.RESIZE_LEFT, EditHandle.RESIZE_RIGHT):
                controller.begin_resize(self, event.scenePos(), self._edit_handle)
        
        event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move - drag operation."""
        if self._edit_handle == EditHandle.NONE:
            super().mouseMoveEvent(event)
            return
        
        controller = self._get_movement_controller()
        if controller:
            scene = self.scene()
            snap_enabled = False
            snap_func = None
            snap_calculator = None
            unit_preference = None
            
            if scene and hasattr(scene, 'grid_system'):
                snap_enabled = scene.grid_system.snap_enabled
                # Try new system first
                if hasattr(scene, '_snap_calculator') and hasattr(scene, '_unit_preference'):
                    snap_calculator = scene._snap_calculator
                    unit_preference = scene._unit_preference
                else:
                    # Fallback to legacy
                    snap_func = scene.grid_system.snap_time
            
            controller.update_drag(
                event.scenePos(),
                snap_enabled,
                snap_func,
                snap_calculator,
                unit_preference
            )
        
        event.accept()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release - end drag operation."""
        if self._edit_handle == EditHandle.NONE:
            super().mouseReleaseEvent(event)
            return
        
        controller = self._get_movement_controller()
        if controller:
            controller.commit_drag()
        
        if self._edit_handle == EditHandle.MOVE:
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        
        self._edit_handle = EditHandle.NONE
        event.accept()
    
    def set_pixels_per_second(self, pps: float):
        """Update pixels per second - also check if waveform should reload at new resolution."""
        # DEFENSIVE: Handle None case
        old_pps = self._pixels_per_second if self._pixels_per_second is not None else 0.0
        old_width = self.rect().width()
        super().set_pixels_per_second(pps)
        
        # PERFORMANCE: Only invalidate cache if zoom actually changed significantly
        # This prevents unnecessary path rebuilding during smooth zoom operations
        # DEFENSIVE: Only compare if old_pps was valid
        if old_pps > 0 and abs(old_pps - pps) > 0.1:  # Only rebuild if zoom changed by >10%
            self._cached_waveform_path = None
            self._cached_path_key = None
    
    def paint(self, painter: QPainter, option, widget=None):
        """
        Paint the block event.
        
        Qt handles viewport culling automatically - paint() is only called
        for items within the viewport.
        """
        rect = self.rect()
        
        # Very small events: minimal drawing
        if rect.width() < 2 or rect.height() < 2:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self._base_color))
            painter.drawRect(rect)
            return
        
        # Check if this should render as a marker (diamond) instead of a clip (rectangle)
        if self.render_as_marker:
            # Render as marker/diamond (similar to MarkerEventItem.paint())
            settings_mgr = get_timeline_settings_manager()
            if settings_mgr:
                shape = settings_mgr.marker_event_shape
                border_width = settings_mgr.marker_event_border_width
                border_darken = settings_mgr.marker_event_border_darken_percent
            else:
                shape = "diamond"
                border_width = 1
                border_darken = 150
            
            if self._selected:
                color = TimelineStyle.SELECTION_COLOR
                border_color = TimelineStyle.SELECTION_COLOR.darker(130)
            elif self._hovered:
                color = self._base_color.lighter(120)
                border_color = self._base_color.darker(130)
            else:
                color = self._base_color
                border_color = self._base_color.darker(border_darken)
            
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(border_color, border_width))
            painter.setBrush(QBrush(color))
            
            cx = rect.width() / 2
            cy = rect.height() / 2
            size = min(rect.width(), rect.height()) / 2 - 2
            
            path = self._create_marker_path(shape, cx, cy, size)
            painter.drawPath(path)
            return
        
        # Get settings (simple lookup, no caching needed)
        settings_mgr = get_timeline_settings_manager()
        from ui.qt_gui.design_system import is_sharp_corners
        if settings_mgr:
            border_radius = 0 if is_sharp_corners() else settings_mgr.block_event_border_radius
            border_width = settings_mgr.block_event_border_width
            label_font_size = settings_mgr.block_event_label_font_size
            border_darken = settings_mgr.block_event_border_darken_percent
            min_width = settings_mgr.waveform_min_width
        else:
            border_radius = 0 if is_sharp_corners() else 3
            border_width = 1
            label_font_size = 10
            border_darken = 150
            min_width = 30
        
        # LOD: skip details for small events
        draw_details = rect.width() >= min_width
        
        # Check if playhead is over this event (for highlighting)
        highlight_at_playhead = False
        if settings_mgr and settings_mgr.highlight_current_event:
            scene = self.scene()
            if scene and hasattr(scene, 'get_playhead_position'):
                playhead_time = scene.get_playhead_position()
                # Check if playhead is within event bounds
                if self.start_time <= playhead_time <= (self.start_time + self.duration):
                    highlight_at_playhead = True
        
        # Calculate colors based on state
        if self._selected:
            color = TimelineStyle.SELECTION_COLOR
            border_color = TimelineStyle.SELECTION_COLOR.darker(130)
        elif highlight_at_playhead:
            # Highlight when playhead is over event
            color = self._base_color.lighter(140)
            border_color = self._base_color.lighter(120)
        elif self._hovered:
            color = self._base_color.lighter(120)
            border_color = self._base_color.darker(130)
        else:
            color = self._base_color
            border_color = self._base_color.darker(border_darken)
        
        # PERFORMANCE: Only use antialiasing for larger events (expensive for many small ones)
        use_antialiasing = rect.width() > 10 and rect.height() > 10
        if use_antialiasing:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw main rectangle
        painter.setPen(QPen(border_color, border_width))
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(rect, border_radius, border_radius)
        
        # draw_details already set above based on event width (proxy for zoom level)
        
        # Draw resize handles when hovered/selected (skip for very small events or low LOD)
        if draw_details and (self._hovered or self._selected) and self._editable and rect.width() > RESIZE_HANDLE_WIDTH * 2:
            handle_color = QColor(TimelineStyle.TEXT_PRIMARY)
            handle_color.setAlpha(80)
            painter.setBrush(QBrush(handle_color))
            painter.setPen(Qt.PenStyle.NoPen)
            
            painter.drawRect(QRectF(0, 0, RESIZE_HANDLE_WIDTH, rect.height()))
            painter.drawRect(QRectF(
                rect.width() - RESIZE_HANDLE_WIDTH, 0,
                RESIZE_HANDLE_WIDTH, rect.height()
            ))
        
        # Draw waveform if enabled and this is a clip event (skip at very low LOD)
        if draw_details and self._should_show_waveform():
            self._draw_waveform(painter, rect)
        
        # Draw label if enabled and wide enough (reduced threshold from 50px to 40px for better visibility)
        # Skip label at low LOD for performance
        show_labels = True
        show_duration_labels = False
        if settings_mgr:
            show_labels = settings_mgr.show_event_labels
            show_duration_labels = settings_mgr.show_event_duration_labels
        
        if draw_details and rect.width() > 40 and self.classification and show_labels:
            painter.setPen(QPen(TimelineStyle.TEXT_PRIMARY))
            font = Typography.default_font()
            font.setPixelSize(label_font_size)
            painter.setFont(font)
            
            text_rect = rect.adjusted(RESIZE_HANDLE_WIDTH + 2, 0, -RESIZE_HANDLE_WIDTH - 2, 0)
            label = self.classification
            if len(label) > 15:
                label = label[:14] + "..."
            
            # Add duration label if enabled
            if show_duration_labels and self.duration > 0:
                duration_text = f"{self.duration:.2f}s"
                # Draw classification label
                painter.drawText(
                    text_rect,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    label
                )
                # Draw duration label below or to the right
                duration_rect = text_rect.adjusted(0, label_font_size + 2, 0, label_font_size + 2)
                painter.drawText(
                    duration_rect,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    duration_text
                )
            else:
                painter.drawText(
                    text_rect,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    label
                )
    
    # ===================================================================
    # Waveform Drawing
    # ===================================================================
    
    def _draw_waveform(self, painter: QPainter, rect: QRectF):
        """
        Draw waveform for this event.
        
        DESIGN:
        - Skip during drag (prevents stretching artifact)
        - Skip if rect too small
        - Use cached path if valid
        - Otherwise load data and build path
        """
        # Skip during drag - shows clean event without stretched waveform
        if self._is_dragging:
            return
        
        # Skip for very small events
        if rect.width() < 5 or rect.height() < 5:
            return
        
        # Build cache key (invalidates when these change)
        path_key = (
            round(self._pixels_per_second, 2),
            round(self.start_time, 4),
            round(self.duration, 4),
            round(rect.height(), 1)
        )
        
        # Use cached path if valid
        if self._cached_path_key == path_key and self._cached_waveform_path is not None:
            self._paint_waveform_path(painter, self._cached_waveform_path)
            return
        
        # Need new path - get waveform data
        waveform_data, audio_duration = self._get_waveform_data()
        if waveform_data is None or audio_duration <= 0:
            # No data yet - waveform_ready signal will trigger repaint
            return
        
        # Build new path
        path = self._build_waveform_path(waveform_data, audio_duration, rect)
        if path is None:
            return
        
        # Cache and draw
        self._cached_waveform_path = path
        self._cached_path_key = path_key
        self._paint_waveform_path(painter, path)
    
    def _paint_waveform_path(self, painter: QPainter, path: QPainterPath):
        """Paint a waveform path with state-based coloring."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        # Get waveform opacity from settings
        settings_mgr = get_timeline_settings_manager()
        opacity = 0.5  # Default
        if settings_mgr:
            opacity = settings_mgr.waveform_opacity
        
        # Color based on state - derive from theme text color, apply opacity setting
        base_alpha = int(255 * opacity)
        if self._selected:
            waveform_color = QColor(TimelineStyle.TEXT_PRIMARY)
            waveform_color.setAlpha(base_alpha)
        elif self._hovered:
            waveform_color = QColor(TimelineStyle.TEXT_PRIMARY)
            waveform_color.setAlpha(int(base_alpha * 0.95))
        else:
            waveform_color = QColor(TimelineStyle.TEXT_SECONDARY)
            waveform_color.setAlpha(int(base_alpha * 0.9))
        
        painter.setPen(QPen(waveform_color, 1.0))
        painter.setBrush(QBrush(waveform_color))
        painter.drawPath(path)
        painter.restore()
    
    def _get_waveform_data(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get waveform data for this event (simple, synchronous).
        
        Clip position (start_time, duration) determines what slice of source audio to show.
        
        Returns:
            Tuple of (waveform_data, clip_duration) or (None, 0.0) if not available
        """
        if not self.audio_id and not self.audio_name:
            return None, 0.0
        
        # Clip times come directly from timeline position
        clip_start = self.start_time
        clip_end = self.start_time + self.duration
        
        # Cache key includes audio ID AND clip times - invalidates on move/resize
        cache_key = f"{self.audio_id or self.audio_name}:{clip_start:.4f}:{clip_end:.4f}"
        
        # Return cached data if valid (key includes clip times!)
        if self._cached_waveform_audio_key == cache_key and self._cached_waveform_data is not None:
            return self._cached_waveform_data
        
        # Load via simple module (uses resolution setting automatically)
        from .waveform_simple import get_waveform_for_event
        
        waveform_data, duration = get_waveform_for_event(
            self.audio_id, self.audio_name, clip_start, clip_end
        )
        
        if waveform_data is not None and duration > 0:
            self._cached_waveform_data = (waveform_data, duration)
            self._cached_waveform_audio_key = cache_key
            return waveform_data, duration
        
        return None, 0.0
    
    def _build_waveform_path(self, waveform_data: np.ndarray, clip_duration: float, rect: QRectF) -> Optional[QPainterPath]:
        """
        Build QPainterPath for waveform.
        
        Path is in event's local coordinates (0,0 to width,height).
        Qt automatically clips to event bounds.
        
        NOTE: waveform_data is ALREADY the clip slice (from waveform_simple module).
        No additional slicing needed here.
        
        Args:
            waveform_data: Amplitude samples (already sliced for this clip)
            clip_duration: Duration of clip in seconds
            rect: Event rect to draw within
            
        Returns:
            QPainterPath or None if data invalid
        """
        # Validate input
        samples = np.asarray(waveform_data).flatten()
        if len(samples) < 2 or clip_duration <= 0:
            return None
        
        # Downsample to max 200 points if needed
        event_width = rect.width()
        max_points = max(2, min(200, int(event_width * 0.8)))
        if len(samples) > max_points:
            indices = np.linspace(0, len(samples) - 1, max_points, dtype=int)
            samples = samples[indices]
        
        # Compute coordinates
        center_y = rect.height() / 2
        max_amp = max(2.0, rect.height() / 2 - 4)
        
        x_coords = np.linspace(0, event_width, len(samples))
        amplitudes = np.abs(samples)
        
        # Normalize
        amp_max = amplitudes.max()
        if amp_max > 0:
            amplitudes = amplitudes / amp_max
        else:
            amplitudes = np.full_like(amplitudes, 0.1)
        amplitudes = np.clip(amplitudes, 0.05, 1.0)
        
        y_top = center_y - amplitudes * max_amp
        y_bottom = center_y + amplitudes * max_amp
        
        # Build polygon (faster than individual lineTo calls)
        polygon = QPolygonF()
        polygon.append(QPointF(0, center_y))
        
        for i in range(len(samples)):
            polygon.append(QPointF(x_coords[i], y_top[i]))
        
        for i in range(len(samples) - 1, -1, -1):
            polygon.append(QPointF(x_coords[i], y_bottom[i]))
        
        polygon.append(QPointF(0, center_y))
        
        path = QPainterPath()
        path.addPolygon(polygon)
        return path
    
class MarkerEventItem(BaseEventItem):
    """
    Instant event item (rendered as marker/diamond).
    
    Features:
    - Drag to move (time and layer)
    - Diamond shape rendering
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['duration'] = 0
        super().__init__(*args, **kwargs)
        
        # Apply Qt styling for marker events
        self._apply_qt_styling("marker_event")
    
    def _calculate_width(self) -> float:
        """Markers have configurable width from settings."""
        settings_mgr = get_timeline_settings_manager()
        if settings_mgr:
            return float(settings_mgr.marker_event_width)
        return float(MARKER_WIDTH)
    
    def _get_alignment_offset(self) -> float:
        """Offset for center alignment with grid lines.
        
        Marker shapes are drawn centered within their rect.
        Offset by -width/2 so the marker's center aligns with the grid line.
        """
        width = self._calculate_width()
        return -width / 2.0
    
    def _on_settings_changed(self, setting_name: str):
        """Handle settings changes - update if marker event styling changed."""
        if setting_name.startswith('marker_event_'):
            if setting_name in ('marker_event_width', 'marker_event_shape'):
                # Width or shape change requires geometry update (shape affects bounding)
                self._update_geometry()
            elif setting_name in ('marker_event_opacity', 'marker_event_z_value', 'marker_event_rotation', 'marker_event_scale',
                                 'marker_event_drop_shadow_enabled', 'marker_event_drop_shadow_blur_radius',
                                 'marker_event_drop_shadow_offset_x', 'marker_event_drop_shadow_offset_y',
                                 'marker_event_drop_shadow_color', 'marker_event_drop_shadow_opacity'):
                # Qt styling changes
                self._apply_qt_styling("marker_event")
            else:
                # Other styling changes (border, etc.) just need repaint
                self.update()
    
    def hoverEnterEvent(self, event):
        """Handle hover enter."""
        self._hovered = True
        if self._editable:
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.update()
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Handle hover leave."""
        self._hovered = False
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.update()
        super().hoverLeaveEvent(event)
    
    # ===================================================================
    # Mouse Events
    # ===================================================================
    
    def mousePressEvent(self, event):
        """Handle mouse press - start drag."""
        if not self._editable or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        
        self._edit_handle = EditHandle.MOVE
        self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        
        # Handle selection
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.setSelected(True)
        else:
            if not self.isSelected():
                scene = self.scene()
                if scene:
                    scene.clearSelection()
                self.setSelected(True)
        
        # Start drag via controller
        controller = self._get_movement_controller()
        if controller:
            controller.begin_move(self, event.scenePos(), self._get_selected_items())
        
        event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move - drag marker."""
        if self._edit_handle == EditHandle.NONE:
            super().mouseMoveEvent(event)
            return
        
        controller = self._get_movement_controller()
        if controller:
            scene = self.scene()
            snap_enabled = False
            snap_func = None
            snap_calculator = None
            unit_preference = None
            
            if scene and hasattr(scene, 'grid_system'):
                snap_enabled = scene.grid_system.snap_enabled
                # Try new system first
                if hasattr(scene, '_snap_calculator') and hasattr(scene, '_unit_preference'):
                    snap_calculator = scene._snap_calculator
                    unit_preference = scene._unit_preference
                else:
                    # Fallback to legacy
                    snap_func = scene.grid_system.snap_time
            
            controller.update_drag(
                event.scenePos(),
                snap_enabled,
                snap_func,
                snap_calculator,
                unit_preference
            )
        
        event.accept()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release - end drag."""
        if self._edit_handle == EditHandle.NONE:
            super().mouseReleaseEvent(event)
            return
        
        controller = self._get_movement_controller()
        if controller:
            controller.commit_drag()
        
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self._edit_handle = EditHandle.NONE
        event.accept()
    
    def _create_marker_path(self, shape: str, cx: float, cy: float, size: float) -> QPainterPath:
        """Create a QPainterPath for the specified marker shape."""
        path = QPainterPath()
        
        if shape == "diamond":
            path.moveTo(cx, cy - size)
            path.lineTo(cx + size, cy)
            path.lineTo(cx, cy + size)
            path.lineTo(cx - size, cy)
            path.closeSubpath()
        
        elif shape == "circle":
            path.addEllipse(cx - size, cy - size, size * 2, size * 2)
        
        elif shape == "square":
            path.addRect(cx - size, cy - size, size * 2, size * 2)
        
        elif shape == "triangle_up":
            path.moveTo(cx, cy - size)
            path.lineTo(cx + size, cy + size)
            path.lineTo(cx - size, cy + size)
            path.closeSubpath()
        
        elif shape == "triangle_down":
            path.moveTo(cx, cy + size)
            path.lineTo(cx + size, cy - size)
            path.lineTo(cx - size, cy - size)
            path.closeSubpath()
        
        elif shape == "triangle_left":
            path.moveTo(cx - size, cy)
            path.lineTo(cx + size, cy - size)
            path.lineTo(cx + size, cy + size)
            path.closeSubpath()
        
        elif shape == "triangle_right":
            path.moveTo(cx + size, cy)
            path.lineTo(cx - size, cy - size)
            path.lineTo(cx - size, cy + size)
            path.closeSubpath()
        
        elif shape == "arrow_up":
            # Arrow pointing up
            arrow_head_size = size * 0.6
            path.moveTo(cx, cy - size)
            path.lineTo(cx + arrow_head_size, cy - size * 0.3)
            path.lineTo(cx + size * 0.3, cy - size * 0.3)
            path.lineTo(cx + size * 0.3, cy + size)
            path.lineTo(cx - size * 0.3, cy + size)
            path.lineTo(cx - size * 0.3, cy - size * 0.3)
            path.lineTo(cx - arrow_head_size, cy - size * 0.3)
            path.closeSubpath()
        
        elif shape == "arrow_down":
            # Arrow pointing down
            arrow_head_size = size * 0.6
            path.moveTo(cx, cy + size)
            path.lineTo(cx + arrow_head_size, cy + size * 0.3)
            path.lineTo(cx + size * 0.3, cy + size * 0.3)
            path.lineTo(cx + size * 0.3, cy - size)
            path.lineTo(cx - size * 0.3, cy - size)
            path.lineTo(cx - size * 0.3, cy + size * 0.3)
            path.lineTo(cx - arrow_head_size, cy + size * 0.3)
            path.closeSubpath()
        
        elif shape == "arrow_left":
            # Arrow pointing left
            arrow_head_size = size * 0.6
            path.moveTo(cx - size, cy)
            path.lineTo(cx - size * 0.3, cy - arrow_head_size)
            path.lineTo(cx - size * 0.3, cy - size * 0.3)
            path.lineTo(cx + size, cy - size * 0.3)
            path.lineTo(cx + size, cy + size * 0.3)
            path.lineTo(cx - size * 0.3, cy + size * 0.3)
            path.lineTo(cx - size * 0.3, cy + arrow_head_size)
            path.closeSubpath()
        
        elif shape == "arrow_right":
            # Arrow pointing right
            arrow_head_size = size * 0.6
            path.moveTo(cx + size, cy)
            path.lineTo(cx + size * 0.3, cy - arrow_head_size)
            path.lineTo(cx + size * 0.3, cy - size * 0.3)
            path.lineTo(cx - size, cy - size * 0.3)
            path.lineTo(cx - size, cy + size * 0.3)
            path.lineTo(cx + size * 0.3, cy + size * 0.3)
            path.lineTo(cx + size * 0.3, cy + arrow_head_size)
            path.closeSubpath()
        
        elif shape == "star":
            # 5-pointed star
            outer_radius = size
            inner_radius = size * 0.4
            points = 5
            for i in range(points * 2):
                angle = (i * pi) / points - pi / 2
                radius = outer_radius if i % 2 == 0 else inner_radius
                x = cx + radius * cos(angle)
                y = cy + radius * sin(angle)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            path.closeSubpath()
        
        elif shape == "cross":
            # X shape
            thickness = size * 0.2
            path.moveTo(cx - size, cy - thickness)
            path.lineTo(cx - thickness, cy - thickness)
            path.lineTo(cx - thickness, cy - size)
            path.lineTo(cx + thickness, cy - size)
            path.lineTo(cx + thickness, cy - thickness)
            path.lineTo(cx + size, cy - thickness)
            path.lineTo(cx + size, cy + thickness)
            path.lineTo(cx + thickness, cy + thickness)
            path.lineTo(cx + thickness, cy + size)
            path.lineTo(cx - thickness, cy + size)
            path.lineTo(cx - thickness, cy + thickness)
            path.lineTo(cx - size, cy + thickness)
            path.closeSubpath()
        
        elif shape == "plus":
            # + shape
            thickness = size * 0.2
            path.moveTo(cx - thickness, cy - size)
            path.lineTo(cx + thickness, cy - size)
            path.lineTo(cx + thickness, cy - thickness)
            path.lineTo(cx + size, cy - thickness)
            path.lineTo(cx + size, cy + thickness)
            path.lineTo(cx + thickness, cy + thickness)
            path.lineTo(cx + thickness, cy + size)
            path.lineTo(cx - thickness, cy + size)
            path.lineTo(cx - thickness, cy + thickness)
            path.lineTo(cx - size, cy + thickness)
            path.lineTo(cx - size, cy - thickness)
            path.lineTo(cx - thickness, cy - thickness)
            path.closeSubpath()
        
        else:
            # Default to diamond if unknown shape
            path.moveTo(cx, cy - size)
            path.lineTo(cx + size, cy)
            path.lineTo(cx, cy + size)
            path.lineTo(cx - size, cy)
            path.closeSubpath()
        
        return path
    
    # ===================================================================
    # Painting & Rendering
    # ===================================================================
    
    def paint(self, painter: QPainter, option, widget=None):
        """Paint the marker with the selected shape.
        
        Note: Qt handles viewport culling automatically - paint() is only called
        for items within the viewport. No manual exposedRect check needed.
        """
        rect = self.rect()
        
        # Get marker event settings or use defaults
        settings_mgr = get_timeline_settings_manager()
        if settings_mgr:
            shape = settings_mgr.marker_event_shape
            border_width = settings_mgr.marker_event_border_width
            border_darken = settings_mgr.marker_event_border_darken_percent
        else:
            shape = "diamond"
            border_width = 1
            border_darken = 150
        
        if self._selected:
            color = TimelineStyle.SELECTION_COLOR
            border_color = TimelineStyle.SELECTION_COLOR.darker(130)
        elif self._hovered:
            color = self._base_color.lighter(120)
            border_color = self._base_color.darker(130)
        else:
            color = self._base_color
            border_color = self._base_color.darker(border_darken)
        
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.setPen(QPen(border_color, border_width))
        painter.setBrush(QBrush(color))
        
        cx = rect.width() / 2
        cy = rect.height() / 2
        size = min(rect.width(), rect.height()) / 2 - 2
        
        path = self._create_marker_path(shape, cx, cy, size)
        painter.drawPath(path)
