"""
Timeline View

QGraphicsView subclass for the timeline with pan and selection.
Supports DAW-standard selection: rubber-band, Shift+Click, Ctrl+Click.
"""

import math

from PyQt6.QtWidgets import QGraphicsView
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QRectF, QTimer
from PyQt6.QtGui import QPainter, QWheelEvent, QMouseEvent, QKeyEvent, QTransform
from enum import Enum

# Local imports
from .style import TimelineStyle as Colors
from ..constants import DEFAULT_PIXELS_PER_SECOND
from .scene import TimelineScene

# Deferred zoom delay in milliseconds
# Lower = more responsive but more CPU usage
# Higher = smoother but slightly delayed final positions
ZOOM_DEFER_DELAY_MS = 60


class ZoomAnchor(Enum):
    """Zoom anchor point options (similar to Qt's ViewportAnchor)"""
    UNDER_MOUSE = "under_mouse"  # Zoom centered on mouse cursor
    VIEW_CENTER = "view_center"  # Zoom centered on viewport center


class TimelineView(QGraphicsView):
    """
    Graphics view for the timeline with pan and selection.
    
    Selection Controls (DAW-standard):
    - Click: Select single item, deselect others
    - Shift+Click: Add to selection / extend range
    - Ctrl/Cmd+Click: Toggle item in selection
    - Ctrl/Cmd+A: Select all events
    - Escape: Deselect all
    - Drag on empty space: Rubber-band selection
    
    Navigation Controls:
    - Middle mouse + drag: Pan
    - Alt + Click: Seek to position
    - Space: Toggle play/pause
    - Home/End: Go to start/end
    - Left/Right arrows: Nudge playhead
    
    Zoom Performance:
    - Uses Qt transform for immediate visual feedback (GPU-accelerated)
    - Coalesces position updates with deferred timer
    - Only updates item positions once per zoom gesture
    
    Signals:
        scroll_changed(offset): Horizontal scroll position changed
        seek_requested(time): User clicked to seek
        space_pressed(): Space key pressed (play/pause toggle)
        select_all_requested(): Ctrl+A pressed
        deselect_all_requested(): Escape pressed
    """
    
    scroll_changed = pyqtSignal(float)  # horizontal offset
    seek_requested = pyqtSignal(float)  # time in seconds
    space_pressed = pyqtSignal()  # Toggle play/pause
    select_all_requested = pyqtSignal()  # Ctrl+A
    deselect_all_requested = pyqtSignal()  # Escape
    
    def __init__(self, scene: TimelineScene, parent=None):
        super().__init__(scene, parent)
        
        from ..constants import (
            MIN_PIXELS_PER_SECOND, MAX_PIXELS_PER_SECOND, ZOOM_FACTOR,
            PIXEL_ZOOM_SENSITIVITY, ANGLE_ZOOM_SENSITIVITY, ZOOM_ACCUMULATOR_THRESHOLD
        )
        
        self._pixels_per_second = DEFAULT_PIXELS_PER_SECOND
        self._min_pixels_per_second = MIN_PIXELS_PER_SECOND
        self._max_pixels_per_second = MAX_PIXELS_PER_SECOND
        self._zoom_factor = ZOOM_FACTOR
        self._zoom_anchor = ZoomAnchor.UNDER_MOUSE  # Default: zoom on mouse cursor
        self._is_panning = False
        self._pan_start = QPointF()
        
        # Smooth zoom constants and accumulator (POC-verified)
        self._pixel_zoom_sensitivity = PIXEL_ZOOM_SENSITIVITY
        self._angle_zoom_sensitivity = ANGLE_ZOOM_SENSITIVITY
        self._zoom_accumulator_threshold = ZOOM_ACCUMULATOR_THRESHOLD
        self._zoom_accumulator = 0.0  # Accumulate small deltas for smooth zoom
        
        # === DEFERRED ZOOM STATE ===
        # Use Qt transform for immediate feedback, defer expensive position updates
        self._zoom_defer_timer = QTimer(self)
        self._zoom_defer_timer.setSingleShot(True)
        self._zoom_defer_timer.timeout.connect(self._apply_deferred_zoom)
        self._pending_pps = None  # Target pixels_per_second (None = no pending zoom)
        self._zoom_base_pps = None  # PPS when transform zooming started
        self._zoom_anchor_time = None  # Time position to keep fixed
        self._zoom_anchor_view_x = None  # View X coordinate of anchor
        self._is_transform_zooming = False  # True while using transform-based zoom
        
        # Configure view with Qt optimizations
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        # Performance optimizations - reduce overhead during zoom/pan
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        # Cache background for grid rendering performance
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        # Default scrollbar policies - will be overridden by settings in TimelineWidget
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # IMPORTANT: Set alignment to top-left to prevent centering when scene is smaller than viewport
        # This ensures layers start at the top and align with layer labels
        # Without this, Qt centers the scene content vertically when it's smaller than viewport
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        # Enable rubber-band selection by default
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRubberBandSelectionMode(Qt.ItemSelectionMode.IntersectsItemShape)
        
        # Enable keyboard focus
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Style
        self.setStyleSheet(f"""
            QGraphicsView {{
                border: none;
                background-color: {Colors.BG_DARK.name()};
            }}
        """)
        
        # Connect scroll bar
        self.horizontalScrollBar().valueChanged.connect(self._on_scroll)
    
    @property
    def pixels_per_second(self) -> float:
        """Get current scale level"""
        return self._pixels_per_second
    
    @property
    def zoom_anchor(self) -> ZoomAnchor:
        """Get current zoom anchor mode"""
        return self._zoom_anchor
    
    def set_zoom_anchor(self, anchor: ZoomAnchor):
        """
        Set zoom anchor point (where zoom is centered).
        
        Args:
            anchor: ZoomAnchor.UNDER_MOUSE or ZoomAnchor.VIEW_CENTER
        """
        self._zoom_anchor = anchor
    
    def set_pixels_per_second(self, pps: float):
        """Set scale level and update all components"""
        # Cancel any pending transform zoom
        self._cancel_transform_zoom()
        
        self._pixels_per_second = pps
        if self.scene():
            self.scene().set_pixels_per_second(pps)
        
        # Update ruler - find TimelineWidget parent
        widget = self.parent()
        while widget:
            if hasattr(widget, '_ruler'):
                widget._ruler.set_pixels_per_second(pps)
                break
            widget = widget.parent() if hasattr(widget, 'parent') else None
    
    def _cancel_transform_zoom(self):
        """Cancel any in-progress transform zoom and reset state."""
        if self._is_transform_zooming:
            self._zoom_defer_timer.stop()
            self.setTransform(QTransform())
            self._is_transform_zooming = False
            self._pending_pps = None
            self._zoom_base_pps = None
            self._zoom_anchor_time = None
            self._zoom_anchor_view_x = None
    
    def _on_scroll(self, value: int):
        """Handle scroll bar changes."""
        self.scroll_changed.emit(float(value))
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for scrolling and zooming (DAW-standard)"""
        # Ctrl+Scroll = Zoom (horizontal/time-based zoom)
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._handle_zoom(event)
            event.accept()
        elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Shift+Scroll = Horizontal scroll
            delta = event.angleDelta().y()
            h_bar = self.horizontalScrollBar()
            h_bar.setValue(h_bar.value() - delta)
            event.accept()
        else:
            # Normal scroll = Vertical scroll
            super().wheelEvent(event)
    
    def _handle_zoom(self, event: QWheelEvent):
        """
        Handle smooth zoom with Ctrl+Scroll (DAW-standard behavior).
        
        HYBRID TRANSFORM + DEFERRED UPDATE APPROACH:
        1. Use Qt view transform for INSTANT visual feedback (GPU-accelerated)
        2. Coalesce zoom events with timer
        3. Update actual item positions once when zoom gesture ends
        
        This provides buttery-smooth zooming regardless of item count.
        """
        # === STEP 1: Get scroll delta (prefer pixelDelta for trackpads) ===
        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()
        
        if not pixel_delta.isNull():
            # High-resolution device (trackpad on macOS)
            zoom_delta = pixel_delta.y() * self._pixel_zoom_sensitivity
        elif not angle_delta.isNull():
            # Traditional mouse wheel
            zoom_delta = angle_delta.y() * self._angle_zoom_sensitivity
        else:
            return
        
        # === STEP 2: Accumulate for very smooth zoom ===
        self._zoom_accumulator += zoom_delta
        
        if abs(self._zoom_accumulator) < self._zoom_accumulator_threshold:
            return
        
        # === STEP 3: Calculate new target PPS ===
        zoom_factor = math.exp(self._zoom_accumulator)
        self._zoom_accumulator = 0.0
        
        # Start from pending PPS if we're already in a zoom gesture
        base_pps = self._pending_pps if self._pending_pps else self._pixels_per_second
        new_pps = base_pps * zoom_factor
        new_pps = max(self._min_pixels_per_second, min(new_pps, self._max_pixels_per_second))
        
        # Skip if no meaningful change
        if abs(new_pps - base_pps) < 0.001:
            return
        
        # === STEP 4: Store anchor on FIRST zoom of gesture ===
        if not self._is_transform_zooming:
            # Starting a new zoom gesture
            self._is_transform_zooming = True
            self._zoom_base_pps = self._pixels_per_second
            
            if self._zoom_anchor == ZoomAnchor.UNDER_MOUSE:
                mouse_pos = event.position()
                scene_pos = self.mapToScene(mouse_pos.toPoint())
                self._zoom_anchor_time = scene_pos.x() / self._pixels_per_second
                self._zoom_anchor_view_x = mouse_pos.x()
            else:
                viewport_center = self.viewport().rect().center()
                scene_center = self.mapToScene(viewport_center)
                self._zoom_anchor_time = scene_center.x() / self._pixels_per_second
                self._zoom_anchor_view_x = viewport_center.x()
        
        # === STEP 5: Apply TRANSFORM for instant visual feedback ===
        # Transform is GPU-accelerated and O(1) regardless of item count
        self._pending_pps = new_pps
        
        # Calculate cumulative scale from base PPS
        scale_x = new_pps / self._zoom_base_pps
        
        # Apply horizontal scale transform centered on anchor
        # Reset and rebuild transform to avoid accumulation errors
        transform = QTransform()
        transform.scale(scale_x, 1.0)
        self.setTransform(transform)
        
        # Adjust scroll to keep anchor point fixed
        # anchor_time * base_pps * scale_x = anchor_time * new_pps
        new_anchor_scene_x = self._zoom_anchor_time * self._zoom_base_pps * scale_x
        current_scroll = self.horizontalScrollBar().value()
        target_scroll = int(new_anchor_scene_x - self._zoom_anchor_view_x)
        self.horizontalScrollBar().setValue(max(0, target_scroll))
        
        # === STEP 6: Start/restart deferred update timer ===
        # When timer fires, we'll update actual item positions
        self._zoom_defer_timer.start(ZOOM_DEFER_DELAY_MS)
    
    def _apply_deferred_zoom(self):
        """
        Apply deferred zoom: update actual item positions and reset transform.
        
        Called when zoom gesture ends (timer fires without new zoom events).
        This is where the O(n) item position update happens, but only ONCE
        per zoom gesture instead of per wheel event.
        """
        if not self._is_transform_zooming or self._pending_pps is None:
            return
        
        new_pps = self._pending_pps
        anchor_time = self._zoom_anchor_time
        anchor_view_x = self._zoom_anchor_view_x
        
        # Clear zoom state BEFORE updates to prevent re-entry
        self._is_transform_zooming = False
        self._pending_pps = None
        self._zoom_base_pps = None
        self._zoom_anchor_time = None
        self._zoom_anchor_view_x = None
        
        # Batch updates for clean transition
        self.setUpdatesEnabled(False)
        try:
            # Reset transform to identity FIRST
            self.setTransform(QTransform())
            
            # Invalidate background cache for fresh grid
            self.resetCachedContent()
            
            # Update actual pixels_per_second
            self._pixels_per_second = new_pps
            
            # Update all item positions (this is the expensive O(n) operation)
            if self.scene():
                self.scene().set_pixels_per_second(new_pps, defer_scene_rect=True)
            
            # Restore anchor position
            if anchor_time is not None and anchor_view_x is not None:
                new_scene_x = anchor_time * new_pps
                target_scroll = int(new_scene_x - anchor_view_x)
                self.horizontalScrollBar().setValue(max(0, target_scroll))
            
            # Update scene rect
            if self.scene():
                self.scene()._update_scene_rect()
            
            # Update ruler
            widget = self.parent()
            while widget:
                if hasattr(widget, '_ruler'):
                    widget._ruler.set_pixels_per_second(new_pps)
                    break
                widget = widget.parent() if hasattr(widget, 'parent') else None
                
        finally:
            self.setUpdatesEnabled(True)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press"""
        if event.button() == Qt.MouseButton.MiddleButton:
            # Middle click = pan
            self._is_panning = True
            self._pan_start = event.position()
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.AltModifier:
                # Alt + click = seek
                scene_pos = self.mapToScene(event.position().toPoint())
                time = max(0, scene_pos.x() / self._pixels_per_second)
                self.seek_requested.emit(time)
                event.accept()
            else:
                # Normal click - let QGraphicsView handle selection
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for panning"""
        if self._is_panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            
            # Update scroll bars
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            h_bar.setValue(h_bar.value() - int(delta.x()))
            v_bar.setValue(v_bar.value() - int(delta.y()))
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts"""
        from PyQt6.QtGui import QKeySequence
        
        # Delete handling
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.scene().keyPressEvent(event)
            return
        
        # Selection shortcuts
        if event.key() == Qt.Key.Key_A and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.select_all_requested.emit()
            event.accept()
            return
        
        if event.key() == Qt.Key.Key_Escape:
            self.deselect_all_requested.emit()
            event.accept()
            return
        
        # Playback shortcuts
        if event.key() == Qt.Key.Key_Space:
            self.space_pressed.emit()
            event.accept()
            return
        
        # Navigation shortcuts (only if no events selected)
        scene = self.scene()
        selected_ids = scene.get_selected_event_ids() if scene else []
        has_selection = len(selected_ids) > 0
        
        # Get settings manager from parent widget if available
        settings_manager = None
        parent = self.parent()
        while parent:
            if hasattr(parent, '_settings_manager'):
                settings_manager = parent._settings_manager
                break
            parent = parent.parent()
        
        # Check configurable shortcuts if settings manager available
        if settings_manager and has_selection:
            # Build current key sequence from event
            modifiers = event.modifiers()
            key = event.key()
            
            # Build modifier string
            mod_parts = []
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                mod_parts.append("Ctrl")
            if modifiers & Qt.KeyboardModifier.AltModifier:
                mod_parts.append("Alt")
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                mod_parts.append("Shift")
            if modifiers & Qt.KeyboardModifier.MetaModifier:
                mod_parts.append("Meta")
            
            # Get key name
            key_name = QKeySequence(key).toString()
            if key_name:
                if mod_parts:
                    current_shortcut = "+".join(mod_parts) + "+" + key_name
                else:
                    current_shortcut = key_name
                
                # Get configured shortcuts
                move_left_shortcut = getattr(settings_manager, 'shortcut_move_event_left', 'Left')
                move_right_shortcut = getattr(settings_manager, 'shortcut_move_event_right', 'Right')
                move_up_shortcut = getattr(settings_manager, 'shortcut_move_event_up_layer', 'Ctrl+Up')
                move_down_shortcut = getattr(settings_manager, 'shortcut_move_event_down_layer', 'Ctrl+Down')
                
                # Normalize shortcuts for comparison (case-insensitive, normalize format)
                def normalize_shortcut(s: str) -> str:
                    # Remove "Key_" prefix if present, normalize case
                    s = s.replace("Key_", "").strip()
                    # Normalize to title case for comparison
                    parts = s.split("+")
                    normalized_parts = [p.strip().title() for p in parts]
                    return "+".join(normalized_parts)
                
                current_normalized = normalize_shortcut(current_shortcut)
                
                # Check each shortcut
                if current_normalized == normalize_shortcut(move_left_shortcut):
                    # Move events left
                    if hasattr(scene, 'nudge_selected_events'):
                        scene.nudge_selected_events(-0.1)  # Move left by 0.1 seconds (or snap interval)
                    event.accept()
                    return
                elif current_normalized == normalize_shortcut(move_right_shortcut):
                    # Move events right
                    if hasattr(scene, 'nudge_selected_events'):
                        scene.nudge_selected_events(0.1)  # Move right by 0.1 seconds (or snap interval)
                    event.accept()
                    return
                elif current_normalized == normalize_shortcut(move_up_shortcut):
                    # Move events up layer
                    if hasattr(scene, 'move_selected_events_layer'):
                        scene.move_selected_events_layer(-1)
                    event.accept()
                    return
                elif current_normalized == normalize_shortcut(move_down_shortcut):
                    # Move events down layer
                    if hasattr(scene, 'move_selected_events_layer'):
                        scene.move_selected_events_layer(1)
                    event.accept()
                    return
        
        # Navigation shortcuts (only when no selection)
        if not has_selection:
            if event.key() == Qt.Key.Key_Home:
                self.seek_requested.emit(0)
                self.horizontalScrollBar().setValue(0)
                event.accept()
                return
            elif event.key() == Qt.Key.Key_End:
                if scene and hasattr(scene, '_duration'):
                    self.seek_requested.emit(scene._duration)
                event.accept()
                return
            elif event.key() == Qt.Key.Key_Left:
                # Nudge playhead left
                if scene and hasattr(scene, 'playhead'):
                    current = scene.playhead.position_seconds
                    self.seek_requested.emit(max(0, current - 0.1))
                event.accept()
                return
            elif event.key() == Qt.Key.Key_Right:
                # Nudge playhead right
                if scene and hasattr(scene, 'playhead'):
                    current = scene.playhead.position_seconds
                    duration = scene._duration if hasattr(scene, '_duration') else float('inf')
                    self.seek_requested.emit(min(duration, current + 0.1))
                event.accept()
                return
        
        super().keyPressEvent(event)
    
    def get_visible_time_range(self) -> tuple:
        """
        Get the visible time range.
        
        Returns:
            Tuple of (start_time, end_time) in seconds
        """
        rect = self.mapToScene(self.viewport().rect()).boundingRect()
        start_time = rect.left() / self._pixels_per_second
        end_time = rect.right() / self._pixels_per_second
        return (max(0, start_time), end_time)
    
    def ensure_time_visible(self, time: float):
        """
        Scroll to ensure a time position is visible.
        
        Args:
            time: Time in seconds to make visible
        """
        x = time * self._pixels_per_second
        rect = self.mapToScene(self.viewport().rect()).boundingRect()
        
        if x < rect.left() or x > rect.right():
            self.centerOn(x, rect.center().y())
    
    def scroll_to_time(self, time: float):
        """
        Scroll to a specific time position (left edge).
        
        Args:
            time: Time in seconds
        """
        x = time * self._pixels_per_second
        self.horizontalScrollBar().setValue(int(x))
    
    def zoom_in(self, factor: float = None):
        """
        Zoom in by the specified factor (DAW-standard).
        
        Args:
            factor: Zoom factor (defaults to ZOOM_FACTOR)
        """
        if factor is None:
            factor = self._zoom_factor
        
        new_pps = self._pixels_per_second * factor
        new_pps = max(self._min_pixels_per_second, min(new_pps, self._max_pixels_per_second))
        
        if new_pps != self._pixels_per_second:
            self._set_zoom_level(new_pps)
    
    def zoom_out(self, factor: float = None):
        """
        Zoom out by the specified factor (DAW-standard).
        
        Args:
            factor: Zoom factor (defaults to ZOOM_FACTOR)
        """
        if factor is None:
            factor = self._zoom_factor
        
        new_pps = self._pixels_per_second / factor
        new_pps = max(self._min_pixels_per_second, min(new_pps, self._max_pixels_per_second))
        
        if new_pps != self._pixels_per_second:
            self._set_zoom_level(new_pps)
    
    def zoom_to_fit(self):
        """
        Zoom to fit entire timeline in view (DAW-standard).
        """
        if not self.scene():
            return
        
        scene = self.scene()
        if not hasattr(scene, '_duration') or scene._duration <= 0:
            return
        
        # Get viewport width
        viewport_width = self.viewport().width()
        if viewport_width <= 0:
            return
        
        # Calculate pixels_per_second needed to fit timeline
        # Add some padding (20% on each side)
        padding_factor = 0.8
        required_pps = (viewport_width * padding_factor) / scene._duration
        
        # Clamp to limits
        new_pps = max(self._min_pixels_per_second, min(required_pps, self._max_pixels_per_second))
        
        self._set_zoom_level(new_pps)
        
        # Scroll to start
        self.horizontalScrollBar().setValue(0)
    
    def zoom_to_selection(self):
        """
        Zoom to fit selected events in view (DAW-standard).
        """
        if not self.scene():
            return
        
        selected_items = self.scene().selectedItems()
        if not selected_items:
            return
        
        # Calculate bounding rect of selected items
        bounding_rect = QRectF()
        
        for item in selected_items:
            item_rect = item.sceneBoundingRect()
            if bounding_rect.isNull():
                bounding_rect = item_rect
            else:
                bounding_rect = bounding_rect.united(item_rect)
        
        if bounding_rect.isNull():
            return
        
        # Get time range of selection
        start_time = bounding_rect.left() / self._pixels_per_second
        end_time = bounding_rect.right() / self._pixels_per_second
        duration = end_time - start_time
        
        if duration <= 0:
            return
        
        # Get viewport width
        viewport_width = self.viewport().width()
        if viewport_width <= 0:
            return
        
        # Calculate pixels_per_second needed to fit selection
        # Add padding (20% on each side)
        padding_factor = 0.6
        required_pps = (viewport_width * padding_factor) / duration
        
        # Clamp to limits
        new_pps = max(self._min_pixels_per_second, min(required_pps, self._max_pixels_per_second))
        
        # Update zoom
        self._set_zoom_level(new_pps)
        
        # Scroll to center selection
        center_time = (start_time + end_time) / 2
        self.scroll_to_time(center_time - (duration / 2))
    
    def reset_zoom(self):
        """
        Reset zoom to default level (DAW-standard).
        """
        self._set_zoom_level(DEFAULT_PIXELS_PER_SECOND)
    
    def _set_zoom_level(self, pps: float, reference_time: float = None):
        """
        Set zoom level by updating pixels_per_second (standard timeline approach).
        
        This recalculates all positions instead of using view transforms.
        Used for programmatic zoom (zoom_in, zoom_out, zoom_to_fit, etc).
        
        Optimized with viewport update batching to reduce glitches.
        
        Args:
            pps: Pixels per second
            reference_time: Optional time to keep fixed (if None, uses zoom_anchor mode)
        """
        # Cancel any pending transform zoom
        self._cancel_transform_zoom()
        
        # Clamp to limits
        pps = max(self._min_pixels_per_second, min(pps, self._max_pixels_per_second))
        
        # Determine reference point
        if reference_time is None:
            # Use zoom anchor mode
            if self._zoom_anchor == ZoomAnchor.VIEW_CENTER:
                viewport_center = self.viewport().rect().center()
                scene_center_before = self.mapToScene(viewport_center)
                reference_time = scene_center_before.x() / self._pixels_per_second if self._pixels_per_second > 0 else 0
            else:
                # UNDER_MOUSE - but we don't have mouse position here, so use viewport center
                viewport_center = self.viewport().rect().center()
                scene_center_before = self.mapToScene(viewport_center)
                reference_time = scene_center_before.x() / self._pixels_per_second if self._pixels_per_second > 0 else 0
        
        # Disable viewport updates during zoom operation (reduces glitches)
        # Note: QGraphicsScene doesn't have setUpdatesEnabled(), only QWidget does
        self.setUpdatesEnabled(False)
        
        try:
            # Update zoom level
            self._pixels_per_second = pps
            
            # Update scene and all items (recalculates positions)
            # Defer scene rect update to avoid viewport updates during zoom
            if self.scene():
                self.scene().set_pixels_per_second(pps, defer_scene_rect=True)
            
            # Update ruler - find TimelineWidget parent
            widget = self.parent()
            while widget:
                if hasattr(widget, '_ruler'):
                    widget._ruler.set_pixels_per_second(pps)
                    break
                widget = widget.parent() if hasattr(widget, 'parent') else None
            
            # Restore reference point at the same time position
            new_x_at_reference = reference_time * pps
            h_bar = self.horizontalScrollBar()
            viewport_width = self.viewport().width()
            new_scroll = int(new_x_at_reference - viewport_width / 2)
            h_bar.setValue(max(0, new_scroll))
            
            # Update scene rect now that zoom is complete
            if self.scene():
                self.scene()._update_scene_rect()
            
        finally:
            # Re-enable updates - Qt will batch all changes into a single repaint
            self.setUpdatesEnabled(True)
