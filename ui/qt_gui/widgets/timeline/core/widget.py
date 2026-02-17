"""
Timeline Widget
================

Self-contained DAW-style timeline widget.

Drop into any panel or window. Manages its own layers, events,
playback, and editing. Communicates changes via signals.

Uses:
- LayerManager for layer management (single source of truth)
- MovementController for drag operations
- TimelineScene for event rendering
- TimelineView for user interaction
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING
from collections import defaultdict
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QSplitter, QDialog, QScrollArea,
    QMenu, QInputDialog, QColorDialog, QSizePolicy, QApplication
)
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QAbstractAnimation, QEventLoop
from PyQt6.QtGui import QPainter, QPainterPath, QPen, QFont, QPolygonF, QColor

# Local imports
from .style import TimelineStyle as Colors, TimelineStyle as Typography
from ..logging import TimelineLog as Log
from ..types import (
    TimelineEvent, TimelineLayer,
    EventMoveResult, EventResizeResult, EventCreateResult, EventDeleteResult, EventSliceResult
)
from ..constants import (
    DEFAULT_PIXELS_PER_SECOND, RULER_HEIGHT, TRACK_HEIGHT, TRACK_SPACING
)
from .style import TimelineStyle
from ..events.layer_manager import LayerManager
from ..events.movement_controller import MovementController
from .scene import TimelineScene
from .view import TimelineView
from ..playback.controller import PlaybackController
from ..grid_system import GridSystem, TimebaseMode
from ..interfaces import PlaybackInterface
from ..events.inspector import EventInspector
from ..settings.storage import TimelineSettingsManager, set_timeline_settings_manager
from ..settings.panel import SettingsPanel, PlayheadFollowMode, TimelineSettings
from ..events.items import BlockEventItem, MarkerEventItem
from ui.qt_gui.design_system import border_radius, Colors

# =============================================================================
# Time Ruler Widget
# =============================================================================

class TimeRuler(QWidget):
    """Time scale display at top of timeline."""
    
    clicked = pyqtSignal(float)
    
    def __init__(self, grid_system: GridSystem, settings_manager=None, parent=None):
        super().__init__(parent)
        self.setFixedHeight(RULER_HEIGHT)
        
        self._grid_system = grid_system
        self._settings_manager = settings_manager
        self._pixels_per_second = DEFAULT_PIXELS_PER_SECOND
        self._scroll_offset = 0
        self._duration = 60
        self._playhead_position = 0
        
        # Reference to scene's grid calculator for consistent intervals
        self._scene_grid_calculator = None
        self._scene_unit_preference = None
        
        self.setStyleSheet(f"background-color: {Colors.BG_MEDIUM.name()};")
        self.setMouseTracking(True)
        
        # Connect to settings changes if settings manager is available
        if self._settings_manager:
            self._settings_manager.settings_changed.connect(self._on_settings_changed)
    
    def set_scene_grid_calculator(self, grid_calculator, unit_preference):
        """Set reference to scene's grid calculator for consistent intervals."""
        self._scene_grid_calculator = grid_calculator
        self._scene_unit_preference = unit_preference
    
    def _on_settings_changed(self, setting_name: str):
        """Handle settings changes - update if event time styling or grid intervals changed."""
        if setting_name.startswith('event_time_') or setting_name.startswith('grid_'):
            self.update()
    
    def set_pixels_per_second(self, pps: float):
        self._pixels_per_second = pps
        self.update()
    
    def set_scroll_offset(self, offset: float):
        self._scroll_offset = offset
        self.update()
    
    def set_duration(self, duration: float):
        self._duration = max(duration, 1)
        self.update()
    
    def set_playhead_position(self, position: float):
        self._playhead_position = position
        self.update()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            time = (event.pos().x() + self._scroll_offset) / self._pixels_per_second
            self.clicked.emit(max(0, time))
    
    def paintEvent(self, event):
        """Paint ruler with tick marks using POC-verified adaptive density algorithm.
        
        Uses the same integer-based indexing as the grid renderer to ensure
        ruler tick marks align perfectly with scene grid lines.
        """
        import math
        from ..constants import MIN_MINOR_LINE_SPACING_PX, MIN_MAJOR_LINE_SPACING_PX
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        painter.fillRect(0, 0, width, height, Colors.BG_MEDIUM)
        painter.setPen(QPen(Colors.BORDER, 1))
        painter.drawLine(0, height - 1, width, height - 1)
        
        # Get base intervals - use scene's grid calculator for consistency with grid renderer
        # This ensures ruler ticks align perfectly with scene grid lines
        if self._scene_grid_calculator and self._scene_unit_preference is not None:
            try:
                major_interval, minor_interval = self._scene_grid_calculator.get_intervals(
                    self._pixels_per_second,
                    self._scene_unit_preference
                )
            except Exception:
                # Fallback to grid_system
                major_interval, minor_interval = self._grid_system.get_major_minor_intervals(
                    self._pixels_per_second
                )
        else:
            # Fallback to grid_system
            major_interval, minor_interval = self._grid_system.get_major_minor_intervals(
                self._pixels_per_second
            )
        
        # === ADAPTIVE DENSITY (POC-verified) ===
        # Apply same skip factor logic as grid renderer for perfect alignment
        
        minor_px = minor_interval * self._pixels_per_second
        
        # Calculate skip factor for minor lines
        if minor_px >= MIN_MINOR_LINE_SPACING_PX:
            minor_skip = 1
        else:
            minor_skip = math.ceil(MIN_MINOR_LINE_SPACING_PX / minor_px)
            minor_skip = self._snap_to_nice_divisor(minor_skip)
        
        display_minor = minor_interval * minor_skip
        display_minor_px = display_minor * self._pixels_per_second
        
        # Calculate skip factor for major lines
        major_px = major_interval * self._pixels_per_second
        
        if major_px >= MIN_MAJOR_LINE_SPACING_PX:
            major_skip = 1
        else:
            major_skip = math.ceil(MIN_MAJOR_LINE_SPACING_PX / major_px)
            major_skip = self._snap_to_nice_divisor(major_skip)
        
        display_major = major_interval * major_skip
        
        # Get font settings from settings manager or use defaults
        FIXED_OFFSET_X = 3
        FIXED_OFFSET_Y = -17
        
        if self._settings_manager:
            font_size = self._settings_manager.event_time_font_size
            font_family = self._settings_manager.event_time_font_family
            major_color = QColor(self._settings_manager.event_time_major_color)
            minor_color = QColor(self._settings_manager.event_time_minor_color)
        else:
            font_size = 10
            font_family = "monospace"
            major_color = Colors.TEXT_PRIMARY
            minor_color = Colors.TEXT_DISABLED
        
        # Set font based on family
        if font_family == "monospace":
            font = Typography.mono_font()
        elif font_family == "small":
            font = Typography.small_font()
        else:
            font = Typography.default_font()
        font.setPixelSize(font_size)
        painter.setFont(font)
        
        # === INTEGER-BASED INDEXING (POC-verified) ===
        # Use integer line indices to avoid floating point bugs
        
        if display_minor_px <= 0:
            return
        
        # Calculate visible range in scene coordinates
        visible_left = self._scroll_offset
        visible_right = self._scroll_offset + width
        
        # Convert to integer line indices
        first_line_idx = max(0, int(visible_left / display_minor_px))
        last_line_idx = int(visible_right / display_minor_px) + 1
        
        # How many display intervals per major?
        if display_minor > 0:
            lines_per_major = max(1, round(display_major / display_minor))
        else:
            lines_per_major = 1
        
        # Draw tick marks using integer indexing
        for i in range(first_line_idx, last_line_idx + 1):
            # X position in scene coordinates
            scene_x = i * display_minor_px
            # Convert to widget coordinates
            x = scene_x - self._scroll_offset
            
            if scene_x < 0:
                continue
            
            # Integer-based major detection (no floating point modulo!)
            is_major = (i % lines_per_major == 0) if lines_per_major > 0 else False
            
            # Use float positions to match grid renderer exactly
            # Qt handles subpixel rendering properly
            if is_major:
                painter.setPen(QPen(major_color, 1))
                painter.drawLine(QPointF(x, height - 15), QPointF(x, height - 1))
                time_secs = i * display_minor
                label = self._grid_system.format_time(time_secs)
                painter.drawText(QPointF(x + FIXED_OFFSET_X, height + FIXED_OFFSET_Y), label)
            else:
                painter.setPen(QPen(minor_color, 1))
                painter.drawLine(QPointF(x, height - 8), QPointF(x, height - 1))
        
        playhead_x = (self._playhead_position * self._pixels_per_second) - self._scroll_offset
        if 0 <= playhead_x <= width:
            painter.setPen(QPen(TimelineStyle.PLAYHEAD_COLOR, 1))
            painter.setBrush(TimelineStyle.PLAYHEAD_COLOR)
            triangle = QPolygonF([
                QPointF(playhead_x - 6, 0),
                QPointF(playhead_x + 6, 0),
                QPointF(playhead_x, 10)
            ])
            painter.drawPolygon(triangle)
    
    @staticmethod
    def _snap_to_nice_divisor(n: int) -> int:
        """Snap to musically/temporally meaningful divisors (POC-verified)."""
        if n <= 1:
            return 1
        nice = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 32, 40, 48, 50, 60, 64, 100]
        for v in nice:
            if v >= n:
                return v
        return n

# =============================================================================
# Layer Labels Widget  
# =============================================================================

class LayerLabels(QWidget):
    """Widget showing layer names on the left side."""
    
    width_changed = pyqtSignal(int)  # Emitted when width changes
    # Note: Layer operations use LayerManager signals (layer_updated, layers_changed)
    
    def __init__(self, layer_manager: LayerManager, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(80)
        self.setMaximumWidth(300)
        self._default_width = 120
        self._layer_manager = layer_manager
        self._timeline_view = None
        self._timeline_scene = None  # Set by TimelineWidget
        self.setContentsMargins(0, 0, 0, 0)
        # IMPORTANT: Expand vertically to fill available space
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background-color: {Colors.BG_MEDIUM.name()};")
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        # Connect to layer manager changes
        self._layer_manager.layers_changed.connect(self._on_layers_changed)
        self._layer_manager.layer_updated.connect(self._on_layer_updated)
        
    def _on_layers_changed(self):
        """Handle layer list changes - re-push sync states and repaint.
        
        When layers are recreated (e.g., BlockUpdated reload), the fresh
        TimelineLayer objects have sync_connection_state='none'.  We ask
        the SSM to re-push so the icons are restored immediately.
        """
        ssm = self._find_any_sync_system_manager()
        if ssm:
            ssm._push_sync_state_to_all_layers()
        self.update()
    
    def _find_any_sync_system_manager(self):
        """Find any live SyncSystemManager (does not require a specific block ID)."""
        from PyQt6.QtWidgets import QApplication
        try:
            from ui.qt_gui.block_panels.show_manager_panel import ShowManagerPanel
        except ImportError:
            return None
        app = QApplication.instance()
        if app:
            for widget in app.allWidgets():
                if (isinstance(widget, ShowManagerPanel)
                        and hasattr(widget, '_sync_system_manager')
                        and widget._sync_system_manager):
                    return widget._sync_system_manager
        return None
    
    def _on_layer_updated(self, layer_id):
        """Handle single layer update - repaint."""
        self.update()
    
    def invalidate_sync_cache(self):
        """Force a repaint.
        
        Called by ShowManager panel when SSM pushes state changes onto
        TimelineLayer objects.  The name is kept for backwards compatibility
        but there is no cache -- just a repaint trigger.
        """
        self.update()
    
    def set_timeline_view(self, view):
        self._timeline_view = view
        self.update()
    
    def set_timeline_scene(self, scene):
        """Set the timeline scene for event operations."""
        self._timeline_scene = scene
    
    def _get_facade_and_command_bus(self):
        """Get facade and command_bus from parent widget chain."""
        parent = self.parent()
        while parent:
            if hasattr(parent, '_facade') and parent._facade:
                facade = parent._facade
                if hasattr(facade, 'command_bus'):
                    return facade, facade.command_bus
            elif hasattr(parent, 'facade') and parent.facade:
                facade = parent.facade
                if hasattr(facade, 'command_bus'):
                    return facade, facade.command_bus
            parent = parent.parent()
        return None, None
    
    def set_layer_visible(self, layer_id: str, visible: bool):
        """Set layer visibility via undoable command.
        
        Uses CommandBus for undo/redo support.
        The TimelineScene handles the layer_updated signal to update
        event positions and force visual repaint.
        """
        from src.application.commands import SetLayerVisibilityCommand
        
        # Get command_bus from parent widget (EditorPanel has facade)
        command_bus = None
        parent = self.parent()
        while parent:
            if hasattr(parent, 'facade') and hasattr(parent.facade, 'command_bus'):
                command_bus = parent.facade.command_bus
                break
            parent = parent.parent()
        
        if not command_bus:
            Log.warning("LayerLabels: Cannot set layer visibility - CommandBus not available")
            return
        
        cmd = SetLayerVisibilityCommand(self._layer_manager, layer_id, visible)
        command_bus.execute(cmd)
    
    def _get_layer_at_y(self, widget_y: int) -> Optional[str]:
        """Get the layer ID at the given widget Y coordinate."""
        scene_top_y = 0.0
        if self._timeline_view and self._timeline_view.scene():
            viewport_top = self._timeline_view.mapToScene(0, 0)
            scene_top_y = viewport_top.y()
            
            scrollbar = self._timeline_view.verticalScrollBar()
            if scrollbar and scrollbar.value() == 0:
                if abs(scene_top_y) > 0.1:
                    scene_top_y = 0.0
        
        # Convert widget Y to scene Y
        scene_y = widget_y + scene_top_y
        
        # Find layer at this position
        return self._layer_manager.get_layer_id_from_y(scene_y)
    
    def _show_context_menu(self, pos):
        """Show context menu for the layer at the given position."""
        layer_id = self._get_layer_at_y(pos.y())
        if not layer_id:
            return
        
        layer = self._layer_manager.get_layer(layer_id)
        if not layer:
            return

        
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 20px;
                border-radius: {border_radius(3)};
            }}
            QMenu::item:selected {{
                background-color: {Colors.ACCENT_BLUE.name()};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {Colors.BORDER.name()};
                margin: 4px 8px;
            }}
        """)
        
        # IMPORTANT: All lambdas use default parameter binding (lid=layer_id) to
        # capture the value at creation time, preventing closure bugs.
        
        # Rename action
        rename_action = menu.addAction("Rename Layer...")
        rename_action.triggered.connect(lambda checked=False, lid=layer_id: self._rename_layer(lid))
        
        # Color action
        color_action = menu.addAction("Change Color...")
        color_action.triggered.connect(lambda checked=False, lid=layer_id: self._change_layer_color(lid))
        
        menu.addSeparator()
        
        # Visibility toggle - uses UNIFIED set_layer_visible (targeted updates + forced repaint)
        if layer.visible:
            hide_action = menu.addAction("Hide Layer")
            hide_action.triggered.connect(lambda checked=False, lid=layer_id: self.set_layer_visible(lid, False))
        else:
            show_action = menu.addAction("Show Layer")
            show_action.triggered.connect(lambda checked=False, lid=layer_id: self.set_layer_visible(lid, True))
        
        # Lock toggle
        if layer.locked:
            unlock_action = menu.addAction("Unlock Layer")
            unlock_action.triggered.connect(lambda checked=False, lid=layer_id: self._toggle_layer_lock(lid, False))
        else:
            lock_action = menu.addAction("Lock Layer")
            lock_action.triggered.connect(lambda checked=False, lid=layer_id: self._toggle_layer_lock(lid, True))
        
        menu.addSeparator()
        
        # Move actions
        move_menu = menu.addMenu("Move Layer")
        move_up = move_menu.addAction("Move Up")
        move_up.triggered.connect(lambda checked=False, lid=layer_id: self._move_layer(lid, -1))
        move_up.setEnabled(layer.index > 0)
        
        move_down = move_menu.addAction("Move Down")
        move_down.triggered.connect(lambda checked=False, lid=layer_id: self._move_layer(lid, 1))
        move_down.setEnabled(layer.index < self._layer_manager.get_layer_count() - 1)
        
        move_top = move_menu.addAction("Move to Top")
        move_top.triggered.connect(lambda checked=False, lid=layer_id: self._move_layer_to(lid, 0))
        move_top.setEnabled(layer.index > 0)
        
        move_bottom = move_menu.addAction("Move to Bottom")
        layer_count = self._layer_manager.get_layer_count()
        move_bottom.triggered.connect(lambda checked=False, lid=layer_id, cnt=layer_count: self._move_layer_to(lid, cnt - 1))
        move_bottom.setEnabled(layer.index < layer_count - 1)
        
        menu.addSeparator()
        
        # Event actions
        select_all = menu.addAction("Select All Events")
        select_all.triggered.connect(lambda checked=False, lid=layer_id: self._select_all_in_layer(lid))
        
        delete_events = menu.addAction("Delete All Events")
        delete_events.triggered.connect(lambda checked=False, lid=layer_id: self._delete_all_in_layer(lid))
        
        menu.addSeparator()
        
        # MA3 Sync actions (for synced or derived layers)
        is_synced = getattr(layer, 'is_synced', False)
        is_derived = getattr(layer, 'derived_from_ma3', False)
        if is_synced or is_derived:
            sync_menu = menu.addMenu("MA3 Sync")
            
            if is_synced:
                disconnect_action = sync_menu.addAction("Disconnect from MA3 Sync")
                disconnect_action.setToolTip(
                    "Detach this layer from MA3 sync. The layer keeps all its data\n"
                    "and becomes an independent derived layer."
                )
                disconnect_action.triggered.connect(
                    lambda checked=False, lid=layer_id: self._detach_sync_layer(lid)
                )
            
            duplicate_action = sync_menu.addAction("Duplicate Layer...")
            duplicate_action.setToolTip(
                "Create a copy of this layer and all its events as a standalone\n"
                "disconnected layer. The original sync is not affected."
            )
            duplicate_action.triggered.connect(
                lambda checked=False, lid=layer_id: self._duplicate_layer(lid)
            )
            
            menu.addSeparator()
        
        # Delete layer action
        delete_action = menu.addAction("Delete Layer")
        delete_action.triggered.connect(lambda checked=False, lid=layer_id: self._delete_layer(lid))
        delete_action.setEnabled(True)
        
        menu.exec(self.mapToGlobal(pos))
    
    def _rename_layer(self, layer_id: str):
        """Rename a layer via dialog (undoable)."""
        layer = self._layer_manager.get_layer(layer_id)
        if not layer:
            return
        
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Layer",
            "Layer name:",
            text=layer.name
        )
        
        if ok and new_name.strip() and new_name.strip() != layer.name:
            facade, command_bus = self._get_facade_and_command_bus()
            if not command_bus:
                from src.utils.message import Log
                Log.warning("LayerLabels._rename_layer: Could not find command_bus")
                return
            
            from src.application.commands import RenameLayerCommand
            # Get callback from parent (EditorPanel) if available
            update_callback = getattr(self.parent(), '_update_events_layer_name', None) if self.parent() else None
            cmd = RenameLayerCommand(self._layer_manager, layer_id, new_name.strip(), update_callback)
            command_bus.execute(cmd)

    def _get_editor_block_id(self) -> Optional[str]:
        """Get the Editor block_id by walking the parent widget chain."""
        timeline_widget = None
        parent = self.parent()
        while parent:
            if isinstance(parent, TimelineWidget):
                timeline_widget = parent
                break
            parent = parent.parent()
        
        if timeline_widget and hasattr(timeline_widget, '_editor_block_id'):
            return timeline_widget._editor_block_id
        
        # Fallback: search for EditorPanel
        try:
            from PyQt6.QtWidgets import QApplication
            from ui.qt_gui.block_panels.editor_panel import EditorPanel
            app = QApplication.instance()
            if app:
                for widget in app.allWidgets():
                    if isinstance(widget, EditorPanel):
                        if timeline_widget and hasattr(widget, 'timeline_widget') and widget.timeline_widget == timeline_widget:
                            return widget.block_id
        except Exception:
            pass
        return None

    def _find_sync_system_manager(self, show_manager_block_id: str):
        """Find the live SyncSystemManager from the ShowManager panel (best-effort)."""
        from PyQt6.QtWidgets import QApplication
        try:
            from ui.qt_gui.block_panels.show_manager_panel import ShowManagerPanel
        except ImportError:
            return None
        app = QApplication.instance()
        if app:
            for widget in app.allWidgets():
                if (isinstance(widget, ShowManagerPanel)
                        and hasattr(widget, 'block_id')
                        and widget.block_id == show_manager_block_id
                        and hasattr(widget, '_sync_system_manager')
                        and widget._sync_system_manager):
                    return widget._sync_system_manager
        return None

    def _detach_sync_layer(self, layer_id: str) -> None:
        """Detach a synced layer from MA3, keeping it as a standalone derived layer.
        
        Uses ShowManagerSettingsManager (via facade) to update the persisted sync
        entity. Does NOT require the ShowManager panel to be open. If the panel
        happens to be open, also attempts to unhook the live MA3 track.
        """
        from src.utils.message import Log
        
        layer = self._layer_manager.get_layer(layer_id)
        if not layer or not getattr(layer, 'is_synced', False):
            return
        
        show_manager_block_id = getattr(layer, 'show_manager_block_id', None)
        if not show_manager_block_id:
            Log.warning("LayerLabels._detach_sync_layer: No show_manager_block_id on layer")
            return
        
        facade, command_bus = self._get_facade_and_command_bus()
        if not facade:
            Log.warning("LayerLabels._detach_sync_layer: Could not find facade")
            return
        
        # Mark the sync entity as disconnected via persisted settings
        try:
            settings_manager = facade.show_manager_settings_manager(show_manager_block_id)
            entity_dict = settings_manager.get_synced_layer("editor", layer.name)
            if entity_dict:
                settings_manager.update_synced_layer("editor", layer.name, {
                    "sync_status": "disconnected",
                    "error_message": "Disconnected by user",
                    "editor_layer_id": None,
                    "editor_block_id": None,
                    "editor_data_item_id": None,
                })
                Log.info(f"LayerLabels: Marked sync entity as disconnected for layer '{layer.name}'")
            else:
                Log.warning(f"LayerLabels._detach_sync_layer: No sync entity found for layer '{layer.name}'")
        except Exception as e:
            Log.warning(f"LayerLabels._detach_sync_layer: Failed to update settings: {e}")
        
        # Best-effort: if the live SyncSystemManager is available, unhook the MA3 track
        ssm = self._find_sync_system_manager(show_manager_block_id)
        if ssm:
            try:
                entity = ssm.get_synced_layer_by_editor_layer(layer.name)
                if entity:
                    ssm.detach_layer(entity.id)
            except Exception as e:
                Log.warning(f"LayerLabels._detach_sync_layer: Live SSM unhook failed (non-critical): {e}")
        
        # Publish BlockUpdated so ShowManager panel (if open) refreshes
        try:
            from src.application.events.event_bus import EventBus
            from src.application.events.block_events import BlockUpdated
            event_bus = EventBus()
            event_bus.publish(BlockUpdated(
                project_id=facade.current_project_id,
                data={
                    "id": show_manager_block_id,
                    "settings_updated": True,
                    "synced_layers_changed": True,
                }
            ))
        except Exception as e:
            Log.warning(f"LayerLabels._detach_sync_layer: Failed to publish event: {e}")
        
        # Update the in-memory layer object for immediate visual feedback
        layer.is_synced = False
        layer.show_manager_block_id = None
        layer.ma3_track_coord = None
        layer.derived_from_ma3 = True
        layer.sync_connection_state = "derived"
        
        # Persist the updated layer properties via command
        if command_bus:
            editor_block_id = self._get_editor_block_id()
            if editor_block_id:
                from src.application.commands.editor_commands import EditorUpdateLayerCommand
                cmd = EditorUpdateLayerCommand(facade, editor_block_id, layer.name, {
                    "is_synced": False,
                    "show_manager_block_id": None,
                    "ma3_track_coord": None,
                    "derived_from_ma3": True,
                })
                command_bus.execute(cmd)
        
        # Repaint (sync state is pushed by SSM via _push_sync_state_to_all_layers)
        self.update()
        Log.info(f"LayerLabels: Detached sync for layer '{layer.name}'")

    def _duplicate_layer(self, layer_id: str) -> None:
        """Duplicate a layer and all its events as a standalone disconnected copy.
        
        The duplicate is initialized as a disconnected layer (is_synced=False).
        If the source was synced or derived from MA3, the copy gets derived_from_ma3=True.
        """
        from src.utils.message import Log
        from PyQt6.QtWidgets import QInputDialog, QMessageBox
        
        layer = self._layer_manager.get_layer(layer_id)
        if not layer:
            return
        
        # Prompt for a name for the duplicate
        default_name = f"{layer.name} (Copy)"
        new_name, ok = QInputDialog.getText(
            self,
            "Duplicate Layer",
            "Name for the duplicate layer:",
            text=default_name
        )
        
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        
        # Check for name collision
        if self._layer_manager.get_layer_by_name(new_name):
            QMessageBox.warning(
                self,
                "Name Already Exists",
                f"A layer named '{new_name}' already exists.\n"
                "Please choose a different name."
            )
            return
        
        facade, command_bus = self._get_facade_and_command_bus()
        if not facade or not command_bus:
            Log.warning("LayerLabels._duplicate_layer: Could not find facade or command_bus")
            return
        
        editor_block_id = self._get_editor_block_id()
        if not editor_block_id:
            Log.warning("LayerLabels._duplicate_layer: Could not find editor_block_id")
            return
        
        # Determine if the duplicate should be marked as derived from MA3
        is_source_synced = getattr(layer, 'is_synced', False)
        is_source_derived = getattr(layer, 'derived_from_ma3', False)
        mark_derived = is_source_synced or is_source_derived
        
        # Step 1: Create the new layer with copied properties
        from src.application.commands.editor_commands import EditorCreateLayerCommand
        layer_properties = {
            "height": layer.height,
            "color": layer.color,
            "visible": layer.visible,
            "locked": layer.locked,
            "group_id": getattr(layer, 'group_id', None),
            "group_name": getattr(layer, 'group_name', None),
            "group_index": getattr(layer, 'group_index', None),
            "is_synced": False,
            "derived_from_ma3": mark_derived,
        }
        create_cmd = EditorCreateLayerCommand(facade, editor_block_id, new_name, layer_properties)
        command_bus.execute(create_cmd)
        
        # Step 2: Copy events from the source layer to the new layer
        from src.application.commands.editor_commands import EditorGetEventsCommand
        from src.application.commands.data_item_commands import AddEventToDataItemCommand
        
        get_events_cmd = EditorGetEventsCommand(facade, editor_block_id, layer_name=layer.name)
        command_bus.execute(get_events_cmd)
        
        events_copied = 0
        for event in get_events_cmd.events:
            data_item_id = event.get("event_data_item_id")
            if not data_item_id:
                continue
            add_cmd = AddEventToDataItemCommand(
                facade,
                data_item_id=data_item_id,
                time=event.get("time", 0.0),
                duration=event.get("duration", 0.0),
                classification=event.get("classification", ""),
                metadata=event.get("metadata"),
                layer_name=new_name,
            )
            command_bus.execute(add_cmd)
            events_copied += 1
        
        self.update()
        Log.info(
            f"LayerLabels: Duplicated layer '{layer.name}' -> '{new_name}' "
            f"({events_copied} events copied)"
        )

    def _change_layer_color(self, layer_id: str):
        """Change layer color via color picker (undoable)."""
        layer = self._layer_manager.get_layer(layer_id)
        if not layer:
            return
        
        current_color = QColor(layer.color) if layer.color else TimelineStyle.get_layer_color(layer.index)
        new_color = QColorDialog.getColor(current_color, self, "Select Layer Color")
        
        if new_color.isValid() and new_color.name() != layer.color:
            facade, command_bus = self._get_facade_and_command_bus()
            if not command_bus:
                from src.utils.message import Log
                Log.warning("LayerLabels._change_layer_color: Could not find command_bus")
                return
            
            from src.application.commands import SetLayerColorCommand
            cmd = SetLayerColorCommand(self._layer_manager, layer_id, new_color.name())
            command_bus.execute(cmd)
    
    def _toggle_layer_lock(self, layer_id: str, locked: bool):
        """Toggle layer lock state (undoable)."""
        facade, command_bus = self._get_facade_and_command_bus()
        if not command_bus:
            from src.utils.message import Log
            Log.warning("LayerLabels._toggle_layer_lock: Could not find command_bus")
            return
        
        from src.application.commands import SetLayerLockCommand
        cmd = SetLayerLockCommand(self._layer_manager, layer_id, locked)
        command_bus.execute(cmd)
    
    def _move_layer(self, layer_id: str, direction: int):
        """Move layer up or down by one position (undoable)."""
        layer = self._layer_manager.get_layer(layer_id)
        if layer:
            new_index = layer.index + direction
            if 0 <= new_index < self._layer_manager.get_layer_count():
                facade, command_bus = self._get_facade_and_command_bus()
                if not command_bus:
                    from src.utils.message import Log
                    Log.warning("LayerLabels._move_layer: Could not find command_bus")
                    return
                
                from src.application.commands import MoveLayerCommand
                cmd = MoveLayerCommand(self._layer_manager, layer_id, new_index)
                command_bus.execute(cmd)
    
    def _move_layer_to(self, layer_id: str, target_index: int):
        """Move layer to a specific position (undoable)."""
        facade, command_bus = self._get_facade_and_command_bus()
        if not command_bus:
            from src.utils.message import Log
            Log.warning("LayerLabels._move_layer_to: Could not find command_bus")
            return
        
        from src.application.commands import MoveLayerCommand
        cmd = MoveLayerCommand(self._layer_manager, layer_id, target_index)
        command_bus.execute(cmd)
    
    def _select_all_in_layer(self, layer_id: str):
        """Select all events in the layer.
        
        Note: Selection is not undoable as per DAW conventions.
        """
        if self._timeline_scene:
            event_ids = []
            for event_id, item in self._timeline_scene.get_all_event_items().items():
                if hasattr(item, 'layer_id') and item.layer_id == layer_id:
                    event_ids.append(event_id)
            if event_ids:
                self._timeline_scene.select_events(event_ids, clear_others=True)
    
    def _delete_all_in_layer(self, layer_id: str):
        """Delete all events in the layer (undoable as single operation).
        
        Uses:
        1. CommandBus macro for single undo step
        2. Batch deletion API for performance
        """
        if not self._timeline_scene:
            return
        
        # Gather events to delete
        event_ids_to_delete = []
        for event_id, item in self._timeline_scene.get_all_event_items().items():
            if hasattr(item, 'layer_id') and item.layer_id == layer_id:
                event_ids_to_delete.append(event_id)
        
        if not event_ids_to_delete:
            return
        
        # Get layer name for macro description
        layer = self._layer_manager.get_layer(layer_id)
        layer_name = layer.name if layer else layer_id
        count = len(event_ids_to_delete)
        
        facade, command_bus = self._get_facade_and_command_bus()
        if not command_bus:
            from src.utils.message import Log
            Log.warning("LayerLabels._delete_all_in_layer: Could not find command_bus")
            return
        
        # Use macro for batch deletion (single undo step)
        command_bus.begin_macro(f"Delete {count} Events from {layer_name}")
        
        try:
            # Use batch API for performance (single signal, suspended updates)
            self._timeline_scene.request_events_delete_batch(event_ids_to_delete)
        finally:
            command_bus.end_macro()
    
    def _delete_layer(self, layer_id: str):
        """Delete the layer completely (undoable).
        
        Uses EditorDeleteLayerCommand to completely remove the layer from:
        - TimelineWidget (UI)
        - EventDataItems (database)
        - Editor block UI state (metadata)
        - ShowManager synced layers (if applicable)
        
        All events in the layer are also deleted.
        
        If the layer is synced with ShowManager, the sync entity will be
        marked as DISCONNECTED so the user can handle it in the ShowManager
        Layer Sync tab (reconnect or remove).
        """
        # Get layer name
        layer = self._layer_manager.get_layer(layer_id)
        if not layer:
            return
        
        layer_name = layer.name
        
        # Get facade and command_bus from parent widget chain
        facade = None
        command_bus = None
        parent = self.parent()
        while parent:
            if hasattr(parent, '_facade') and parent._facade:
                facade = parent._facade
                if hasattr(facade, 'command_bus'):
                    command_bus = facade.command_bus
                break
            elif hasattr(parent, 'facade') and parent.facade:
                facade = parent.facade
                if hasattr(facade, 'command_bus'):
                    command_bus = facade.command_bus
                break
            parent = parent.parent()
        
        if not facade or not command_bus:
            from src.utils.message import Log
            Log.warning("LayerLabels._delete_layer: Could not find facade or command_bus")
            return
        
        # Get Editor block_id from TimelineWidget or EditorPanel
        editor_block_id = None
        timeline_widget = None
        
        # Walk up parent chain to find TimelineWidget
        parent = self.parent()
        while parent:
            if isinstance(parent, TimelineWidget):
                timeline_widget = parent
                break
            parent = parent.parent()
        
        if timeline_widget and hasattr(timeline_widget, '_editor_block_id'):
            editor_block_id = timeline_widget._editor_block_id
        else:
            # Try to find EditorPanel parent
            try:
                from PyQt6.QtWidgets import QApplication
                from ui.qt_gui.block_panels.editor_panel import EditorPanel
                
                app = QApplication.instance()
                if app:
                    for widget in app.allWidgets():
                        if isinstance(widget, EditorPanel):
                            # Check if this TimelineWidget belongs to this EditorPanel
                            if timeline_widget and hasattr(widget, 'timeline_widget') and widget.timeline_widget == timeline_widget:
                                editor_block_id = widget.block_id
                                break
            except Exception:
                pass
        
        if not editor_block_id:
            # Fallback to old behavior (UI-only deletion)
            from src.application.commands import DeleteLayerCommand
            cmd = DeleteLayerCommand(self._layer_manager, layer_id)
            command_bus.execute(cmd)
            return
        
        # Use EditorDeleteLayerCommand for complete deletion
        from src.application.commands import EditorDeleteLayerCommand
        
        cmd = EditorDeleteLayerCommand(
            facade=facade,
            block_id=editor_block_id,
            layer_name=layer_name
        )
        command_bus.execute(cmd)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        scene_top_y = 0.0
        if self._timeline_view and self._timeline_view.scene():
            viewport_top = self._timeline_view.mapToScene(0, 0)
            scene_top_y = viewport_top.y()
            
            scrollbar = self._timeline_view.verticalScrollBar()
            if scrollbar and scrollbar.value() == 0:
                if abs(scene_top_y) > 0.1:
                    scene_top_y = 0.0
        
        painter.fillRect(0, 0, width, height, Colors.BG_MEDIUM)
        painter.setPen(QPen(Colors.BORDER, 1))
        painter.drawLine(width - 1, 0, width - 1, height)
        painter.setClipRect(0, 0, width, height)
        
        font = Typography.default_font()
        font.setPixelSize(11)
        painter.setFont(font)
        
        # Render layers in visual order (by index)
        # Track which groups we've already rendered to avoid duplicate headers
        processed_groups: set = set()
        all_layers = self._layer_manager.get_all_layers()
        
        # Build map of groups for separator line rendering
        grouped_layers: Dict[Optional[str], List[TimelineLayer]] = defaultdict(list)
        for layer in all_layers:
            if layer.visible and layer.group_id:
                grouped_layers[layer.group_id].append(layer)
        
        # Sort grouped layers by group_index within each group
        for group_id in grouped_layers:
            grouped_layers[group_id].sort(key=lambda l: (l.group_index if l.group_index is not None else 999, l.index))
        
        # Edge case: handle groups with no visible children - ensure they still render as a single child
        # This is already handled because we only add to grouped_layers when layers are visible and have group_id
        
        # Render all layers in order
        for layer in all_layers:
            if not layer.visible:
                continue
            
            # Render group header if this is the first layer in a group
            is_first_in_group = False
            if layer.group_id and layer.group_id not in processed_groups:
                processed_groups.add(layer.group_id)
                self._render_group_header(painter, layer, scene_top_y, width, height)
                is_first_in_group = True
            
            # Render the layer with appropriate indentation
            is_grouped = layer.group_id is not None
            # Skip top separator for first layer in group (header already provides separation)
            self._render_layer(painter, layer, scene_top_y, width, height, is_grouped=is_grouped, skip_top_separator=is_first_in_group)
        
        # Note: Group separator lines removed - layer separators already provide visual separation
        # The group header at the top of each group is sufficient for visual grouping
    
    def _render_group_header(self, painter: QPainter, first_layer: TimelineLayer, scene_top_y: float, width: int, height: int):
        """Render a group header divider above the first child layer (as a folder/separator)."""
        if not first_layer.group_name:
            return
        
        GROUP_HEADER_HEIGHT = 18  # Height of group header divider
        
        # Position header above the first layer (header takes up space, calculated in get_layer_y_position)
        # The first layer's y position already accounts for the header height
        first_layer_y = self._layer_manager.get_layer_y_position(first_layer.id)
        header_y = int(first_layer_y - scene_top_y - GROUP_HEADER_HEIGHT)
        
        if header_y + GROUP_HEADER_HEIGHT < 0 or header_y > height:
            return
        
        # Draw the divider background (acts as a folder header)
        divider_bg_color = QColor(Colors.BG_MEDIUM)
        divider_bg_color = divider_bg_color.darker(115)  # 15% darker for better contrast
        painter.fillRect(0, header_y, width, GROUP_HEADER_HEIGHT, divider_bg_color)
        
        # Draw divider top border line only (bottom is handled by layer separator)
        painter.setPen(QPen(Colors.BORDER, 2))
        painter.drawLine(0, header_y, width, header_y)  # Top border only
        
        # Group header text (smaller, fits within divider)
        painter.setPen(Colors.TEXT_PRIMARY)
        header_font = Typography.default_font()
        header_font.setPixelSize(9)  # Smaller font to fit in divider
        header_font.setBold(True)
        painter.setFont(header_font)
        
        group_name = first_layer.group_name
        available_width = width - 16
        metrics = painter.fontMetrics()
        
        # Truncate group name if too long
        if metrics.horizontalAdvance(group_name) > available_width:
            while len(group_name) > 3 and metrics.horizontalAdvance(group_name + "...") > available_width:
                group_name = group_name[:-1]
            group_name = group_name + "..."
        
        # Place text centered vertically within the divider
        text_height = metrics.height()
        text_y = header_y + (GROUP_HEADER_HEIGHT - text_height) // 2
        text_rect = QRectF(8, float(text_y), float(available_width), float(text_height))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            group_name
        )
    
    def _render_layer(self, painter: QPainter, layer: TimelineLayer, scene_top_y: float, width: int, height: int, is_grouped: bool, skip_top_separator: bool = False):
        """Render a single layer with optional indentation for grouped layers.
        
        Args:
            painter: QPainter to draw with
            layer: Layer to render
            scene_top_y: Top Y position of the visible scene area
            width: Width of the widget
            height: Height of the widget
            is_grouped: Whether this layer is part of a group
            skip_top_separator: If True, skip drawing the top separator line
        """
        scene_y = self._layer_manager.get_layer_y_position(layer.id)
        widget_y = int(scene_y - scene_top_y)
        layer_height = layer.height
        
        if widget_y + layer_height < 0 or widget_y > height:
            return
        
        # Indentation for grouped layers
        indent_offset = 20 if is_grouped else 0
        indicator_x = 4 + indent_offset
        
        # Color indicator
        if layer.color:
            color = QColor(layer.color)
        else:
            color = TimelineStyle.get_layer_color(layer.index)
        
        indicator_y = int(widget_y + 8)
        indicator_height = int(layer_height - 16)
        painter.fillRect(indicator_x, indicator_y, 8, indicator_height, color)
        
        # Sync state indicator (right-aligned)
        # Reserve space for the icon so the label truncation accounts for it
        sync_state = self._get_sync_connection_state(layer)
        icon_width = 0
        if sync_state != "none":
            icon_width = 16  # icon size + padding
        
        # Label text
        painter.setPen(Colors.TEXT_PRIMARY)
        font = Typography.default_font()
        font.setPixelSize(11)
        painter.setFont(font)
        
        label = layer.name if layer.name else f"Layer {layer.index + 1}"
        
        # Dynamic truncation based on available width (account for indentation and icon)
        available_width = width - 24 - indent_offset - icon_width
        metrics = painter.fontMetrics()
        if metrics.horizontalAdvance(label) > available_width:
            # Truncate with ellipsis
            while len(label) > 3 and metrics.horizontalAdvance(label + "...") > available_width:
                label = label[:-1]
            label = label + "..."
        
        text_x = 16 + indent_offset
        text_rect = QRectF(float(text_x), float(widget_y), float(width - text_x - 4 - icon_width), float(layer_height))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            label
        )
        
        # Draw sync state icon (right-aligned)
        if sync_state != "none":
            self._draw_sync_indicator(painter, layer, width, widget_y, layer_height, sync_state)
        
        # Connector line for grouped layers (vertical line from group header)
        if is_grouped:
            painter.setPen(QPen(Colors.BORDER, 1))
            # Draw vertical connector line on the left
            connector_x = 14
            painter.drawLine(connector_x, int(widget_y), connector_x, int(widget_y + layer_height))

    def _get_sync_connection_state(self, layer: TimelineLayer) -> str:
        """Read the sync connection state directly from the layer.
        
        The SSM is the single source of truth and pushes state changes
        onto TimelineLayer.sync_connection_state.  This method is a
        zero-cost read -- no lookups, no cache, no settings manager.
        
        Returns:
            "active"               - Sync entity exists, track is hooked, MA3 connected
            "diverged"             - MA3 and Editor events differ, awaiting user resolution
            "awaiting_connection"  - No MA3 connection; waiting for MA3 to reconnect
            "disconnected"         - is_synced=True but sync is broken/unavailable
            "derived"              - Layer was detached from MA3 (derived_from_ma3=True)
            "none"                 - Regular layer, no sync involvement
        """
        state = getattr(layer, 'sync_connection_state', 'none')
        return state

    # _check_live_sync_state removed -- sync state is now pushed directly
    # onto TimelineLayer.sync_connection_state by the SyncSystemManager.

    def _draw_sync_indicator(self, painter: QPainter, layer: TimelineLayer, width: int, widget_y: int, layer_height: float, sync_state: str):
        """Draw a small sync state indicator icon on the right side of the layer label.
        
        - "active": Two-arrow sync symbol in accent green
        - "diverged": Exclamation-in-triangle warning symbol in yellow/amber
        - "awaiting_connection": Signal/antenna with X in red-orange (no MA3 connection)
        - "disconnected": Broken link / warning symbol in amber/orange
        - "derived": Small diamond/tag origin marker in muted secondary color
        """
        icon_size = 10
        icon_x = width - icon_size - 6  # 6px right margin
        icon_y = int(widget_y + (layer_height - icon_size) / 2)
        
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        cx = icon_x + icon_size / 2
        cy = icon_y + icon_size / 2
        r = icon_size / 2 - 1
        
        if sync_state == "active":
            # Active sync: draw a bidirectional arrow (sync symbol) in green
            icon_color = QColor(Colors.ACCENT_GREEN)
            painter.setPen(QPen(icon_color, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            # Top arc (right-to-left)
            path = QPainterPath()
            path.moveTo(cx + r * 0.7, cy - r * 0.3)
            path.arcTo(cx - r, cy - r, r * 2, r * 2, 20, 140)
            painter.drawPath(path)
            # Arrowhead on top arc
            end_x = cx - r * 0.7
            end_y = cy - r * 0.3
            painter.drawLine(QPointF(end_x, end_y), QPointF(end_x + 3, end_y - 2))
            painter.drawLine(QPointF(end_x, end_y), QPointF(end_x + 2, end_y + 3))
            
            # Bottom arc (left-to-right)
            path2 = QPainterPath()
            path2.moveTo(cx - r * 0.7, cy + r * 0.3)
            path2.arcTo(cx - r, cy - r, r * 2, r * 2, 200, 140)
            painter.drawPath(path2)
            # Arrowhead on bottom arc
            end_x2 = cx + r * 0.7
            end_y2 = cy + r * 0.3
            painter.drawLine(QPointF(end_x2, end_y2), QPointF(end_x2 - 3, end_y2 + 2))
            painter.drawLine(QPointF(end_x2, end_y2), QPointF(end_x2 - 2, end_y2 - 3))
        
        elif sync_state == "diverged":
            # Diverged: exclamation-in-triangle warning in yellow/amber
            icon_color = Colors.STATUS_WARNING
            painter.setPen(QPen(icon_color, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            # Draw a small triangle
            tri_path = QPainterPath()
            tri_path.moveTo(cx, cy - r)           # top
            tri_path.lineTo(cx + r, cy + r * 0.8)  # bottom-right
            tri_path.lineTo(cx - r, cy + r * 0.8)  # bottom-left
            tri_path.closeSubpath()
            painter.drawPath(tri_path)
            
            # Exclamation mark inside
            painter.setPen(QPen(icon_color, 1.8))
            painter.drawLine(QPointF(cx, cy - r * 0.4), QPointF(cx, cy + r * 0.15))
            painter.setBrush(icon_color)
            painter.drawEllipse(QPointF(cx, cy + r * 0.5), 0.8, 0.8)
        
        elif sync_state == "awaiting_connection":
            # Awaiting MA3 connection: signal/antenna icon with X in red-orange
            icon_color = Colors.ACCENT_ORANGE
            painter.setPen(QPen(icon_color, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            # Draw three arc signal waves (antenna icon)
            for i, scale in enumerate([0.4, 0.65, 0.9]):
                arc_r = r * scale
                painter.drawArc(
                    int(cx - arc_r), int(cy - arc_r),
                    int(arc_r * 2), int(arc_r * 2),
                    30 * 16, 120 * 16  # Start 30deg, span 120deg (top-right quadrant)
                )
            
            # Small X through center to indicate "no signal"
            x_size = r * 0.35
            painter.setPen(QPen(icon_color, 1.8))
            painter.drawLine(QPointF(cx - x_size, cy - x_size), QPointF(cx + x_size, cy + x_size))
            painter.drawLine(QPointF(cx + x_size, cy - x_size), QPointF(cx - x_size, cy + x_size))
        
        elif sync_state == "disconnected":
            # Disconnected sync: broken link / warning indicator in amber/orange
            icon_color = QColor(Colors.STATUS_WARNING)
            painter.setPen(QPen(icon_color, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            # Draw a broken chain-link: two C-shaped halves with a gap
            # Left half-link
            path_left = QPainterPath()
            path_left.moveTo(cx - 1, cy - r)
            path_left.arcTo(cx - r - 1, cy - r, r * 1.2, r * 2, 90, 180)
            painter.drawPath(path_left)
            
            # Right half-link
            path_right = QPainterPath()
            path_right.moveTo(cx + 1, cy - r)
            path_right.arcTo(cx - r * 0.2 + 1, cy - r, r * 1.2, r * 2, 90, -180)
            painter.drawPath(path_right)
            
            # Small slash through the gap to emphasize "broken"
            painter.drawLine(
                QPointF(cx + 2, cy - 2),
                QPointF(cx - 2, cy + 2)
            )
        
        elif sync_state == "derived":
            # Derived from MA3: draw a small diamond/tag origin marker in muted color
            icon_color = QColor(Colors.TEXT_SECONDARY)
            painter.setPen(QPen(icon_color, 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            s = icon_size / 2 - 1  # half-size of diamond
            
            # Draw a diamond shape
            path = QPainterPath()
            path.moveTo(cx, cy - s)       # top
            path.lineTo(cx + s, cy)        # right
            path.lineTo(cx, cy + s)        # bottom
            path.lineTo(cx - s, cy)        # left
            path.closeSubpath()
            painter.drawPath(path)
            
            # Small dot in center
            painter.setBrush(icon_color)
            painter.drawEllipse(QPointF(cx, cy), 1.5, 1.5)
        
        painter.restore()
    
    def resizeEvent(self, event):
        """Emit width changed signal when resized."""
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self.width_changed.emit(event.size().width())
    
    def get_default_width(self) -> int:
        """Get the default width for the layer labels."""
        return self._default_width

# =============================================================================
# Main Timeline Widget
# =============================================================================

class TimelineWidget(QWidget):
    """
    Self-contained DAW-style timeline widget.
    
    Drop into any panel or window. Manages its own layers, events,
    playback, and editing. Communicates changes via signals.
    
    Signals (Output):
        events_moved(list): List[EventMoveResult] - events were moved
        events_resized(list): List[EventResizeResult] - events were resized
        event_deleted(object): EventDeleteResult - event was deleted
        event_created(object): EventCreateResult - new event created
        event_sliced(object): EventSliceResult - event was sliced/split
        selection_changed(list): List[str] - selected event IDs changed
        position_changed(float): Playhead position changed
        playback_state_changed(bool): Play/pause state changed
        
        Legacy signals (for backwards compatibility):
        event_selected(str): First selected event ID
    """
    
    # New typed signals
    events_moved = pyqtSignal(list)      # List[EventMoveResult]
    events_resized = pyqtSignal(list)    # List[EventResizeResult]
    event_created = pyqtSignal(object)   # EventCreateResult
    event_deleted = pyqtSignal(object)   # EventDeleteResult (single)
    events_deleted = pyqtSignal(list)    # List[EventDeleteResult] (batch - for performance)
    event_sliced = pyqtSignal(object)    # EventSliceResult
    selection_changed = pyqtSignal(list) # List[str]
    position_changed = pyqtSignal(float)
    playback_state_changed = pyqtSignal(bool)
    
    # Legacy signals (backwards compatibility)
    event_selected = pyqtSignal(str)
    
    def __init__(self, parent=None, preferences_repo=None):
        super().__init__(parent)
        
        # Settings manager (single source of truth for UI settings)
        self._settings_manager = TimelineSettingsManager(preferences_repo, self)
        set_timeline_settings_manager(self._settings_manager)
        
        self._pixels_per_second = self._settings_manager.default_pixels_per_second
        
        # Debounce timers for settings saves (only save on drag end)
        self._layer_column_save_timer = QTimer(self)
        self._layer_column_save_timer.setSingleShot(True)
        self._layer_column_save_timer.setInterval(300)  # 300ms after drag stops
        self._layer_column_save_timer.timeout.connect(self._save_layer_column_width)
        self._pending_layer_column_width = None
        
        self._inspector_save_timer = QTimer(self)
        self._inspector_save_timer.setSingleShot(True)
        self._inspector_save_timer.setInterval(300)
        self._inspector_save_timer.timeout.connect(self._save_inspector_width)
        self._pending_inspector_width = None
        
        # Debounce timer for scroll position saves
        self._scroll_save_timer = QTimer(self)
        self._scroll_save_timer.setSingleShot(True)
        self._scroll_save_timer.setInterval(500)  # 500ms after scrolling stops
        self._scroll_save_timer.timeout.connect(self._save_scroll_position)
        
        # Continuous smooth scrolling for playhead follow
        # Will be initialized after _view is created in _setup_ui
        self._scroll_animation: Optional[QPropertyAnimation] = None  # Keep for fallback
        self._target_scroll_x: Optional[float] = None  # Target scroll position for continuous following
        self._scroll_lerp_factor = 0.25  # Interpolation factor (0.25 = smooth, catches up in ~4 frames at 60fps)
        
        # Create core components
        self._layer_manager = LayerManager()
        self._grid_system = GridSystem()
        self._playback_controller = PlaybackController(self)
        
        # Scene with layer manager
        self._scene = TimelineScene(self._layer_manager)
        self._scene.grid_system = self._grid_system
        # Sync new timing system with grid system
        if hasattr(self._scene, '_sync_grid_calculator'):
            self._scene._sync_grid_calculator()
        
        # Movement controller
        self._movement_controller = MovementController(self._scene, self._layer_manager)
        self._scene.set_movement_controller(self._movement_controller)
        
        # Viewport event filter will be installed after _view is created in _setup_ui()
        self._viewport_update_timer = None
        
        # Apply all saved settings to components
        self._apply_saved_settings()
        
        self._setup_ui()
        self._connect_signals()
        
        # Connect ruler to scene's grid calculator for consistent intervals
        self._sync_ruler_with_scene()
        
        # Connect settings changes to update components
        self._settings_manager.settings_changed.connect(self._on_setting_changed)
    
    def load_all_waveforms_now(self) -> None:
        """
        Manually trigger loading all waveforms for currently loaded events.
        
        Useful for:
        - Eliminating all scrolling lag
        - Pre-loading before a long session
        - Ensuring smooth playback
        """
        from ..logging import TimelineLog as Log
        Log.info("TimelineWidget: Loading all waveforms...")
        self._schedule_staged_waveform_loading()
    
    def eventFilter(self, obj, event):
        """Filter viewport events to detect scrolling/zooming."""
        # Viewport event filter can be used for other purposes in the future
        return super().eventFilter(obj, event)
    
    def _sync_ruler_with_scene(self):
        """Sync ruler with scene's grid calculator for consistent grid line intervals."""
        if hasattr(self._scene, '_grid_calculator') and hasattr(self._scene, '_unit_preference'):
            self._ruler.set_scene_grid_calculator(
                self._scene._grid_calculator,
                self._scene._unit_preference
            )
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._sync_layer_labels_height)
    
    def _setup_ui(self):
        """Setup the UI layout."""
        # Ensure widget expands to fill available space
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setHandleWidth(4)
        self._main_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._main_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {Colors.BORDER.name()};
            }}
            QSplitter::handle:hover {{
                background-color: {Colors.ACCENT_BLUE.name()};
            }}
        """)
        
        # Timeline container - holds top row, content, and toolbar as a grouped unit
        timeline_container = QFrame()
        timeline_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        timeline_container.setStyleSheet(f"background-color: {Colors.BG_DARK.name()};")
        timeline_layout = QVBoxLayout(timeline_container)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(0)
        
        # ========== TOP ROW (ruler, corner, event sources) ==========
        # Fixed height, expands horizontally with the container
        self._top_row = QFrame()
        self._top_row.setFixedHeight(RULER_HEIGHT)
        self._top_row.setStyleSheet(f"background-color: {Colors.BG_MEDIUM.name()};")
        top_layout = QHBoxLayout(self._top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        
        self._corner = QFrame()
        self._corner.setFixedSize(120, RULER_HEIGHT)
        self._corner.setStyleSheet(f"background-color: {Colors.BG_MEDIUM.name()};")
        top_layout.addWidget(self._corner)
        
        self._ruler = TimeRuler(self._grid_system, self._settings_manager)
        self._ruler.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top_layout.addWidget(self._ruler, 1)
        
        self._top_right_widget = None
        self._top_right_container = QWidget()
        self._top_right_container.setFixedHeight(RULER_HEIGHT)
        self._top_right_container.setStyleSheet(f"background-color: {Colors.BG_MEDIUM.name()};")
        self._top_right_container.setVisible(False)
        self._top_right_layout = QHBoxLayout(self._top_right_container)
        self._top_right_layout.setContentsMargins(8, 0, 8, 0)
        self._top_right_layout.setSpacing(4)
        top_layout.addWidget(self._top_right_container, 0)
        
        timeline_layout.addWidget(self._top_row)
        
        # Main content area with draggable splitter between layer labels and timeline view
        self._content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._content_splitter.setHandleWidth(4)
        self._content_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._content_splitter.setStyleSheet(f"""
            QSplitter::handle:horizontal {{
                background-color: {Colors.BORDER.name()};
            }}
            QSplitter::handle:horizontal:hover {{
                background-color: {Colors.ACCENT_BLUE.name()};
            }}
        """)
        
        self._layer_labels = LayerLabels(self._layer_manager)
        self._content_splitter.addWidget(self._layer_labels)
        
        self._view = TimelineView(self._scene)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._apply_scrollbar_settings()  # Apply saved scrollbar visibility settings
        
        # Initialize smooth scroll animation for playhead follow (fallback for PAGE mode)
        self._scroll_animation = QPropertyAnimation(self._view.horizontalScrollBar(), b"value", self)
        self._scroll_animation.setDuration(100)  # 100ms for smooth but responsive scrolling during playback
        self._scroll_animation.setEasingCurve(QEasingCurve.Type.OutCubic)  # Smooth deceleration
        
        self._content_splitter.addWidget(self._view)
        
        # Set initial sizes from saved settings
        saved_column_width = self._settings_manager.layer_column_width
        self._content_splitter.setSizes([saved_column_width, 600])
        self._content_splitter.setStretchFactor(0, 0)  # Layer labels don't stretch
        self._content_splitter.setStretchFactor(1, 1)  # Timeline view stretches
        
        # Sync corner widget width with layer column + splitter handle
        splitter_handle_width = self._content_splitter.handleWidth()
        self._corner.setFixedWidth(saved_column_width + splitter_handle_width)
        
        # Sync corner width with layer labels width
        self._layer_labels.width_changed.connect(self._on_layer_labels_width_changed)
        self._content_splitter.splitterMoved.connect(self._on_content_splitter_moved)
        
        self._view.verticalScrollBar().valueChanged.connect(lambda: self._layer_labels.update())
        self._view.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)
        # Note: Horizontal scroll handled via scroll_changed signal in _connect_signals()
        # to avoid duplicate connections and properly update ruler
        self._layer_labels.set_timeline_view(self._view)
        self._layer_labels.set_timeline_scene(self._scene)
        
        # ========== CONTENT AREA (layer labels + timeline view) ==========
        # Expands both horizontally and vertically
        timeline_layout.addWidget(self._content_splitter, 1)
        
        # ========== BOTTOM TOOLBAR (play/pause, clock, settings) ==========
        # Fixed height, expands horizontally with the container
        self._toolbar = self._create_toolbar()
        timeline_layout.addWidget(self._toolbar)
        
        self._main_splitter.addWidget(timeline_container)
        
        self._inspector_container = self._create_inspector_panel()
        self._main_splitter.addWidget(self._inspector_container)
        
        # Apply saved inspector settings
        saved_inspector_width = self._settings_manager.inspector_width
        self._main_splitter.setSizes([600, saved_inspector_width])
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 0)
        
        # Track inspector width changes
        self._main_splitter.splitterMoved.connect(self._on_main_splitter_moved)
        
        # Apply inspector visibility
        self._inspector_container.setVisible(self._settings_manager.inspector_visible)
        
        main_layout.addWidget(self._main_splitter)
        
        # Restore scroll position after a short delay (after layout is complete)
        QTimer.singleShot(100, self._restore_scroll_position)
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Colors.BG_DARK.name()};
            }}
        """)
    
    def _apply_saved_settings(self):
        """
        Apply all saved settings from TimelineSettingsManager to components.
        
        Called once during initialization to restore user preferences.
        """
        # Grid/snap settings
        self._grid_system.snap_enabled = self._settings_manager.snap_enabled
        self._grid_system.settings.snap_enabled = self._settings_manager.snap_enabled
        # Snap interval mode (auto, 1f, 5f, 10f, etc.)
        self._grid_system.set_snap_interval_mode(self._settings_manager.snap_interval_mode)
        # Note: Grid intervals are automatically calculated from timebase/FPS.
        # No manual multipliers - grid adapts to zoom level.
        self._grid_system.settings.show_grid_lines = self._settings_manager.show_grid_lines
        
        # Sync new timing system if scene is available
        if hasattr(self, '_scene') and hasattr(self._scene, '_sync_grid_calculator'):
            self._scene._sync_grid_calculator()
        
        # Sync ruler with scene's grid calculator
        if hasattr(self, '_ruler'):
            self._sync_ruler_with_scene()
        
        # Layer settings - apply default height to layer manager
        self._layer_manager.set_default_layer_height(self._settings_manager.default_layer_height)
        
        # Zoom level
        self._pixels_per_second = self._settings_manager.default_pixels_per_second
        
        Log.debug(f"TimelineWidget: Applied saved settings - snap_enabled={self._settings_manager.snap_enabled}, "
                  f"show_grid_lines={self._settings_manager.show_grid_lines}, "
                  f"default_layer_height={self._settings_manager.default_layer_height}")
    
    def _on_setting_changed(self, setting_name: str):
        """
        Handle settings changes and update the relevant component.
        
        Called when a setting is changed via the settings manager.
        """
        # Grid/snap settings
        if setting_name == 'snap_enabled':
            self._grid_system.snap_enabled = self._settings_manager.snap_enabled
            self._grid_system.settings.snap_enabled = self._settings_manager.snap_enabled
            # Sync new timing system if available
            if hasattr(self, '_scene') and hasattr(self._scene, '_snap_calculator'):
                self._scene._snap_calculator.snap_enabled = self._settings_manager.snap_enabled
        elif setting_name == 'snap_interval_mode':
            # Update grid system with new snap interval mode
            self._grid_system.set_snap_interval_mode(self._settings_manager.snap_interval_mode)
        elif setting_name == 'snap_to_grid':
            # snap_to_grid is used similarly to snap_enabled
            pass
        # Note: grid_major/minor_interval_multiplier settings removed.
        # Grid intervals are automatically calculated from timebase/FPS.
        elif setting_name == 'show_grid_lines':
            self._grid_system.settings.show_grid_lines = self._settings_manager.show_grid_lines
            # Sync new timing system if available
            if hasattr(self, '_scene') and hasattr(self._scene, '_grid_renderer'):
                self._scene._grid_renderer.show_grid_lines = self._settings_manager.show_grid_lines
            if hasattr(self, '_scene'):
                self._scene.update()  # Redraw to show/hide grid
        
        # Layer settings
        elif setting_name == 'default_layer_height':
            self._layer_manager.set_default_layer_height(self._settings_manager.default_layer_height)
        
        # Scrollbar settings
        elif setting_name in ('vertical_scrollbar_always_visible', 'horizontal_scrollbar_always_visible'):
            self._apply_scrollbar_settings()
        
        # Zoom settings
        elif setting_name == 'default_pixels_per_second':
            self._pixels_per_second = self._settings_manager.default_pixels_per_second
        
        # Block event styling - update all block events when styling changes
        elif setting_name.startswith('block_event_'):
            # Update all block event items
            for event_id, item in self._scene.get_all_event_items().items():
                if isinstance(item, BlockEventItem):
                    if setting_name == 'block_event_height':
                        # Height change requires geometry update
                        if hasattr(item, '_update_geometry'):
                            item._update_geometry()
                    elif setting_name in ('block_event_opacity', 'block_event_z_value', 'block_event_rotation', 'block_event_scale',
                                         'block_event_drop_shadow_enabled', 'block_event_drop_shadow_blur_radius',
                                         'block_event_drop_shadow_offset_x', 'block_event_drop_shadow_offset_y',
                                         'block_event_drop_shadow_color', 'block_event_drop_shadow_opacity'):
                        # Qt styling changes
                        if hasattr(item, '_apply_qt_styling'):
                            item._apply_qt_styling("block_event")
                    else:
                        # Other styling changes (border, font, etc.) just need repaint
                        if hasattr(item, 'update'):
                            item.update()
            self._scene.update()
        
        # Marker event styling - update all marker events when styling changes
        elif setting_name.startswith('marker_event_'):
            # Update all marker event items
            for event_id, item in self._scene.get_all_event_items().items():
                if isinstance(item, MarkerEventItem):
                    if setting_name in ('marker_event_width', 'marker_event_shape'):
                        # Width or shape change requires geometry update
                        if hasattr(item, '_update_geometry'):
                            item._update_geometry()
                    elif setting_name in ('marker_event_opacity', 'marker_event_z_value', 'marker_event_rotation', 'marker_event_scale',
                                         'marker_event_drop_shadow_enabled', 'marker_event_drop_shadow_blur_radius',
                                         'marker_event_drop_shadow_offset_x', 'marker_event_drop_shadow_offset_y',
                                         'marker_event_drop_shadow_color', 'marker_event_drop_shadow_opacity'):
                        # Qt styling changes
                        if hasattr(item, '_apply_qt_styling'):
                            item._apply_qt_styling("marker_event")
                    else:
                        # Other styling changes (border, etc.) just need repaint
                        if hasattr(item, 'update'):
                            item.update()
            
            # Also update block events if marker width changed (they use it as minimum)
            if setting_name == 'marker_event_width':
                for event_id, item in self._scene.get_all_event_items().items():
                    if isinstance(item, BlockEventItem):
                        if hasattr(item, '_update_geometry'):
                            item._update_geometry()
            
            self._scene.update()
        
        # Inspector settings are handled by the inspector widget itself
    
    def _apply_scrollbar_settings(self):
        """Apply scrollbar visibility settings from settings manager."""
        if not hasattr(self, '_view') or not self._view:
            return
            
        from PyQt6.QtCore import Qt
        
        # Vertical scrollbar
        if self._settings_manager.vertical_scrollbar_always_visible:
            self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        else:
            self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Horizontal scrollbar
        if self._settings_manager.horizontal_scrollbar_always_visible:
            self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        else:
            self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    
    def set_vertical_scrollbar_always_visible(self, always_visible: bool):
        """Set vertical scrollbar visibility (persists to settings)."""
        self._settings_manager.vertical_scrollbar_always_visible = always_visible
        self._apply_scrollbar_settings()
    
    def set_horizontal_scrollbar_always_visible(self, always_visible: bool):
        """Set horizontal scrollbar visibility (persists to settings)."""
        self._settings_manager.horizontal_scrollbar_always_visible = always_visible
        self._apply_scrollbar_settings()
    
    def _on_scroll_changed(self):
        """Handle scroll position change - debounce save."""
        self._scroll_save_timer.start()
    
    def _save_scroll_position(self):
        """Save current scroll position after debounce delay."""
        if not hasattr(self, '_view') or not self._view:
            return
        
        h_bar = self._view.horizontalScrollBar()
        v_bar = self._view.verticalScrollBar()
        
        if h_bar:
            self._settings_manager.last_scroll_x = h_bar.value()
        if v_bar:
            self._settings_manager.last_scroll_y = v_bar.value()
    
    def _restore_scroll_position(self):
        """Restore scroll position from settings (or default to top-left)."""
        if not hasattr(self, '_view') or not self._view:
            return
        
        h_bar = self._view.horizontalScrollBar()
        v_bar = self._view.verticalScrollBar()
        
        if self._settings_manager.restore_scroll_position:
            if h_bar:
                h_bar.setValue(self._settings_manager.last_scroll_x)
            if v_bar:
                v_bar.setValue(self._settings_manager.last_scroll_y)
        else:
            if h_bar:
                h_bar.setValue(0)
            if v_bar:
                v_bar.setValue(0)
    
    def scroll_to_top_left(self):
        """Scroll view to top-left position."""
        if hasattr(self, '_view') and self._view:
            h_bar = self._view.horizontalScrollBar()
            v_bar = self._view.verticalScrollBar()
            if h_bar:
                h_bar.setValue(0)
            if v_bar:
                v_bar.setValue(0)
    
    def _pre_warm_viewport_cache(self):
        """
        Pre-warm viewport cache by triggering paint events for visible items.
        
        PERFORMANCE: Replicates what happens during first scroll - triggers paint()
        for all visible items, which builds waveform paths, caches geometry, etc.
        This eliminates the lag on first horizontal scroll.
        
        Uses Qt's update() mechanism to trigger natural paint events.
        """
        if not self._view or not self._scene:
            return
        
        # Get visible viewport region in scene coordinates
        viewport_rect = self._view.viewport().rect()
        visible_rect = self._view.mapToScene(viewport_rect).boundingRect()
        
        # Expand to include items just outside viewport (pre-warm scroll buffer)
        # This ensures smooth scrolling in both directions
        padding = 500  # pixels - enough to cover typical scroll distance
        visible_rect.adjust(-padding, -padding, padding, padding)
        
        # Trigger update for visible region - this causes Qt to paint all items in region
        # which naturally builds all caches (waveform paths, geometry, etc.)
        self._scene.update(visible_rect)
        
        # Also trigger view update to ensure viewport is painted
        # Note: Removed processEvents() call - it was blocking the UI during load.
        # Qt will handle paint events naturally without blocking.
        self._view.update()
    
    
    def _schedule_staged_waveform_loading(self):
        """
        Pre-load all waveforms synchronously with progress feedback.
        
        Uses sync loading (which works reliably) with processEvents() 
        to keep UI responsive. Shows progress in status bar.
        """
        if not self._scene:
            return
        
        from ..events.items import BlockEventItem
        from ..logging import TimelineLog as Log
        from PyQt6.QtWidgets import QApplication
        
        # Collect ALL items that need waveforms
        items_to_load = []
        for item in self._scene._event_items.values():
            if not isinstance(item, BlockEventItem):
                continue
            if not item._should_show_waveform():
                continue
            items_to_load.append(item)
        
        total = len(items_to_load)
        if total == 0:
            return
        
        Log.info(f"TimelineWidget: Pre-loading {total} waveforms...")
        
        # Pre-load in batches with UI updates
        BATCH_SIZE = 5  # Small batches for responsive UI
        loaded = 0
        
        got_waveform = 0
        for i, item in enumerate(items_to_load):
            try:
                # Trigger sync load by calling _get_waveform_data()
                # This loads and caches the waveform
                wf_data, dur = item._get_waveform_data()
                if wf_data is not None and dur > 0:
                    got_waveform += 1
                loaded += 1
            except Exception as e:
                Log.debug(f"TimelineWidget: Failed to pre-load waveform: {e}")
            
            # Process events every batch to keep UI responsive
            if (i + 1) % BATCH_SIZE == 0:
                QApplication.processEvents()
        
        Log.info(f"TimelineWidget: Pre-loaded {loaded}/{total} waveforms (waveform data: {got_waveform})")
    
    
    def _sync_layer_labels_height(self):
        """Sync layer labels - no longer sets fixed height to allow proper expansion."""
        # Note: Previously this set a fixed height which constrained the content_splitter.
        # Now layer_labels uses QSizePolicy.Expanding and grows with its container.
        pass
    
    def resizeEvent(self, event):
        """Handle widget resize events."""
        super().resizeEvent(event)
    
    def _on_layer_labels_width_changed(self, width: int):
        """Sync corner widget width with layer labels width + splitter handle."""
        if hasattr(self, '_corner') and hasattr(self, '_content_splitter'):
            splitter_handle_width = self._content_splitter.handleWidth()
            self._corner.setFixedWidth(width + splitter_handle_width)
    
    def _on_content_splitter_moved(self, pos: int, index: int):
        """Handle splitter being dragged - sync corner width, debounce save."""
        if hasattr(self, '_layer_labels') and hasattr(self, '_corner') and hasattr(self, '_content_splitter'):
            width = self._layer_labels.width()
            splitter_handle_width = self._content_splitter.handleWidth()
            self._corner.setFixedWidth(width + splitter_handle_width)
            # Debounce: store pending value and restart timer
            self._pending_layer_column_width = width
            self._layer_column_save_timer.start()
    
    def _save_layer_column_width(self):
        """Save layer column width after debounce delay (on drag end)."""
        if self._pending_layer_column_width is not None:
            self._settings_manager.layer_column_width = self._pending_layer_column_width
            self._pending_layer_column_width = None
    
    def _on_main_splitter_moved(self, pos: int, index: int):
        """Handle main splitter moved - debounce save inspector width."""
        if hasattr(self, '_inspector_container') and self._inspector_container.isVisible():
            sizes = self._main_splitter.sizes()
            if len(sizes) >= 2:
                # Debounce: store pending value and restart timer
                self._pending_inspector_width = sizes[1]
                self._inspector_save_timer.start()
    
    def _save_inspector_width(self):
        """Save inspector width after debounce delay (on drag end)."""
        if self._pending_inspector_width is not None:
            self._settings_manager.inspector_width = self._pending_inspector_width
            self._pending_inspector_width = None
    
    def _create_inspector_panel(self) -> QWidget:
        container = QWidget()
        container.setMinimumWidth(200)
        container.setMaximumWidth(400)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        header = QFrame()
        header.setFixedHeight(26)
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_MEDIUM.name()};
                border-left: 1px solid {Colors.BORDER.name()};
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        title = QLabel("Event Inspector")
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        layout.addWidget(header)
        
        self._event_inspector = EventInspector()
        self._event_inspector.set_grid_system(self._grid_system)
        layout.addWidget(self._event_inspector, 1)
        
        self._settings_panel = SettingsPanel(self._grid_system, timeline_widget=self)
        self._settings_panel.settings_changed.connect(self._on_settings_changed)
        self._settings_panel.follow_mode_changed.connect(self._on_follow_mode_changed)
        
        return container
    
    def _on_settings_changed(self):
        """Handle settings panel changes - update timeline display."""
        self._ruler.update()
        # Update all event items to reflect label/duration/highlight changes
        if hasattr(self._scene, '_event_items'):
            for item in self._scene._event_items.values():
                item.update()
        self._scene.update()
    
    def _on_follow_mode_changed(self, mode: PlayheadFollowMode):
        Log.debug(f"TimelineWidget: Follow mode changed to {mode}")
    
    def _show_settings_dialog(self):
        """
        Show the timeline settings dialog.
        
        Creates a modal dialog containing the settings panel, properly sized
        and centered relative to the parent window.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Timeline Settings")
        dialog.setModal(False)
        
        # Fixed width to prevent stretching - SettingsPanel has fixed width of 320
        dialog.setFixedWidth(320)
        dialog.setMinimumHeight(450)
        
        # Size policy prevents horizontal stretching
        size_policy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        dialog.setSizePolicy(size_policy)
        
        # Apply styling
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BG_DARK.name()};
            }}
        """)
        
        # Main layout
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)
        
        # Create or reuse settings panel
        if not self._settings_panel:
            self._settings_panel = SettingsPanel(self._grid_system, timeline_widget=self)
            self._settings_panel.settings_changed.connect(self._on_settings_changed)
            self._settings_panel.follow_mode_changed.connect(self._on_follow_mode_changed)
        
        # Temporarily reparent settings panel to dialog
        original_parent = self._settings_panel.parent()
        self._settings_panel.setParent(None)
        layout.addWidget(self._settings_panel)
        
        # Store references for cleanup
        dialog._original_parent = original_parent
        dialog._settings_panel = self._settings_panel
        
        # Restore panel to original parent when dialog closes
        def on_dialog_finished():
            if hasattr(dialog, '_settings_panel') and dialog._settings_panel:
                panel = dialog._settings_panel
                layout.removeWidget(panel)
                if hasattr(dialog, '_original_parent'):
                    panel.setParent(dialog._original_parent)
        
        dialog.finished.connect(on_dialog_finished)
        
        # Close button at bottom
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(8, 8, 8, 8)
        btn_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(dialog.close)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.BG_LIGHT.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 6px 16px;
            }}
            QPushButton:hover {{
                background-color: {Colors.HOVER.name()};
            }}
        """)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        # Center dialog relative to parent window
        parent_window = self.window()
        if parent_window:
            parent_rect = parent_window.geometry()
            dialog_rect = dialog.geometry()
            x = parent_rect.x() + (parent_rect.width() - dialog_rect.width()) // 2
            y = parent_rect.y() + (parent_rect.height() - dialog_rect.height()) // 2
            dialog.move(x, y)
        
        dialog.exec()
    
    def _create_toolbar(self) -> QWidget:
        """Create bottom toolbar (play/pause, clock, settings) - fixed at bottom."""
        toolbar = QFrame()
        toolbar.setFixedHeight(40)
        toolbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        toolbar.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_MEDIUM.name()};
                border-top: 1px solid {Colors.BORDER.name()};
            }}
        """)
        
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        btn_style = f"""
            QPushButton {{
                background-color: {Colors.BG_LIGHT.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(3)};
                padding: 4px 8px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {Colors.HOVER.name()};
            }}
            QPushButton:pressed {{
                background-color: {Colors.BG_DARK.name()};
            }}
            QPushButton:checked {{
                background-color: {Colors.ACCENT_BLUE.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
        """
        
        transport_frame = QFrame()
        transport_layout = QHBoxLayout(transport_frame)
        transport_layout.setContentsMargins(0, 0, 0, 0)
        transport_layout.setSpacing(4)
        
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedWidth(50)
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setStyleSheet(btn_style)
        transport_layout.addWidget(self._stop_btn)
        
        self._play_btn = QPushButton("Play")
        self._play_btn.setFixedWidth(60)
        self._play_btn.setCheckable(True)
        self._play_btn.clicked.connect(self._on_play_toggle)
        self._play_btn.setStyleSheet(btn_style)
        transport_layout.addWidget(self._play_btn)
        
        layout.addWidget(transport_frame)
        
        self._position_label = QLabel("00:00.00")
        self._position_label.setFixedWidth(80)
        self._position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._position_label.setStyleSheet(f"""
            QLabel {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.ACCENT_GREEN.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(3)};
                padding: 4px;
                font-family: monospace;
                font-size: 12px;
            }}
        """)
        layout.addWidget(self._position_label)
        
        # Container for custom toolbar widgets (e.g., event source selector)
        self._toolbar_custom_container = QWidget()
        self._toolbar_custom_layout = QHBoxLayout(self._toolbar_custom_container)
        self._toolbar_custom_layout.setContentsMargins(0, 0, 0, 0)
        self._toolbar_custom_layout.setSpacing(8)
        self._toolbar_custom_widget = None
        layout.addWidget(self._toolbar_custom_container)
        
        layout.addStretch()
        
        self._settings_btn = QPushButton("\u2699")
        self._settings_btn.setFixedWidth(32)
        self._settings_btn.setToolTip("Timeline Settings")
        self._settings_btn.clicked.connect(self._show_settings_dialog)
        self._settings_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.BG_LIGHT.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(3)};
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: {Colors.HOVER.name()};
            }}
            QPushButton:pressed {{
                background-color: {Colors.ACCENT_BLUE.name()};
            }}
        """)
        layout.addWidget(self._settings_btn)
        
        return toolbar
    
    def _connect_signals(self):
        """Connect internal signals."""
        # View signals
        self._view.scroll_changed.connect(self._on_horizontal_scroll_changed)
        self._view.seek_requested.connect(self._on_seek)
        self._view.space_pressed.connect(self._playback_controller.toggle_playback)
        self._view.select_all_requested.connect(self._scene.select_all)
        self._view.deselect_all_requested.connect(self._scene.deselect_all)
        
        # Scene signals - new typed signals
        self._scene.selection_changed.connect(self._on_selection_changed)
        self._scene.events_moved.connect(self._on_events_moved)
        self._scene.events_resized.connect(self._on_events_resized)
        self._scene.event_deleted.connect(self._on_event_delete_requested)
        self._scene.events_deleted.connect(self._on_events_delete_requested_batch)
        self._scene.event_created.connect(self._on_event_created)
        self._scene.event_sliced.connect(self._on_event_sliced)
        self._scene.playhead_seeked.connect(self._on_seek)
        
        # Movement controller status
        self._movement_controller.status_message.connect(self._on_status_message)
        
        # Ruler click
        self._ruler.clicked.connect(self._on_seek)
        
        # Playback controller
        self._playback_controller.position_changed.connect(self._on_position_changed)
        self._playback_controller.playback_started.connect(lambda: self._on_playback_state_changed(True))
        self._playback_controller.playback_paused.connect(lambda: self._on_playback_state_changed(False))
        self._playback_controller.playback_stopped.connect(lambda: self._on_playback_state_changed(False))
    
    # =========================================================================
    # Signal Handlers
    # =========================================================================
    
    def _on_events_moved(self, results: List[EventMoveResult]):
        """Forward move results."""
        self.events_moved.emit(results)
        
        # Legacy signal for backwards compatibility
        for result in results:
            layer = self._layer_manager.get_layer(result.new_layer_id)
            layer_index = layer.index if layer else 0
            # Don't emit legacy signal - it causes issues
    
    def _on_events_resized(self, results: List[EventResizeResult]):
        """Forward resize results."""
        self.events_resized.emit(results)
    
    def _on_event_created(self, result: EventCreateResult):
        """Forward event creation."""
        self.event_created.emit(result)
    
    def _on_event_sliced(self, result: EventSliceResult):
        """Forward event slice result."""
        self.event_sliced.emit(result)
    
    def _on_event_delete_requested(self, event_id: str):
        """Handle single event deletion request (wraps in batch for unified pathway)."""
        Log.debug(f"[DELETE DEBUG] TimelineWidget._on_event_delete_requested: Single event deletion for {event_id}, wrapping in batch")
        # Use scene batch pathway so payload type matches EventDeleteResult.
        if hasattr(self._scene, 'request_events_delete_batch'):
            self._scene.request_events_delete_batch([event_id])
    
    def _on_events_delete_requested_batch(self, delete_results: List[EventDeleteResult]):
        """Handle batch event deletion request (optimized for large selections).
        
        This method:
        1. Receives EventDeleteResult objects with event data already gathered
        2. Removes all events with updates suspended
        3. Emits a single batch signal with all results
        
        This is significantly faster than N individual deletions.
        
        IMPORTANT: Only emits events_deleted (batch), NOT individual event_deleted signals,
        to prevent double-deletion which causes index corruption in the data layer.
        
        Args:
            delete_results: List[EventDeleteResult] - events to delete with their data
        """
        if not delete_results:
            Log.warning("[DELETE DEBUG] TimelineWidget._on_events_delete_requested_batch: Empty delete_results list")
            return
        
        Log.debug(f"[DELETE DEBUG] TimelineWidget._on_events_delete_requested_batch: Received {len(delete_results)} delete results")
        
        # Extract event IDs from EventDeleteResult objects
        event_ids = [result.event_id for result in delete_results]
        
        # Remove duplicates to prevent removing the same event twice
        # Create a mapping from event_id to EventDeleteResult (preserves first occurrence)
        unique_results = []
        seen_ids = set()
        for result in delete_results:
            if result.event_id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.event_id)
        
        unique_event_ids = [r.event_id for r in unique_results]
        
        if len(unique_event_ids) != len(event_ids):
            Log.warning(f"[DELETE DEBUG] TimelineWidget._on_events_delete_requested_batch: Removed {len(event_ids) - len(unique_event_ids)} duplicates. Unique IDs: {unique_event_ids}")
        
        Log.debug(f"[DELETE DEBUG] TimelineWidget._on_events_delete_requested_batch: Processing {len(unique_results)} unique events")
        
        # Remove events from scene
        # Use batch removal if available, otherwise remove individually
        actually_removed = []
        if hasattr(self._scene, 'remove_events_batch'):
            actually_removed = self._scene.remove_events_batch(unique_event_ids)
        else:
            # Fallback: remove individually
            for event_id in unique_event_ids:
                if self._scene.remove_event(event_id):
                    actually_removed.append(event_id)
        
        Log.debug(f"[DELETE DEBUG] TimelineWidget._on_events_delete_requested_batch: Scene removed {len(actually_removed)} events: {actually_removed}")
        
        # Filter results to only include events that were actually removed
        removed_set = set(actually_removed)
        filtered_results = [r for r in unique_results if r.event_id in removed_set]
        not_removed = [r.event_id for r in unique_results if r.event_id not in removed_set]
        
        if not_removed:
            Log.warning(f"[DELETE DEBUG] TimelineWidget._on_events_delete_requested_batch: {len(not_removed)} events weren't removed from scene: {not_removed}")
            
        # Emit batch signal with results
        if filtered_results:
            Log.debug(f"[DELETE DEBUG] TimelineWidget._on_events_delete_requested_batch: Emitting events_deleted signal for {len(filtered_results)} events: {[r.event_id for r in filtered_results]}")
            self.events_deleted.emit(filtered_results)
        else:
            Log.warning("[DELETE DEBUG] TimelineWidget._on_events_delete_requested_batch: No results to emit (no events were removed)")
    
    def _on_selection_changed(self, selected_ids: List[str]):
        """Handle selection change."""
        self.selection_changed.emit(selected_ids)
        
        # Legacy signal
        if selected_ids:
            self.event_selected.emit(selected_ids[0])
        
        # Update inspector
        selected_data = []
        valid_selected_ids = []
        for eid in selected_ids:
            
            # Validate event exists in scene before processing
            if hasattr(self._scene, "_event_items") and eid not in self._scene._event_items:
                from ..logging import TimelineLog as Log
                Log.warning(f"TimelineWidget: Selected event {eid} not found in scene - skipping (stale selection)")
                continue
            
            item = self._scene.get_event_item(eid)
            if item:
                valid_selected_ids.append(eid)
                # Get actual layer name from layer_id
                layer = self._layer_manager.get_layer(item.layer_id)
                layer_name = layer.name if layer else item.classification
                layer_index = self._layer_manager.get_layer_index(item.layer_id)
                
                # Build metadata dict for EventInspector (includes audio fields for waveform)
                metadata_for_inspector = dict(item.user_data) if item.user_data else {}
                if item.audio_id:
                    metadata_for_inspector['audio_id'] = item.audio_id
                if item.audio_name:
                    metadata_for_inspector['audio_name'] = item.audio_name
                # Add clip times for inspector waveform (derived from timeline position)
                metadata_for_inspector['clip_start_time'] = item.start_time
                metadata_for_inspector['clip_end_time'] = item.start_time + item.duration
                
                selected_data.append({
                    'event_id': item.event_id,
                    'start_time': item.start_time,
                    'duration': item.duration,
                    'end_time': item.start_time + item.duration,
                    'classification': item.classification,
                    'layer_id': item.layer_id,
                    'layer_name': layer_name,  # Actual layer name (for display)
                    'layer_index': layer_index,
                    'metadata': metadata_for_inspector
                })
        
        # Clear selection if all events were invalid (stale selection)
        if selected_ids and not valid_selected_ids:
            from ..logging import TimelineLog as Log
            Log.warning(f"TimelineWidget: All {len(selected_ids)} selected events were invalid - clearing selection")
            if hasattr(self._scene, 'deselect_all'):
                self._scene.deselect_all()
            elif hasattr(self._scene, 'clearSelection'):  # Fallback to QGraphicsScene method
                self._scene.clearSelection()
        
        self._event_inspector.update_selection(selected_data)
    
    def _on_status_message(self, message: str, is_error: bool):
        """Handle status messages (for future status bar)."""
        level = "ERROR" if is_error else "INFO"
        Log.debug(f"TimelineWidget [{level}]: {message}")
    
    # =========================================================================
    # Transport Controls
    # =========================================================================
    
    def _on_play_toggle(self, checked: bool):
        if checked:
            # Sync playhead position before playing - use scene playhead as source of truth
            scene_playhead_pos = self._scene.playhead.position_seconds if hasattr(self._scene, 'playhead') else self._playback_controller.position
            if abs(scene_playhead_pos - self._playback_controller.position) > 0.01:
                # Playhead was moved by user, sync playback controller
                self._playback_controller.seek(scene_playhead_pos)
            self._play_btn.setText("Pause")
            self._playback_controller.play()
        else:
            self._play_btn.setText("Play")
            self._playback_controller.pause()
    
    def _on_stop(self):
        self._play_btn.setChecked(False)
        self._play_btn.setText("Play")
        self._playback_controller.stop()
    
    def _on_seek(self, time: float):
        if self._grid_system.snap_enabled:
            # Try new SnapCalculator first
            if hasattr(self._scene, '_snap_calculator') and hasattr(self._scene, '_unit_preference'):
                time = self._scene._snap_calculator.snap_time(
                    time,
                    self._pixels_per_second,
                    self._scene._unit_preference
                )
            else:
                # Fallback to legacy
                time = self._grid_system.snap_time(time, self._pixels_per_second)
        
        self._playback_controller.seek(time)
        self._scene.set_playhead_position(time)
        self._ruler.set_playhead_position(time)
        self._update_position_display(time)
    
    def _on_position_changed(self, seconds: float):
        self._scene.set_playhead_position(seconds)
        self._ruler.set_playhead_position(seconds)
        self._update_position_display(seconds)
        self.position_changed.emit(seconds)
        self._apply_playhead_follow(seconds)
    
    def _smooth_scroll_to_time(self, time: float):
        """
        Set target scroll position for continuous smooth following.
        The scroll position will smoothly interpolate towards this target each frame.
        
        Args:
            time: Time in seconds to scroll to
        """
        target_x = time * self._view.pixels_per_second
        h_bar = self._view.horizontalScrollBar()
        
        # Clamp to valid range
        target_x = max(0.0, min(target_x, float(h_bar.maximum())))
        self._target_scroll_x = target_x
    
    def _update_continuous_scroll(self):
        """
        Update scroll position using continuous interpolation towards target.
        Called every frame during playback for smooth following.
        """
        if self._target_scroll_x is None:
            return
        
        h_bar = self._view.horizontalScrollBar()
        current_x = float(h_bar.value())
        target_x = self._target_scroll_x
        
        # Check if we're close enough (within 1 pixel)
        if abs(target_x - current_x) < 1.0:
            h_bar.setValue(int(target_x))
            return
        
        # Linear interpolation for smooth continuous motion
        new_x = current_x + (target_x - current_x) * self._scroll_lerp_factor
        h_bar.setValue(int(new_x))
    
    def _apply_playhead_follow(self, seconds: float):
        settings = self._settings_panel.settings
        
        if settings.follow_during_playback_only and not self._playback_controller.is_playing:
            # Clear target when not following
            self._target_scroll_x = None
            return
        
        mode = settings.follow_mode
        
        if mode == PlayheadFollowMode.OFF:
            # Clear target when off
            self._target_scroll_x = None
            return
        
        start_time, end_time = self._view.get_visible_time_range()
        visible_duration = end_time - start_time
        
        if mode == PlayheadFollowMode.PAGE:
            # PAGE mode uses instant jumps (original behavior)
            margin = visible_duration * 0.1
            if seconds > end_time - margin:
                new_start = seconds - (visible_duration * 0.1)
                self._view.scroll_to_time(max(0, new_start))
                self._target_scroll_x = None
            elif seconds < start_time + margin:
                new_start = seconds - (visible_duration * 0.9)
                self._view.scroll_to_time(max(0, new_start))
                self._target_scroll_x = None
        
        elif mode == PlayheadFollowMode.SMOOTH:
            # SMOOTH mode: continuously follow at 75% position
            # Always calculate target and let continuous scroll handle smooth following
            target_position = 0.75
            new_start = seconds - (visible_duration * target_position)
            self._smooth_scroll_to_time(max(0, new_start))
            # Apply continuous scroll update every frame
            self._update_continuous_scroll()
        
        elif mode == PlayheadFollowMode.CENTER:
            # CENTER mode: continuously keep playhead centered
            # Always calculate target and let continuous scroll handle smooth following
            new_start = seconds - (visible_duration / 2)
            self._smooth_scroll_to_time(max(0, new_start))
            # Apply continuous scroll update every frame
            self._update_continuous_scroll()
    
    def _on_playback_state_changed(self, is_playing: bool):
        """Handle playback state change"""
        self._play_btn.setChecked(is_playing)
        self._play_btn.setText("Pause" if is_playing else "Play")
        self.playback_state_changed.emit(is_playing)
    
    def _update_position_display(self, seconds: float):
        self._position_label.setText(self._grid_system.format_time(seconds))
    
    def _on_horizontal_scroll_changed(self, offset: float):
        """Handle horizontal scroll position change - update ruler and save position.
        
        Args:
            offset: Horizontal scroll offset in pixels
        """
        self._ruler.set_scroll_offset(offset)
        # Also save scroll position (debounced)
        self._scroll_save_timer.start()
    
    # =========================================================================
    # Layer Config Preservation (Single Source of Truth for Layer Heights)
    # =========================================================================
    
    def _preserve_layer_configs(self) -> Dict[tuple, Dict[str, Any]]:
        """
        Preserve current layer configurations before clearing.
        
        Returns:
            Dict mapping (group_key, layer_name) -> config dict (height, color, visible, locked)
        """
        preserved = {}
        for layer in self._layer_manager.get_all_layers():
            group_key = layer.group_id or layer.group_name
            key = (group_key, layer.name)
            preserved[key] = {
                'height': layer.height,
                'color': layer.color,
                'visible': layer.visible,
                'locked': layer.locked,
            }
        return preserved
    
    def _restore_layer_configs(self, preserved: Dict[tuple, Dict[str, Any]]) -> None:
        """
        Restore layer configurations after layers are recreated.
        
        Matches layers by (group_key, name) and applies preserved settings.
        
        Args:
            preserved: Dict from _preserve_layer_configs()
        """
        if not preserved:
            return
        
        for layer in self._layer_manager.get_all_layers():
            group_key = layer.group_id or layer.group_name
            key = (group_key, layer.name)
            config = preserved.get(key)
            if not config:
                continue
            self._layer_manager.update_layer(
                layer.id,
                height=config.get('height'),
                color=config.get('color'),
                visible=config.get('visible'),
                locked=config.get('locked'),
            )
    
    # =========================================================================
    # Public API - Data Input
    # =========================================================================
    
    def set_events(self, events: List[TimelineEvent], editable: bool = True) -> None:
        """
        Load events into the timeline.
        
        Clears existing events. Layers MUST exist before calling this method.
        Events are mapped to existing layers by _visual_layer_name or classification.
        Throws ValueError if layers don't exist.
        
        Args:
            events: List of TimelineEvent objects
            editable: Whether events should be editable
        """
        self._view.setUpdatesEnabled(False)
        
        preserved_scroll = None
        if self._view:
            v_bar = self._view.verticalScrollBar()
            h_bar = self._view.horizontalScrollBar()
            preserved_scroll = {
                'vertical': v_bar.value(),
                'horizontal': h_bar.value(),
            }
        
        # Preserve layer configurations before clearing (keyed by name)
        preserved_layers = self._preserve_layer_configs()
        
        # Clear all events before reloading
        # NOTE: Previously had "synced layer protection" that preserved synced layer events,
        # but this caused MA3 sync issues - old event positions would persist because
        # clear_events_except_layers() skipped synced layers. Since EventDataItems
        # (including MA3 sync items) contain the authoritative data, a full clear + reload
        # is the correct approach. The MA3 sync EventDataItem has the updated event positions.
        self._scene.clear_events()
        # CRITICAL: Never clear layer manager - layers must exist before adding events
        # Layers are created explicitly via EditorCreateLayerCommand -> _restore_layer_state()
        # Clearing would destroy layers that were just restored
        self._scene.editable = editable
        
        if not events:
            from src.utils.message import Log
            Log.debug("TimelineWidget: No events to display")
            self._view.setUpdatesEnabled(True)
            return
        
        # Build mapping from existing layers - layers MUST exist before adding events
        # Layers are created explicitly via EditorCreateLayerCommand -> _restore_layer_state()
        existing_layers = self._layer_manager.get_all_layers()
        from src.utils.message import Log
        Log.info(f"[LAYER_CREATE] TimelineWidget.set_events() - found {len(existing_layers)} existing layers")
        Log.info(f"[LAYER_CREATE]   Existing layer names: {[l.name for l in existing_layers]}")
        Log.info(f"[LAYER_CREATE]   LayerManager ID: {id(self._layer_manager)}, _layers dict size: {len(self._layer_manager._layers)}, _order size: {len(self._layer_manager._order)}")
        
        layer_mapping = {}
        for layer in existing_layers:
            normalized_name = self._normalize_layer_name(layer.name)
            layer_mapping[normalized_name] = layer.id
            # Also map by exact name (no normalization) for direct matches - ALWAYS add exact name
            layer_mapping[layer.name] = layer.id
        
        # Log warning for events that don't match existing layers
        unmatched_classifications = set()
        for event in events:
            if event.layer_id is None:
                classification = event.user_data.get('_visual_layer_name') if event.user_data else event.classification
                if classification:
                    normalized_classification = self._normalize_layer_name(classification)
                    if normalized_classification not in layer_mapping and classification not in layer_mapping:
                        unmatched_classifications.add(classification)
        
        if unmatched_classifications:
            Log.warning(
                f"[LAYER_CREATE] TimelineWidget: Events have classifications for non-existent layers: {unmatched_classifications}. "
                f"These events will cause errors (NO NEW LAYERS CREATED). "
                f"Existing layers: {[l.name for l in existing_layers]}"
            )
        
        # Restore preserved layer configurations (heights, colors, etc.)
        self._restore_layer_configs(preserved_layers)
        
        # Add events and track source-to-layer mapping for grouping
        max_time = 0
        source_to_layers: Dict[str, Dict[str, Any]] = {}  # source_id -> {name, layer_ids}
        
        for event in events:
            # Determine layer_id - events should already have layer_id set from EventLayers
            # Only support layer_id (no classification-based matching - layers created from EventLayers)
            layer_id = event.layer_id
            
            # If layer_id is None, try to find by _source_layer_name (from EventLayer)
            if layer_id is None:
                source_layer_name = event.user_data.get('_source_layer_name') if event.user_data else None
                if source_layer_name:
                    # Find layer by name (should exist since it was created from EventLayer)
                    layer = self._layer_manager.get_layer_by_name(source_layer_name)
                    if layer:
                        layer_id = layer.id
                    else:
                        # Try normalized name
                        normalized_name = self._normalize_layer_name(source_layer_name)
                        layer = self._layer_manager.get_layer_by_name(normalized_name)
                        if layer:
                            layer_id = layer.id
            
            # If still no layer_id, THROW ERROR - layers must exist (created from EventLayers)
            if layer_id is None:
                from src.utils.message import Log
                import traceback
                stack = traceback.extract_stack()
                caller_info = f"{stack[-2].filename}:{stack[-2].lineno}" if len(stack) >= 2 else "unknown"
                existing_layers = [l.name for l in self._layer_manager.get_all_layers()]
                source_layer_name = event.user_data.get('_source_layer_name') if event.user_data else None
                error_msg = (
                    f"TimelineWidget.set_events(): Event has no layer_id and source_layer_name '{source_layer_name}' "
                    f"does not match any existing layer. Event classification: '{event.classification}'. "
                    f"Existing layers: {existing_layers}. Layers must be created from EventLayers before adding events. "
                    f"Caller: {caller_info}"
                )
                Log.error(f"[LAYER_CREATE] {error_msg}")
                raise ValueError(error_msg)
            
            # Track source-to-layer mapping for grouping
            source_id = event.user_data.get('_source_item_id') if event.user_data else None
            source_name = event.user_data.get('_source_item_name') if event.user_data else None
            if source_id and source_name and layer_id:
                if source_id not in source_to_layers:
                    source_to_layers[source_id] = {
                        'name': source_name,
                        'layer_ids': set()
                    }
                source_to_layers[source_id]['layer_ids'].add(layer_id)
            
            self._scene.add_event(
                event_id=event.id,
                start_time=event.time,
                duration=event.duration,
                classification=event.classification,
                layer_id=layer_id,
                audio_id=event.audio_id,
                audio_name=event.audio_name,
                user_data=event.user_data,
                editable=editable
            )
            
            end_time = event.time + event.duration
            if end_time > max_time:
                max_time = end_time
        
        # Assign grouping metadata to layers based on source EventDataItem
        # Group layers that share the same source_id
        for source_id, source_info in source_to_layers.items():
            # Sort layer IDs by their visual index (maintain display order)
            layer_ids_in_group = sorted(
                source_info['layer_ids'],
                key=lambda lid: self._layer_manager.get_layer(lid).index if self._layer_manager.get_layer(lid) else 999
            )
            
            # Edge case: if no layers, skip (can't create group without layers)
            if not layer_ids_in_group:
                continue
            
            # Assign grouping info to all layers in this group
            # Edge case: if only one layer, treat it as the single child (group_index=0)
            for group_index, layer_id in enumerate(layer_ids_in_group):
                layer = self._layer_manager.get_layer(layer_id)
                if not layer:
                    continue
                group_id = layer.group_id or source_id
                group_name = layer.group_name or source_info['name']
                self._layer_manager.update_layer(
                    layer_id,
                    group_id=group_id,
                    group_name=group_name,
                    group_index=group_index
                )
        
        timeline_duration = max_time + 1.0
        self._scene.set_duration(timeline_duration)
        self._ruler.set_duration(timeline_duration)
        self._playback_controller.set_duration(timeline_duration)
        
        if preserved_scroll and self._view:
            v_bar = self._view.verticalScrollBar()
            h_bar = self._view.horizontalScrollBar()
            h_bar.setValue(preserved_scroll['horizontal'])
            max_v_scroll = v_bar.maximum()
            if max_v_scroll > 0:
                v_bar.setValue(min(preserved_scroll['vertical'], max_v_scroll))
        
        # Don't clear selection after loading events - this interferes with user selections.
        # Selection validation happens in _on_selection_changed() which filters out invalid events.
        
        self._view.setUpdatesEnabled(True)
        
        if self._settings_panel:
            QTimer.singleShot(50, self._settings_panel._update_layer_controls)
        
        self._sync_layer_labels_height()
        QTimer.singleShot(10, self._sync_layer_labels_height)
        
        Log.info(f"TimelineWidget: Loaded {len(events)} events across {self._layer_manager.get_layer_count()} layers")
        
        # Pre-warm caches (triggers paint events for visible items)
        self._pre_warm_viewport_cache()
        
        # Schedule staged waveform loading (async, batched to avoid memory explosion)
        self._schedule_staged_waveform_loading()
    
    def set_events_from_dicts(self, events: List[Dict[str, Any]], editable: bool = True) -> None:
        """
        Set events from list of dictionaries (legacy format).
        
        Args:
            events: List of event dicts with 'id', 'time', 'duration', 'classification', 'metadata'
            editable: Whether events should be editable
        """
        timeline_events = []
        for e in events:
            timeline_events.append(TimelineEvent(
                id=e.get('id', str(len(timeline_events))),
                time=float(e.get('time', 0)),
                duration=float(e.get('duration', 0)),
                classification=e.get('classification', 'Event'),
                layer_id=e.get('layer_id'),
                metadata=e.get('metadata', {})
            ))
        
        self.set_events(timeline_events, editable=editable)
    
    @staticmethod
    def _normalize_layer_name(layer_name: str) -> str:
        """
        Normalize layer name to canonical form, removing block-specific prefixes.
        
        When data is pulled from another block, layer names may contain source block
        prefixes (e.g., "Editor1_DetectOnsets1_Separator1_vocals_events_edited: onset").
        This function extracts the semantic layer name (e.g., "Vocals") to make
        layer names block-agnostic and reflect current block ownership.
        
        CRITICAL: Preserves specific layer names like "101_<Kick>" that are already
        meaningful layer identifiers (not block prefixes).
        
        Args:
            layer_name: Full layer name potentially with block prefixes
            
        Returns:
            Normalized canonical layer name (e.g., "Vocals", "Drums", "Bass", or "101_<Kick>")
        """
        if not layer_name:
            return layer_name
        
        # CRITICAL: If layer name looks like a specific identifier (e.g., "101_<Kick>", "TC_101_<Snare>"),
        # preserve it as-is. Don't extract keywords from it.
        # Pattern: Starts with number, contains underscores and angle brackets or similar
        import re
        if re.match(r'^\d+_<.+>$', layer_name) or re.match(r'^TC_\d+_<.+>$', layer_name):
            # This is a specific layer identifier - preserve it
            return layer_name
        
        # If already a simple canonical name (no underscores, no colons, short), return as-is
        if '_' not in layer_name and ':' not in layer_name and len(layer_name) < 20:
            # Already normalized (e.g., "Vocals", "Drums")
            return layer_name
        
        layer_lower = layer_name.lower()
        
        # Map common patterns to canonical names
        # Look for instrument/type keywords in the name
        keyword_map = {
            'vocals': 'Vocals',
            'vocal': 'Vocals',
            'drums': 'Drums',
            'drum': 'Drums',
            'bass': 'Bass',
            'other': 'Other',
            'kick': 'Kick',
            'snare': 'Snare',
            'hihat': 'HiHat',
            'hi-hat': 'HiHat',
            'hi hat': 'HiHat',
            'cymbal': 'Cymbal',
            'tom': 'Tom',
            'percussion': 'Percussion',
        }
        
        # Check for keywords (prioritize longer matches first)
        # BUT: Only extract if the name looks like it has block prefixes (multiple underscores, colons)
        # Don't extract from simple names like "101_<Kick>"
        has_block_prefixes = ':' in layer_name or layer_name.count('_') > 2
        if has_block_prefixes:
            for keyword, canonical in sorted(keyword_map.items(), key=lambda x: -len(x[0])):
                if keyword in layer_lower:
                    return canonical
        
        # If no keyword found, try to extract meaningful segment
        # Split by common delimiters
        parts = layer_name.replace(':', '_').split('_')
        
        # Only extract if we have multiple parts suggesting block prefixes
        if len(parts) > 3:
            # Look for segments that look like instrument/type names
            for segment in reversed(parts):
                segment = segment.strip()
                if not segment:
                    continue
                
                segment_lower = segment.lower()
                
                # Check if segment matches a keyword
                for keyword, canonical in keyword_map.items():
                    if keyword in segment_lower:
                        return canonical
                
                # If segment is short, alphabetic, and looks like a name (not a UUID/number)
                if 2 < len(segment) < 15 and segment.replace('-', '').isalpha():
                    return segment.capitalize()
        
        # Fallback: return as-is if we can't normalize
        # This preserves user-created layer names that don't match patterns
        return layer_name

    def set_events_from_data_items(self, event_data_items, editable: bool = True) -> None:
        """
        Convenience method to load from EchoZero EventDataItem objects.

        Args:
            event_data_items: List of EventDataItem objects
            editable: Whether events should be editable
        """
        try:
            from src.shared.domain.entities import EventDataItem
        except ImportError:
            Log.error("EventDataItem not available. Use set_events() instead.")
            return
        
        if isinstance(event_data_items, EventDataItem):
            items = [event_data_items]
        else:
            items = list(event_data_items) if event_data_items else []
        
        if not items:
            # Use set_events with empty list to properly preserve/clear
            self.set_events([], editable=editable)
            return
        
        # SINGLE SOURCE OF TRUTH: Create TimelineLayers ONLY from EventLayers in EventDataItems
        # KEY FIX: Each EventLayer in each EventDataItem gets its OWN TimelineLayer
        # Key is (item.id, layer.name) to allow same layer name in different groups
        event_layers_by_key = {}  # (item_id, layer_name) -> (EventLayer, source_item)
        all_events = []
        seen_event_ids = set()
        seen_times = set()
        
        for item in items:
            if not isinstance(item, EventDataItem):
                continue
            
            # Check if item has native EventLayers (single source of truth)
            has_internal_layers = (
                hasattr(item, '_layers') and 
                item._layers and 
                len(item._layers) > 0 and
                (not hasattr(item, '_events') or not item._events or len(item._events) == 0)
            )
            if has_internal_layers:
                # Use EventLayers as single source of truth
                for layer in item._layers:
                    layer_name = layer.name
                    layer_events = layer.get_events()
                    
                    # Track this EventLayer with UNIQUE key per (EventDataItem, LayerName)
                    # This ensures each EventLayer gets its own TimelineLayer
                    layer_key = (item.id, layer_name)
                    event_layers_by_key[layer_key] = (layer, item)
                    
                    if not layer_events:
                        continue
                    
                    # Process events in this layer
                    for i, event in enumerate(layer_events):
                        # SINGLE SOURCE OF TRUTH: Always use domain Event's UUID
                        # Domain Events MUST have stable UUIDs - fail hard if missing
                        event_id = getattr(event, 'id', None)
                        if not event_id:
                            from src.utils.message import Log
                            error_msg = (
                                f"TimelineWidget.set_events_from_data_items(): "
                                f"Event at index {i} in layer '{layer_name}' has no ID. "
                                f"All events must have stable UUIDs. Event: time={event.time}, "
                                f"classification={event.classification}, duration={event.duration}"
                            )
                            Log.error(error_msg)
                            raise ValueError(error_msg)
                        
                        # Ensure UUID is a string (should always be, but defensive)
                        event_id = str(event_id)
                        
                        if event_id in seen_event_ids:
                            continue
                        seen_event_ids.add(event_id)
                        
                        time_key = (event.time, event.classification, item.id, layer.id)
                        if time_key in seen_times:
                            continue
                        seen_times.add(time_key)
                        
                        # Extract audio fields from domain event metadata
                        event_audio_id = event.metadata.get('audio_id') if event.metadata else None
                        event_audio_name = event.metadata.get('audio_name') if event.metadata else None
                        
                        # Build user_data (domain metadata minus audio fields)
                        # SINGLE SOURCE OF TRUTH: Start with event.metadata from database
                        # This ensures render_as_marker and all other metadata fields are preserved
                        user_data = dict(event.metadata) if event.metadata else {}
                        
                        # Remove fields that are handled separately (audio_id, audio_name) or are UI-only
                        user_data.pop('audio_id', None)
                        user_data.pop('audio_name', None)
                        user_data.pop('clip_start_time', None)
                        user_data.pop('clip_end_time', None)
                        
                        # Add source tracking fields (UI-only, don't overwrite database metadata)
                        # These are added AFTER copying event.metadata to ensure database fields take precedence
                        user_data['_source_item_id'] = item.id
                        user_data['_source_item_name'] = item.name
                        user_data['_source_layer_id'] = layer.id
                        user_data['_source_layer_name'] = layer.name
                        user_data['_original_classification'] = event.classification
                        # Store the unique layer key for mapping events to correct TimelineLayer
                        user_data['_layer_key'] = f"{item.id}:{layer_name}"
                        
                        # SINGLE SOURCE OF TRUTH: Use TimelineEvent.from_event() for conversion
                        # This handles all normalization (duration, render_as_marker, metadata mapping)
                        # TimelineEvent.from_event() already uses event.id as the TimelineEvent.id
                        timeline_event = TimelineEvent.from_event(
                            event=event,
                            layer_id=None,  # Will be set after layers are created
                            audio_id=event_audio_id,
                            audio_name=event_audio_name,
                            user_data=user_data,
                        )
                        
                        # Verify ID consistency (TimelineEvent.from_event() should use event.id)
                        if timeline_event.id != event_id:
                            from src.utils.message import Log
                            Log.warning(
                                f"TimelineWidget: TimelineEvent.id ({timeline_event.id}) != event.id ({event_id}). "
                                f"TimelineEvent.from_event() should preserve event.id. Overriding to match."
                            )
                            timeline_event.id = event_id
                        
                        
                        all_events.append(timeline_event)
            else:
                # LEGACY: No EventLayers - skip this item (blocks should output layers)
                from src.utils.message import Log
                Log.warning(
                    f"TimelineWidget.set_events_from_data_items(): "
                    f"EventDataItem '{item.name}' has no EventLayers. "
                    f"Blocks should output EventLayers for proper layer handling. Skipping item."
                )
                continue
        
        # CREATE TimelineLayers from EventLayers (single source of truth)
        # Each (EventDataItem, LayerName) combination gets its own TimelineLayer
        # Existing layers keyed by (group_id, name) to match properly
        existing_layers_by_key = {}
        existing_layers_by_name = {}
        for l in self._layer_manager.get_all_layers():
            if l.group_id:
                existing_layers_by_key[(l.group_id, l.name)] = l
            if l.name:
                existing_layers_by_name.setdefault(l.name, []).append(getattr(l, "group_id", None))
        layer_key_to_id = {}  # (item_id, layer_name) -> timeline_layer_id
        current_index = len(self._layer_manager.get_all_layers())
        
        for (item_id, layer_name), (event_layer, source_item) in event_layers_by_key.items():
            layer_key = (item_id, layer_name)
            group_id_key = source_item.id
            if hasattr(source_item, "metadata") and source_item.metadata:
                meta_group_id = source_item.metadata.get("group_id")
                if isinstance(meta_group_id, str) and meta_group_id.startswith("tc_"):
                    group_id_key = meta_group_id
            existing_key = (group_id_key, layer_name)
            
            if existing_key in existing_layers_by_key:
                # Use existing layer - it's already correctly grouped
                existing_layer = existing_layers_by_key[existing_key]
                layer_key_to_id[layer_key] = existing_layer.id
            else:
                # Create new TimelineLayer for this (EventDataItem, LayerName) combination
                group_name = None
                if hasattr(source_item, "metadata") and source_item.metadata:
                    group_name = source_item.metadata.get("group_name")
                    source_tag = source_item.metadata.get("source")
                    if (not group_name) and source_tag in {"ma3", "ma3_sync"}:
                        if layer_name and layer_name[0].isdigit():
                            head = layer_name.split("_", 1)[0]
                            if head.isdigit():
                                group_name = f"TC {head}"
                if not group_name:
                    group_name = source_item.name
                group_id = source_item.id
                if hasattr(source_item, "metadata") and source_item.metadata:
                    meta_group_id = source_item.metadata.get("group_id")
                    if isinstance(meta_group_id, str) and meta_group_id.startswith("tc_"):
                        group_id = meta_group_id

                # Determine sync metadata (only applies to MA3-sourced items)
                source_meta = getattr(source_item, "metadata", {}) or {}
                is_synced = bool(
                    source_meta.get("_synced_from_ma3") is True
                    or source_meta.get("source") in {"ma3", "ma3_sync"}
                )
                ma3_track_coord = (
                    source_meta.get("_ma3_track_coord")
                    or source_meta.get("ma3_track_coord")
                    or source_meta.get("ma3_coord")
                )
                show_manager_block_id = (
                    source_meta.get("_show_manager_block_id")
                    or source_meta.get("show_manager_block_id")
                )
                layer_meta = getattr(event_layer, "metadata", {}) if event_layer else {}
                if not ma3_track_coord:
                    ma3_track_coord = (
                        layer_meta.get("_ma3_track_coord")
                        or layer_meta.get("ma3_track_coord")
                        or layer_meta.get("ma3_coord")
                    )
                if not show_manager_block_id:
                    show_manager_block_id = (
                        layer_meta.get("_show_manager_block_id")
                        or layer_meta.get("show_manager_block_id")
                    )
                if is_synced and (not ma3_track_coord or not show_manager_block_id):
                    # Fallback: try event metadata within the layer
                    for evt in event_layer.get_events():
                        evt_meta = getattr(evt, "metadata", {}) or {}
                        if not ma3_track_coord:
                            ma3_track_coord = (
                                evt_meta.get("_ma3_track_coord")
                                or evt_meta.get("ma3_track_coord")
                                or evt_meta.get("ma3_coord")
                            )
                        if not show_manager_block_id:
                            show_manager_block_id = (
                                evt_meta.get("_show_manager_block_id")
                                or evt_meta.get("show_manager_block_id")
                            )
                        if ma3_track_coord and show_manager_block_id:
                            break
                if not ma3_track_coord or not show_manager_block_id:
                    is_synced = False

                created_layer = self._layer_manager.create_layer(
                    name=layer_name,
                    layer_id=f"layer_{current_index}",
                    index=current_index,
                    height=40.0,  # Default height
                    color=None,  # Auto color
                    group_id=group_id,
                    group_name=group_name,
                    group_index=None,
                    is_synced=is_synced,
                    show_manager_block_id=show_manager_block_id,
                    ma3_track_coord=ma3_track_coord
                )
                
                # Set grouping metadata - group by source EventDataItem
                if created_layer:
                    layer_key_to_id[layer_key] = created_layer.id
                    current_index += 1

        # Set layer_id on events based on (item_id, layer_name) key
        for event in all_events:
            layer_key_str = event.user_data.get('_layer_key')
            if layer_key_str:
                # Parse "item_id:layer_name" back to tuple
                parts = layer_key_str.split(':', 1)
                if len(parts) == 2:
                    layer_key = (parts[0], parts[1])
                    if layer_key in layer_key_to_id:
                        event.layer_id = layer_key_to_id[layer_key]
        
        # Set events in timeline
        self.set_events(all_events, editable=editable)
        
        # Don't clear selection after loading events - this interferes with user selections.
        # Selection validation happens in _on_selection_changed() which filters out invalid events.
    
    def add_event(self, event: TimelineEvent) -> str:
        """Add a single event. Returns event ID. THROWS ERROR if layer doesn't exist."""
        if event.layer_id is None:
            # Try to find layer by classification, but THROW ERROR if not found
            normalized_name = self._normalize_layer_name(event.classification)
            layer = self._layer_manager.get_layer_by_name(normalized_name)
            if not layer:
                from src.utils.message import Log
                import traceback
                stack = traceback.extract_stack()
                caller_info = f"{stack[-2].filename}:{stack[-2].lineno}" if len(stack) >= 2 else "unknown"
                existing_layers = [l.name for l in self._layer_manager.get_all_layers()]
                error_msg = f"TimelineWidget.add_event(): Layer for classification '{event.classification}' (normalized: '{normalized_name}') does not exist. Existing layers: {existing_layers}. Layer must be created explicitly before adding events. Caller: {caller_info}"
                Log.error(f"[LAYER_CREATE] {error_msg}")
                raise ValueError(error_msg)
            layer_id = layer.id
        else:
            layer_id = event.layer_id
        
        self._scene.add_event(
            event_id=event.id,
            start_time=event.time,
            duration=event.duration,
            classification=event.classification,
            layer_id=layer_id,
            audio_id=event.audio_id,
            audio_name=event.audio_name,
            user_data=event.user_data
        )
        
        # Update timeline duration to match the last event's end time
        # Scene already updates its internal duration, but we need to sync ruler and playback controller
        event_end_time = event.time + event.duration
        current_duration = self._scene._duration
        if event_end_time > current_duration - 1.0:  # Add 1s padding if event extends close to end
            new_duration = event_end_time + 1.0
            self._scene.set_duration(new_duration)
            self._ruler.set_duration(new_duration)
            self._playback_controller.set_duration(new_duration)
        
        QTimer.singleShot(0, self._sync_layer_labels_height)
        return event.id
    
    def update_event(self, event_id: str, **kwargs) -> bool:
        """Update an existing event in place."""
        old_duration = self._scene._duration
        result = self._scene.update_event(event_id, **kwargs)
        
        # If duration changed (event extended beyond timeline), sync all components
        if result and self._scene._duration != old_duration:
            new_duration = self._scene._duration
            self._ruler.set_duration(new_duration)
            self._playback_controller.set_duration(new_duration)
        
        return result
    
    def remove_event(self, event_id: str) -> bool:
        """Remove an event."""
        return self._scene.remove_event(event_id)
    
    def get_event(self, event_id: str) -> Optional[TimelineEvent]:
        """Get event data by ID."""
        return self._scene.get_event_data(event_id)
    
    def get_all_events(self) -> List[TimelineEvent]:
        """Get all events."""
        return self._scene.get_all_events_data()
    
    def clear(self) -> None:
        """Remove all events and layers."""
        self._scene.clear_events()
        self._layer_manager.clear()
        QTimer.singleShot(0, self._sync_layer_labels_height)
    
    # =========================================================================
    # Public API - Layers
    # =========================================================================
    
    def set_layers(self, layers: List[TimelineLayer]) -> None:
        """Set explicit layer configuration (call BEFORE set_events)."""
        self._layer_manager.set_layers(layers)
    
    def add_layer(self, layer: TimelineLayer) -> str:
        """Add a new layer. Returns layer ID."""
        from src.utils.message import Log
        import traceback
        stack = traceback.extract_stack()
        caller_info = f"{stack[-2].filename}:{stack[-2].lineno}" if len(stack) >= 2 else "unknown"
        Log.debug(f"[LAYER_CREATE] TimelineWidget.add_layer() called for '{layer.name}' from {caller_info}")
        created = self._layer_manager.create_layer(
            name=layer.name,
            layer_id=layer.id,
            height=layer.height,
            color=layer.color
        )
        return created.id
    
    def remove_layer(self, layer_id: str) -> bool:
        """Remove a layer."""
        return self._layer_manager.delete_layer(layer_id)
    
    def get_layers(self) -> List[TimelineLayer]:
        """Get current layer configuration."""
        return self._layer_manager.get_all_layers()
    
    def get_layer_names(self) -> List[str]:
        """Get layer names (legacy compatibility)."""
        return self._layer_manager.get_layer_names()
    
    def get_layer_height(self, layer_index: int) -> float:
        """
        Get height for a specific layer by index.
        
        Args:
            layer_index: Index of the layer
            
        Returns:
            Height in pixels
        """
        return self._layer_manager.get_layer_height_by_index(layer_index)
    
    def set_layer_height(self, layer_index: int, height: float) -> None:
        """
        Set height for a specific layer by index.
        
        Args:
            layer_index: Index of the layer
            height: Height in pixels (minimum 20, maximum 200)
        """
        layer = self._layer_manager.get_layer_at_index(layer_index)
        if layer:
            self._layer_manager.update_layer(layer.id, height=height)
    
    def set_layer_visible(self, layer_id: str, visible: bool):
        """
        Set layer visibility with TARGETED event updates (undoable).
        
        Uses CommandBus for undo/redo support.
        Only events on the changed layer and events BELOW it will be updated.
        Events above the changed layer are not touched (optimization).
        Forces immediate visual repaint.
        
        Args:
            layer_id: ID of the layer to modify
            visible: True to show, False to hide
        """
        from src.application.commands import SetLayerVisibilityCommand
        
        if not self._facade.command_bus:
            Log.warning("TimelineWidget: Cannot set layer visibility - CommandBus not initialized")
            return
        
        cmd = SetLayerVisibilityCommand(self._layer_manager, layer_id, visible)
        self._facade.command_bus.execute(cmd)
    
    # =========================================================================
    # Public API - Layer Operations (Handler-Based)
    # =========================================================================
    
    def load_layer(
        self,
        layer_config: TimelineLayer,
        events: List[TimelineEvent],
    ) -> str:
        """
        Load a single layer with its events using the appropriate handler.
        
        This is the standard layer loading operation. The handler is automatically
        selected based on the layer type (regular vs sync).
        
        Args:
            layer_config: Layer configuration (TimelineLayer)
            events: Events to load into the layer
            
        Returns:
            Layer ID of the created/updated layer
        """
        from ..handlers import get_handler_for_layer
        
        # Create or get the layer
        existing = self._layer_manager.get_layer_by_name(layer_config.name)
        if existing:
            layer = existing
            Log.debug(f"TimelineWidget.load_layer: Using existing layer '{layer.name}'")
        else:
            layer = self._layer_manager.create_layer(
                name=layer_config.name,
                layer_id=layer_config.id,
                height=layer_config.height,
                color=layer_config.color,
                group_id=layer_config.group_id,
                group_name=layer_config.group_name,
                group_index=layer_config.group_index,
                is_synced=layer_config.is_synced,
                show_manager_block_id=layer_config.show_manager_block_id,
                ma3_track_coord=layer_config.ma3_track_coord,
                derived_from_ma3=getattr(layer_config, 'derived_from_ma3', False),
            )
            Log.debug(f"TimelineWidget.load_layer: Created layer '{layer.name}'")
        
        # Get appropriate handler
        handler = get_handler_for_layer(layer, self._scene, self._layer_manager)
        
        # Notify handler of layer creation
        handler.on_layer_created(layer)
        
        # Load events via handler
        loaded = handler.load_events(layer, events)
        Log.info(f"TimelineWidget.load_layer: Loaded {loaded} events into layer '{layer.name}'")
        
        # Sync layer labels
        QTimer.singleShot(0, self._sync_layer_labels_height)
        
        return layer.id
    
    def reload_layer(self, layer_id: str) -> bool:
        """
        Reload a single layer from its data source.
        
        Clears events in the layer and prepares for fresh data.
        The actual data fetch is handled by the caller (EditorPanel).
        
        Args:
            layer_id: ID of the layer to reload
            
        Returns:
            True if reload preparation succeeded
        """
        from ..handlers import get_handler_for_layer
        
        layer = self._layer_manager.get_layer(layer_id)
        if not layer:
            Log.warning(f"TimelineWidget.reload_layer: Layer '{layer_id}' not found")
            return False
        
        # Get appropriate handler
        handler = get_handler_for_layer(layer, self._scene, self._layer_manager)
        
        # Reload via handler (clears events, prepares for fresh data)
        result = handler.reload(layer)
        
        if result:
            Log.info(f"TimelineWidget.reload_layer: Prepared layer '{layer.name}' for reload")
        else:
            Log.warning(f"TimelineWidget.reload_layer: Failed to reload layer '{layer.name}'")
        
        return result
    
    def clear_layer_events(self, layer_id: str) -> int:
        """
        Clear all events in a specific layer.
        
        Args:
            layer_id: ID of the layer to clear
            
        Returns:
            Number of events cleared
        """
        return self._scene.clear_events_in_layer(layer_id)
    
    def get_synced_layers(self) -> List[TimelineLayer]:
        """
        Get all synced layers.
        
        Returns:
            List of layers where is_synced=True
        """
        return [
            layer for layer in self._layer_manager.get_all_layers()
            if getattr(layer, 'is_synced', False)
        ]
    
    def get_regular_layers(self) -> List[TimelineLayer]:
        """
        Get all regular (non-synced) layers.
        
        Returns:
            List of layers where is_synced=False
        """
        return [
            layer for layer in self._layer_manager.get_all_layers()
            if not getattr(layer, 'is_synced', False)
        ]
    
    def get_layer_handler(self, layer_id: str):
        """
        Get the handler for a specific layer.
        
        Args:
            layer_id: ID of the layer
            
        Returns:
            LayerHandler instance for this layer, or None if layer not found
        """
        from ..handlers import get_handler_for_layer
        
        layer = self._layer_manager.get_layer(layer_id)
        if not layer:
            return None
        
        return get_handler_for_layer(layer, self._scene, self._layer_manager)
    
    def validate_layer(self, layer_id: str) -> List[str]:
        """
        Validate a layer's state.
        
        Args:
            layer_id: ID of the layer to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        handler = self.get_layer_handler(layer_id)
        if not handler:
            return [f"Layer '{layer_id}' not found"]
        
        layer = self._layer_manager.get_layer(layer_id)
        return handler.validate(layer)
    
    # =========================================================================
    # Public API - Selection
    # =========================================================================
    
    def select_events(self, event_ids: List[str], clear_existing: bool = True) -> None:
        """Select events by ID."""
        self._scene.select_events(event_ids, clear_existing)
    
    def select_all(self) -> None:
        """Select all events."""
        self._scene.select_all()
    
    def clear_selection(self) -> None:
        """Deselect all events."""
        self._scene.deselect_all()
    
    def delete_selected(self) -> List[str]:
        """Delete selected events. Returns deleted IDs (uses batch API)."""
        selected = self._scene.get_selected_event_ids()
        if selected:
            # Use batch API - collect all event data first, then remove and emit
            results = []
            for event_id in selected:
                event_data = self._scene.get_event_data(event_id)
                if event_data:
                    results.append(EventDeleteResult(event_id=event_id, event_data=event_data))
            
            # Remove all at once
            self._scene.remove_events_batch(selected)
            
            # Emit batch signal
            if results:
                self.events_deleted.emit(results)
        return selected
    
    @property
    def selected_event_ids(self) -> List[str]:
        """IDs of currently selected events."""
        return self._scene.get_selected_event_ids()
    
    def get_selected_events(self) -> List[str]:
        """Get IDs of selected events (legacy)."""
        return self._scene.get_selected_event_ids()
    
    def get_selected_events_data(self) -> List[Dict[str, Any]]:
        """Get full data for selected events (legacy)."""
        result = []
        for eid in self._scene.get_selected_event_ids():
            item = self._scene.get_event_item(eid)
            if item:
                # Build metadata dict with audio fields for backward compatibility
                metadata = dict(item.user_data) if item.user_data else {}
                if item.audio_id:
                    metadata['audio_id'] = item.audio_id
                if item.audio_name:
                    metadata['audio_name'] = item.audio_name
                
                result.append({
                    'event_id': item.event_id,
                    'start_time': item.start_time,
                    'duration': item.duration,
                    'end_time': item.start_time + item.duration,
                    'classification': item.classification,
                    'layer_id': item.layer_id,
                    'audio_id': item.audio_id,
                    'audio_name': item.audio_name,
                    'metadata': metadata  # Legacy field
                })
        return result
    
    # =========================================================================
    # Public API - Playback
    # =========================================================================
    
    def set_audio_source(self, file_path: str) -> bool:
        """Set audio file for playback."""
        from ..playback.controller import SimpleAudioPlayer
        player = SimpleAudioPlayer()
        if player.load(file_path):
            self._playback_controller.set_audio_backend(player)
            return True
        return False
    
    def set_playback_controller(self, backend: Optional[PlaybackInterface]):
        """Connect an audio playback backend."""
        self._playback_controller.set_audio_backend(backend)
    
    def play(self) -> None:
        """Start playback."""
        self._playback_controller.play()
    
    def pause(self) -> None:
        """Pause playback."""
        self._playback_controller.pause()
    
    def stop(self) -> None:
        """Stop playback."""
        self._playback_controller.stop()
    
    def seek(self, time: float) -> None:
        """Seek to specific time."""
        self._on_seek(time)
    
    @property
    def duration(self) -> float:
        """Total timeline duration."""
        return self._scene._duration
    
    @duration.setter
    def duration(self, value: float):
        self.set_duration(value)
    
    def set_duration(self, seconds: float):
        """Set total timeline duration."""
        self._scene.set_duration(seconds)
        self._ruler.set_duration(seconds)
        self._playback_controller.set_duration(seconds)
    
    def get_duration(self) -> float:
        """Get timeline duration (legacy)."""
        return self._scene._duration
    
    @property
    def current_time(self) -> float:
        """Current playhead position."""
        return self._playback_controller.position
    
    @current_time.setter
    def current_time(self, value: float):
        self._on_seek(value)
    
    @property
    def is_playing(self) -> bool:
        """Whether playback is active."""
        return self._playback_controller.is_playing
    
    @property
    def position(self) -> float:
        """Get current playhead position (legacy)."""
        return self._playback_controller.position
    
    # =========================================================================
    # Public API - View
    # =========================================================================
    
    def zoom_to_fit(self) -> None:
        """Zoom to fit all events."""
        self._view.zoom_to_fit()
    
    def zoom_to_selection(self) -> None:
        """Zoom to fit selected events."""
        self._view.zoom_to_selection()
    
    def scroll_to_time(self, time: float) -> None:
        """Scroll view to show specific time."""
        self._view.scroll_to_time(time)
    
    def scroll_to_playhead(self) -> None:
        """Scroll view to show playhead."""
        self._view.scroll_to_time(self._playback_controller.position)
    
    def zoom_in(self, factor: float = None):
        self._view.zoom_in(factor)
    
    def zoom_out(self, factor: float = None):
        self._view.zoom_out(factor)
    
    def reset_zoom(self):
        self._view.reset_zoom()
    
    @property
    def zoom_level(self) -> float:
        return self._view.pixels_per_second
    
    # =========================================================================
    # Public API - Configuration
    # =========================================================================
    
    def set_editable(self, editable: bool) -> None:
        """Enable/disable event editing."""
        self._scene.editable = editable
    
    def set_snap_enabled(self, enabled: bool) -> None:
        """Enable/disable grid snapping (persists to settings)."""
        self._settings_manager.snap_enabled = enabled
        self._grid_system.snap_enabled = enabled
        self._grid_system.settings.snap_enabled = enabled
    
    
    def set_show_grid_lines(self, show: bool) -> None:
        """Show/hide grid lines (persists to settings)."""
        self._settings_manager.show_grid_lines = show
        self._grid_system.settings.show_grid_lines = show
        if hasattr(self, '_scene'):
            self._scene.update()
    
    def set_default_layer_height(self, height: int) -> None:
        """Set default height for new layers (persists to settings)."""
        self._settings_manager.default_layer_height = height
        self._layer_manager.set_default_layer_height(height)
    
    def show_settings(self) -> None:
        """Show settings dialog."""
        self._show_settings_dialog()
    
    @property
    def grid_system(self) -> GridSystem:
        return self._grid_system
    
    @property
    def event_inspector(self) -> EventInspector:
        return self._event_inspector
    
    def set_data_item_repo(self, repo):
        """
        Set data_item_repo for direct audio item lookup by ID.
        
        Propagates to scene (for waveform_simple) and inspector (for clip playback).
        
        Args:
            repo: DataItemRepository instance
        """
        # Set on scene (which propagates to waveform_simple)
        if self._scene:
            self._scene.set_data_item_repo(repo)
        # Set on inspector (which uses it for clip playback)
        if self._event_inspector:
            self._event_inspector.set_data_item_repo(repo)
    
    def set_audio_lookup_callback(self, callback):
        """Deprecated: Use set_data_item_repo() instead. Kept for backward compatibility."""
        # Legacy: propagate callback to scene and inspector
        if self._scene:
            self._scene.set_audio_lookup_callback(callback)
        if self._event_inspector:
            self._event_inspector.set_audio_lookup_callback(callback)
    
    def set_event_update_callback(self, callback):
        """
        Set event update callback for event inspector.
        
        Args:
            callback: Function that accepts (event_id: str, metadata: Dict[str, Any]) 
                     and updates the event. Should return True if successful.
        """
        self._event_update_callback = callback
        # Set on scene so event items can access it for context menu updates
        if self._scene:
            self._scene.set_event_update_callback(callback)
        # Also set on inspector (for any inspector-specific updates)
        if hasattr(self, '_event_inspector') and self._event_inspector:
            self._event_inspector.set_event_update_callback(callback)
    
    @property
    def settings_panel(self) -> SettingsPanel:
        return self._settings_panel
    
    @property
    def settings(self) -> TimelineSettings:
        return self._settings_panel.settings
    
    def show_inspector(self, show: bool = True):
        """Show or hide the inspector panel (persists to settings)."""
        self._inspector_container.setVisible(show)
        self._settings_manager.inspector_visible = show
    
    def toggle_inspector(self):
        """Toggle inspector panel visibility (persists to settings)."""
        self.show_inspector(not self._inspector_container.isVisible())
    
    def is_inspector_visible(self) -> bool:
        """Check if inspector panel is visible."""
        return self._inspector_container.isVisible()
    
    @property
    def settings_manager(self) -> TimelineSettingsManager:
        """Get the settings manager for this timeline widget."""
        return self._settings_manager
    
    def set_top_right_widget(self, widget: QWidget):
        if self._top_right_widget:
            self._top_right_layout.removeWidget(self._top_right_widget)
            self._top_right_widget.setParent(None)
        
        self._top_right_widget = widget
        widget.setParent(self._top_right_container)
        self._top_right_layout.addWidget(widget)
        self._top_right_container.setVisible(True)
    
    def clear_top_right_widget(self):
        if self._top_right_widget:
            self._top_right_layout.removeWidget(self._top_right_widget)
            self._top_right_widget.setParent(None)
            self._top_right_widget = None
            self._top_right_container.setVisible(False)
    
    def set_toolbar_widget(self, widget: QWidget):
        """Add a custom widget to the bottom toolbar (after position label)."""
        if self._toolbar_custom_widget:
            self._toolbar_custom_layout.removeWidget(self._toolbar_custom_widget)
            self._toolbar_custom_widget.setParent(None)
        
        self._toolbar_custom_widget = widget
        widget.setParent(self._toolbar_custom_container)
        self._toolbar_custom_layout.addWidget(widget)
    
    def clear_toolbar_widget(self):
        """Remove custom widget from the bottom toolbar."""
        if self._toolbar_custom_widget:
            self._toolbar_custom_layout.removeWidget(self._toolbar_custom_widget)
            self._toolbar_custom_widget.setParent(None)
            self._toolbar_custom_widget = None
    
    def set_follow_mode(self, mode: PlayheadFollowMode):
        self._settings_panel.set_follow_mode(mode)
    
    def set_zoom_anchor(self, anchor):
        from .view import ZoomAnchor
        if isinstance(anchor, str):
            anchor = ZoomAnchor.UNDER_MOUSE if anchor == "under_mouse" else ZoomAnchor.VIEW_CENTER
        self._view.set_zoom_anchor(anchor)
    
    @property
    def zoom_anchor(self):
        return self._view.zoom_anchor
    
    # Internal access (for advanced use)
    def scene(self):
        """Get the internal scene (for advanced use)."""
        return self._scene
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Clean up event inspector first (stops timers, threads, and audio playback)
        # This must happen before other cleanup to prevent hanging on close
        if hasattr(self, '_event_inspector') and self._event_inspector:
            # Get reference to clip player before clear() (clear() stops it but doesn't cleanup)
            clip_player = getattr(self._event_inspector, '_clip_player', None)
            
            # Clear selection (stops timers, threads, and stops clip player)
            # This must be called first to stop active operations
            self._event_inspector.clear()
            
            # Cleanup clip player (clear() stops it but doesn't call cleanup())
            if clip_player:
                try:
                    clip_player.cleanup()
                except Exception as e:
                    Log.warning(f"TimelineWidget: Error cleaning up clip player: {e}")
        
        self._playback_controller.cleanup()
        self._scene.clear_events()
        
        Log.debug("TimelineWidget: Cleanup complete")
