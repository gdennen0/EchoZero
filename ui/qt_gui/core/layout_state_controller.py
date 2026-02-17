"""
Layout State Controller

Central controller for managing window layout state.
Handles save/restore, import/export, and window registration.

This replaces the broken Qt saveState() approach with a deterministic
manual state system that works for dynamically-created dock widgets.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from PyQt6.QtWidgets import QMainWindow, QDockWidget
from PyQt6.QtCore import Qt, QObject, pyqtSignal

from src.utils.message import Log
from .window_state_types import (
    LayoutState, WindowState, TabGroupInfo, WindowGeometry,
    DockPosition, IStatefulWindow
)

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class LayoutStateController(QObject):
    """
    Central controller for window layout state.
    
    Responsibilities:
    - Register/unregister windows
    - Collect state from all windows
    - Save/restore layout state
    - Export/import layout files
    - Detect and manage tab groups
    
    Key Design:
    - Windows are responsible for their internal state (IStatefulWindow)
    - This controller handles positions, visibility, tab groups
    - State is deterministic - no Qt timing hacks
    """
    
    # Signals
    layout_saved = pyqtSignal()
    layout_restored = pyqtSignal()
    layout_reset = pyqtSignal()
    
    # Layout file extension
    LAYOUT_FILE_EXTENSION = ".ezlayout"
    
    def __init__(self, main_window: QMainWindow, facade: "ApplicationFacade"):
        super().__init__()
        self.main_window = main_window
        self.facade = facade
        
        # Registry of all managed windows
        # window_id -> (dock_widget, window_type, block_type, block_id)
        self._windows: Dict[str, Dict[str, Any]] = {}
        
        # Current tab groups (detected from Qt or restored from state)
        self._tab_groups: List[TabGroupInfo] = []
    
    # ==================== Window Registration ====================
    
    def register_window(
        self,
        window_id: str,
        dock_widget: QDockWidget,
        window_type: str,
        block_type: Optional[str] = None,
        block_id: Optional[str] = None
    ) -> None:
        """
        Register a window for state management.
        
        Args:
            window_id: Unique identifier (e.g., "node_editor", "block_panel_123")
            dock_widget: The QDockWidget instance
            window_type: Type of window (e.g., "node_editor", "block_panel")
            block_type: For block panels, the block type (e.g., "PlotEvents")
            block_id: For block panels, the specific block ID
        """
        self._windows[window_id] = {
            "dock": dock_widget,
            "type": window_type,
            "block_type": block_type,
            "block_id": block_id
        }
        Log.debug(f"LayoutStateController: Registered window {window_id} (type={window_type})")
    
    def unregister_window(self, window_id: str) -> None:
        """Unregister a window from state management."""
        if window_id in self._windows:
            del self._windows[window_id]
            # Remove from any tab groups
            for tg in self._tab_groups:
                if window_id in tg.window_ids:
                    tg.window_ids.remove(window_id)
            Log.debug(f"LayoutStateController: Unregistered window {window_id}")
    
    def get_registered_windows(self) -> List[str]:
        """Get list of all registered window IDs."""
        return list(self._windows.keys())
    
    # ==================== State Capture ====================
    
    def capture_layout_state(self) -> LayoutState:
        """
        Capture current layout state from all registered windows.
        
        Returns:
            LayoutState with current positions, tab groups, internal states
        """
        # Main window geometry
        geom = self.main_window.geometry()
        main_geom = WindowGeometry(geom.x(), geom.y(), geom.width(), geom.height())
        main_maximized = self.main_window.isMaximized()
        
        # Capture state from each window
        windows: List[WindowState] = []
        
        for window_id, info in self._windows.items():
            dock: QDockWidget = info["dock"]
            
            # Get geometry
            dock_geom = dock.geometry()
            geometry = WindowGeometry(
                dock_geom.x(), dock_geom.y(),
                dock_geom.width(), dock_geom.height()
            )
            
            # Determine dock position
            is_floating = dock.isFloating()
            Log.debug(f"Capturing state for {window_id}: isFloating()={is_floating}, "
                     f"isVisible()={dock.isVisible()}, geometry={dock_geom.x()},{dock_geom.y()},{dock_geom.width()},{dock_geom.height()}")
            
            if is_floating:
                dock_position = DockPosition.FLOATING
            else:
                dock_position = self._get_dock_position(dock)
                Log.debug(f"  Determined dock position: {dock_position.value}")
            
            # Get internal state from window (if it implements IStatefulWindow)
            internal_state = {}
            widget = dock.widget()
            if widget and isinstance(widget, IStatefulWindow):
                try:
                    internal_state = widget.get_internal_state()
                except Exception as e:
                    Log.warning(f"Failed to get internal state from {window_id}: {e}")
            
            # Also check if dock itself implements IStatefulWindow
            if isinstance(dock, IStatefulWindow):
                try:
                    dock_state = dock.get_internal_state()
                    internal_state.update(dock_state)
                except Exception as e:
                    Log.warning(f"Failed to get internal state from dock {window_id}: {e}")
            
            window_state = WindowState(
                window_id=window_id,
                window_type=info["type"],
                block_type=info.get("block_type"),
                block_id=info.get("block_id"),
                dock_position=dock_position,
                geometry=geometry,
                is_visible=dock.isVisible(),
                tab_group_id=self._find_tab_group_for_window(window_id),
                internal_state=internal_state
            )
            windows.append(window_state)
        
        # Detect tab groups from current Qt state
        tab_groups = self._detect_tab_groups()
        
        return LayoutState(
            main_window_geometry=main_geom,
            main_window_maximized=main_maximized,
            windows=windows,
            tab_groups=tab_groups,
            version="1.0",
            timestamp=time.time()
        )
    
    def _get_dock_position(self, dock: QDockWidget) -> DockPosition:
        """Determine the dock position of a non-floating dock."""
        # Use Qt's dockWidgetArea() to get the actual dock area
        area = self.main_window.dockWidgetArea(dock)
        
        # Map Qt dock areas to our DockPosition enum
        if area == Qt.DockWidgetArea.LeftDockWidgetArea:
            return DockPosition.LEFT
        elif area == Qt.DockWidgetArea.RightDockWidgetArea:
            return DockPosition.RIGHT
        elif area == Qt.DockWidgetArea.TopDockWidgetArea:
            return DockPosition.TOP
        elif area == Qt.DockWidgetArea.BottomDockWidgetArea:
            return DockPosition.BOTTOM
        else:
            # Fallback: use geometry heuristic
            main_rect = self.main_window.centralWidget().geometry() if self.main_window.centralWidget() else self.main_window.geometry()
            dock_rect = dock.geometry()
            
            dock_center = dock_rect.center()
            main_center = main_rect.center()
            
            dx = dock_center.x() - main_center.x()
            dy = dock_center.y() - main_center.y()
            
            if abs(dx) > abs(dy):
                return DockPosition.LEFT if dx < 0 else DockPosition.RIGHT
            else:
                return DockPosition.TOP if dy < 0 else DockPosition.BOTTOM
    
    def _detect_tab_groups(self) -> List[TabGroupInfo]:
        """
        Detect current tab groups from Qt's dock structure.
        
        Uses tabifiedDockWidgets() to find which docks are grouped together.
        """
        tab_groups: List[TabGroupInfo] = []
        processed_docks = set()
        
        for window_id, info in self._windows.items():
            dock: QDockWidget = info["dock"]
            
            if dock in processed_docks or dock.isFloating():
                continue
            
            # Get all docks tabified with this one
            tabified = self.main_window.tabifiedDockWidgets(dock)
            
            if tabified:
                # This dock is part of a tab group
                group_window_ids = [window_id]
                processed_docks.add(dock)
                
                for other_dock in tabified:
                    # Find window_id for this dock
                    for other_id, other_info in self._windows.items():
                        if other_info["dock"] == other_dock:
                            group_window_ids.append(other_id)
                            processed_docks.add(other_dock)
                            break
                
                if len(group_window_ids) >= 2:
                    # Determine which tab is active (visible)
                    active_index = 0
                    for i, wid in enumerate(group_window_ids):
                        d = self._windows[wid]["dock"]
                        if d.isVisible() and not d.visibleRegion().isEmpty():
                            active_index = i
                            break
                    
                    # Get dock position from first dock in group
                    dock_position = self._get_dock_position(dock)
                    
                    tab_group = TabGroupInfo(
                        dock_position=dock_position,
                        window_ids=group_window_ids,
                        active_index=active_index
                    )
                    tab_groups.append(tab_group)
        
        return tab_groups
    
    def _find_tab_group_for_window(self, window_id: str) -> Optional[str]:
        """Find tab group ID for a window, if it's in a tab group."""
        for i, tg in enumerate(self._tab_groups):
            if window_id in tg.window_ids:
                return f"tab_group_{i}"
        return None
    
    # ==================== Save/Restore ====================
    
    def save_to_session(self) -> bool:
        """
        Save current layout state to session storage.
        
        Returns:
            True if saved successfully
        """
        try:
            state = self.capture_layout_state()
            
            # Debug: Log what we're saving
            Log.debug(f"Saving layout state:")
            for w in state.windows:
                Log.debug(f"  - {w.window_id}: position={w.dock_position.value}, "
                         f"geometry={w.geometry.to_tuple() if w.geometry else None}, "
                         f"visible={w.is_visible}")
            for tg in state.tab_groups:
                Log.debug(f"  - Tab group: {tg.window_ids}, active={tg.active_index}")
            
            state_dict = state.to_dict()
            
            result = self.facade.set_session_state("window_layout", state_dict)
            
            if result.success:
                Log.info(f"Layout state saved ({len(state.windows)} windows, {len(state.tab_groups)} tab groups)")
                self.layout_saved.emit()
                return True
            else:
                Log.error(f"Failed to save layout state: {result.message}")
                return False
        except Exception as e:
            Log.error(f"Error saving layout state: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def restore_from_session(self) -> bool:
        """
        Restore layout state from session storage.
        
        Returns:
            True if restored successfully, False if no state or error
        """
        try:
            result = self.facade.get_session_state("window_layout")
            
            if not result.success or not result.data:
                Log.info("No saved layout state found")
                return False
            
            state = LayoutState.from_dict(result.data)
            
            if state.version < "1.0":
                Log.info("Old layout format detected, using default layout")
                return False
            
            # Debug: Log what we're restoring
            Log.debug(f"Restoring layout state:")
            for w in state.windows:
                Log.debug(f"  - {w.window_id}: position={w.dock_position.value}, "
                         f"geometry={w.geometry.to_tuple() if w.geometry else None}, "
                         f"visible={w.is_visible}")
            for tg in state.tab_groups:
                Log.debug(f"  - Tab group: {tg.window_ids}, active={tg.active_index}")
            
            return self._apply_layout_state(state)
        except Exception as e:
            Log.error(f"Error restoring layout state: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _apply_layout_state(self, state: LayoutState) -> bool:
        """
        Apply a layout state to the current windows.
        
        Args:
            state: LayoutState to apply
            
        Returns:
            True if applied successfully
        """
        try:
            
            # Restore main window geometry
            if state.main_window_geometry:
                g = state.main_window_geometry
                self.main_window.setGeometry(g.x, g.y, g.width, g.height)
            
            if state.main_window_maximized:
                self.main_window.showMaximized()
            
            # Build lookup: block_type -> window_id (for matching across projects)
            # This allows layouts to work across projects with same block types
            current_by_type: Dict[str, List[str]] = {}
            for window_id, info in self._windows.items():
                block_type = info.get("block_type")
                if block_type:
                    if block_type not in current_by_type:
                        current_by_type[block_type] = []
                    current_by_type[block_type].append(window_id)
            
            # Track which saved windows map to which current windows
            saved_to_current: Dict[str, str] = {}
            
            # First pass: Match by window_type for non-block-panel windows
            for ws in state.windows:
                if ws.window_type != "block_panel":
                    if ws.window_id in self._windows:
                        saved_to_current[ws.window_id] = ws.window_id
            
            # Second pass: Match block panels by block_type
            type_usage: Dict[str, int] = {}  # Track how many of each type we've used
            for ws in state.windows:
                if ws.window_type == "block_panel" and ws.block_type:
                    if ws.block_type in current_by_type:
                        idx = type_usage.get(ws.block_type, 0)
                        available = current_by_type[ws.block_type]
                        if idx < len(available):
                            saved_to_current[ws.window_id] = available[idx]
                            type_usage[ws.block_type] = idx + 1
            
            # Apply window states
            for ws in state.windows:
                current_id = saved_to_current.get(ws.window_id)
                if not current_id or current_id not in self._windows:
                    Log.debug(f"Skipping window {ws.window_id} - not found in current windows")
                    continue
                
                info = self._windows[current_id]
                dock: QDockWidget = info["dock"]
                
                Log.debug(f"Applying state to {current_id}: position={ws.dock_position.value}")
                
                # Set visibility
                if ws.is_visible:
                    dock.show()
                else:
                    dock.hide()
                
                # Set position
                if ws.dock_position == DockPosition.FLOATING:
                    dock.setFloating(True)
                    if ws.geometry:
                        g = ws.geometry
                        dock.setGeometry(g.x, g.y, g.width, g.height)
                        Log.debug(f"  Set floating geometry: ({g.x}, {g.y}, {g.width}, {g.height})")
                else:
                    dock.setFloating(False)
                    qt_area = self._dock_position_to_qt(ws.dock_position)
                    self.main_window.addDockWidget(qt_area, dock)
                    Log.debug(f"  Docked to area: {ws.dock_position.value}")
                
                # Restore internal state
                if ws.internal_state:
                    widget = dock.widget()
                    if widget and isinstance(widget, IStatefulWindow):
                        try:
                            widget.restore_internal_state(ws.internal_state)
                        except Exception as e:
                            Log.warning(f"Failed to restore internal state for {current_id}: {e}")
                    
                    if isinstance(dock, IStatefulWindow):
                        try:
                            dock.restore_internal_state(ws.internal_state)
                        except Exception as e:
                            Log.warning(f"Failed to restore internal state for dock {current_id}: {e}")
            
            # Restore tab groups (translate window IDs)
            self._tab_groups = []
            for tg in state.tab_groups:
                translated_ids = []
                for wid in tg.window_ids:
                    current_id = saved_to_current.get(wid)
                    if current_id and current_id in self._windows:
                        translated_ids.append(current_id)
                
                if len(translated_ids) >= 2:
                    new_tg = TabGroupInfo(
                        dock_position=tg.dock_position,
                        window_ids=translated_ids,
                        active_index=min(tg.active_index, len(translated_ids) - 1)
                    )
                    self._tab_groups.append(new_tg)
                    
                    # Create tab group in Qt
                    self._create_qt_tab_group(new_tg)
            
            Log.info(f"Layout state applied ({len(saved_to_current)} windows matched, {len(self._tab_groups)} tab groups)")
            self.layout_restored.emit()
            return True
            
        except Exception as e:
            Log.error(f"Error applying layout state: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _dock_position_to_qt(self, position: DockPosition) -> Qt.DockWidgetArea:
        """Convert DockPosition to Qt.DockWidgetArea."""
        mapping = {
            DockPosition.LEFT: Qt.DockWidgetArea.LeftDockWidgetArea,
            DockPosition.RIGHT: Qt.DockWidgetArea.RightDockWidgetArea,
            DockPosition.TOP: Qt.DockWidgetArea.TopDockWidgetArea,
            DockPosition.BOTTOM: Qt.DockWidgetArea.BottomDockWidgetArea,
        }
        return mapping.get(position, Qt.DockWidgetArea.TopDockWidgetArea)
    
    def _create_qt_tab_group(self, tab_group: TabGroupInfo) -> None:
        """Create a tab group in Qt by tabifying docks."""
        if len(tab_group.window_ids) < 2:
            return
        
        # Get the first dock as anchor
        first_id = tab_group.window_ids[0]
        if first_id not in self._windows:
            return
        
        anchor_dock = self._windows[first_id]["dock"]
        
        # Tabify remaining docks onto anchor
        for wid in tab_group.window_ids[1:]:
            if wid not in self._windows:
                continue
            dock = self._windows[wid]["dock"]
            if not dock.isFloating():
                self.main_window.tabifyDockWidget(anchor_dock, dock)
        
        # Raise the active tab
        if 0 <= tab_group.active_index < len(tab_group.window_ids):
            active_id = tab_group.window_ids[tab_group.active_index]
            if active_id in self._windows:
                self._windows[active_id]["dock"].raise_()
    
    # ==================== Layout File Import/Export ====================
    
    def export_layout(self, file_path: str) -> bool:
        """
        Export current layout to a file.
        
        Args:
            file_path: Path to save the layout file
            
        Returns:
            True if exported successfully
        """
        try:
            state = self.capture_layout_state()
            state_dict = state.to_dict()
            
            # Ensure correct extension
            path = Path(file_path)
            if path.suffix != self.LAYOUT_FILE_EXTENSION:
                path = path.with_suffix(self.LAYOUT_FILE_EXTENSION)
            
            with open(path, 'w') as f:
                json.dump(state_dict, f, indent=2)
            
            Log.info(f"Layout exported to {path}")
            return True
        except Exception as e:
            Log.error(f"Failed to export layout: {e}")
            return False
    
    def import_layout(self, file_path: str) -> bool:
        """
        Import and apply a layout from a file.
        
        Args:
            file_path: Path to the layout file
            
        Returns:
            True if imported and applied successfully
        """
        try:
            path = Path(file_path)
            if not path.exists():
                Log.error(f"Layout file not found: {path}")
                return False
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            state = LayoutState.from_dict(data)
            
            if state.version < "1.0":
                Log.warning("Layout file has incompatible version")
                return False
            
            return self._apply_layout_state(state)
        except json.JSONDecodeError as e:
            Log.error(f"Invalid layout file format: {e}")
            return False
        except Exception as e:
            Log.error(f"Failed to import layout: {e}")
            return False
    
    # ==================== Reset ====================
    
    def reset_to_default(self) -> None:
        """
        Reset all windows to default state (all floating, visible).
        
        This is the initial state for a new project.
        """
        for window_id, info in self._windows.items():
            dock: QDockWidget = info["dock"]
            
            # Make floating and visible
            dock.setFloating(True)
            dock.show()
            
            # Position cascading from top-left
            idx = list(self._windows.keys()).index(window_id)
            dock.setGeometry(100 + idx * 30, 100 + idx * 30, 800, 600)
        
        self._tab_groups = []
        Log.info("Layout reset to default (all windows floating)")
        self.layout_reset.emit()
