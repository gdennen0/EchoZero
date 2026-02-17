"""
Dock State Manager

Simple dock state management using Qt's native saveState/restoreState.
This replaces the 596-line LayoutStateController with ~100 lines.

Key insight: Qt's saveState() handles ALL dock positioning including:
- Dock areas (top/bottom/left/right)
- Floating windows (position + size)  
- Tab groups (which docks are tabbed, order, active tab)
- Visibility

We just need to:
1. Create all docks before restoreState()
2. Give each dock a unique objectName()
3. Save internal state (zoom, scroll) separately
"""
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from PyQt6.QtWidgets import QMainWindow, QDockWidget
from PyQt6.QtCore import QByteArray

from src.utils.message import Log
from .window_state_types import IStatefulWindow

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class DockStateManager:
    """
    Minimal dock state manager using Qt's native state system.
    
    Usage:
        manager = DockStateManager(main_window, facade)
        
        # Register docks (for internal state tracking)
        manager.register_dock("node_editor", dock)
        
        # Save on close
        manager.save_state()
        
        # Restore on open (after creating all docks)
        manager.restore_state()
    """
    
    UI_STATE_TYPE_QT_DOCK = "qt_dock_state"
    UI_STATE_TYPE_DOCK_INTERNAL = "dock_internal_states"
    UI_STATE_TYPE_MAIN_WINDOW_GEOMETRY = "main_window_geometry"
    UI_STATE_TYPE_OPEN_PANELS = "open_block_panels"
    
    def __init__(self, main_window: QMainWindow, facade: "ApplicationFacade"):
        self.main_window = main_window
        self.facade = facade
        # Track docks for internal state (zoom, scroll, etc.)
        self._docks: Dict[str, QDockWidget] = {}
    
    def register_dock(self, dock_id: str, dock: QDockWidget) -> None:
        """Register a dock for internal state tracking."""
        self._docks[dock_id] = dock
        # Ensure unique object name for Qt state
        if not dock.objectName():
            dock.setObjectName(dock_id)
        Log.debug(f"DockStateManager: Registered {dock_id}")
    
    def unregister_dock(self, dock_id: str) -> None:
        """Unregister a dock."""
        if dock_id in self._docks:
            del self._docks[dock_id]
            Log.debug(f"DockStateManager: Unregistered {dock_id}")
    
    def get_registered_docks(self) -> List[str]:
        """Get list of registered dock IDs."""
        return list(self._docks.keys())
    
    def save_state(self) -> bool:
        """
        Save dock state to UI state (project-specific).
        
        Saves:
        1. Qt native dock state (positions, tabs, visibility)
        2. Main window geometry
        3. Internal state for each dock (zoom, scroll, etc.)
        4. List of open block panel IDs
        """
        try:
            if not self.facade.get_current_project_id():
                Log.warning("DockStateManager: No project loaded, skipping state save")
                return False
            
            # 1. Save Qt native state
            qt_state = self.main_window.saveState()
            qt_state_str = qt_state.toBase64().data().decode('utf-8')
            result = self.facade.set_ui_state(self.UI_STATE_TYPE_QT_DOCK, None, {"state": qt_state_str})
            if not result.success:
                Log.warning(f"DockStateManager: Failed to save Qt dock state: {result.message}")
            
            # 2. Save main window geometry
            geo = self.main_window.geometry()
            result = self.facade.set_ui_state(self.UI_STATE_TYPE_MAIN_WINDOW_GEOMETRY, None, {
                "x": geo.x(),
                "y": geo.y(),
                "width": geo.width(),
                "height": geo.height(),
                "maximized": self.main_window.isMaximized()
            })
            if not result.success:
                Log.warning(f"DockStateManager: Failed to save main window geometry: {result.message}")
            
            # 3. Save internal states (zoom, scroll, etc.)
            internal_states = {}
            for dock_id, dock in self._docks.items():
                widget = dock.widget()
                state = {}
                
                # Try dock itself
                if isinstance(dock, IStatefulWindow):
                    state.update(dock.get_internal_state())
                
                # Try widget inside dock
                if widget and isinstance(widget, IStatefulWindow):
                    state.update(widget.get_internal_state())
                
                if state:
                    internal_states[dock_id] = state
            
            if internal_states:
                result = self.facade.set_ui_state(self.UI_STATE_TYPE_DOCK_INTERNAL, None, internal_states)
                if not result.success:
                    Log.warning(f"DockStateManager: Failed to save dock internal states: {result.message}")
            
            # 4. Save list of open block panels (for recreation on load)
            open_panels = []
            for dock_id, dock in self._docks.items():
                if dock_id.startswith("block_panel_"):
                    block_id = dock_id.replace("block_panel_", "")
                    open_panels.append({
                        "block_id": block_id,
                        "dock_id": dock_id
                    })
            
            if open_panels:
                result = self.facade.set_ui_state(self.UI_STATE_TYPE_OPEN_PANELS, None, {"panels": open_panels})
                if not result.success:
                    Log.warning(f"DockStateManager: Failed to save open panels: {result.message}")
            
            Log.info(f"DockStateManager: Saved state for {len(self._docks)} docks")
            return True
            
        except Exception as e:
            Log.error(f"DockStateManager: Failed to save state: {e}")
            return False
    
    def restore_state(self) -> bool:
        """
        Restore dock state from UI state (project-specific).
        
        IMPORTANT: All docks must be created and registered BEFORE calling this.
        """
        try:
            if not self.facade.get_current_project_id():
                Log.info("DockStateManager: No project loaded, skipping state restore")
                return False
            
            # 1. Restore main window geometry
            result = self.facade.get_ui_state(self.UI_STATE_TYPE_MAIN_WINDOW_GEOMETRY, None)
            if result and result.success and result.data:
                g = result.data
                if g.get("maximized"):
                    self.main_window.showMaximized()
                else:
                    self.main_window.setGeometry(
                        g.get("x", 100),
                        g.get("y", 100),
                        g.get("width", 1600),
                        g.get("height", 800)
                    )
            
            # 2. Restore Qt native state
            result = self.facade.get_ui_state(self.UI_STATE_TYPE_QT_DOCK, None)
            if result and result.success and result.data:
                qt_state_data = result.data
                qt_state_str = qt_state_data.get("state")
                if qt_state_str:
                    qt_state = QByteArray.fromBase64(qt_state_str.encode('utf-8'))
                    success = self.main_window.restoreState(qt_state)
                    if success:
                        Log.info("DockStateManager: Qt state restored successfully")
                    else:
                        Log.warning("DockStateManager: Qt restoreState returned False")
                else:
                    Log.info("DockStateManager: No Qt state data in saved state")
            else:
                Log.info("DockStateManager: No saved Qt state, using defaults")
                return False
            
            # 3. Restore internal states
            result = self.facade.get_ui_state(self.UI_STATE_TYPE_DOCK_INTERNAL, None)
            if result and result.success and result.data:
                internal_states = result.data
                for dock_id, state in internal_states.items():
                    if dock_id in self._docks:
                        dock = self._docks[dock_id]
                        widget = dock.widget()
                        
                        # Try dock itself
                        if isinstance(dock, IStatefulWindow):
                            try:
                                dock.restore_internal_state(state)
                            except Exception as e:
                                Log.warning(f"Failed to restore dock state for {dock_id}: {e}")
                        
                        # Try widget inside dock
                        if widget and isinstance(widget, IStatefulWindow):
                            try:
                                widget.restore_internal_state(state)
                            except Exception as e:
                                Log.warning(f"Failed to restore widget state for {dock_id}: {e}")
            
            Log.info(f"DockStateManager: Restored state for {len(self._docks)} docks")
            return True
            
        except Exception as e:
            Log.error(f"DockStateManager: Failed to restore state: {e}")
            return False
    
    def get_open_panel_ids(self) -> List[str]:
        """Get list of block IDs for panels that were open."""
        result = self.facade.get_ui_state(self.UI_STATE_TYPE_OPEN_PANELS, None)
        if result and result.success and result.data:
            panels_data = result.data
            panels = panels_data.get("panels", [])
            return [p["block_id"] for p in panels if isinstance(p, dict) and "block_id" in p]
        return []





