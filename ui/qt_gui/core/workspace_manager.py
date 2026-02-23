"""
Workspace Manager

Unified workspace state management using PyQt6Ads CDockManager.
Manages dock positions, layout perspectives, and internal window states.

CDockManager handles all dock positioning including:
- Dock areas (top/bottom/left/right/center)
- Floating windows (position + size)
- Tab groups (which docks are tabbed, order, active tab)
- Split views, auto-hide sidebars, visibility

This manager adds:
- Named layout perspectives (save/restore/list/delete) via CDockManager API
- Session state (zoom, scroll, viewport) consolidated here
- Open block panel tracking for recreation on project load
- Default layout for first-launch experience

Layout state is shared across application modes (production/developer).
Mode-specific dock visibility is handled by MainWindow._apply_mode_visibility(),
not by separate saved layouts.
"""
from typing import Dict, Any, Optional, List, TYPE_CHECKING

import PyQt6Ads as ads
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtCore import QByteArray

from src.utils.message import Log
from .window_state_types import IStatefulWindow

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class WorkspaceManager:
    """
    Unified workspace state manager backed by PyQt6Ads CDockManager.

    Handles all layout persistence through a single interface:
    - CDockManager state (positions, tabs, splits, visibility)
    - Main window geometry
    - Internal dock states (zoom, scroll, etc. via IStatefulWindow)
    - Open block panel tracking
    - Named layout perspectives
    """

    UI_STATE_QT_DOCK = "qt_dock_state"
    UI_STATE_DOCK_INTERNAL = "dock_internal_states"
    UI_STATE_MAIN_GEOMETRY = "main_window_geometry"
    UI_STATE_OPEN_PANELS = "open_block_panels"
    UI_STATE_SESSION = "workspace_session"

    def __init__(self, main_window: QMainWindow, dock_manager: ads.CDockManager, facade: "ApplicationFacade"):
        self.main_window = main_window
        self.dock_manager = dock_manager
        self.facade = facade
        self._docks: Dict[str, ads.CDockWidget] = {}

    # ======================== Dock Registration ========================

    def register_dock(self, dock_id: str, dock: ads.CDockWidget) -> None:
        """Register a dock for state tracking."""
        self._docks[dock_id] = dock
        if not dock.objectName():
            dock.setObjectName(dock_id)
        Log.debug(f"WorkspaceManager: Registered {dock_id}")

    def unregister_dock(self, dock_id: str) -> None:
        """Unregister a dock."""
        if dock_id in self._docks:
            del self._docks[dock_id]
            Log.debug(f"WorkspaceManager: Unregistered {dock_id}")

    def get_registered_docks(self) -> List[str]:
        """Get list of registered dock IDs."""
        return list(self._docks.keys())

    # ======================== State Save / Restore ========================

    def save_state(self) -> bool:
        """Save complete workspace state to project storage."""
        try:
            if not self.facade.get_current_project_id():
                Log.warning("WorkspaceManager: No project loaded, skipping save")
                return False

            dock_state = self.dock_manager.saveState()
            dock_state_str = dock_state.toBase64().data().decode('utf-8')
            self.facade.set_ui_state(self.UI_STATE_QT_DOCK, None, {"state": dock_state_str})

            geo = self.main_window.geometry()
            self.facade.set_ui_state(self.UI_STATE_MAIN_GEOMETRY, None, {
                "x": geo.x(),
                "y": geo.y(),
                "width": geo.width(),
                "height": geo.height(),
                "maximized": self.main_window.isMaximized()
            })

            internal_states = {}
            for dock_id, dock in self._docks.items():
                widget = dock.widget()
                state = {}
                if isinstance(dock, IStatefulWindow):
                    state.update(dock.get_internal_state())
                if widget and isinstance(widget, IStatefulWindow):
                    state.update(widget.get_internal_state())
                if state:
                    internal_states[dock_id] = state

            if internal_states:
                self.facade.set_ui_state(self.UI_STATE_DOCK_INTERNAL, None, internal_states)

            open_panels = []
            for dock_id, dock in self._docks.items():
                if dock_id.startswith("block_panel_"):
                    block_id = dock_id.replace("block_panel_", "")
                    open_panels.append({"block_id": block_id, "dock_id": dock_id})

            self.facade.set_ui_state(self.UI_STATE_OPEN_PANELS, None, {"panels": open_panels})

            Log.info(f"WorkspaceManager: Saved state ({len(self._docks)} docks)")
            return True

        except Exception as e:
            Log.error(f"WorkspaceManager: Failed to save state: {e}")
            return False

    def restore_state(self) -> bool:
        """Restore workspace state from project storage.

        All docks must be created and registered BEFORE calling this.
        """
        try:
            if not self.facade.get_current_project_id():
                Log.info("WorkspaceManager: No project loaded, skipping restore")
                return False

            result = self.facade.get_ui_state(self.UI_STATE_MAIN_GEOMETRY, None)
            if result and result.success and result.data:
                g = result.data
                if g.get("maximized"):
                    self.main_window.showMaximized()
                else:
                    self.main_window.setGeometry(
                        g.get("x", 100), g.get("y", 100),
                        g.get("width", 1600), g.get("height", 800)
                    )

            result = self.facade.get_ui_state(self.UI_STATE_QT_DOCK, None)
            if result and result.success and result.data:
                dock_state_str = result.data.get("state")
                if dock_state_str:
                    dock_state = QByteArray.fromBase64(dock_state_str.encode('utf-8'))
                    success = self.dock_manager.restoreState(dock_state)
                    if success:
                        Log.info("WorkspaceManager: CDockManager state restored")
                    else:
                        Log.warning("WorkspaceManager: CDockManager restoreState returned False")
                else:
                    return False
            else:
                return False

            result = self.facade.get_ui_state(self.UI_STATE_DOCK_INTERNAL, None)
            if result and result.success and result.data:
                for dock_id, state in result.data.items():
                    if dock_id in self._docks:
                        dock = self._docks[dock_id]
                        widget = dock.widget()
                        if isinstance(dock, IStatefulWindow):
                            try:
                                dock.restore_internal_state(state)
                            except Exception as e:
                                Log.warning(f"Failed to restore dock state for {dock_id}: {e}")
                        if widget and isinstance(widget, IStatefulWindow):
                            try:
                                widget.restore_internal_state(state)
                            except Exception as e:
                                Log.warning(f"Failed to restore widget state for {dock_id}: {e}")

            Log.info(f"WorkspaceManager: Restored state ({len(self._docks)} docks)")
            return True

        except Exception as e:
            Log.error(f"WorkspaceManager: Failed to restore state: {e}")
            return False

    # ======================== Open Panels ========================

    def get_open_panel_ids(self) -> List[str]:
        """Get list of block IDs for panels that were open when last saved."""
        result = self.facade.get_ui_state(self.UI_STATE_OPEN_PANELS, None)
        if result and result.success and result.data:
            panels = result.data.get("panels", [])
            return [p["block_id"] for p in panels if isinstance(p, dict) and "block_id" in p]
        return []

    # ======================== Session State ========================

    def save_session(self, key: str, value: Any) -> None:
        """Save a session value (zoom, scroll, viewport, etc.)."""
        try:
            if not self.facade.get_current_project_id():
                return
            result = self.facade.get_ui_state(self.UI_STATE_SESSION, None)
            session = result.data if (result and result.success and result.data) else {}
            session[key] = value
            self.facade.set_ui_state(self.UI_STATE_SESSION, None, session)
        except Exception as e:
            Log.warning(f"WorkspaceManager: Failed to save session key '{key}': {e}")

    def get_session(self, key: str, default: Any = None) -> Any:
        """Get a session value."""
        try:
            if not self.facade.get_current_project_id():
                return default
            result = self.facade.get_ui_state(self.UI_STATE_SESSION, None)
            if result and result.success and result.data:
                return result.data.get(key, default)
        except Exception:
            pass
        return default

    # ======================== Layout Perspectives ========================

    def save_preset(self, name: str) -> bool:
        """Save current layout as a named perspective."""
        try:
            self.dock_manager.addPerspective(name)
            Log.info(f"WorkspaceManager: Saved perspective '{name}'")
            return True
        except Exception as e:
            Log.error(f"WorkspaceManager: Failed to save perspective '{name}': {e}")
            return False

    def restore_preset(self, name: str) -> bool:
        """Restore a named layout perspective."""
        try:
            names = self.dock_manager.perspectiveNames()
            if name not in names:
                Log.warning(f"WorkspaceManager: Perspective '{name}' not found")
                return False
            self.dock_manager.openPerspective(name)
            Log.info(f"WorkspaceManager: Restored perspective '{name}'")
            return True
        except Exception as e:
            Log.error(f"WorkspaceManager: Failed to restore perspective '{name}': {e}")
            return False

    def list_presets(self) -> List[str]:
        """List all saved perspective names."""
        try:
            return list(self.dock_manager.perspectiveNames())
        except Exception:
            return []

    def delete_preset(self, name: str) -> bool:
        """Delete a named perspective."""
        try:
            self.dock_manager.removePerspective(name)
            Log.info(f"WorkspaceManager: Deleted perspective '{name}'")
            return True
        except Exception as e:
            Log.error(f"WorkspaceManager: Failed to delete perspective '{name}': {e}")
            return False

    # ======================== Default Layout ========================

    def apply_default_layout(self) -> None:
        """Apply the default first-launch layout.

        All docks start hidden.  ``_apply_mode_visibility`` in MainWindow
        is responsible for showing the correct docks for the active mode.
        """
        for _dock_id, dock in self._docks.items():
            dock.toggleView(False)

        Log.info("WorkspaceManager: Applied default layout (all docks hidden)")
