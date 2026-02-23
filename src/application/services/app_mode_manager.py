"""
Application Mode Manager

Controls whether EchoZero runs in Production or Developer mode.

Production mode: Simplified UI, frozen block graph, template-based project.
Developer mode: Full UI, block creation, connection editing, template authoring.

The mode is determined at startup from AppSettingsManager.debug_mode and the
ECHOZERO_DEV_MODE environment variable. Components subscribe to mode_changed
to reconfigure themselves when the mode switches (requires restart for full
effect, but signal is provided for future runtime switching).
"""
import os
from enum import Enum
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal


class AppMode(Enum):
    PRODUCTION = "production"
    DEVELOPER = "developer"


class AppModeManager(QObject):
    """
    Centralized mode state with reactive signal propagation.

    Components check ``is_production`` at init time and optionally connect
    to ``mode_changed`` for runtime updates.
    """

    mode_changed = pyqtSignal(object)  # AppMode enum value

    def __init__(self, initial_mode: AppMode, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._mode = initial_mode

    @property
    def mode(self) -> AppMode:
        return self._mode

    @property
    def is_production(self) -> bool:
        return self._mode == AppMode.PRODUCTION

    @property
    def is_developer(self) -> bool:
        return self._mode == AppMode.DEVELOPER

    def switch_mode(self, mode: AppMode) -> None:
        if mode != self._mode:
            self._mode = mode
            self.mode_changed.emit(mode)

    @staticmethod
    def resolve_initial_mode(debug_mode_setting: bool) -> AppMode:
        """Determine the startup mode from settings and environment.

        Priority (highest first):
        1. ECHOZERO_DEV_MODE environment variable (any truthy value)
        2. debug_mode from AppSettingsManager (user toggled developer mode)
        3. When running from source (not frozen): always Developer
        4. When frozen (packaged): read default_mode from packaging_config.json
        5. Final fallback: Production
        """
        import sys

        env_val = os.environ.get("ECHOZERO_DEV_MODE", "").strip()
        if env_val and env_val not in ("0", "false", "no", "off"):
            return AppMode.DEVELOPER
        if debug_mode_setting:
            return AppMode.DEVELOPER

        # Running from source = developer by default
        if not getattr(sys, "frozen", False):
            return AppMode.DEVELOPER

        # Frozen (packaged) build: check build-time default
        try:
            import json
            from src.utils.paths import get_app_install_dir
            install_dir = get_app_install_dir()
            if install_dir:
                config_path = install_dir / "packaging_config.json"
                if config_path.is_file():
                    data = json.loads(config_path.read_text(encoding="utf-8"))
                    default_mode = data.get("default_mode", "").lower()
                    if default_mode == "developer":
                        return AppMode.DEVELOPER
        except Exception:
            pass

        return AppMode.PRODUCTION
