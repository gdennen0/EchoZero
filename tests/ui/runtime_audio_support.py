"""Compatibility wrapper for historical runtime-audio support imports.
Exists to keep the legacy runtime-audio support path stable while the tests live in smaller support modules.
Connects wrapper imports to the behavior-owned controller and widget support slices.
"""

from tests.ui.runtime_audio_controller_support import *  # noqa: F401,F403
from tests.ui.runtime_audio_widget_support import *  # noqa: F401,F403

__all__ = [name for name in globals() if name.startswith("test_")]
