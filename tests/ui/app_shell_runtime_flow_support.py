"""Compatibility wrapper for historical app-shell runtime-flow support imports.
Exists to keep the legacy runtime-flow support path stable while the tests live in smaller support modules.
Connects wrapper imports to the behavior-owned runtime-flow support slices.
"""

from tests.ui.app_shell_runtime_flow_audio_support import *  # noqa: F401,F403
from tests.ui.app_shell_runtime_flow_pipeline_support import *  # noqa: F401,F403
from tests.ui.app_shell_runtime_flow_project_support import *  # noqa: F401,F403
from tests.ui.app_shell_runtime_flow_settings_support import *  # noqa: F401,F403

__all__ = [name for name in globals() if name.startswith("test_")]
