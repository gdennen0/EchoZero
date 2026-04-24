"""Behavior-oriented app-shell runtime-flow test wrapper.
Keeps the historical test path stable while importing the smaller runtime-flow case modules.
"""

from tests.ui.app_shell_runtime_flow_audio_cases import *  # noqa: F401,F403
from tests.ui.app_shell_runtime_flow_pipeline_cases import *  # noqa: F401,F403
from tests.ui.app_shell_runtime_flow_project_cases import *  # noqa: F401,F403
from tests.ui.app_shell_runtime_flow_settings_cases import *  # noqa: F401,F403
