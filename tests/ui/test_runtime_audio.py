"""Behavior-oriented runtime-audio test wrapper.
Keeps the historical test path stable while importing the smaller runtime-audio case modules.
"""

from tests.ui.runtime_audio_controller_cases import *  # noqa: F401,F403
from tests.ui.runtime_audio_widget_cases import *  # noqa: F401,F403
