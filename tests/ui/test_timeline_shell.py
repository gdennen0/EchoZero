"""Behavior-oriented timeline-shell test wrapper.
Keeps the historical test path stable while importing the smaller timeline-shell case modules.
"""

from tests.ui.timeline_shell_support import _selection_test_presentation
from tests.ui.timeline_shell_contract_actions_cases import *  # noqa: F401,F403
from tests.ui.timeline_shell_interactions_cases import *  # noqa: F401,F403
from tests.ui.timeline_shell_layout_cases import *  # noqa: F401,F403
from tests.ui.timeline_shell_object_info_cases import *  # noqa: F401,F403
from tests.ui.timeline_shell_transfer_cases import *  # noqa: F401,F403
