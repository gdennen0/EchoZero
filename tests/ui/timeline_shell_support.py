"""Compatibility wrapper for historical timeline-shell support imports.
Exists to keep the legacy support path stable while the test cases live in smaller support modules.
Connects wrapper imports to the shared fixtures and behavior-owned support slices.
"""

from tests.ui.timeline_shell_contract_actions_support import *  # noqa: F401,F403
from tests.ui.timeline_shell_interactions_support import *  # noqa: F401,F403
from tests.ui.timeline_shell_layout_support import *  # noqa: F401,F403
from tests.ui.timeline_shell_object_info_support import *  # noqa: F401,F403
from tests.ui.timeline_shell_shared_support import *  # noqa: F401,F403
from tests.ui.timeline_shell_transfer_support import *  # noqa: F401,F403

__all__ = [
    name
    for name in globals()
    if name.startswith("test_") or name == "_selection_test_presentation"
]
