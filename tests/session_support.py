"""Compatibility wrapper for historical session support imports.
Exists to keep the legacy session support path stable while the cases live in smaller support modules.
Connects wrapper imports to the shared fixtures and behavior-owned session slices.
"""

from tests.session_dirty_create_support import *  # noqa: F401,F403
from tests.session_edge_cases_support import *  # noqa: F401,F403
from tests.session_lifecycle_support import *  # noqa: F401,F403
from tests.session_save_graph_support import *  # noqa: F401,F403
from tests.session_shared_support import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("__")]
