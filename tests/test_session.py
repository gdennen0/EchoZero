"""Behavior-oriented session test wrapper.
Keeps the historical test path stable while importing the smaller session case modules.
"""

from tests.session_dirty_and_create_cases import *  # noqa: F401,F403
from tests.session_edge_cases_cases import *  # noqa: F401,F403
from tests.session_lifecycle_cases import *  # noqa: F401,F403
from tests.session_save_and_graph_cases import *  # noqa: F401,F403
