"""Behavior-oriented persistence test wrapper.
Keeps the historical test path stable while importing the smaller persistence case modules.
"""

from tests.persistence_core_cases import *  # noqa: F401,F403
from tests.persistence_integrity_cases import *  # noqa: F401,F403
from tests.persistence_layers_cases import *  # noqa: F401,F403
from tests.persistence_roundtrip_cases import *  # noqa: F401,F403
