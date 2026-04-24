"""Compatibility wrapper for historical persistence support imports.
Exists to keep the legacy persistence support path stable while the cases live in smaller support modules.
Connects wrapper imports to the shared fixture seam and behavior-owned persistence slices.
"""

from tests.persistence_core_support import *  # noqa: F401,F403
from tests.persistence_integrity_support import *  # noqa: F401,F403
from tests.persistence_layers_support import *  # noqa: F401,F403
from tests.persistence_roundtrip_support import *  # noqa: F401,F403
from tests.persistence_shared_support import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("__")]
