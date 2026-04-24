"""Public run-mixin root for the Foundry window.
Exists to keep `main_window.py` importing a stable run surface while the implementation stays split by concern.
Connects run-tab builders, run/artifact actions, and summary formatting into one mixin seam.
"""

from __future__ import annotations

from echozero.foundry.ui.main_window_run_actions_mixin import (
    _FoundryWindowRunActionsMixin,
)
from echozero.foundry.ui.main_window_run_build_mixin import (
    _FoundryWindowRunBuildMixin,
)
from echozero.foundry.ui.main_window_run_summary_mixin import (
    _FoundryWindowRunSummaryMixin,
)


class FoundryWindowRunMixin(
    _FoundryWindowRunBuildMixin,
    _FoundryWindowRunActionsMixin,
    _FoundryWindowRunSummaryMixin,
):
    """Stable Foundry run/artifact mixin root for the main window shell."""
