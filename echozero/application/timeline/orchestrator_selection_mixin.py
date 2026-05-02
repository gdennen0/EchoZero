"""Legacy selection support shim for timeline application owners.
Exists to keep the old mixin import surface stable while `TimelineMutator` owns selection truth.
Connects compatibility imports to the lower-level selection and event-batch helpers only.
"""

from __future__ import annotations

from echozero.application.timeline.orchestrator_event_batch_mixin import (
    TimelineOrchestratorEventBatchMixin,
)

__all__ = ["TimelineOrchestratorSelectionMixin"]


class TimelineOrchestratorSelectionMixin(TimelineOrchestratorEventBatchMixin):
    """Compatibility shim for the old selection mixin root."""

    pass
