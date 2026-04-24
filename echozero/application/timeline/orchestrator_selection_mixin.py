"""Public selection/edit mixin root for the timeline orchestrator.
Exists to keep `orchestrator.py` importing one stable selection surface while the implementation stays split by concern.
Connects selection-state and event-edit helpers into the canonical timeline-orchestrator seam.
"""

from __future__ import annotations

from echozero.application.timeline.orchestrator_event_batch_mixin import (
    TimelineOrchestratorEventBatchMixin,
)

__all__ = ["TimelineOrchestratorSelectionMixin"]


class TimelineOrchestratorSelectionMixin(TimelineOrchestratorEventBatchMixin):
    """Stable public mixin seam for selection and event-edit orchestrator behavior."""

    pass
