"""MA3 event adapter for EchoZero automator plans.
Exists so generated automator events can use the existing direct MA3 event writer.
Connects clean-sheet automator intent to MA3EventSnapshot without cursor side effects.
"""

from __future__ import annotations

from echozero.application.ma3_show.models import AutomatorEvent
from echozero.infrastructure.sync.ma3_adapter import MA3EventSnapshot


def automator_event_to_ma3_snapshot(event: AutomatorEvent) -> MA3EventSnapshot:
    """Convert one automator event into the existing MA3 direct event DTO."""

    return MA3EventSnapshot(
        event_id=event.id,
        label=event.label,
        start=event.start_seconds,
        end=event.start_seconds,
        cmd=event.command,
        cue_ref=event.cue_ref,
        notes=str(event.metadata.get("notes") or ""),
        payload_ref=f"echozero://automator-events/{event.id}",
    )
