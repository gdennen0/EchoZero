"""In-memory timeline repository implementation for the new application architecture."""

from echozero.application.shared.ids import TimelineId
from echozero.application.timeline.models import Timeline
from echozero.application.timeline.repository import TimelineRepository


class InMemoryTimelineRepository(TimelineRepository):
    def __init__(self) -> None:
        self._timelines: dict[TimelineId, Timeline] = {}

    def get(self, timeline_id: TimelineId) -> Timeline:
        return self._timelines[timeline_id]

    def save(self, timeline: Timeline) -> None:
        self._timelines[timeline.id] = timeline
