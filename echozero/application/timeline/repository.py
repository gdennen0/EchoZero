"""Timeline persistence boundary for the new EchoZero application layer."""

from abc import ABC, abstractmethod

from echozero.application.timeline.models import Timeline
from echozero.application.shared.ids import TimelineId


class TimelineRepository(ABC):
    @abstractmethod
    def get(self, timeline_id: TimelineId) -> Timeline:
        raise NotImplementedError

    @abstractmethod
    def save(self, timeline: Timeline) -> None:
        raise NotImplementedError
