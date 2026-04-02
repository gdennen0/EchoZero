"""Simple query path for timeline presentation assembly."""

from dataclasses import dataclass

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.models import Timeline


@dataclass(slots=True)
class TimelineQueries:
    """Read-side access to assembled timeline presentation."""

    assembler: TimelineAssembler

    def get_presentation(self, timeline: Timeline, session: Session) -> TimelinePresentation:
        return self.assembler.assemble(timeline=timeline, session=session)
