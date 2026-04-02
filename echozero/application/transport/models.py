"""Transport application models."""

from dataclasses import dataclass

from echozero.application.shared.enums import FollowMode
from echozero.application.shared.ranges import TimeRange


@dataclass(slots=True)
class TransportState:
    is_playing: bool = False
    playhead: float = 0.0
    loop_enabled: bool = False
    loop_region: TimeRange | None = None
    preroll_enabled: bool = False
    follow_mode: FollowMode = FollowMode.PAGE
