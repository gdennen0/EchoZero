"""Transport application models.
Exists to keep transport state and external command data typed at the app boundary.
Connects UI, OSC, and MA3 controls to one canonical playback contract.
"""

from dataclasses import dataclass
from enum import StrEnum

from echozero.application.shared.enums import FollowMode
from echozero.application.shared.ranges import TimeRange


class ExternalTransportAction(StrEnum):
    """Canonical actions accepted from external transport controllers."""

    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    TOGGLE = "toggle"
    SEEK = "seek"
    MOVE = "move"
    JUMP_PREVIOUS_SECTION = "jump_previous_section"
    JUMP_NEXT_SECTION = "jump_next_section"


@dataclass(frozen=True, slots=True)
class ExternalTransportCommand:
    """Normalized external transport request at the EchoZero application boundary."""

    action: ExternalTransportAction
    position_seconds: float | None = None
    delta_seconds: float | None = None
    source: str | None = None
    request_id: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(slots=True)
class TransportState:
    is_playing: bool = False
    playhead: float = 0.0
    loop_enabled: bool = False
    loop_region: TimeRange | None = None
    preroll_enabled: bool = False
    follow_mode: FollowMode = FollowMode.CENTER
