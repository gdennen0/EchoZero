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
    """Canonical app transport state including the return-to-start playback anchor."""

    is_playing: bool = False
    playhead: float = 0.0
    playback_start: float | None = None
    loop_enabled: bool = False
    loop_region: TimeRange | None = None
    preroll_enabled: bool = False
    follow_mode: FollowMode = FollowMode.CENTER

    def __post_init__(self) -> None:
        playhead = max(0.0, float(self.playhead))
        playback_start = playhead if self.playback_start is None else self.playback_start
        self.playhead = playhead
        self.playback_start = max(0.0, float(playback_start))

    @property
    def playback_start_seconds(self) -> float:
        """Return the position playback should return to when pausing."""

        if self.playback_start is None:
            return max(0.0, float(self.playhead))
        return max(0.0, float(self.playback_start))

    def set_playback_start(self, position: float) -> None:
        """Move both the visible playhead and the return-to-start anchor."""

        next_position = max(0.0, float(position))
        self.playhead = next_position
        self.playback_start = next_position

    def set_playback_home(self, position: float, *, move_playhead: bool = False) -> None:
        """Move the return-to-start anchor, optionally moving the visible playhead."""

        next_position = max(0.0, float(position))
        self.playback_start = next_position
        if move_playhead:
            self.playhead = next_position
