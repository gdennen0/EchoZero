"""
Transport: Play/pause/stop/seek state machine.

Exists because transport state is more than a boolean — it's a state machine with
rules about valid transitions. Separating it from the engine keeps the callback
lean and makes testing trivial.

Inspired by: Every DAW transport bar. Reaper's 5-state model simplified to 3
(STOPPED, PLAYING, PAUSED) because EchoZero doesn't need RECORDING or FAST_FORWARD.
"""

from __future__ import annotations

from enum import Enum, auto

from echozero.audio.clock import Clock


class TransportState(Enum):
    """Playback state. Three states, clean transitions."""
    STOPPED = auto()   # position = 0 or last seek, not outputting audio
    PLAYING = auto()   # outputting audio, clock advancing
    PAUSED = auto()    # position held, not outputting audio


class Transport:
    """Transport controls wrapping the master clock.

    State machine:
        STOPPED → play() → PLAYING
        PLAYING → pause() → PAUSED
        PLAYING → stop() → STOPPED (resets to 0 or last-seek)
        PAUSED  → play() → PLAYING (resume)
        PAUSED  → stop() → STOPPED (resets to 0)
        STOPPED → stop() → no-op
        PLAYING → play() → no-op

    Seek is valid in any state.
    """

    __slots__ = ("_clock", "_state", "_stop_position")

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._state: TransportState = TransportState.STOPPED
        self._stop_position: int = 0  # where to return on stop

    @property
    def state(self) -> TransportState:
        return self._state

    @property
    def is_playing(self) -> bool:
        return self._state == TransportState.PLAYING

    @property
    def is_paused(self) -> bool:
        return self._state == TransportState.PAUSED

    @property
    def is_stopped(self) -> bool:
        return self._state == TransportState.STOPPED

    def play(self) -> None:
        """Start or resume playback."""
        if self._state == TransportState.PLAYING:
            return
        self._state = TransportState.PLAYING

    def pause(self) -> None:
        """Pause playback, holding current position."""
        if self._state == TransportState.PLAYING:
            self._state = TransportState.PAUSED

    def stop(self) -> None:
        """Stop playback, return to start or last-set stop position."""
        if self._state == TransportState.STOPPED:
            return
        self._state = TransportState.STOPPED
        self._clock.seek(self._stop_position)

    def seek(self, position_samples: int) -> None:
        """Jump to position. Valid in any state."""
        self._clock.seek(position_samples)
        if self._state == TransportState.STOPPED:
            self._stop_position = max(0, position_samples)

    def seek_seconds(self, seconds: float) -> None:
        """Jump to position in seconds."""
        self.seek(int(seconds * self._clock.sample_rate))

    def toggle_play_pause(self) -> None:
        """Toggle between play and pause. From stopped, starts playback."""
        if self._state == TransportState.PLAYING:
            self.pause()
        else:
            self.play()

    def return_to_start(self) -> None:
        """Return to position 0 regardless of state."""
        self._stop_position = 0
        self._clock.seek(0)
        if self._state != TransportState.PLAYING:
            self._state = TransportState.STOPPED
