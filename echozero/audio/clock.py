"""
Clock: Sample-accurate master clock driven by the audio callback.

Exists because everything in a DAW needs a single source of truth for time.
The audio callback advances the clock. Subscribers (UI playhead, event triggering,
OSC output, MIDI sync) read position or receive callbacks.

Inspired by: Reaper's master timeline, Ableton's global transport clock,
Logic's SPL (Sample Position Locator).

The clock is THE arbiter of time. Nothing else keeps its own counter.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


class ClockSubscriber(Protocol):
    """Anything that wants to know the current playback position.

    Called from the audio callback thread — implementations MUST be fast
    (no allocations, no locks, no I/O). If you need to do heavy work,
    copy the position atomically and process it on another thread.
    """

    def on_clock_tick(self, position_samples: int, sample_rate: int) -> None:
        """Called every audio callback with the current position."""
        ...


@dataclass
class LoopRegion:
    """Defines a loop range in samples. Both inclusive."""
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"Loop start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"Loop end must be > start, got start={self.start} end={self.end}")


class Clock:
    """Sample-accurate master clock.

    Thread safety model:
    - position is read/written atomically via a lock-free int (Python's GIL
      makes int reads/writes atomic, but we use a lock for compound operations
      like advance-and-check-loop).
    - Subscribers are called from the audio thread. Add/remove only when stopped.

    Usage:
        clock = Clock(sample_rate=44100)
        clock.add_subscriber(my_playhead)
        # Audio callback calls clock.advance(frames) every buffer
    """

    __slots__ = (
        "_position", "_sample_rate", "_subscribers",
        "_loop", "_loop_enabled", "_lock",
    )

    def __init__(self, sample_rate: int = 44100) -> None:
        self._position: int = 0
        self._sample_rate: int = sample_rate
        self._subscribers: list[ClockSubscriber] = []
        self._loop: LoopRegion | None = None
        self._loop_enabled: bool = False
        self._lock = threading.Lock()

    @property
    def position(self) -> int:
        """Current position in samples. Safe to read from any thread."""
        return self._position

    @property
    def position_seconds(self) -> float:
        """Current position in seconds."""
        return self._position / self._sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        self._sample_rate = value

    def seek(self, position_samples: int) -> None:
        """Jump to a position. Safe from any thread."""
        with self._lock:
            self._position = max(0, position_samples)

    def seek_seconds(self, seconds: float) -> None:
        """Jump to a position in seconds."""
        self.seek(int(seconds * self._sample_rate))

    def advance(self, frames: int) -> int:
        """Advance the clock by `frames` samples. Called from audio callback.

        Returns the position BEFORE the advance (the position for this buffer).
        Handles loop wrapping if a loop region is active.
        """
        with self._lock:
            read_pos = self._position
            new_pos = self._position + frames

            # Loop wrapping
            if self._loop_enabled and self._loop is not None:
                if new_pos >= self._loop.end:
                    overshoot = new_pos - self._loop.end
                    loop_len = self._loop.end - self._loop.start
                    new_pos = self._loop.start + (overshoot % loop_len)

            self._position = new_pos

        # Notify subscribers (outside lock to prevent deadlocks)
        for sub in self._subscribers:
            sub.on_clock_tick(read_pos, self._sample_rate)

        return read_pos

    def set_loop(self, start: int, end: int) -> None:
        """Set the loop region. Validates immediately."""
        region = LoopRegion(start, end)
        with self._lock:
            self._loop = region

    def set_loop_seconds(self, start: float, end: float) -> None:
        """Set loop region in seconds."""
        self.set_loop(
            int(start * self._sample_rate),
            int(end * self._sample_rate),
        )

    @property
    def loop_enabled(self) -> bool:
        return self._loop_enabled

    @loop_enabled.setter
    def loop_enabled(self, value: bool) -> None:
        with self._lock:
            self._loop_enabled = value

    @property
    def loop_region(self) -> LoopRegion | None:
        return self._loop

    def add_subscriber(self, sub: ClockSubscriber) -> None:
        """Add a clock subscriber. Only call when transport is stopped."""
        if sub not in self._subscribers:
            self._subscribers.append(sub)

    def remove_subscriber(self, sub: ClockSubscriber) -> None:
        """Remove a subscriber. Only call when transport is stopped."""
        try:
            self._subscribers.remove(sub)
        except ValueError:
            pass

    def reset(self) -> None:
        """Reset position to zero."""
        with self._lock:
            self._position = 0
