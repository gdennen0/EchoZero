"""
Playback timebase and SMPTE conversion contracts.
Exists because playback, UI, export, and sync need one deterministic timecode policy.
Connects sample-accurate transport truth to canonical frame/timecode conversions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Final


_TIMECODE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(\d{2}):([0-5]\d):([0-5]\d)([:;\.])(\d{2})\s*$"
)
_DF_FPS_RATIOS: Final[set[tuple[int, int]]] = {(30000, 1001), (60000, 1001)}
_LEGACY_RATE_TOLERANCE: Final[float] = 1e-6


class TimecodeDisplayPolicy(str, Enum):
    """Supported timeline time label policies for one playback context."""

    SMPTE = "smpte"
    CLOCK = "clock"


@dataclass(frozen=True, slots=True)
class TimebaseSpec:
    """Canonical playback timebase configuration for one active playback context."""

    nominal_fps: int
    fps_numerator: int
    fps_denominator: int = 1
    drop_frame: bool = False
    start_frame_offset: int = 0
    display_policy: TimecodeDisplayPolicy = TimecodeDisplayPolicy.SMPTE

    def __post_init__(self) -> None:
        if self.nominal_fps < 1:
            raise ValueError("TimebaseSpec nominal_fps must be >= 1")
        if self.fps_numerator < 1:
            raise ValueError("TimebaseSpec fps_numerator must be >= 1")
        if self.fps_denominator < 1:
            raise ValueError("TimebaseSpec fps_denominator must be >= 1")
        if self.start_frame_offset < 0:
            raise ValueError("TimebaseSpec start_frame_offset must be >= 0")
        if self.drop_frame and (self.fps_numerator, self.fps_denominator) not in _DF_FPS_RATIOS:
            raise ValueError("TimebaseSpec drop_frame requires an SMPTE-compatible drop-frame rate")
        if self.drop_frame and self.nominal_fps not in {30, 60}:
            raise ValueError("TimebaseSpec drop_frame requires nominal_fps of 30 or 60")

    @property
    def fps(self) -> float:
        """Return effective frames-per-second for this timebase."""

        return float(self.fps_numerator) / float(self.fps_denominator)

    @classmethod
    def from_legacy_fps(
        cls,
        fps: float | int | None,
        *,
        drop_frame: bool = False,
        start_frame_offset: int = 0,
        display_policy: TimecodeDisplayPolicy = TimecodeDisplayPolicy.SMPTE,
    ) -> TimebaseSpec:
        """Build a canonical spec from legacy project/export `timecode_fps` values."""

        if fps is None:
            resolved_rate = 30.0
        else:
            resolved_rate = float(fps)
        if abs(resolved_rate - 24.0) <= _LEGACY_RATE_TOLERANCE:
            return cls(
                nominal_fps=24,
                fps_numerator=24,
                fps_denominator=1,
                drop_frame=drop_frame,
                start_frame_offset=start_frame_offset,
                display_policy=display_policy,
            )
        if abs(resolved_rate - 25.0) <= _LEGACY_RATE_TOLERANCE:
            return cls(
                nominal_fps=25,
                fps_numerator=25,
                fps_denominator=1,
                drop_frame=drop_frame,
                start_frame_offset=start_frame_offset,
                display_policy=display_policy,
            )
        if abs(resolved_rate - 29.97) <= _LEGACY_RATE_TOLERANCE:
            return cls(
                nominal_fps=30,
                fps_numerator=30000,
                fps_denominator=1001,
                drop_frame=drop_frame,
                start_frame_offset=start_frame_offset,
                display_policy=display_policy,
            )
        if abs(resolved_rate - 30.0) <= _LEGACY_RATE_TOLERANCE:
            return cls(
                nominal_fps=30,
                fps_numerator=30,
                fps_denominator=1,
                drop_frame=drop_frame,
                start_frame_offset=start_frame_offset,
                display_policy=display_policy,
            )
        raise ValueError(
            f"Unsupported timebase fps={resolved_rate}. Supported: 24, 25, 29.97, 30"
        )


@dataclass(frozen=True, slots=True)
class SmpteTimecode:
    """Parsed or computed SMPTE timecode components."""

    hours: int
    minutes: int
    seconds: int
    frames: int
    drop_frame: bool


@dataclass(frozen=True, slots=True)
class TimecodeCodec:
    """Convert between samples, seconds, frame indices, and SMPTE labels."""

    timebase: TimebaseSpec

    def samples_to_frames(self, sample_position: int, sample_rate: int) -> int:
        """Convert sample position to the nearest frame index."""

        if sample_position < 0:
            raise ValueError("samples_to_frames sample_position must be >= 0")
        if sample_rate < 1:
            raise ValueError("samples_to_frames sample_rate must be >= 1")
        numerator = sample_position * self.timebase.fps_numerator
        denominator = sample_rate * self.timebase.fps_denominator
        return int(round(float(numerator) / float(denominator)))

    def frames_to_samples(self, frame_index: int, sample_rate: int) -> int:
        """Convert frame index to the nearest sample position."""

        if frame_index < 0:
            raise ValueError("frames_to_samples frame_index must be >= 0")
        if sample_rate < 1:
            raise ValueError("frames_to_samples sample_rate must be >= 1")
        numerator = frame_index * sample_rate * self.timebase.fps_denominator
        denominator = self.timebase.fps_numerator
        return int(round(float(numerator) / float(denominator)))

    def seconds_to_frames(self, seconds: float) -> int:
        """Convert elapsed seconds to the nearest frame index."""

        if seconds < 0:
            raise ValueError("seconds_to_frames seconds must be >= 0")
        numerator = float(seconds) * float(self.timebase.fps_numerator)
        denominator = float(self.timebase.fps_denominator)
        return int(round(numerator / denominator))

    def frames_to_seconds(self, frame_index: int) -> float:
        """Convert frame index into elapsed seconds."""

        if frame_index < 0:
            raise ValueError("frames_to_seconds frame_index must be >= 0")
        numerator = frame_index * self.timebase.fps_denominator
        return float(numerator) / float(self.timebase.fps_numerator)

    def timecode_from_frames(self, frame_index: int) -> SmpteTimecode:
        """Render one SMPTE value for a frame index using this codec's timebase."""

        if frame_index < 0:
            raise ValueError("timecode_from_frames frame_index must be >= 0")
        with_offset = frame_index + self.timebase.start_frame_offset
        if self.timebase.drop_frame:
            return self._drop_frame_timecode_from_frames(with_offset)
        return self._non_drop_timecode_from_frames(with_offset)

    def frames_from_timecode(self, value: SmpteTimecode | str) -> int:
        """Resolve frame index from parsed SMPTE components or one SMPTE string."""

        timecode = self.parse_timecode(value) if isinstance(value, str) else value
        nominal = self.timebase.nominal_fps
        total_time_seconds = (timecode.hours * 3600) + (timecode.minutes * 60) + timecode.seconds
        frame_number = (total_time_seconds * nominal) + timecode.frames
        if self.timebase.drop_frame:
            frame_number -= self._dropped_frame_count(timecode.hours, timecode.minutes)
        if frame_number < self.timebase.start_frame_offset:
            raise ValueError(
                "frames_from_timecode resolved frame precedes start_frame_offset"
            )
        return frame_number - self.timebase.start_frame_offset

    def parse_timecode(self, value: str) -> SmpteTimecode:
        """Parse one SMPTE string using this codec's timebase rules."""

        match = _TIMECODE_RE.match(str(value))
        if match is None:
            raise ValueError(f"Invalid SMPTE timecode: {value!r}")
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        frames = int(match.group(5))
        nominal = self.timebase.nominal_fps
        if frames >= nominal:
            raise ValueError(
                f"Invalid SMPTE timecode frame component {frames}; must be < {nominal}"
            )
        drop_frame = self.timebase.drop_frame
        if drop_frame and seconds == 0 and (minutes % 10 != 0):
            dropped_each_minute = self._drop_frames_per_minute()
            if frames < dropped_each_minute:
                raise ValueError(
                    f"Invalid drop-frame SMPTE label {value!r}: frames 00-{dropped_each_minute - 1:02d} "
                    "are skipped at each minute boundary except every 10th minute"
                )
        return SmpteTimecode(
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            frames=frames,
            drop_frame=drop_frame,
        )

    def format_timecode(
        self,
        value: SmpteTimecode,
        *,
        frame_separator: str | None = None,
    ) -> str:
        """Format one SMPTE value as HH:MM:SS<sep>FF."""

        if frame_separator is None:
            separator = ";" if value.drop_frame else ":"
        else:
            separator = str(frame_separator)
        return (
            f"{value.hours:02d}:{value.minutes:02d}:{value.seconds:02d}"
            f"{separator}{value.frames:02d}"
        )

    def format_timecode_from_frames(
        self,
        frame_index: int,
        *,
        frame_separator: str | None = None,
    ) -> str:
        """Format SMPTE directly from a frame index."""

        return self.format_timecode(
            self.timecode_from_frames(frame_index),
            frame_separator=frame_separator,
        )

    def _non_drop_timecode_from_frames(self, frame_index: int) -> SmpteTimecode:
        nominal = self.timebase.nominal_fps
        frames_per_day = nominal * 60 * 60 * 24
        frame_number = frame_index % frames_per_day
        hours = frame_number // (nominal * 3600)
        frame_number %= nominal * 3600
        minutes = frame_number // (nominal * 60)
        frame_number %= nominal * 60
        seconds = frame_number // nominal
        frames = frame_number % nominal
        return SmpteTimecode(
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            frames=frames,
            drop_frame=False,
        )

    def _drop_frame_timecode_from_frames(self, frame_index: int) -> SmpteTimecode:
        nominal = self.timebase.nominal_fps
        dropped_each_minute = self._drop_frames_per_minute()
        frames_per_hour = (nominal * 3600) - (dropped_each_minute * 54)
        frames_per_24_hours = frames_per_hour * 24
        frames_per_10_minutes = (nominal * 600) - (dropped_each_minute * 9)
        frames_per_minute = (nominal * 60) - dropped_each_minute
        real_frame_number = frame_index % frames_per_24_hours
        tens = real_frame_number // frames_per_10_minutes
        remainder = real_frame_number % frames_per_10_minutes
        dropped = dropped_each_minute * 9 * tens
        if remainder >= dropped_each_minute:
            dropped += dropped_each_minute * (
                (remainder - dropped_each_minute) // frames_per_minute
            )
        label_frame_number = real_frame_number + dropped
        hours = label_frame_number // (nominal * 3600)
        label_frame_number %= nominal * 3600
        minutes = label_frame_number // (nominal * 60)
        label_frame_number %= nominal * 60
        seconds = label_frame_number // nominal
        frames = label_frame_number % nominal
        return SmpteTimecode(
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            frames=frames,
            drop_frame=True,
        )

    def _dropped_frame_count(self, hours: int, minutes: int) -> int:
        total_minutes = (hours * 60) + minutes
        dropped_each_minute = self._drop_frames_per_minute()
        return dropped_each_minute * (total_minutes - (total_minutes // 10))

    def _drop_frames_per_minute(self) -> int:
        return 2 if self.timebase.nominal_fps == 30 else 4


def format_clock_label(seconds: float) -> str:
    """Format a MM:SS.ss readout for non-SMPTE UI contexts."""

    clamped = max(0.0, float(seconds))
    mins = int(clamped // 60)
    secs = clamped - (mins * 60)
    return f"{mins:02d}:{secs:05.2f}"


__all__ = [
    "SmpteTimecode",
    "TimebaseSpec",
    "TimecodeCodec",
    "TimecodeDisplayPolicy",
    "format_clock_label",
]
