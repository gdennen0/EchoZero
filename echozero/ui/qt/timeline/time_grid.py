from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil, floor


class TimelineGridMode(str, Enum):
    OFF = "off"
    AUTO = "auto"
    BEAT = "beat"


@dataclass(frozen=True, slots=True)
class GridLine:
    time_seconds: float
    role: str = "minor"


@dataclass(frozen=True, slots=True)
class SnapResolution:
    time_seconds: float
    kind: str


_AUTO_GRID_STEPS_SECONDS = (
    1.0 / 30.0,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
)

_BEAT_GRID_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)


def grid_step_seconds(
    *,
    pixels_per_second: float,
    mode: TimelineGridMode | str,
    bpm: float | None = None,
    min_spacing_px: float = 60.0,
) -> float | None:
    normalized_mode = TimelineGridMode(str(mode))
    if normalized_mode is TimelineGridMode.OFF:
        return None

    pps = max(1.0, float(pixels_per_second))
    if normalized_mode is TimelineGridMode.BEAT and bpm and bpm > 0:
        beat_seconds = 60.0 / float(bpm)
        for multiplier in _BEAT_GRID_MULTIPLIERS:
            candidate = beat_seconds * multiplier
            if candidate * pps >= min_spacing_px:
                return candidate
        return beat_seconds * _BEAT_GRID_MULTIPLIERS[-1]

    for candidate in _AUTO_GRID_STEPS_SECONDS:
        if candidate * pps >= min_spacing_px:
            return candidate
    return _AUTO_GRID_STEPS_SECONDS[-1]


def visible_grid_lines(
    *,
    scroll_x: float,
    pixels_per_second: float,
    content_width: float,
    mode: TimelineGridMode | str,
    bpm: float | None = None,
    min_spacing_px: float = 60.0,
) -> list[GridLine]:
    step = grid_step_seconds(
        pixels_per_second=pixels_per_second,
        mode=mode,
        bpm=bpm,
        min_spacing_px=min_spacing_px,
    )
    if step is None:
        return []

    pps = max(1.0, float(pixels_per_second))
    start_second = max(0.0, float(scroll_x) / pps)
    end_second = max(start_second, (float(scroll_x) + max(1.0, float(content_width))) / pps)
    start_index = max(0, int(floor(start_second / step)) - 1)
    end_index = int(ceil(end_second / step)) + 1

    normalized_mode = TimelineGridMode(str(mode))
    lines: list[GridLine] = []
    for index in range(start_index, max(start_index, end_index) + 1):
        time_seconds = float(index) * step
        role = "minor"
        if normalized_mode is TimelineGridMode.BEAT and bpm and bpm > 0:
            beat_seconds = 60.0 / float(bpm)
            beats = round(time_seconds / beat_seconds) if beat_seconds else 0
            if abs(time_seconds - (beats * beat_seconds)) <= 1e-6:
                role = "bar" if beats % 4 == 0 else "beat"
        elif index % 4 == 0:
            role = "major"
        lines.append(GridLine(time_seconds=time_seconds, role=role))
    return lines


def resolve_snap_time(
    time_seconds: float,
    *,
    pixels_per_second: float,
    mode: TimelineGridMode | str,
    bpm: float | None = None,
    threshold_px: float,
    event_times: tuple[float, ...] = (),
    playhead_time: float | None = None,
    min_spacing_px: float = 60.0,
) -> SnapResolution | None:
    pps = max(1.0, float(pixels_per_second))
    threshold_seconds = max(0.0, float(threshold_px) / pps)
    if threshold_seconds <= 0.0:
        return None

    candidates: list[SnapResolution] = []
    step = grid_step_seconds(
        pixels_per_second=pps,
        mode=mode,
        bpm=bpm,
        min_spacing_px=min_spacing_px,
    )
    if step is not None and step > 0.0:
        snapped = round(float(time_seconds) / step) * step
        candidates.append(SnapResolution(time_seconds=max(0.0, snapped), kind="grid"))

    for event_time in event_times:
        candidates.append(SnapResolution(time_seconds=max(0.0, float(event_time)), kind="event"))

    if playhead_time is not None:
        candidates.append(SnapResolution(time_seconds=max(0.0, float(playhead_time)), kind="playhead"))

    best_candidate: SnapResolution | None = None
    best_distance = threshold_seconds + 1e-9
    for candidate in candidates:
        distance = abs(candidate.time_seconds - float(time_seconds))
        if distance > threshold_seconds:
            continue
        if distance < best_distance:
            best_candidate = candidate
            best_distance = distance
    return best_candidate
