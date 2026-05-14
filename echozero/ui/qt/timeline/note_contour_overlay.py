"""Note contour overlay drawing helpers."""

from __future__ import annotations

from PyQt6.QtGui import QPainterPath


def contour_samples_from_events(events: object) -> tuple[tuple[float, float], ...]:
    samples: list[tuple[float, float]] = []
    for event in tuple(events or ()):  # presentation events
        start = float(getattr(event, "start", getattr(event, "start_seconds", 0.0)) or 0.0)
        metadata = getattr(event, "metadata", {}) or {}
        midi = metadata.get("midi_note") if isinstance(metadata, dict) else None
        if midi is None:
            classifications = getattr(event, "classifications", {}) or {}
            midi = classifications.get("midi_note") if isinstance(classifications, dict) else None
        try:
            pitch = float(midi)
        except (TypeError, ValueError):
            pitch = 60.0
        samples.append((start, pitch))
    return tuple(sorted(samples))


def build_note_contour_path(
    samples: tuple[tuple[float, float], ...],
    *,
    scroll_x: float,
    pixels_per_second: float,
    content_start_x: float,
    top: float,
    row_height: float,
) -> QPainterPath | None:
    if len(samples) < 2:
        return None
    pitches = [pitch for _time, pitch in samples]
    low = min(pitches)
    high = max(pitches)
    span = max(1.0, high - low)
    path = QPainterPath()
    for index, (time_seconds, pitch) in enumerate(samples):
        x = content_start_x + (float(time_seconds) * pixels_per_second) - scroll_x
        normalized = (pitch - low) / span
        y = top + row_height - (normalized * row_height)
        if index == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    return path


__all__ = ["build_note_contour_path", "contour_samples_from_events"]
