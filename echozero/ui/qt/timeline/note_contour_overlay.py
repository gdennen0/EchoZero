"""
note_contour_overlay.py: Geometry helpers for rendering pitch contour overlays.
Exists to keep contour extraction display math out of the large canvas paint mixin.
Connects note-contour event layers to a smooth waveform overlay path.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QPainterPath

from echozero.ui.FEEL import NOTE_CONTOUR_ROW_PADDING_PX
from echozero.ui.qt.timeline.blocks.waveform_lane import waveform_x_for_time


@dataclass(frozen=True, slots=True)
class NoteContourSample:
    """One note segment rendered as a point on the contour."""

    start: float
    end: float
    midi_note: int
    label: str = ""

    @property
    def center_time(self) -> float:
        return self.start + ((self.end - self.start) * 0.5)


def contour_samples_from_events(events: Iterable[Any]) -> list[NoteContourSample]:
    """Extract note contour samples from presentation events."""

    samples: list[NoteContourSample] = []
    for event in events:
        detection_metadata = getattr(event, "detection_metadata", {}) or {}
        midi_note = detection_metadata.get("midi_note")
        if not isinstance(midi_note, int):
            continue
        start = float(getattr(event, "start", 0.0))
        end = float(getattr(event, "end", start))
        samples.append(
            NoteContourSample(
                start=start,
                end=max(start, end),
                midi_note=midi_note,
                label=str(getattr(event, "label", "") or ""),
            )
        )
    return samples


def build_note_contour_path(
    samples: list[NoteContourSample],
    *,
    scroll_x: float,
    pixels_per_second: float,
    content_start_x: float,
    top: float,
    row_height: float,
) -> QPainterPath | None:
    """Build a smooth Bezier path from note contour samples."""

    if len(samples) < 2:
        return None
    midi_values = sorted({sample.midi_note for sample in samples})
    min_midi = midi_values[0]
    max_midi = midi_values[-1]
    if min_midi == max_midi:
        min_midi -= 1
        max_midi += 1

    points = [
        QPointF(
            waveform_x_for_time(
                sample.center_time,
                scroll_x=scroll_x,
                pixels_per_second=pixels_per_second,
                content_start_x=content_start_x,
            ),
            note_contour_y_for_midi(
                sample.midi_note,
                min_midi=min_midi,
                max_midi=max_midi,
                top=top,
                row_height=row_height,
            ),
        )
        for sample in samples
    ]
    path = QPainterPath(points[0])
    for current, following in zip(points, points[1:], strict=False):
        midpoint_x = current.x() + ((following.x() - current.x()) * 0.5)
        path.cubicTo(
            QPointF(midpoint_x, current.y()),
            QPointF(midpoint_x, following.y()),
            following,
        )
    return path


def note_contour_y_for_midi(
    midi_note: int,
    *,
    min_midi: int,
    max_midi: int,
    top: float,
    row_height: float,
) -> float:
    """Map MIDI note space into waveform row space."""

    inner_top = float(top) + float(NOTE_CONTOUR_ROW_PADDING_PX)
    inner_bottom = float(top) + float(row_height) - float(NOTE_CONTOUR_ROW_PADDING_PX)
    usable_height = max(1.0, inner_bottom - inner_top)
    span = max(1, int(max_midi) - int(min_midi))
    clamped = min(max(int(midi_note), int(min_midi)), int(max_midi))
    normalized = float(clamped - int(min_midi)) / float(span)
    return inner_bottom - (normalized * usable_height)
