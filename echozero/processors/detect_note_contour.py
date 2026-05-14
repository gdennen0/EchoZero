"""Detect note-contour processor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

from echozero.domain.types import AudioData, Event, EventData, Layer
from echozero.errors import ExecutionError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok


@dataclass(frozen=True, slots=True)
class PitchFrame:
    time: float
    frequency_hz: float


def _midi_from_frequency(frequency_hz: float) -> int:
    if frequency_hz <= 0:
        return 0
    return int(round(69 + 12 * math.log2(float(frequency_hz) / 440.0)))


def _note_label(midi_note: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    octave = midi_note // 12 - 1
    return f"{names[midi_note % 12]}{octave}"


class DetectNoteContourProcessor:
    """Detect monophonic note regions and return them as timeline events."""

    def __init__(self, pitch_track_fn: Callable[..., list[PitchFrame]] | None = None) -> None:
        self._pitch_track_fn = pitch_track_fn or self._default_pitch_track

    def execute(self, block_id: str, context: ExecutionContext) -> Result[EventData]:
        context.progress_bus.publish(
            ProgressReport(block_id=block_id, phase="detect_note_contour", percent=0.0)
        )
        audio = context.get_input(block_id, "audio_in", AudioData)
        if audio is None:
            return err(ExecutionError(f"Block '{block_id}' has no audio input"))
        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))
        min_note_duration = float(block.settings.get("min_note_duration", 0.05))
        try:
            frames = list(self._pitch_track_fn(audio.file_path, audio.sample_rate, block.settings))
        except TypeError:
            frames = list(self._pitch_track_fn(audio.file_path, audio.sample_rate))
        except Exception as exc:
            return err(ExecutionError(f"Note contour detection failed for block '{block_id}': {exc}"))
        events = _events_from_pitch_frames(
            frames,
            block_id=block_id,
            min_note_duration=min_note_duration,
        )
        context.progress_bus.publish(
            ProgressReport(block_id=block_id, phase="detect_note_contour", percent=1.0)
        )
        return ok(EventData(layers=(Layer(id=f"{block_id}_notes", name="Notes", events=tuple(events)),)))

    @staticmethod
    def _default_pitch_track(*_args: Any, **_kwargs: Any) -> list[PitchFrame]:
        raise NotImplementedError("Default note contour detection requires an injected pitch tracker")


def _events_from_pitch_frames(
    frames: list[PitchFrame],
    *,
    block_id: str,
    min_note_duration: float,
) -> list[Event]:
    if not frames:
        return []
    ordered = sorted(frames, key=lambda frame: float(frame.time))
    events: list[Event] = []
    current_midi: int | None = None
    current_start = 0.0
    last_time = float(ordered[0].time)
    for frame in ordered:
        midi = _midi_from_frequency(float(frame.frequency_hz))
        time = float(frame.time)
        if current_midi is None:
            current_midi = midi
            current_start = time
        elif midi != current_midi:
            _append_note_event(events, block_id, current_midi, current_start, time, min_note_duration)
            current_midi = midi
            current_start = time
        last_time = time
    frame_step = 0.05 if len(ordered) < 2 else max(0.01, float(ordered[-1].time) - float(ordered[-2].time))
    _append_note_event(events, block_id, current_midi or 0, current_start, last_time + frame_step, min_note_duration)
    return events


def _append_note_event(
    events: list[Event],
    block_id: str,
    midi_note: int,
    start: float,
    end: float,
    min_note_duration: float,
) -> None:
    duration = max(0.0, float(end) - float(start))
    if duration < min_note_duration:
        return
    events.append(
        Event(
            id=f"{block_id}_note_{len(events)}",
            time=float(start),
            duration=duration,
            classifications={"note": _note_label(midi_note)},
            metadata={"midi_note": midi_note, "frequency_note": _note_label(midi_note)},
            origin=block_id,
        )
    )


__all__ = ["DetectNoteContourProcessor", "PitchFrame"]
