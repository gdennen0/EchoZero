"""
DetectNoteContourProcessor: Extract a monophonic note contour from audio.
Exists because operators need a simple pitch-over-time lane for melodic source layers.
Used by the canonical object-action pipeline path to persist note-contour event layers.
"""

from __future__ import annotations

from collections.abc import Callable
from math import log2
from typing import NamedTuple

import numpy as np

from echozero.domain.types import AudioData, Event, EventData, Layer
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok


class PitchFrame(NamedTuple):
    """One voiced pitch estimate for a frame."""

    time_seconds: float
    frequency_hz: float


PitchTrackFn = Callable[[str, int, int, int, float, float], list[PitchFrame]]

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
NOTE_COLOR_BY_PITCH_CLASS = {
    "C": "#ff6b6b",
    "C#": "#ff8e5c",
    "D": "#ffb14f",
    "D#": "#ffd166",
    "E": "#d4e157",
    "F": "#8bd17c",
    "F#": "#4dd0a8",
    "G": "#4fc3f7",
    "G#": "#5c9cff",
    "A": "#7e8cff",
    "A#": "#b388ff",
    "B": "#f48fb1",
}


def midi_to_note_name(midi_note: int) -> str:
    """Convert a MIDI note number into note-name text."""

    octave = (int(midi_note) // 12) - 1
    return f"{NOTE_NAMES[int(midi_note) % 12]}{octave}"


def midi_to_frequency(midi_note: int) -> float:
    """Convert a MIDI note number into frequency in Hz."""

    return 440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0))


def note_color_hex(note_name: str) -> str:
    """Return a stable display color for one note name."""

    text = str(note_name or "").strip()
    if not text:
        return "#57a0ff"
    pitch_class = text[:-1] if text[-1].isdigit() else text
    return NOTE_COLOR_BY_PITCH_CLASS.get(pitch_class, "#57a0ff")


def frequency_to_midi(frequency_hz: float) -> float:
    """Convert frequency in Hz to fractional MIDI note space."""

    if frequency_hz <= 0.0:
        raise ValueError(f"frequency_hz must be > 0, got {frequency_hz}")
    return 69.0 + (12.0 * log2(float(frequency_hz) / 440.0))


def _default_pitch_track(
    file_path: str,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> list[PitchFrame]:
    """Estimate one voiced pitch contour with librosa pYIN."""

    try:
        import librosa
    except Exception as exc:  # pragma: no cover
        raise NotImplementedError("librosa is required for note contour extraction") from exc

    audio, loaded_sample_rate = librosa.load(file_path, sr=sample_rate, mono=True)
    if audio.size == 0:
        return []
    f0, voiced_flags, _voiced_probs = librosa.pyin(
        audio,
        fmin=float(min_frequency_hz),
        fmax=float(max_frequency_hz),
        sr=int(loaded_sample_rate),
        frame_length=int(frame_length),
        hop_length=int(hop_length),
    )
    times = librosa.times_like(f0, sr=int(loaded_sample_rate), hop_length=int(hop_length))
    frames: list[PitchFrame] = []
    for frequency_hz, voiced, time_seconds in zip(
        np.asarray(f0),
        np.asarray(voiced_flags),
        np.asarray(times),
        strict=False,
    ):
        if not bool(voiced) or not np.isfinite(frequency_hz) or float(frequency_hz) <= 0.0:
            continue
        frames.append(PitchFrame(float(time_seconds), float(frequency_hz)))
    return frames


class DetectNoteContourProcessor:
    """Estimate a simplified pitch contour and emit one event layer of note segments."""

    def __init__(self, pitch_track_fn: PitchTrackFn | None = None) -> None:
        self._pitch_track_fn = pitch_track_fn or _default_pitch_track

    def execute(self, block_id: str, context: ExecutionContext) -> Result[EventData]:
        """Read upstream audio, detect a note contour, and emit note-segment events."""

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="detect_note_contour",
                percent=0.0,
                message="Starting note contour extraction",
            )
        )
        audio = context.get_input(block_id, "audio_in", AudioData)
        if audio is None:
            return err(
                ExecutionError(
                    f"Block '{block_id}' has no audio input — connect an audio source to 'audio_in'"
                )
            )
        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))

        settings = block.settings
        frame_length = int(settings.get("frame_length", 4096))
        hop_length = int(settings.get("hop_length", 1024))
        min_note_midi = int(settings.get("min_note_midi", 36))
        max_note_midi = int(settings.get("max_note_midi", 72))
        min_note_length = float(settings.get("min_note_length", 0.08))

        if frame_length < 256:
            return err(ValidationError(f"frame_length must be >= 256, got {frame_length}"))
        if hop_length < 64:
            return err(ValidationError(f"hop_length must be >= 64, got {hop_length}"))
        if max_note_midi <= min_note_midi:
            return err(
                ValidationError(
                    "max_note_midi must be greater than min_note_midi, "
                    f"got {min_note_midi}..{max_note_midi}"
                )
            )
        if min_note_length < 0.0:
            return err(
                ValidationError(f"min_note_length must be >= 0.0, got {min_note_length}")
            )

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="detect_note_contour",
                percent=0.15,
                message="Estimating note contour",
            )
        )
        min_frequency_hz = midi_to_frequency(min_note_midi)
        max_frequency_hz = midi_to_frequency(max_note_midi)
        try:
            frames = self._pitch_track_fn(
                audio.file_path,
                int(audio.sample_rate),
                frame_length,
                hop_length,
                min_frequency_hz,
                max_frequency_hz,
            )
        except NotImplementedError:
            return err(
                ExecutionError(
                    f"Note contour backend not available for block '{block_id}'. Install librosa."
                )
            )
        except Exception as exc:
            return err(ExecutionError(f"Note contour extraction failed for block '{block_id}': {exc}"))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="detect_note_contour",
                percent=0.75,
                message=f"Building note contour from {len(frames)} voiced frames",
            )
        )
        segments = _segment_pitch_frames(
            frames,
            min_note_length=min_note_length,
            fallback_duration=max(0.01, float(hop_length) / max(1.0, float(audio.sample_rate))),
        )
        events: list[Event] = []
        for index, segment in enumerate(segments):
            note_name = midi_to_note_name(segment.midi_note)
            events.append(
                Event(
                    id=f"{block_id}_note_contour_{index}",
                    time=segment.start_time,
                    duration=max(0.01, segment.end_time - segment.start_time),
                    classifications={"note": note_name},
                    metadata={
                        "note": note_name,
                        "color": note_color_hex(note_name),
                        "detection": {
                            "midi_note": segment.midi_note,
                            "note_name": note_name,
                            "frequency_hz": round(segment.frequency_hz, 3),
                            "sample_count": segment.sample_count,
                        }
                    },
                    origin=block_id,
                )
            )

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="detect_note_contour",
                percent=1.0,
                message=f"Note contour complete — {len(events)} note segment(s)",
            )
        )
        return ok(
            EventData(
                layers=(
                    Layer(
                        id=f"{block_id}_note_contour",
                        name="Notes",
                        events=tuple(events),
                    ),
                )
            )
        )


class _PitchSegment(NamedTuple):
    start_time: float
    end_time: float
    midi_note: int
    frequency_hz: float
    sample_count: int


def _segment_pitch_frames(
    frames: list[PitchFrame],
    *,
    min_note_length: float,
    fallback_duration: float,
) -> list[_PitchSegment]:
    """Collapse frame-wise pitch estimates into contiguous note segments."""

    if not frames:
        return []
    ordered = sorted(frames, key=lambda frame: frame.time_seconds)
    frame_step = fallback_duration
    if len(ordered) > 1:
        deltas = [
            max(0.0, ordered[index + 1].time_seconds - ordered[index].time_seconds)
            for index in range(len(ordered) - 1)
        ]
        positive_deltas = [delta for delta in deltas if delta > 0.0]
        if positive_deltas:
            frame_step = max(fallback_duration, min(positive_deltas))

    segments: list[_PitchSegment] = []
    current_midi = int(round(frequency_to_midi(ordered[0].frequency_hz)))
    current_start = float(ordered[0].time_seconds)
    current_frequencies = [float(ordered[0].frequency_hz)]
    previous_time = float(ordered[0].time_seconds)

    def flush_segment(end_time: float) -> None:
        duration = max(frame_step, float(end_time) - current_start)
        if duration < min_note_length:
            return
        segments.append(
            _PitchSegment(
                start_time=current_start,
                end_time=float(end_time),
                midi_note=current_midi,
                frequency_hz=float(sum(current_frequencies) / max(1, len(current_frequencies))),
                sample_count=len(current_frequencies),
            )
        )

    for frame in ordered[1:]:
        frame_time = float(frame.time_seconds)
        frame_midi = int(round(frequency_to_midi(frame.frequency_hz)))
        is_gap = (frame_time - previous_time) > (frame_step * 1.75)
        if frame_midi != current_midi or is_gap:
            flush_segment(previous_time + frame_step)
            current_midi = frame_midi
            current_start = frame_time
            current_frequencies = [float(frame.frequency_hz)]
        else:
            current_frequencies.append(float(frame.frequency_hz))
        previous_time = frame_time

    flush_segment(previous_time + frame_step)
    return segments
