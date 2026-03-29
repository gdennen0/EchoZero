"""
TranscribeNotesProcessor: ML-based note transcription from audio.
Exists because note extraction (pitch, duration, velocity) is essential for
melodic analysis — especially bass lines and harmonic content.
Used by ExecutionEngine when running blocks of type 'TranscribeNotes'.

Supports two backends via dependency injection:
- basic_pitch (Spotify's deep learning model) — default, best quality
- librosa (pyin) — lightweight fallback

The injectable function pattern allows testing without ML dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, NamedTuple

from echozero.domain.types import AudioData, Event, EventData, Layer
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok


# ---------------------------------------------------------------------------
# Note result type
# ---------------------------------------------------------------------------

class NoteInfo(NamedTuple):
    """A single transcribed note."""

    start_time: float
    end_time: float
    midi_note: int
    velocity: int


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI number to note name (e.g. 60 → C4)."""
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def midi_to_frequency(midi: int) -> float:
    """Convert MIDI number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


# ---------------------------------------------------------------------------
# Transcription function signature for DI
# ---------------------------------------------------------------------------

TranscribeFn = Callable[
    [
        str,    # file_path
        int,    # sample_rate
        float,  # onset_threshold
        float,  # frame_threshold
        float,  # min_note_length (seconds)
        float,  # min_frequency (Hz)
        float,  # max_frequency (Hz)
    ],
    list[NoteInfo],
]


def _default_transcribe(
    file_path: str,
    sample_rate: int,
    onset_threshold: float,
    frame_threshold: float,
    min_note_length: float,
    min_frequency: float,
    max_frequency: float,
) -> list[NoteInfo]:
    """Transcribe notes using basic_pitch. Requires basic-pitch installed."""
    raise NotImplementedError(
        "Default transcription requires basic-pitch. Provide a custom transcribe_fn."
    )


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class TranscribeNotesProcessor:
    """Transcribes audio into note events with pitch, duration, and velocity."""

    def __init__(self, transcribe_fn: TranscribeFn | None = None) -> None:
        self._transcribe_fn = transcribe_fn or _default_transcribe

    def execute(self, block_id: str, context: ExecutionContext) -> Result[EventData]:
        """Read upstream audio, transcribe notes, return EventData."""
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="transcribe_notes",
                percent=0.0,
                message="Starting note transcription",
            )
        )

        # Read audio input
        audio = context.get_input(block_id, "audio_in", AudioData)
        if audio is None:
            return err(
                ExecutionError(
                    f"Block '{block_id}' has no audio input — "
                    f"connect an audio source to 'audio_in'"
                )
            )

        # Read settings
        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))

        settings = block.settings
        onset_threshold = settings.get("onset_threshold", 0.5)
        frame_threshold = settings.get("frame_threshold", 0.3)
        min_note_length = settings.get("min_note_length", 0.058)
        min_frequency = settings.get("min_frequency", 27.5)
        max_frequency = settings.get("max_frequency", 4186.0)

        # Validate
        if not isinstance(onset_threshold, (int, float)) or not (0.0 <= onset_threshold <= 1.0):
            return err(ValidationError(
                f"onset_threshold must be between 0.0 and 1.0, got {onset_threshold}"
            ))
        if not isinstance(frame_threshold, (int, float)) or not (0.0 <= frame_threshold <= 1.0):
            return err(ValidationError(
                f"frame_threshold must be between 0.0 and 1.0, got {frame_threshold}"
            ))
        if not isinstance(min_frequency, (int, float)) or min_frequency < 1.0:
            return err(ValidationError(
                f"min_frequency must be >= 1.0 Hz, got {min_frequency}"
            ))
        if not isinstance(max_frequency, (int, float)) or max_frequency <= min_frequency:
            return err(ValidationError(
                f"max_frequency must be > min_frequency, got {max_frequency}"
            ))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="transcribe_notes",
                percent=0.1,
                message="Running note transcription model",
            )
        )

        # Run transcription
        try:
            notes = self._transcribe_fn(
                file_path=audio.file_path,
                sample_rate=audio.sample_rate,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                min_note_length=min_note_length,
                min_frequency=min_frequency,
                max_frequency=max_frequency,
            )
        except NotImplementedError:
            return err(ExecutionError(
                f"Note transcription backend not available for block '{block_id}'. "
                f"Install basic-pitch or provide a custom transcribe_fn."
            ))
        except Exception as exc:
            return err(ExecutionError(
                f"Note transcription failed for block '{block_id}': {exc}"
            ))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="transcribe_notes",
                percent=0.8,
                message=f"Building events from {len(notes)} notes",
            )
        )

        # Group notes by note name → one layer per pitch
        notes_by_name: dict[str, list[NoteInfo]] = defaultdict(list)
        for note in notes:
            name = midi_to_note_name(note.midi_note)
            notes_by_name[name].append(note)

        # Build layers
        layers: list[Layer] = []
        event_counter = 0
        for note_name in sorted(notes_by_name.keys()):
            note_list = notes_by_name[note_name]
            events: list[Event] = []
            for note in note_list:
                events.append(Event(
                    id=f"{block_id}_note_{event_counter}",
                    time=note.start_time,
                    duration=note.end_time - note.start_time,
                    classifications={"note": note_name},
                    metadata={
                        "midi_note": note.midi_note,
                        "velocity": note.velocity,
                        "frequency_hz": midi_to_frequency(note.midi_note),
                        "origin_block": block_id,
                    },
                    origin=block_id,
                ))
                event_counter += 1
            layers.append(Layer(
                id=f"{block_id}_{note_name}",
                name=note_name,
                events=tuple(events),
            ))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="transcribe_notes",
                percent=1.0,
                message=f"Transcription complete — {event_counter} notes in {len(layers)} layers",
            )
        )

        return ok(EventData(layers=tuple(layers)))


