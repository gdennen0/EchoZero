"""
Note-contour extraction pipeline template.
Exists to turn one audio layer into a notes/event layer from monophonic pitch tracking.
Registers with the pipeline registry on import.
"""

from __future__ import annotations

from echozero.pipelines.block_specs import DetectNoteContour, LoadAudio
from echozero.pipelines.params import KnobWidget, knob
from echozero.pipelines.pipeline import PersistenceMapping, Pipeline
from echozero.pipelines.registry import pipeline_template


@pipeline_template(
    id="extract_note_contour",
    name="Extract Notes",
    description="Detect a monophonic note contour and persist it as timeline events.",
    knobs={
        "audio_file": knob(
            "",
            label="Audio File",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".wav", ".mp3", ".flac", ".aiff"),
        ),
        "sample_rate": knob(
            22050,
            label="Sample Rate",
            min_value=8000,
            max_value=96000,
            step=1,
            advanced=True,
        ),
        "frame_length": knob(
            2048,
            label="Frame Length",
            min_value=256,
            max_value=8192,
            step=256,
            advanced=True,
        ),
        "hop_length": knob(
            512,
            label="Hop Length",
            min_value=64,
            max_value=4096,
            step=64,
            advanced=True,
        ),
        "min_note_duration": knob(
            0.05,
            label="Minimum Note Duration",
            min_value=0.0,
            max_value=2.0,
            step=0.01,
        ),
    },
)
def build_extract_note_contour(
    audio_file: str = "",
    sample_rate: int = 22050,
    frame_length: int = 2048,
    hop_length: int = 512,
    min_note_duration: float = 0.05,
) -> Pipeline:
    """Build a LoadAudio -> DetectNoteContour pipeline."""

    pipeline = Pipeline("extract_note_contour", name="Extract Notes")
    audio = pipeline.add(
        LoadAudio(file_path=audio_file, target_sample_rate=sample_rate),
        id="load_audio",
    )
    notes = pipeline.add(
        DetectNoteContour(
            sample_rate=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            min_note_duration=min_note_duration,
        ),
        id="detect_note_contour",
        audio_in=audio.audio_out,
    )
    pipeline.output(
        "Notes",
        notes.events_out,
        data_type="event",
        persistence=PersistenceMapping(target="layer_take"),
    )
    return pipeline
