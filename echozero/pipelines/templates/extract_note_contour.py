"""
Note contour extraction pipeline template: LoadAudio -> DetectNoteContour.
Exists to provide a simple pitch-over-time action for melodic and bass-oriented source audio.
Registers with the pipeline registry so the canonical object-action path can execute it.
"""

from echozero.pipelines.block_specs import DetectNoteContour, LoadAudio
from echozero.pipelines.params import knob
from echozero.pipelines.pipeline import Pipeline
from echozero.pipelines.registry import pipeline_template


@pipeline_template(
    id="extract_note_contour",
    name="Extract Notes",
    description="Estimate the current note contour over time and emit a simple notes layer.",
    knobs={
        "audio_file": knob(""),
        "min_note_midi": knob(36, label="Min Note (MIDI)", min_value=12, max_value=96, step=1),
        "max_note_midi": knob(72, label="Max Note (MIDI)", min_value=24, max_value=108, step=1),
        "frame_length": knob(
            4096,
            label="Frame Length",
            min_value=256,
            max_value=16384,
            step=1,
            advanced=True,
        ),
        "hop_length": knob(
            1024,
            label="Hop Length",
            min_value=64,
            max_value=8192,
            step=1,
            advanced=True,
        ),
        "min_note_length": knob(
            0.08,
            label="Min Segment Length (s)",
            min_value=0.01,
            max_value=2.0,
            step=0.01,
        ),
    },
)
def build_extract_note_contour(
    audio_file="",
    min_note_midi=36,
    max_note_midi=72,
    frame_length=4096,
    hop_length=1024,
    min_note_length=0.08,
) -> Pipeline:
    """Build a LoadAudio -> DetectNoteContour pipeline."""

    pipeline = Pipeline("extract_note_contour", name="Extract Notes")
    load_audio = pipeline.add(
        LoadAudio(file_path=audio_file, target_sample_rate=44100),
        id="load_audio",
    )
    detect_note_contour = pipeline.add(
        DetectNoteContour(
            min_note_midi=min_note_midi,
            max_note_midi=max_note_midi,
            frame_length=frame_length,
            hop_length=hop_length,
            min_note_length=min_note_length,
        ),
        id="detect_note_contour",
        audio_in=load_audio.audio_out,
    )
    pipeline.output("Notes", detect_note_contour.events_out)
    return pipeline
