"""
Song section extraction pipeline template: LoadAudio -> DetectSongSections.
Exists to auto-generate section cue layers from source song audio with MFCC sequence pooling.
Registers with the pipeline registry on import.
"""

from echozero.pipelines.block_specs import DetectSongSections, LoadAudio
from echozero.pipelines.params import KnobWidget, knob
from echozero.pipelines.pipeline import Pipeline
from echozero.pipelines.registry import pipeline_template


@pipeline_template(
    id="extract_song_sections",
    name="Extract Song Sections",
    description=(
        "Auto-generate section cues from song audio using MFCC features with "
        "sequence-style temporal pooling inspired by MusicSegmentationML."
    ),
    knobs={
        "sample_rate": knob(
            22050,
            label="Sample Rate",
            min_value=8000,
            max_value=96000,
            step=1,
            advanced=True,
        ),
        "n_mfcc": knob(
            20,
            label="MFCC Features",
            min_value=8,
            max_value=64,
            step=1,
        ),
        "n_fft": knob(
            8192,
            label="FFT Window",
            min_value=512,
            max_value=32768,
            step=1,
            advanced=True,
        ),
        "hop_length": knob(
            4096,
            label="Hop Length",
            min_value=64,
            max_value=16384,
            step=1,
            advanced=True,
        ),
        "history_pool_frames": knob(
            160,
            label="History Pool Frames",
            min_value=8,
            max_value=1000,
            step=1,
            description=(
                "How many MFCC frames are pooled into each sequential decision window. "
                "Inspired by the article's pooled sequence input approach."
            ),
        ),
        "boundary_sensitivity": knob(
            0.60,
            label="Boundary Sensitivity",
            widget=KnobWidget.SLIDER,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description="Higher values detect more section boundaries.",
        ),
        "min_section_seconds": knob(
            8.0,
            label="Minimum Section Length (s)",
            min_value=1.0,
            max_value=60.0,
            step=0.5,
        ),
        "max_sections": knob(
            14,
            label="Max Sections",
            min_value=2,
            max_value=40,
            step=1,
            advanced=True,
        ),
        "similarity_threshold": knob(
            0.84,
            label="Repeat Similarity",
            widget=KnobWidget.SLIDER,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            advanced=True,
        ),
        "intro_tail_seconds": knob(
            14.0,
            label="Intro Cap (s)",
            min_value=2.0,
            max_value=45.0,
            step=0.5,
            advanced=True,
        ),
        "end_tail_seconds": knob(
            16.0,
            label="End Tail (s)",
            min_value=2.0,
            max_value=45.0,
            step=0.5,
            advanced=True,
        ),
    },
)
def build_extract_song_sections(
    sample_rate=22050,
    n_mfcc=20,
    n_fft=8192,
    hop_length=4096,
    history_pool_frames=160,
    boundary_sensitivity=0.60,
    min_section_seconds=8.0,
    max_sections=14,
    similarity_threshold=0.84,
    intro_tail_seconds=14.0,
    end_tail_seconds=16.0,
):
    """Build a LoadAudio -> DetectSongSections pipeline."""

    pipeline = Pipeline("extract_song_sections", name="Extract Song Sections")

    load = pipeline.add(LoadAudio(), id="load_audio")
    detect_sections = pipeline.add(
        DetectSongSections(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            history_pool_frames=history_pool_frames,
            boundary_sensitivity=boundary_sensitivity,
            min_section_seconds=min_section_seconds,
            max_sections=max_sections,
            similarity_threshold=similarity_threshold,
            intro_tail_seconds=intro_tail_seconds,
            end_tail_seconds=end_tail_seconds,
        ),
        audio_in=load.audio_out,
        id="detect_song_sections",
    )

    pipeline.output("sections", detect_sections.events_out)

    return pipeline
