"""Drum classification pipeline template."""

from echozero.pipelines.block_specs import Classify, DetectOnsets, LoadAudio
from echozero.pipelines.params import KnobWidget, knob
from echozero.pipelines.pipeline import Pipeline
from echozero.pipelines.registry import pipeline_template


@pipeline_template(
    id="drum_classification",
    name="Drum Classification",
    description="Detect onsets in drum audio and classify the resulting events.",
    knobs={
        "audio_file": knob(
            "",
            label="Audio File",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".wav", ".mp3", ".flac", ".aiff"),
        ),
        "classify_model_path": knob(
            "",
            label="Classification Model",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".pth",),
            description="PyTorch model for drum event classification",
        ),
        "classify_device": knob(
            "auto",
            label="Device",
            widget=KnobWidget.DROPDOWN,
            options=("auto", "cpu", "cuda"),
        ),
        "classify_batch_size": knob(
            32,
            label="Batch Size",
            min_value=1,
            max_value=512,
            step=1,
        ),
    },
)
def build_drum_classification(
    audio_file="",
    classify_model_path="",
    classify_device="auto",
    classify_batch_size=32,
) -> Pipeline:
    """Build a LoadAudio -> DetectOnsets -> Classify pipeline."""
    pipeline = Pipeline("drum_classification", name="Drum Classification")
    audio = pipeline.add(
        LoadAudio(file_path=audio_file, target_sample_rate=44100),
        id="load_audio",
    )
    onsets = pipeline.add(
        DetectOnsets(),
        id="detect_onsets",
        audio_in=audio.audio_out,
    )
    classified = pipeline.add(
        Classify(
            model_path=classify_model_path,
            device=classify_device,
            batch_size=classify_batch_size,
        ),
        id="classify",
        events_in=onsets.events_out,
    )
    pipeline.output("drum_classified_events", classified.events_out)
    return pipeline
