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
        "onset_threshold": knob(
            0.3,
            label="Onset Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            maps_to_setting="threshold",
        ),
        "onset_min_gap": knob(
            0.04,
            label="Onset Min Gap",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            maps_to_setting="min_gap",
        ),
        "onset_method": knob(
            "default",
            label="Onset Method",
            widget=KnobWidget.DROPDOWN,
            options=("default", "hfc", "complex"),
            maps_to_setting="method",
        ),
        "onset_backtrack": knob(
            True,
            label="Onset Backtrack",
            maps_to_setting="backtrack",
        ),
        "onset_timing_offset_ms": knob(
            0.0,
            label="Onset Timing Offset (ms)",
            min_value=-100.0,
            max_value=100.0,
            step=1.0,
            maps_to_setting="timing_offset_ms",
        ),
        "classify_model_path": knob(
            "",
            label="Classification Model",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".manifest.json", ".pth"),
            description="PyTorch model for drum event classification",
            maps_to_setting="model_path",
        ),
        "classify_device": knob(
            "auto",
            label="Device",
            widget=KnobWidget.DROPDOWN,
            options=("auto", "cpu", "cuda"),
            maps_to_setting="device",
        ),
        "classify_batch_size": knob(
            32,
            label="Batch Size",
            min_value=1,
            max_value=512,
            step=1,
            maps_to_setting="batch_size",
        ),
    },
)
def build_drum_classification(
    audio_file="",
    onset_threshold=0.3,
    onset_min_gap=0.04,
    onset_method="default",
    onset_backtrack=True,
    onset_timing_offset_ms=0.0,
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
        DetectOnsets(
            threshold=onset_threshold,
            min_gap=onset_min_gap,
            method=onset_method,
            backtrack=onset_backtrack,
            timing_offset_ms=onset_timing_offset_ms,
        ),
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
        audio_in=audio.audio_out,
    )
    pipeline.output("drum_classified_events", classified.events_out)
    return pipeline
