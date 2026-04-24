"""
Full Analysis pipeline template: LoadAudio → Separate → DetectOnsets (per stem) → Classify drums.

The flagship pipeline. Separates a mix into stems, detects onsets on each,
and classifies drum events. This is the one-click "analyze everything" workflow.
"""

from echozero.pipelines.block_specs import (
    AudioFilter,
    Classify,
    DetectOnsets,
    LoadAudio,
    Separator,
)
from echozero.pipelines.params import KnobWidget, knob
from echozero.pipelines.pipeline import Pipeline
from echozero.pipelines.registry import pipeline_template


@pipeline_template(
    id="full_analysis",
    name="Full Analysis",
    description="Separate stems, detect onsets per stem, classify drums.",
    knobs={
        "audio_file": knob(
            "",
            label="Audio File",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".wav", ".mp3", ".flac", ".aiff"),
        ),
        "threshold": knob(
            0.3,
            label="Sensitivity",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            description="Onset detection threshold (lower = more events)",
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
            advanced=True,
            maps_to_setting="timing_offset_ms",
        ),
        "model": knob(
            "htdemucs",
            label="Separation Model",
            widget=KnobWidget.DROPDOWN,
            options=("htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"),
        ),
        "device": knob(
            "auto",
            label="Device",
            widget=KnobWidget.DROPDOWN,
            options=("auto", "cpu", "cuda"),
        ),
        "classify_model_path": knob(
            "",
            label="Classification Model",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".manifest.json", ".pth"),
            description="PyTorch model for drum classification",
            advanced=True,
            maps_to_setting="model_path",
        ),
    },
)
def build_full_analysis(
    audio_file="",
    threshold=0.3,
    onset_method="default",
    onset_backtrack=True,
    onset_timing_offset_ms=0.0,
    model="htdemucs",
    device="auto",
    classify_model_path="",
) -> Pipeline:
    """Build the full analysis pipeline."""
    p = Pipeline("full_analysis", name="Full Analysis")

    load = p.add(LoadAudio(file_path=audio_file), id="load")
    sep = p.add(Separator(model=model, device=device), id="sep", audio_in=load.audio_out)

    drums_onsets = p.add(
        DetectOnsets(
            threshold=threshold,
            method=onset_method,
            backtrack=onset_backtrack,
            timing_offset_ms=onset_timing_offset_ms,
        ),
        id="drums_onsets",
        audio_in=sep.drums_out,
    )
    bass_onsets = p.add(
        DetectOnsets(
            threshold=threshold,
            method=onset_method,
            backtrack=onset_backtrack,
            timing_offset_ms=onset_timing_offset_ms,
        ),
        id="bass_onsets",
        audio_in=sep.bass_out,
    )
    vocals_onsets = p.add(
        DetectOnsets(
            threshold=threshold,
            method=onset_method,
            backtrack=onset_backtrack,
            timing_offset_ms=onset_timing_offset_ms,
        ),
        id="vocals_onsets",
        audio_in=sep.vocals_out,
    )
    other_onsets = p.add(
        DetectOnsets(
            threshold=threshold,
            method=onset_method,
            backtrack=onset_backtrack,
            timing_offset_ms=onset_timing_offset_ms,
        ),
        id="other_onsets",
        audio_in=sep.other_out,
    )

    # Classify drums if a model is provided
    if classify_model_path:
        drums_classified = p.add(
            Classify(model_path=classify_model_path, device=device),
            id="drums_classify",
            events_in=drums_onsets.events_out,
            audio_in=sep.drums_out,
        )
        p.output("drums_classified", drums_classified.events_out)
    else:
        p.output("drums_onsets", drums_onsets.events_out)

    p.output("bass_onsets", bass_onsets.events_out)
    p.output("vocals_onsets", vocals_onsets.events_out)
    p.output("other_onsets", other_onsets.events_out)

    return p
