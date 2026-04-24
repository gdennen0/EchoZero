"""
Song drum event extraction pipeline template: LoadAudio -> SeparateAudio -> DetectOnsets -> BinaryDrumClassify.
Exists because song audio should expose one action that isolates drums before kick/snare extraction.
Registers with the pipeline registry on import.
"""

from echozero.pipelines.block_specs import BinaryDrumClassify, DetectOnsets, LoadAudio, Separator
from echozero.pipelines.params import KnobWidget, knob
from echozero.pipelines.pipeline import Pipeline
from echozero.pipelines.registry import pipeline_template


@pipeline_template(
    id="extract_song_drum_events",
    name="Extract Song Drum Events",
    description="Separate drums from song audio, detect onsets, and build kick/snare layers.",
    knobs={
        "audio_file": knob(
            "",
            label="Audio File",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".wav", ".mp3", ".flac", ".aiff"),
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
        "shifts": knob(
            1,
            label="Quality Shifts",
            min_value=0,
            max_value=10,
            step=1,
            description="More shifts = better quality, slower",
            advanced=True,
        ),
        "kick_model_path": knob(
            "",
            label="Kick Model",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".manifest.json", ".pth"),
        ),
        "snare_model_path": knob(
            "",
            label="Snare Model",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".manifest.json", ".pth"),
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
        "positive_threshold": knob(
            0.5,
            label="Positive Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        ),
    },
)
def build_extract_song_drum_events(
    audio_file="",
    model="htdemucs",
    device="auto",
    shifts=1,
    kick_model_path="",
    snare_model_path="",
    onset_threshold=0.3,
    onset_min_gap=0.04,
    onset_method="default",
    onset_backtrack=True,
    onset_timing_offset_ms=0.0,
    positive_threshold=0.5,
) -> Pipeline:
    """Build a song-audio pipeline that extracts kick/snare event layers."""
    pipeline = Pipeline("extract_song_drum_events", name="Extract Song Drum Events")
    audio = pipeline.add(
        LoadAudio(file_path=audio_file, target_sample_rate=44100),
        id="load_audio",
    )
    drums = pipeline.add(
        Separator(
            model=model,
            device=device,
            shifts=shifts,
            two_stems="drums",
            output_format="wav",
            mp3_bitrate=320,
        ),
        id="separate_drums",
        audio_in=audio.audio_out,
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
        audio_in=drums.drums_out,
    )
    classified = pipeline.add(
        BinaryDrumClassify(
            kick_model_path=kick_model_path,
            snare_model_path=snare_model_path,
            device=device,
            positive_threshold=positive_threshold,
        ),
        id="classify_drums",
        audio_in=drums.drums_out,
        events_in=onsets.events_out,
    )
    pipeline.output("classified_drums", classified.events_out)
    return pipeline
