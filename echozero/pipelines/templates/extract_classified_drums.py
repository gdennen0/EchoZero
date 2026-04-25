"""Extract classified drums pipeline template."""

from echozero.pipelines.block_specs import AudioFilter, BinaryDrumClassify, DetectOnsets, LoadAudio
from echozero.pipelines.params import KnobWidget, knob
from echozero.pipelines.pipeline import Pipeline
from echozero.pipelines.registry import pipeline_template


@pipeline_template(
    id="extract_classified_drums",
    name="Extract Classified Drums",
    description="Detect drum stem onsets and classify them into kick/snare layers.",
    knobs={
        "audio_file": knob(
            "",
            label="Audio File",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".wav", ".mp3", ".flac", ".aiff"),
        ),
        "kick_model_path": knob(
            "",
            label="Kick Model",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".manifest.json", ".pth"),
            maps_to_block="classify_drums",
        ),
        "snare_model_path": knob(
            "",
            label="Snare Model",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".manifest.json", ".pth"),
            maps_to_block="classify_drums",
        ),
        "classify_device": knob(
            "auto",
            label="Device",
            widget=KnobWidget.DROPDOWN,
            options=("auto", "cpu", "cuda"),
            maps_to_block="classify_drums",
            maps_to_setting="device",
        ),
        "kick_positive_threshold": knob(
            0.50,
            label="Kick Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description=(
                "Stage 2 (classification): minimum kick model confidence required to keep"
                " a detected candidate as a kick event."
            ),
            maps_to_block="classify_drums",
        ),
        "snare_positive_threshold": knob(
            0.65,
            label="Snare Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description=(
                "Stage 2 (classification): minimum snare model confidence required to keep"
                " a detected candidate as a snare event."
            ),
            maps_to_block="classify_drums",
        ),
        "kick_filter_enabled": knob(
            True,
            label="Kick Filter Enabled",
            widget=KnobWidget.TOGGLE,
            maps_to_block="kick_filter",
            maps_to_setting="enabled",
        ),
        "kick_filter_freq": knob(
            180.0,
            label="Kick Filter Cutoff",
            widget=KnobWidget.FREQUENCY,
            min_value=20.0,
            max_value=4_000.0,
            step=1.0,
            maps_to_block="kick_filter",
            maps_to_setting="freq",
        ),
        "kick_onset_threshold": knob(
            0.25,
            label="Kick Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            description=(
                "Stage 1 (detection): onset sensitivity for kick candidates before"
                " classification. Lower values create more candidate events."
            ),
            maps_to_block="kick_onsets",
            maps_to_setting="threshold",
        ),
        "snare_filter_enabled": knob(
            True,
            label="Snare Filter Enabled",
            widget=KnobWidget.TOGGLE,
            maps_to_block="snare_filter",
            maps_to_setting="enabled",
        ),
        "snare_filter_freq": knob(
            180.0,
            label="Snare Filter Cutoff",
            widget=KnobWidget.FREQUENCY,
            min_value=20.0,
            max_value=8_000.0,
            step=1.0,
            maps_to_block="snare_filter",
            maps_to_setting="freq",
        ),
        "snare_onset_threshold": knob(
            0.3,
            label="Snare Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            description=(
                "Stage 1 (detection): onset sensitivity for snare candidates before"
                " classification. Lower values create more candidate events."
            ),
            maps_to_block="snare_onsets",
            maps_to_setting="threshold",
        ),
        "kick_filter_type": knob(
            "lowpass",
            label="Kick Filter",
            widget=KnobWidget.DROPDOWN,
            options=("lowpass", "highpass", "bandpass"),
            advanced=True,
            maps_to_block="kick_filter",
            maps_to_setting="filter_type",
        ),
        "kick_onset_min_gap": knob(
            0.08,
            label="Kick Onset Min Gap",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            advanced=True,
            maps_to_block="kick_onsets",
            maps_to_setting="min_gap",
        ),
        "kick_onset_method": knob(
            "default",
            label="Kick Onset Method",
            widget=KnobWidget.DROPDOWN,
            options=("default", "hfc", "complex"),
            advanced=True,
            maps_to_block="kick_onsets",
            maps_to_setting="method",
        ),
        "kick_onset_backtrack": knob(
            True,
            label="Kick Onset Backtrack",
            advanced=True,
            maps_to_block="kick_onsets",
            maps_to_setting="backtrack",
        ),
        "kick_onset_timing_offset_ms": knob(
            0.0,
            label="Kick Onset Timing Offset (ms)",
            min_value=-100.0,
            max_value=100.0,
            step=1.0,
            advanced=True,
            maps_to_block="kick_onsets",
            maps_to_setting="timing_offset_ms",
        ),
        "snare_filter_type": knob(
            "highpass",
            label="Snare Filter",
            widget=KnobWidget.DROPDOWN,
            options=("lowpass", "highpass", "bandpass"),
            advanced=True,
            maps_to_block="snare_filter",
            maps_to_setting="filter_type",
        ),
        "snare_onset_min_gap": knob(
            0.05,
            label="Snare Onset Min Gap",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            advanced=True,
            maps_to_block="snare_onsets",
            maps_to_setting="min_gap",
        ),
        "snare_onset_method": knob(
            "default",
            label="Snare Onset Method",
            widget=KnobWidget.DROPDOWN,
            options=("default", "hfc", "complex"),
            advanced=True,
            maps_to_block="snare_onsets",
            maps_to_setting="method",
        ),
        "snare_onset_backtrack": knob(
            True,
            label="Snare Onset Backtrack",
            advanced=True,
            maps_to_block="snare_onsets",
            maps_to_setting="backtrack",
        ),
        "snare_onset_timing_offset_ms": knob(
            0.0,
            label="Snare Onset Timing Offset (ms)",
            min_value=-100.0,
            max_value=100.0,
            step=1.0,
            advanced=True,
            maps_to_block="snare_onsets",
            maps_to_setting="timing_offset_ms",
        ),
        "assignment_mode": knob(
            "independent",
            label="Assignment Mode",
            widget=KnobWidget.DROPDOWN,
            options=("independent", "exclusive_max"),
            advanced=True,
            maps_to_block="classify_drums",
        ),
        "winner_margin": knob(
            0.05,
            label="Winner Margin",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            advanced=True,
            maps_to_block="classify_drums",
        ),
        "event_match_window_ms": knob(
            40.0,
            label="Match Window (ms)",
            min_value=0.0,
            max_value=200.0,
            step=1.0,
            advanced=True,
            maps_to_block="classify_drums",
        ),
    },
)
def build_extract_classified_drums(
    audio_file="",
    kick_model_path="",
    snare_model_path="",
    classify_device="auto",
    kick_positive_threshold=0.50,
    snare_positive_threshold=0.65,
    kick_filter_enabled=True,
    kick_filter_freq=180.0,
    kick_onset_threshold=0.25,
    snare_filter_enabled=True,
    snare_filter_freq=180.0,
    snare_onset_threshold=0.3,
    kick_filter_type="lowpass",
    kick_onset_min_gap=0.08,
    kick_onset_method="default",
    kick_onset_backtrack=True,
    kick_onset_timing_offset_ms=0.0,
    snare_filter_type="highpass",
    snare_onset_min_gap=0.05,
    snare_onset_method="default",
    snare_onset_backtrack=True,
    snare_onset_timing_offset_ms=0.0,
    assignment_mode="independent",
    winner_margin=0.05,
    event_match_window_ms=40.0,
) -> Pipeline:
    """Build split kick/snare branches over one drums source."""
    pipeline = Pipeline("extract_classified_drums", name="Extract Classified Drums")
    audio = pipeline.add(
        LoadAudio(file_path=audio_file, target_sample_rate=44100),
        id="load_audio",
    )
    kick_filter = pipeline.add(
        AudioFilter(
            enabled=kick_filter_enabled,
            filter_type=kick_filter_type,
            freq=kick_filter_freq,
        ),
        id="kick_filter",
        audio_in=audio.audio_out,
    )
    kick_onsets = pipeline.add(
        DetectOnsets(
            threshold=kick_onset_threshold,
            min_gap=kick_onset_min_gap,
            method=kick_onset_method,
            backtrack=kick_onset_backtrack,
            timing_offset_ms=kick_onset_timing_offset_ms,
        ),
        id="kick_onsets",
        audio_in=kick_filter.audio_out,
    )
    snare_filter = pipeline.add(
        AudioFilter(
            enabled=snare_filter_enabled,
            filter_type=snare_filter_type,
            freq=snare_filter_freq,
        ),
        id="snare_filter",
        audio_in=audio.audio_out,
    )
    snare_onsets = pipeline.add(
        DetectOnsets(
            threshold=snare_onset_threshold,
            min_gap=snare_onset_min_gap,
            method=snare_onset_method,
            backtrack=snare_onset_backtrack,
            timing_offset_ms=snare_onset_timing_offset_ms,
        ),
        id="snare_onsets",
        audio_in=snare_filter.audio_out,
    )
    classified = pipeline.add(
        BinaryDrumClassify(
            kick_model_path=kick_model_path,
            snare_model_path=snare_model_path,
            device=classify_device,
            kick_positive_threshold=kick_positive_threshold,
            snare_positive_threshold=snare_positive_threshold,
            assignment_mode=assignment_mode,
            winner_margin=winner_margin,
            event_match_window_ms=event_match_window_ms,
        ),
        id="classify_drums",
        audio_in=audio.audio_out,
        events_in=kick_onsets.events_out,
        kick_events_in=kick_onsets.events_out,
        snare_events_in=snare_onsets.events_out,
    )
    pipeline.output("classified_drums", classified.events_out)
    return pipeline
