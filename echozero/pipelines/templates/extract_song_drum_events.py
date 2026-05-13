"""
Song drum event extraction pipeline template: LoadAudio -> SeparateAudio -> DetectOnsets -> BinaryDrumClassify.
Exists because song audio should expose one action that isolates drums before per-class extraction.
Registers with the pipeline registry on import.
"""

from echozero.application.event_flows.drum_events import apply_drum_event_sensitivity_preset
from echozero.pipelines.block_specs import (
    AudioFilter,
    BinaryDrumClassify,
    DetectOnsets,
    LoadAudio,
    Separator,
)
from echozero.pipelines.params import KnobWidget, knob
from echozero.pipelines.pipeline import Pipeline
from echozero.pipelines.registry import pipeline_template


@pipeline_template(
    id="extract_song_drum_events",
    name="Extract Song Drum Events",
    description="Separate drums from song audio, detect onsets, and build selected drum event layers.",
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
        "include_drums_stem_layer": knob(
            False,
            label="Add Drums Stem Layer",
            widget=KnobWidget.TOGGLE,
            description="Add the separated drums stem as an audio layer in the timeline.",
        ),
        "include_bass_stem_layer": knob(
            False,
            label="Add Bass Stem Layer",
            widget=KnobWidget.TOGGLE,
            description="Add the separated bass stem as an audio layer in the timeline.",
        ),
        "include_vocals_stem_layer": knob(
            False,
            label="Add Vocals Stem Layer",
            widget=KnobWidget.TOGGLE,
            description="Add the separated vocals stem as an audio layer in the timeline.",
        ),
        "include_other_stem_layer": knob(
            False,
            label="Add Other Stem Layer",
            widget=KnobWidget.TOGGLE,
            description="Add the separated other stem as an audio layer in the timeline.",
        ),
        "target_drum_labels": knob(
            ("kick", "snare"),
            label="Drum Outputs",
            widget=KnobWidget.MULTI_SELECT,
            options=("kick", "snare", "clap", "cymbal"),
            description=(
                "Choose which installed drum-model outputs to build. Installed bundles show up "
                "as selectable targets in the action settings surface."
            ),
            maps_to_block="classify_drums",
            maps_to_setting="target_labels",
        ),
        "sensitivity_preset": knob(
            "balanced",
            label="Sensitivity",
            widget=KnobWidget.DROPDOWN,
            options=("more_events", "balanced", "fewer_events", "custom"),
            description=(
                "Choose how eagerly EchoZero finds drum events. Custom leaves advanced "
                "threshold and onset controls untouched."
            ),
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
        "clap_model_path": knob(
            "",
            label="Clap Model",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".manifest.json", ".pth"),
            maps_to_block="classify_drums",
        ),
        "cymbal_model_path": knob(
            "",
            label="Cymbal Model",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".manifest.json", ".pth"),
            maps_to_block="classify_drums",
        ),
        "kick_positive_threshold": knob(
            0.50,
            label="Kick Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description=(
                "Stage 2 (classification): minimum kick model confidence required to"
                " promote a detected candidate. Lower-scoring candidates still persist"
                " as demoted reviewable events."
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
                "Stage 2 (classification): minimum snare model confidence required to"
                " promote a detected candidate. Lower-scoring candidates still persist"
                " as demoted reviewable events."
            ),
            maps_to_block="classify_drums",
        ),
        "positive_threshold": knob(
            0.60,
            label="Default Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description=(
                "Fallback classification threshold for any selected drum label that does not "
                "have its own label-specific threshold override."
            ),
            maps_to_block="classify_drums",
        ),
        "clap_positive_threshold": knob(
            0.60,
            label="Clap Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description=(
                "Stage 2 (classification): minimum clap model confidence required to"
                " promote a detected candidate."
            ),
            maps_to_block="classify_drums",
        ),
        "cymbal_positive_threshold": knob(
            0.60,
            label="Cymbal Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            description=(
                "Stage 2 (classification): minimum cymbal model confidence required to"
                " promote a detected candidate."
            ),
            maps_to_block="classify_drums",
        ),
        "clap_min_event_peak": knob(
            0.0015,
            label="Clap Noise Peak Floor",
            min_value=0.0,
            max_value=0.1,
            step=0.0005,
            description="Reject clap candidates whose source window never reaches this peak level.",
            maps_to_block="classify_drums",
        ),
        "clap_min_event_rms": knob(
            0.0003,
            label="Clap Noise RMS Floor",
            min_value=0.0,
            max_value=0.05,
            step=0.0001,
            description="Reject clap candidates whose source window energy is below this RMS level.",
            maps_to_block="classify_drums",
        ),
        "clap_min_separation_ms": knob(
            55.0,
            label="Clap Dedup Window (ms)",
            min_value=0.0,
            max_value=500.0,
            step=1.0,
            description="Demote duplicate clap hits that land within this window of a stronger hit.",
            maps_to_block="classify_drums",
        ),
        "cymbal_min_event_peak": knob(
            0.0008,
            label="Cymbal Noise Peak Floor",
            min_value=0.0,
            max_value=0.1,
            step=0.0005,
            description="Reject cymbal candidates whose source window never reaches this peak level.",
            maps_to_block="classify_drums",
        ),
        "cymbal_min_event_rms": knob(
            0.00015,
            label="Cymbal Noise RMS Floor",
            min_value=0.0,
            max_value=0.05,
            step=0.0001,
            description="Reject cymbal candidates whose source window energy is below this RMS level.",
            maps_to_block="classify_drums",
        ),
        "cymbal_min_separation_ms": knob(
            90.0,
            label="Cymbal Dedup Window (ms)",
            min_value=0.0,
            max_value=500.0,
            step=1.0,
            description="Demote duplicate cymbal hits that land within this window of a stronger hit.",
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
            220.0,
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
            900.0,
            label="Snare Filter Cutoff",
            widget=KnobWidget.FREQUENCY,
            min_value=20.0,
            max_value=8_000.0,
            step=1.0,
            maps_to_block="snare_filter",
            maps_to_setting="freq",
        ),
        "snare_onset_threshold": knob(
            0.30,
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
        "clap_filter_enabled": knob(
            True,
            label="Clap Filter Enabled",
            widget=KnobWidget.TOGGLE,
            maps_to_block="clap_filter",
            maps_to_setting="enabled",
        ),
        "clap_filter_freq": knob(
            1_200.0,
            label="Clap Filter Center",
            widget=KnobWidget.FREQUENCY,
            min_value=100.0,
            max_value=8_000.0,
            step=10.0,
            maps_to_block="clap_filter",
            maps_to_setting="freq",
        ),
        "clap_onset_threshold": knob(
            0.35,
            label="Clap Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            description="Stage 1 (detection): onset sensitivity for clap candidates before classification.",
            maps_to_block="clap_onsets",
            maps_to_setting="threshold",
        ),
        "cymbal_filter_enabled": knob(
            True,
            label="Cymbal Filter Enabled",
            widget=KnobWidget.TOGGLE,
            maps_to_block="cymbal_filter",
            maps_to_setting="enabled",
        ),
        "cymbal_filter_freq": knob(
            3_200.0,
            label="Cymbal Filter Cutoff",
            widget=KnobWidget.FREQUENCY,
            min_value=200.0,
            max_value=16_000.0,
            step=10.0,
            maps_to_block="cymbal_filter",
            maps_to_setting="freq",
        ),
        "cymbal_onset_threshold": knob(
            0.40,
            label="Cymbal Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            description="Stage 1 (detection): onset sensitivity for cymbal candidates before classification.",
            maps_to_block="cymbal_onsets",
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
            "hfc",
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
        "clap_filter_type": knob(
            "bandpass",
            label="Clap Filter",
            widget=KnobWidget.DROPDOWN,
            options=("lowpass", "highpass", "bandpass"),
            advanced=True,
            maps_to_block="clap_filter",
            maps_to_setting="filter_type",
        ),
        "clap_filter_q": knob(
            1.2,
            label="Clap Filter Q",
            min_value=0.2,
            max_value=10.0,
            step=0.1,
            advanced=True,
            maps_to_block="clap_filter",
            maps_to_setting="Q",
        ),
        "clap_onset_min_gap": knob(
            0.055,
            label="Clap Onset Min Gap",
            min_value=0.0,
            max_value=1.0,
            step=0.005,
            advanced=True,
            maps_to_block="clap_onsets",
            maps_to_setting="min_gap",
        ),
        "clap_onset_method": knob(
            "hfc",
            label="Clap Onset Method",
            widget=KnobWidget.DROPDOWN,
            options=("default", "hfc", "complex"),
            advanced=True,
            maps_to_block="clap_onsets",
            maps_to_setting="method",
        ),
        "clap_onset_backtrack": knob(
            True,
            label="Clap Onset Backtrack",
            advanced=True,
            maps_to_block="clap_onsets",
            maps_to_setting="backtrack",
        ),
        "clap_onset_timing_offset_ms": knob(
            0.0,
            label="Clap Onset Timing Offset (ms)",
            min_value=-100.0,
            max_value=100.0,
            step=1.0,
            advanced=True,
            maps_to_block="clap_onsets",
            maps_to_setting="timing_offset_ms",
        ),
        "cymbal_filter_type": knob(
            "highpass",
            label="Cymbal Filter",
            widget=KnobWidget.DROPDOWN,
            options=("lowpass", "highpass", "bandpass"),
            advanced=True,
            maps_to_block="cymbal_filter",
            maps_to_setting="filter_type",
        ),
        "cymbal_filter_q": knob(
            1.0,
            label="Cymbal Filter Q",
            min_value=0.2,
            max_value=10.0,
            step=0.1,
            advanced=True,
            maps_to_block="cymbal_filter",
            maps_to_setting="Q",
        ),
        "cymbal_onset_min_gap": knob(
            0.09,
            label="Cymbal Onset Min Gap",
            min_value=0.0,
            max_value=1.0,
            step=0.005,
            advanced=True,
            maps_to_block="cymbal_onsets",
            maps_to_setting="min_gap",
        ),
        "cymbal_onset_method": knob(
            "hfc",
            label="Cymbal Onset Method",
            widget=KnobWidget.DROPDOWN,
            options=("default", "hfc", "complex"),
            advanced=True,
            maps_to_block="cymbal_onsets",
            maps_to_setting="method",
        ),
        "cymbal_onset_backtrack": knob(
            True,
            label="Cymbal Onset Backtrack",
            advanced=True,
            maps_to_block="cymbal_onsets",
            maps_to_setting="backtrack",
        ),
        "cymbal_onset_timing_offset_ms": knob(
            0.0,
            label="Cymbal Onset Timing Offset (ms)",
            min_value=-100.0,
            max_value=100.0,
            step=1.0,
            advanced=True,
            maps_to_block="cymbal_onsets",
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
def build_extract_song_drum_events(
    audio_file="",
    model="htdemucs",
    device="auto",
    shifts=1,
    target_drum_labels=("kick", "snare"),
    sensitivity_preset="balanced",
    include_drums_stem_layer=False,
    include_bass_stem_layer=False,
    include_vocals_stem_layer=False,
    include_other_stem_layer=False,
    kick_model_path="",
    snare_model_path="",
    clap_model_path="",
    cymbal_model_path="",
    kick_positive_threshold=0.50,
    snare_positive_threshold=0.65,
    positive_threshold=0.60,
    clap_positive_threshold=0.60,
    cymbal_positive_threshold=0.60,
    clap_min_event_peak=0.0015,
    clap_min_event_rms=0.0003,
    clap_min_separation_ms=55.0,
    cymbal_min_event_peak=0.0008,
    cymbal_min_event_rms=0.00015,
    cymbal_min_separation_ms=90.0,
    kick_filter_enabled=True,
    kick_filter_freq=220.0,
    kick_onset_threshold=0.25,
    snare_filter_enabled=True,
    snare_filter_freq=900.0,
    snare_onset_threshold=0.30,
    clap_filter_enabled=True,
    clap_filter_freq=1_200.0,
    clap_onset_threshold=0.35,
    cymbal_filter_enabled=True,
    cymbal_filter_freq=3_200.0,
    cymbal_onset_threshold=0.40,
    kick_filter_type="lowpass",
    kick_onset_min_gap=0.08,
    kick_onset_method="default",
    kick_onset_backtrack=True,
    kick_onset_timing_offset_ms=0.0,
    snare_filter_type="highpass",
    snare_onset_min_gap=0.05,
    snare_onset_method="hfc",
    snare_onset_backtrack=True,
    snare_onset_timing_offset_ms=0.0,
    clap_filter_type="bandpass",
    clap_filter_q=1.2,
    clap_onset_min_gap=0.055,
    clap_onset_method="hfc",
    clap_onset_backtrack=True,
    clap_onset_timing_offset_ms=0.0,
    cymbal_filter_type="highpass",
    cymbal_filter_q=1.0,
    cymbal_onset_min_gap=0.09,
    cymbal_onset_method="hfc",
    cymbal_onset_backtrack=True,
    cymbal_onset_timing_offset_ms=0.0,
    assignment_mode="independent",
    winner_margin=0.05,
    event_match_window_ms=40.0,
) -> Pipeline:
    """Build a song-audio pipeline that extracts selected drum event layers."""
    if isinstance(target_drum_labels, str):
        target_labels = tuple(
            _normalize_drum_label(label)
            for label in target_drum_labels.split(",")
            if label.strip()
        )
    else:
        target_labels = tuple(
            _normalize_drum_label(label)
            for label in target_drum_labels
            if str(label).strip()
        )
    if not target_labels:
        raise ValueError("Select at least one drum label to extract.")
    compiled_values = apply_drum_event_sensitivity_preset(
        {
            "positive_threshold": positive_threshold,
            "kick_positive_threshold": kick_positive_threshold,
            "snare_positive_threshold": snare_positive_threshold,
            "clap_positive_threshold": clap_positive_threshold,
            "cymbal_positive_threshold": cymbal_positive_threshold,
            "kick_onset_threshold": kick_onset_threshold,
            "snare_onset_threshold": snare_onset_threshold,
            "clap_onset_threshold": clap_onset_threshold,
            "cymbal_onset_threshold": cymbal_onset_threshold,
        },
        sensitivity=sensitivity_preset,
    )
    positive_threshold = compiled_values["positive_threshold"]
    kick_positive_threshold = compiled_values["kick_positive_threshold"]
    snare_positive_threshold = compiled_values["snare_positive_threshold"]
    clap_positive_threshold = compiled_values["clap_positive_threshold"]
    cymbal_positive_threshold = compiled_values["cymbal_positive_threshold"]
    kick_onset_threshold = compiled_values["kick_onset_threshold"]
    snare_onset_threshold = compiled_values["snare_onset_threshold"]
    clap_onset_threshold = compiled_values["clap_onset_threshold"]
    cymbal_onset_threshold = compiled_values["cymbal_onset_threshold"]
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
            include_drums_stem_layer=include_drums_stem_layer,
            include_bass_stem_layer=include_bass_stem_layer,
            include_vocals_stem_layer=include_vocals_stem_layer,
            include_other_stem_layer=include_other_stem_layer,
            output_format="wav",
            mp3_bitrate=320,
        ),
        id="separate_drums",
        audio_in=audio.audio_out,
    )
    kick_filter = pipeline.add(
        AudioFilter(
            enabled=kick_filter_enabled,
            filter_type=kick_filter_type,
            freq=kick_filter_freq,
        ),
        id="kick_filter",
        audio_in=drums.drums_out,
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
        audio_in=drums.drums_out,
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
    clap_filter = pipeline.add(
        AudioFilter(
            enabled=clap_filter_enabled,
            filter_type=clap_filter_type,
            freq=clap_filter_freq,
            Q=clap_filter_q,
        ),
        id="clap_filter",
        audio_in=drums.drums_out,
    )
    clap_onsets = pipeline.add(
        DetectOnsets(
            threshold=clap_onset_threshold,
            min_gap=clap_onset_min_gap,
            method=clap_onset_method,
            backtrack=clap_onset_backtrack,
            timing_offset_ms=clap_onset_timing_offset_ms,
        ),
        id="clap_onsets",
        audio_in=clap_filter.audio_out,
    )
    cymbal_filter = pipeline.add(
        AudioFilter(
            enabled=cymbal_filter_enabled,
            filter_type=cymbal_filter_type,
            freq=cymbal_filter_freq,
            Q=cymbal_filter_q,
        ),
        id="cymbal_filter",
        audio_in=drums.drums_out,
    )
    cymbal_onsets = pipeline.add(
        DetectOnsets(
            threshold=cymbal_onset_threshold,
            min_gap=cymbal_onset_min_gap,
            method=cymbal_onset_method,
            backtrack=cymbal_onset_backtrack,
            timing_offset_ms=cymbal_onset_timing_offset_ms,
        ),
        id="cymbal_onsets",
        audio_in=cymbal_filter.audio_out,
    )
    classified = pipeline.add(
        BinaryDrumClassify(
            target_labels=target_labels,
            kick_model_path=kick_model_path,
            snare_model_path=snare_model_path,
            clap_model_path=clap_model_path,
            cymbal_model_path=cymbal_model_path,
            device=device,
            positive_threshold=positive_threshold,
            kick_positive_threshold=kick_positive_threshold,
            snare_positive_threshold=snare_positive_threshold,
            clap_positive_threshold=clap_positive_threshold,
            cymbal_positive_threshold=cymbal_positive_threshold,
            clap_min_event_peak=clap_min_event_peak,
            clap_min_event_rms=clap_min_event_rms,
            clap_min_separation_ms=clap_min_separation_ms,
            cymbal_min_event_peak=cymbal_min_event_peak,
            cymbal_min_event_rms=cymbal_min_event_rms,
            cymbal_min_separation_ms=cymbal_min_separation_ms,
            assignment_mode=assignment_mode,
            winner_margin=winner_margin,
            event_match_window_ms=event_match_window_ms,
        ),
        id="classify_drums",
        audio_in=drums.drums_out,
        events_in=snare_onsets.events_out,
        clap_audio_in=clap_filter.audio_out,
        cymbal_audio_in=cymbal_filter.audio_out,
        kick_events_in=kick_onsets.events_out,
        snare_events_in=snare_onsets.events_out,
        clap_events_in=clap_onsets.events_out,
        cymbal_events_in=cymbal_onsets.events_out,
    )
    pipeline.output("drums", drums.drums_out)
    pipeline.output("bass", drums.bass_out)
    pipeline.output("vocals", drums.vocals_out)
    pipeline.output("other", drums.other_out)
    pipeline.output("classified_drums", classified.events_out)
    return pipeline


def _normalize_drum_label(raw_label: object) -> str:
    """Normalize operator-facing percussion aliases to runtime bundle labels."""
    label = str(raw_label).strip().lower()
    if label in {"symbol", "cymbol"}:
        return "cymbal"
    return label
