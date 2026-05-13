"""
Compact drum-event flow contract tests.
Exists to prove user-facing event type and sensitivity controls compile to pipeline knobs.
Connects action settings UX to existing extract-drum pipeline parameters without persistence side effects.
"""

from __future__ import annotations

from types import SimpleNamespace

from echozero.application.event_flows.drum_event_extraction_v2 import (
    DrumEventExtractionRequest,
    build_drum_event_extraction_result,
    build_drum_event_layer_take_drafts,
)
from echozero.application.event_flows.drum_events import (
    apply_drum_event_sensitivity_preset,
    compile_drum_event_sensitivity_knobs,
    model_readiness_from_fields,
    normalize_drum_event_labels,
)


def test_drum_event_label_normalization_dedupes_cymbal_aliases() -> None:
    """Common cymbal aliases normalize before event selection reaches the pipeline."""

    assert normalize_drum_event_labels(("kick", "cymbol", "symbol", "clap")) == (
        "kick",
        "cymbal",
        "clap",
    )
    request = DrumEventExtractionRequest(
        labels=("cymbol", "symbol", "clap"),
        source_audio_by_label={"symbol": "cymbal.wav"},
    )
    assert request.labels == ("cymbal", "clap")
    assert request.source_audio_by_label["cymbal"] == "cymbal.wav"


def test_v2_clap_and_cymbal_lanes_do_not_fall_back_to_kick_inputs() -> None:
    """V2 lanes stay empty/missing instead of borrowing kick candidates or sources."""

    kick_candidate = SimpleNamespace(id="kick-1")
    result = build_drum_event_extraction_result(
        DrumEventExtractionRequest(labels=("clap", "cymbal")),
        candidate_events_by_label={"kick": (kick_candidate,)},
        source_audio_by_label={"kick": "kick.wav"},
        audition_source_by_label={"kick": "kick-preview.wav"},
    )

    clap_lane = result.lane_for("clap")
    cymbal_lane = result.lane_for("cymbal")
    assert clap_lane is not None
    assert cymbal_lane is not None
    assert clap_lane.candidate_events == ()
    assert clap_lane.audition_source_ref is None
    assert cymbal_lane.candidate_events == ()
    assert cymbal_lane.source_audio_ref is None


def test_v2_layer_take_drafts_preserve_per_label_audition_sources() -> None:
    """Persistence/playback drafts use the lane source for each selected label."""

    clap_event = SimpleNamespace(id="clap-1")
    cymbal_event = SimpleNamespace(id="cymbal-1")
    result = build_drum_event_extraction_result(
        DrumEventExtractionRequest(
            labels=("clap", "cymbol"),
            source_audio_by_label={"clap": "clap.wav", "cymbol": "cymbal.wav"},
        ),
        promoted_events_by_label={"clap": (clap_event,), "cymbal": (cymbal_event,)},
    )

    drafts = {draft.label: draft for draft in build_drum_event_layer_take_drafts(result)}
    assert drafts["clap"].events == (clap_event,)
    assert drafts["clap"].playback_source_ref == "clap.wav"
    assert drafts["cymbal"].events == (cymbal_event,)
    assert drafts["cymbal"].playback_source_ref == "cymbal.wav"


def test_sensitivity_preset_compiles_to_existing_detection_and_classification_knobs() -> None:
    """Compact sensitivity maps to conservative onset and classifier thresholds."""

    more_events = compile_drum_event_sensitivity_knobs("more_events")
    balanced = compile_drum_event_sensitivity_knobs("balanced")
    fewer_events = compile_drum_event_sensitivity_knobs("fewer_events")

    assert more_events["kick_onset_threshold"] < balanced["kick_onset_threshold"]
    assert more_events["kick_positive_threshold"] < balanced["kick_positive_threshold"]
    assert fewer_events["cymbal_onset_threshold"] > balanced["cymbal_onset_threshold"]
    assert fewer_events["snare_positive_threshold"] > balanced["snare_positive_threshold"]
    assert compile_drum_event_sensitivity_knobs("custom") == {}


def test_balanced_sensitivity_preserves_intentional_raw_custom_values() -> None:
    """Legacy/custom raw settings are not stomped by the default compact preset."""

    values = {
        "kick_onset_threshold": 0.25,
        "kick_positive_threshold": 0.50,
        "snare_positive_threshold": 0.65,
    }

    assert apply_drum_event_sensitivity_preset(values, sensitivity="balanced") == values
    assert apply_drum_event_sensitivity_preset(values, sensitivity="fewer_events")[
        "kick_onset_threshold"
    ] == 0.22


def test_model_readiness_summarizes_ready_missing_and_select_states() -> None:
    """Model status can be shown compactly without exposing paths as primary controls."""

    fields = (
        SimpleNamespace(
            key="kick_model_path",
            value="/models/kick.manifest.json",
            enabled=True,
            options=(SimpleNamespace(metadata={"status": "ready"}),),
        ),
        SimpleNamespace(
            key="snare_model_path",
            value="",
            enabled=True,
            options=(SimpleNamespace(metadata={"status": "ready"}),),
        ),
        SimpleNamespace(key="clap_model_path", value="", enabled=False, options=()),
    )

    readiness = {item.label: item for item in model_readiness_from_fields(fields)}

    assert readiness["kick"].status == "ready"
    assert readiness["snare"].status == "select"
    assert readiness["clap"].status == "missing"
