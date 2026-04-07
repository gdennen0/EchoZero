from echozero.domain.types import Event
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.real_data_fixture import _event_label, _fixture_layer_color, _fmt_time, build_real_data_variants
from echozero.ui.qt.timeline.style import TIMELINE_STYLE, fixture_color, fixture_take_action_label


def test_fmt_time_mm_ss_cc():
    assert _fmt_time(0.0) == "00:00.00"
    assert _fmt_time(8.25) == "00:08.25"
    assert _fmt_time(125.5) == "02:05.50"


def test_event_label_prefers_classification_value():
    event = Event(
        id="e1",
        time=1.0,
        duration=0.1,
        classifications={"class": "kick"},
        metadata={},
        origin="pipeline",
    )
    assert _event_label(event) == "Kick"


def test_real_data_variants_include_expected_views():
    presentation = build_demo_app().presentation()
    variants = build_real_data_variants(presentation)

    assert {"real_default", "real_scrolled", "real_zoomed_in", "real_zoomed_out"} <= set(variants)


def test_real_data_fixture_defaults_resolve_from_timeline_style_tokens():
    assert _fixture_layer_color("song") == fixture_color("song")
    assert _fixture_layer_color("drums") == fixture_color("drums")
    assert _fixture_layer_color("unknown-layer") == TIMELINE_STYLE.fixture.fallback_audio_lane_hex
    assert fixture_take_action_label("overwrite_main") == "Overwrite Main"
