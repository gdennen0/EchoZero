"""Tests for the param/widget system (echozero.pipelines.params) and resources."""

from __future__ import annotations

from pathlib import Path

import pytest

from echozero.pipelines.params import (
    Knob,
    KnobWidget,
    extract_knobs,
    knob,
    validate_bindings,
)
# resources.py was removed (dead code) — ModelRef/parse_model_ref tests removed


# ── KnobWidget enum ────────────────────────────────────────────────────────


class TestKnobWidget:
    def test_all_expected_widgets_exist(self):
        expected = {
            "AUTO", "SLIDER", "DROPDOWN", "TOGGLE", "TEXT", "NUMBER",
            "FILE_PICKER", "MODEL_PICKER", "COLOR_PICKER",
            "FREQUENCY", "GAIN", "TIME_RANGE", "MULTI_SELECT",
        }
        actual = {w.name for w in KnobWidget}
        assert actual == expected


# ── knob() builder — AUTO inference ────────────────────────────────────────


class TestAutoInference:
    def test_bool_infers_toggle(self):
        p = knob(True, label="Enable")
        assert p.widget == KnobWidget.TOGGLE

    def test_int_with_range_infers_slider(self):
        p = knob(5, min_value=0, max_value=10)
        assert p.widget == KnobWidget.SLIDER

    def test_float_with_range_infers_slider(self):
        p = knob(0.5, min_value=0.0, max_value=1.0)
        assert p.widget == KnobWidget.SLIDER

    def test_str_with_options_infers_dropdown(self):
        p = knob("a", options=("a", "b", "c"))
        assert p.widget == KnobWidget.DROPDOWN

    def test_str_without_options_infers_text(self):
        p = knob("hello")
        assert p.widget == KnobWidget.TEXT

    def test_str_with_file_types_infers_file_picker(self):
        p = knob("model.pth", file_types=(".pth", ".pt"))
        assert p.widget == KnobWidget.FILE_PICKER

    def test_path_infers_file_picker(self):
        p = knob(Path("/some/file.wav"))
        assert p.widget == KnobWidget.FILE_PICKER

    def test_int_without_range_infers_number(self):
        p = knob(42)
        assert p.widget == KnobWidget.NUMBER

    def test_float_without_range_infers_number(self):
        p = knob(3.14)
        assert p.widget == KnobWidget.NUMBER

    def test_list_with_options_infers_multi_select(self):
        p = knob(["a"], options=("a", "b", "c"))
        assert p.widget == KnobWidget.MULTI_SELECT


# ── knob() builder — explicit widgets ──────────────────────────────────────


class TestExplicitWidgets:
    def test_frequency_defaults(self):
        p = knob(440.0, widget=KnobWidget.FREQUENCY)
        assert p.min_value == 20.0
        assert p.max_value == 20_000.0
        assert p.units == "Hz"
        assert p.log_scale is True

    def test_gain_defaults(self):
        p = knob(0.0, widget=KnobWidget.GAIN)
        assert p.min_value == -48.0
        assert p.max_value == 48.0
        assert p.units == "dB"

    def test_frequency_custom_range(self):
        p = knob(100.0, widget=KnobWidget.FREQUENCY, min_value=50.0, max_value=5000.0)
        assert p.min_value == 50.0
        assert p.max_value == 5000.0

    def test_explicit_slider(self):
        p = knob(5, widget=KnobWidget.SLIDER, min_value=0, max_value=10)
        assert p.widget == KnobWidget.SLIDER


# ── knob() builder — validation ────────────────────────────────────────────


class TestValidation:
    def test_slider_without_range_raises(self):
        with pytest.raises(ValueError, match="requires both min_value and max_value"):
            knob(5, widget=KnobWidget.SLIDER)

    def test_dropdown_without_options_raises(self):
        with pytest.raises(ValueError, match="requires non-empty options"):
            knob("a", widget=KnobWidget.DROPDOWN)

    def test_min_gte_max_raises(self):
        with pytest.raises(ValueError, match="must be less than"):
            knob(5, widget=KnobWidget.SLIDER, min_value=10, max_value=5)

    def test_default_out_of_range_raises(self):
        with pytest.raises(ValueError, match="outside range"):
            knob(20, widget=KnobWidget.SLIDER, min_value=0, max_value=10)

    def test_default_not_in_options_raises(self):
        with pytest.raises(ValueError, match="not in options"):
            knob("z", options=("a", "b", "c"))

    def test_negative_step_raises(self):
        with pytest.raises(ValueError, match="step must be positive"):
            knob(5, min_value=0, max_value=10, step=-1)

    def test_zero_step_raises(self):
        with pytest.raises(ValueError, match="step must be positive"):
            knob(5, min_value=0, max_value=10, step=0)


# ── Knob is frozen ──────────────────────────────────────────────────────


class TestFrozen:
    def test_Knob_immutable(self):
        p = knob(0.5, min_value=0.0, max_value=1.0)
        with pytest.raises(AttributeError):
            p.default = 0.9  # type: ignore


# ── extract_knobs ──────────────────────────────────────────────────────────


class TestExtractParams:
    def test_extracts_Knob_defaults(self):
        def my_pipeline(
            threshold: float = knob(0.5, min_value=0.0, max_value=1.0, label="Threshold"),
            method: str = knob("rms", options=("rms", "spectral_flux"), label="Method"),
            verbose: bool = False,  # not a Knob — should be ignored
        ):
            pass

        params = extract_knobs(my_pipeline)
        assert set(params.keys()) == {"threshold", "method"}
        assert params["threshold"].label == "Threshold"
        assert params["method"].widget == KnobWidget.DROPDOWN

    def test_empty_function(self):
        def no_params():
            pass

        assert extract_knobs(no_params) == {}

    def test_mixed_args(self):
        def mixed(
            audio_path: str,  # positional, no default
            gain: float = knob(0.0, widget=KnobWidget.GAIN, label="Gain"),
        ):
            pass

        params = extract_knobs(mixed)
        assert "gain" in params
        assert "audio_path" not in params


# ── validate_bindings ───────────────────────────────────────────────────────


class TestValidateBindings:
    @pytest.fixture
    def sample_params(self):
        return {
            "threshold": knob(0.5, min_value=0.0, max_value=1.0),
            "method": knob("rms", options=("rms", "spectral_flux")),
            "enabled": knob(True),
        }

    def test_valid_bindings(self, sample_params):
        errors = validate_bindings(sample_params, {"threshold": 0.8, "method": "rms"})
        assert errors == []

    def test_unknown_key(self, sample_params):
        errors = validate_bindings(sample_params, {"bogus": 42})
        assert any("Unknown" in e for e in errors)

    def test_wrong_type(self, sample_params):
        errors = validate_bindings(sample_params, {"threshold": "not a float"})
        assert any("expected float" in e for e in errors)

    def test_int_to_float_coercion(self, sample_params):
        errors = validate_bindings(sample_params, {"threshold": 1})
        assert errors == []

    def test_out_of_range(self, sample_params):
        errors = validate_bindings(sample_params, {"threshold": 5.0})
        assert any("outside" in e for e in errors)

    def test_invalid_option(self, sample_params):
        errors = validate_bindings(sample_params, {"method": "invalid"})
        assert any("not in" in e for e in errors)

    def test_empty_bindings_ok(self, sample_params):
        errors = validate_bindings(sample_params, {})
        assert errors == []


# ── Integration: realistic pipeline builder ─────────────────────────────────


class TestIntegration:
    def test_realistic_pipeline(self):
        """Simulates what a real pipeline builder function would look like."""

        def onset_detection(
            threshold: float = knob(
                0.5, min_value=0.0, max_value=1.0, step=0.01,
                label="Detection Threshold",
                description="Sensitivity of onset detection",
                group="Detection",
            ),
            method: str = knob(
                "spectral_flux",
                options=("spectral_flux", "rms", "complex_domain"),
                label="Method",
                group="Detection",
            ),
            min_frequency: float = knob(
                200.0,
                widget=KnobWidget.FREQUENCY,
                label="Min Frequency",
                group="Filter",
            ),
            output_gain: float = knob(
                0.0,
                widget=KnobWidget.GAIN,
                label="Output Gain",
                group="Output",
                advanced=True,
            ),
            debug: bool = knob(False, label="Debug Mode", hidden=True),
        ):
            pass

        params = extract_knobs(onset_detection)
        assert len(params) == 5

        # All are Knob
        assert all(isinstance(p, Knob) for p in params.values())

        # Groups
        groups = {p.group for p in params.values() if p.group}
        assert groups == {"Detection", "Filter", "Output"}

        # Validate good bindings
        errors = validate_bindings(params, {
            "threshold": 0.7,
            "method": "rms",
            "min_frequency": 500.0,
        })
        assert errors == []

        # Validate bad bindings
        errors = validate_bindings(params, {
            "threshold": 5.0,  # out of range
            "method": "nope",  # invalid option
        })
        assert len(errors) == 2
