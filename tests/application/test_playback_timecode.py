"""Tests for playback timebase and SMPTE conversion contracts."""

from __future__ import annotations

import pytest

from echozero.application.playback.timecode import (
    TimebaseSpec,
    TimecodeCodec,
    TimecodeDisplayPolicy,
    format_clock_label,
)


def test_from_legacy_fps_defaults_to_30_ndf() -> None:
    spec = TimebaseSpec.from_legacy_fps(None)

    assert spec.nominal_fps == 30
    assert (spec.fps_numerator, spec.fps_denominator) == (30, 1)
    assert spec.drop_frame is False


def test_from_legacy_fps_supports_29_97() -> None:
    spec = TimebaseSpec.from_legacy_fps(29.97)

    assert spec.nominal_fps == 30
    assert (spec.fps_numerator, spec.fps_denominator) == (30000, 1001)
    assert spec.drop_frame is False


def test_from_legacy_fps_rejects_out_of_scope_rate() -> None:
    with pytest.raises(ValueError, match="Unsupported timebase fps"):
        TimebaseSpec.from_legacy_fps(23.976)


def test_drop_frame_requires_drop_compatible_ratio() -> None:
    with pytest.raises(ValueError, match="drop_frame requires an SMPTE-compatible"):
        TimebaseSpec.from_legacy_fps(30, drop_frame=True)


def test_samples_frames_roundtrip_for_29_97() -> None:
    codec = TimecodeCodec(TimebaseSpec.from_legacy_fps(29.97))

    samples = codec.frames_to_samples(12345, sample_rate=48000)
    roundtrip_frames = codec.samples_to_frames(samples, sample_rate=48000)

    assert abs(roundtrip_frames - 12345) <= 1


def test_non_drop_roundtrip_timecode_labels() -> None:
    codec = TimecodeCodec(TimebaseSpec.from_legacy_fps(30))

    for frame_index in (0, 1, 29, 30, 1799, 1800, 108000, 259199):
        label = codec.format_timecode_from_frames(frame_index)
        assert codec.frames_from_timecode(label) == frame_index


def test_drop_frame_formats_minute_boundary_skip() -> None:
    codec = TimecodeCodec(TimebaseSpec.from_legacy_fps(29.97, drop_frame=True))

    assert codec.format_timecode_from_frames(1799) == "00:00:59;29"
    assert codec.format_timecode_from_frames(1800) == "00:01:00;02"


def test_drop_frame_rejects_skipped_frame_labels() -> None:
    codec = TimecodeCodec(TimebaseSpec.from_legacy_fps(29.97, drop_frame=True))

    with pytest.raises(ValueError, match="Invalid drop-frame SMPTE label"):
        codec.frames_from_timecode("00:01:00;00")


def test_drop_frame_roundtrip_labels_across_large_slice() -> None:
    codec = TimecodeCodec(TimebaseSpec.from_legacy_fps(29.97, drop_frame=True))

    for frame_index in range(0, 20000, 137):
        label = codec.format_timecode_from_frames(frame_index)
        assert codec.frames_from_timecode(label) == frame_index


def test_start_offset_applies_on_format_and_parse() -> None:
    spec = TimebaseSpec.from_legacy_fps(30, start_frame_offset=30)
    codec = TimecodeCodec(spec)

    assert codec.format_timecode_from_frames(0) == "00:00:01:00"
    assert codec.frames_from_timecode("00:00:01:00") == 0


def test_clock_label_formatter_clamps_negative_seconds() -> None:
    assert format_clock_label(-1.0) == "00:00.00"
    assert format_clock_label(61.25) == "01:01.25"


def test_display_policy_can_be_explicitly_set() -> None:
    spec = TimebaseSpec.from_legacy_fps(
        25,
        display_policy=TimecodeDisplayPolicy.CLOCK,
    )

    assert spec.display_policy is TimecodeDisplayPolicy.CLOCK
