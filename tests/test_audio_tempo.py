from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np

from echozero.persistence import audio_tempo


def test_filename_bpm_hint_parses_common_import_name() -> None:
    hint = audio_tempo._filename_bpm_hint(
        Path("/tmp/NoahKahan_You'reGonnaGoFar_85bpm_SMPTE_v01.wav")
    )

    assert hint == 85.0


def test_resolve_reported_bpm_prefers_nearby_filename_hint() -> None:
    resolved = audio_tempo._resolve_reported_bpm(
        86.1328125,
        filename_bpm_hint=85.0,
    )

    assert resolved == 85.0


def test_load_tempo_analysis_audio_prefers_program_channel_when_ltc_is_detected(
    monkeypatch,
) -> None:
    left_channel = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    right_channel = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    stereo = np.column_stack((left_channel, right_channel))

    fake_librosa = types.SimpleNamespace(
        load=lambda *_args, **_kwargs: (right_channel, 22050),
        resample=lambda samples, *, orig_sr, target_sr: samples,
    )
    fake_soundfile = types.SimpleNamespace(
        read=lambda *_args, **_kwargs: (stereo, 48000),
    )

    monkeypatch.setattr(audio_tempo, "_program_channel_index", lambda _path: 1)
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    samples, sample_rate = audio_tempo._load_tempo_analysis_audio(
        Path("/tmp/fake.wav"),
        sample_rate=22050,
    )

    assert sample_rate == 22050
    assert np.array_equal(samples, right_channel)


def test_load_tempo_analysis_audio_falls_back_to_plain_mono_load_when_no_ltc(
    monkeypatch,
) -> None:
    mono = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    fake_librosa = types.SimpleNamespace(
        load=lambda *_args, **_kwargs: (mono, 22050),
    )

    monkeypatch.setattr(audio_tempo, "_program_channel_index", lambda _path: None)
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    samples, sample_rate = audio_tempo._load_tempo_analysis_audio(
        Path("/tmp/fake.wav"),
        sample_rate=22050,
    )

    assert sample_rate == 22050
    assert np.array_equal(samples, mono)
