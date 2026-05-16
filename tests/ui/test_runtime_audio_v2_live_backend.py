"""First dev-gated live-backend coverage for audio engine v2."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from echozero.application.audio_engine_v2.live_engine import V2LiveAudioEngine
from echozero.application.playback.engine_selection import build_runtime_audio_engine
from echozero.audio.engine import AudioEngine
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController
from tests.ui.runtime_audio_shared_support import (
    _audio_presentation,
    _fake_stream_factory,
)


def _loader(_path: str) -> tuple[np.ndarray, int]:
    return np.ones(4096, dtype=np.float32), 44100


def test_runtime_audio_engine_selection_keeps_v1_fallback() -> None:
    engine = build_runtime_audio_engine(
        backend_name="v1",
        stream_factory=_fake_stream_factory,
    )

    try:
        assert isinstance(engine, AudioEngine)
        assert engine.backend_name == "sounddevice"
    finally:
        engine.shutdown()


def test_runtime_audio_controller_selects_v2_backend_for_live_callback_flow() -> None:
    streams = []

    def _stream_factory(**kwargs):
        stream = _fake_stream_factory(**kwargs)
        streams.append(stream)
        return stream

    engine = build_runtime_audio_engine(
        backend_name="v2",
        channels=2,
        stream_factory=_stream_factory,
    )
    assert isinstance(engine, V2LiveAudioEngine)
    controller = TimelineRuntimeAudioController(engine=engine, audio_loader=_loader)
    presentation = _audio_presentation()

    try:
        controller.build_for_presentation(presentation)
        controller.play()

        assert streams and streams[-1].started is True
        assert controller.is_playing() is True

        first = np.zeros((256, 2), dtype=np.float32)
        streams[-1].callback(first, 256, None, None)

        assert float(np.max(np.abs(first))) > 0.0
        assert engine.rt_graph_identity_full_hash
        assert controller.current_time_seconds() > 0.0

        muted = replace(
            presentation,
            layers=[replace(presentation.layers[0], muted=True)],
        )
        controller.apply_mix_state(muted)

        _ramp_down = np.zeros((256, 2), dtype=np.float32)
        streams[-1].callback(_ramp_down, 256, None, None)
        muted_block = np.zeros((256, 2), dtype=np.float32)
        streams[-1].callback(muted_block, 256, None, None)
        np.testing.assert_allclose(muted_block, 0.0)

        controller.pause()
        paused = np.full((256, 2), 9.0, dtype=np.float32)
        streams[-1].callback(paused, 256, None, None)
        np.testing.assert_allclose(paused, 0.0)
    finally:
        controller.shutdown()


def test_v2_live_backend_callback_honors_direct_route_and_no_output_mix() -> None:
    engine = V2LiveAudioEngine(
        sample_rate=44100,
        channels=4,
        stream_factory=_fake_stream_factory,
    )
    direct = engine.create_track(
        "__ez_route__direct",
        np.ones(512, dtype=np.float32),
        44100,
        output_bus="outputs_3_3",
    )
    silent = engine.create_track(
        "__ez_route__silent",
        np.ones(512, dtype=np.float32),
        44100,
        output_bus="none",
    )

    try:
        engine.replace_tracks([direct, silent])
        engine.play()
        out = np.zeros((128, 4), dtype=np.float32)
        engine._audio_callback(out, 128, None, None)

        np.testing.assert_allclose(out[:, :2], 0.0)
        assert float(np.max(np.abs(out[:, 2]))) > 0.0
        np.testing.assert_allclose(out[:, 3], 0.0)
    finally:
        engine.shutdown()
