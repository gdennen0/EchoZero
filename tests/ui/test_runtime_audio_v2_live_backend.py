"""First dev-gated live-backend coverage for audio engine v2."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from echozero.application.audio_engine_v2.live_engine import V2LiveAudioEngine
from echozero.application.playback.engine_selection import (
    ENGINE_BACKEND_ENV,
    build_runtime_audio_engine,
    selected_audio_engine_backend,
)
from echozero.application.playback.process_service import PlaybackProcessService
from echozero.application.settings import AudioOutputRuntimeConfig
from echozero.audio.engine import AudioEngine
from echozero.audio.output_backend import AudioOutputConfig
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController
from tests.ui.runtime_audio_shared_support import (
    _audio_presentation,
    _fake_stream_factory,
)


def _loader(_path: str) -> tuple[np.ndarray, int]:
    return np.ones(4096, dtype=np.float32), 44100


class _FakeOutputBackend:
    name = "fake-output"

    def __init__(self) -> None:
        self.streams = []

    def resolve_output_config(
        self,
        *,
        sample_rate: int | None,
        channels: int | None,
        buffer_size: int,
        output_device: int | str | None,
        stream_blocksize: int | None,
        stream_latency: str | float | None,
        prime_output_buffers_using_stream_callback: bool,
    ) -> AudioOutputConfig:
        return AudioOutputConfig(
            sample_rate=sample_rate or 44100,
            channels=channels or 2,
            buffer_size=buffer_size,
            blocksize=stream_blocksize or 0,
            latency=stream_latency or "low",
            prime_output_buffers_using_stream_callback=(
                prime_output_buffers_using_stream_callback
            ),
            output_device=output_device,
            requested_output_device=output_device,
            resolved_output_device=output_device,
            requested_sample_rate=sample_rate,
            requested_channels=channels,
            device_max_output_channels=channels or 2,
            hardware_resolution_reason="injected-test-backend",
            sample_rate_resolution_reason="requested" if sample_rate is not None else "default",
            channel_resolution_reason="requested" if channels is not None else "default",
        )

    def open_output_stream(self, callback, config: AudioOutputConfig):
        stream = _fake_stream_factory(
            callback=callback,
            samplerate=config.sample_rate,
            blocksize=config.blocksize,
            channels=config.channels,
            latency=config.latency,
            device=config.output_device,
            prime_output_buffers_using_stream_callback=(
                config.prime_output_buffers_using_stream_callback
            ),
        )
        self.streams.append(stream)
        return stream


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


def test_runtime_audio_engine_selection_defaults_to_v2() -> None:
    engine = build_runtime_audio_engine(
        channels=2,
        stream_factory=_fake_stream_factory,
    )

    try:
        assert selected_audio_engine_backend() == "v2"
        assert isinstance(engine, V2LiveAudioEngine)
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


def test_v2_live_backend_applies_mix_edit_while_playing_as_rt_ramp() -> None:
    backend = _FakeOutputBackend()
    engine = V2LiveAudioEngine(sample_rate=44100, channels=2, backend=backend)
    track = engine.create_track(
        "__ez_route__bed",
        np.ones(2048, dtype=np.float32),
        44100,
        volume=1.0,
    )

    try:
        engine.replace_tracks([track])
        engine.play()
        before = np.zeros((128, 2), dtype=np.float32)
        engine._audio_callback(before, 128, None, None)

        changed = engine.apply_track_mix_updates(
            {"__ez_route__bed": (False, 0.25, track.output_bus)}
        )
        after = np.zeros((128, 2), dtype=np.float32)
        engine._audio_callback(after, 128, None, None)

        assert changed is True
        assert engine.last_discontinuity_reason == "mix-update"
        assert engine.last_ramp_reason == "mix-update"
        assert float(after[0, 0]) > float(after[-1, 0])
        assert float(after[-1, 0]) == pytest.approx(0.25)
        assert any(
            event.get("kind") == "v2-command-batch-applied"
            for event in engine.recent_runtime_events
        )
    finally:
        engine.shutdown()


def test_v2_live_backend_applies_solo_edit_while_playing_as_rt_ramp() -> None:
    backend = _FakeOutputBackend()
    engine = V2LiveAudioEngine(sample_rate=44100, channels=2, backend=backend)
    bed = engine.create_track(
        "__ez_route__bed",
        np.ones(2048, dtype=np.float32),
        44100,
        volume=0.25,
    )
    lead = engine.create_track(
        "__ez_route__lead",
        np.ones(2048, dtype=np.float32),
        44100,
        volume=0.25,
    )

    try:
        engine.replace_tracks([bed, lead])
        engine.play()
        before = np.zeros((128, 2), dtype=np.float32)
        engine._audio_callback(before, 128, None, None)

        changed = engine.apply_track_mix_updates(
            {
                "__ez_route__bed": (False, 0.25, bed.output_bus, False),
                "__ez_route__lead": (False, 0.25, lead.output_bus, True),
            }
        )
        after = np.zeros((128, 2), dtype=np.float32)
        engine._audio_callback(after, 128, None, None)

        assert changed is True
        assert 0.25 < float(after[0, 0]) < 0.5
        assert float(after[-1, 0]) == pytest.approx(0.25)
        assert engine.get_track("__ez_route__lead").solo is True
    finally:
        engine.shutdown()


def test_v2_live_backend_route_edit_while_playing_commits_graph_crossfade() -> None:
    engine = V2LiveAudioEngine(sample_rate=44100, channels=4, stream_factory=_fake_stream_factory)
    track = engine.create_track(
        "__ez_route__bed",
        np.ones(2048, dtype=np.float32),
        44100,
        output_bus=None,
    )

    try:
        engine.replace_tracks([track])
        engine.play()
        before = np.zeros((128, 4), dtype=np.float32)
        engine._audio_callback(before, 128, None, None)

        changed = engine.apply_track_mix_updates({"__ez_route__bed": (False, 1.0, "outputs_3_3")})
        after = np.zeros((128, 4), dtype=np.float32)
        engine._audio_callback(after, 128, None, None)

        assert changed is True
        assert engine.last_discontinuity_reason == "route-update"
        assert engine.last_ramp_reason == "route-update"
        assert float(np.max(np.abs(after[:, 2]))) > 0.0
        assert float(after[0, 0]) > float(after[-1, 0])
    finally:
        engine.shutdown()


def test_v2_live_backend_seek_pause_stop_keep_callback_time_and_tail_sane() -> None:
    engine = V2LiveAudioEngine(sample_rate=1000, channels=2, stream_factory=_fake_stream_factory)
    track = engine.create_track("bed", np.ones(3000, dtype=np.float32), 1000)

    try:
        engine.replace_tracks([track])
        engine.play()
        engine.seek_seconds(1.0)
        playing = np.zeros((100, 2), dtype=np.float32)
        engine._audio_callback(
            playing,
            100,
            {"currentTime": 1.0, "outputBufferDacTime": 1.025},
            None,
        )
        assert engine.clock.position_seconds == pytest.approx(1.1)
        assert engine.audible_time_seconds == pytest.approx(1.075, abs=1e-3)

        engine.pause()
        paused = np.ones((100, 2), dtype=np.float32)
        engine._audio_callback(paused, 100, None, None)
        assert float(np.max(np.abs(paused))) <= 1.0
        paused_tail = np.ones((100, 2), dtype=np.float32)
        engine._audio_callback(paused_tail, 100, None, None)
        np.testing.assert_allclose(paused_tail, 0.0)
        assert engine.clock.position_seconds == pytest.approx(1.1)

        engine.stop()
        stopped = np.ones((100, 2), dtype=np.float32)
        engine._audio_callback(stopped, 100, None, None)
        np.testing.assert_allclose(stopped, 0.0)
        assert engine.clock.position_seconds == pytest.approx(0.0)
        assert engine.audible_time_seconds == pytest.approx(0.0)
    finally:
        engine.shutdown()


def test_v2_live_backend_preview_overlay_is_callback_mixed_and_self_clearing() -> None:
    engine = V2LiveAudioEngine(sample_rate=10, channels=2, stream_factory=_fake_stream_factory)

    try:
        played = engine.play_overlay(np.arange(6, dtype=np.float32), 10, volume=0.5)

        assert played is True
        assert engine.overlay_active is True
        first = np.zeros((4, 2), dtype=np.float32)
        second = np.zeros((4, 2), dtype=np.float32)
        engine._audio_callback(first, 4, None, None)
        engine._audio_callback(second, 4, None, None)

        assert float(np.max(np.abs(first))) > 0.0
        assert engine.overlay_active is False
        assert any(event.get("kind") == "overlay-start" for event in engine.recent_runtime_events)
    finally:
        engine.shutdown()


def test_v2_env_selected_controller_reconfigure_device_preserves_v2_backend(
    monkeypatch,
) -> None:
    backends = [_FakeOutputBackend(), _FakeOutputBackend()]

    def _backend_factory(**_kwargs):
        return backends.pop(0)

    monkeypatch.setenv(ENGINE_BACKEND_ENV, "v2")
    monkeypatch.setattr(
        "echozero.application.audio_engine_v2.live_engine.SounddeviceBackend",
        _backend_factory,
    )
    controller = TimelineRuntimeAudioController(
        audio_loader=lambda _path: (np.ones(1024, dtype=np.float32), 44100),
    )
    presentation = _audio_presentation()

    try:
        assert isinstance(controller.engine, V2LiveAudioEngine)
        controller.build_for_presentation(presentation)
        controller.play()
        controller.seek(0.01)

        result = controller.reconfigure_device(device_spec={"channels": 4})

        assert isinstance(controller.engine, V2LiveAudioEngine)
        assert result["reason"] == "settings-change"
        assert result["output_channels"] == 4
        assert controller.is_playing() is True
    finally:
        controller.shutdown()


def test_process_service_build_controller_selects_v2_from_env_with_fake_backend(
    monkeypatch,
) -> None:
    backend = _FakeOutputBackend()

    monkeypatch.setenv(ENGINE_BACKEND_ENV, "v2")
    monkeypatch.setattr(
        "echozero.application.audio_engine_v2.live_engine.SounddeviceBackend",
        lambda **_kwargs: backend,
    )
    service = PlaybackProcessService.__new__(PlaybackProcessService)
    service._profile_index = 1
    service._base_audio_config = AudioOutputRuntimeConfig(
        sample_rate=48000,
        channels=4,
        stream_latency="low",
        stream_blocksize=128,
        prime_output_buffers_using_stream_callback=False,
    )

    controller = service._build_controller()

    try:
        assert isinstance(controller.engine, V2LiveAudioEngine)
        assert controller.engine.sample_rate == 48000
        assert controller.engine.output_channels == 4
        assert controller.engine.stream_blocksize == 128
        assert controller.engine.output_config.hardware_resolution_reason == (
            "injected-test-backend"
        )
    finally:
        controller.shutdown()


def test_process_service_honors_explicit_high_latency_config_with_fake_backend(
    monkeypatch,
) -> None:
    backend = _FakeOutputBackend()

    monkeypatch.setenv(ENGINE_BACKEND_ENV, "v2")
    monkeypatch.setattr(
        "echozero.application.audio_engine_v2.live_engine.SounddeviceBackend",
        lambda **_kwargs: backend,
    )
    service = PlaybackProcessService.__new__(PlaybackProcessService)
    service._profile_index = 1
    service._base_audio_config = AudioOutputRuntimeConfig(
        sample_rate=48000,
        channels=2,
        stream_latency="high",
        stream_blocksize=None,
        prime_output_buffers_using_stream_callback=True,
    )

    controller = service._build_controller()

    try:
        assert isinstance(controller.engine, V2LiveAudioEngine)
        assert controller.engine.stream_latency == "high"
        assert controller.engine.stream_blocksize == 1024
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
