from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from echozero.audio.engine import AudioEngine
from echozero.testing.analysis_mocks import write_test_tone_wav
from echozero.testing.app_flow import AppFlowHarness
from echozero.testing.playback_capture import PlaybackSegment, capture_playback_video
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController

_TEST_TEMP_ROOT = Path("C:/Users/griff/.codex/memories/test_playback_capture")


def _resolved_test_temp_root() -> Path:
    if _TEST_TEMP_ROOT.is_absolute():
        return _TEST_TEMP_ROOT
    return Path(tempfile.gettempdir()) / ".codex" / "memories" / "test_playback_capture"


def _repo_local_temp_root() -> Path:
    root = _resolved_test_temp_root() / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


class _FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs.get("callback")
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def _fake_stream_factory(**kwargs):
    return _FakeStream(**kwargs)


def test_capture_playback_video_writes_simulated_audio_and_video():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(working_dir_root=temp_root / "working")

    try:
        runtime_audio = TimelineRuntimeAudioController(engine=AudioEngine(stream_factory=_fake_stream_factory))
        harness.install_runtime_audio(runtime_audio)
        audio_path = write_test_tone_wav(
            temp_root / "fixtures" / "capture-tone.wav",
            duration_seconds=0.75,
            frequency_hz=220.0,
        )
        harness.runtime.add_song_from_path("Capture Song", audio_path)
        harness.widget.set_presentation(harness.runtime.presentation())
        harness._app.processEvents()

        layer = harness.presentation().layers[0]
        result = capture_playback_video(
            harness=harness,
            runtime_audio=runtime_audio,
            output_dir=temp_root / "capture-output",
            segments=[PlaybackSegment(label="main", layer_id=str(layer.layer_id), duration_seconds=0.5)],
            fps=10,
            callback_frames=1024,
        )

        rendered, sample_rate = sf.read(result.audio_path, dtype="float32")

        assert result.video_path.exists()
        assert result.audio_path.exists()
        assert result.video_path.name == "simulated-playback.mp4"
        assert result.audio_path.name == "simulated-playback.wav"
        assert result.frame_count >= 2
        assert sample_rate == runtime_audio.engine.sample_rate
        assert np.max(np.abs(rendered)) > 0.0
        assert result.proof_classification == "simulated_playback_capture"
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)
