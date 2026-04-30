from __future__ import annotations

"""Simulated playback capture helper.

This helper advances the engine offline and captures widget frames for deterministic
artifact generation. It is useful for narrow regression checks, but it is not a
human-path demo surface and must not be presented as proof of real operator-visible
playback behavior.
"""

import shutil
import subprocess
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from PyQt6.QtWidgets import QApplication

from echozero.application.timeline.intents import Play
from echozero.testing.app_flow import AppFlowHarness
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController


@dataclass(slots=True, frozen=True)
class PlaybackSegment:
    label: str
    layer_id: str
    duration_seconds: float


@dataclass(slots=True, frozen=True)
class PlaybackCaptureResult:
    video_path: Path
    audio_path: Path
    frame_count: int
    duration_seconds: float
    proof_classification: str = "simulated_playback_capture"


def capture_playback_video(
    *,
    harness: AppFlowHarness,
    runtime_audio: TimelineRuntimeAudioController,
    output_dir: str | Path,
    segments: list[PlaybackSegment],
    intro_hold_seconds: float = 0.0,
    fps: int = 30,
    callback_frames: int = 1024,
) -> PlaybackCaptureResult:
    if fps < 1:
        raise ValueError("fps must be >= 1")
    if callback_frames < 1:
        raise ValueError("callback_frames must be >= 1")
    if not segments:
        raise ValueError("segments must not be empty")

    output_root = Path(output_dir)
    frames_dir = output_root / "frames"
    output_root.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance()
    if app is None:
        raise RuntimeError("capture_playback_video requires an active QApplication")

    engine = runtime_audio.engine
    sample_rate = int(engine.sample_rate)
    channels = max(1, int(getattr(engine, "_channels", 1)))
    frame_interval_samples = max(1, int(round(sample_rate / fps)))

    frame_index = 0
    captured_audio: list[np.ndarray] = []
    total_samples = 0

    def snap() -> None:
        nonlocal frame_index
        frame_path = frames_dir / f"{frame_index:04d}.png"
        if not harness.widget.grab().save(str(frame_path)):
            raise RuntimeError(f"Failed to save frame: {frame_path}")
        frame_index += 1

    def set_route(layer_id: str) -> None:
        presentation = harness.runtime.presentation()
        harness.widget.set_presentation(
            replace(
                presentation,
                selected_layer_id=layer_id,
                selected_layer_ids=[layer_id],
                selected_take_id=None,
            )
        )
        app.processEvents()

    def render_samples(sample_count: int) -> None:
        nonlocal total_samples
        remaining = max(0, int(sample_count))
        while remaining > 0:
            chunk_frames = min(callback_frames, remaining)
            outdata = np.zeros((chunk_frames, channels), dtype=np.float32)
            engine._audio_callback(outdata, chunk_frames, None, None)  # noqa: SLF001
            captured_audio.append(outdata[:, 0].copy())
            total_samples += chunk_frames
            harness.widget._on_runtime_tick()  # noqa: SLF001
            app.processEvents()
            remaining -= chunk_frames

    snap()
    if intro_hold_seconds > 0:
        for _ in range(max(1, int(round(intro_hold_seconds * fps)))):
            snap()

    return _capture_sequence(
        harness=harness,
        runtime_audio=runtime_audio,
        output_root=output_root,
        segments=segments,
        fps=fps,
        frame_interval_samples=frame_interval_samples,
        render_samples=render_samples,
        set_route=set_route,
        snap=snap,
        captured_audio=captured_audio,
        total_samples_ref=lambda: total_samples,
        sample_rate=sample_rate,
    )


def _capture_sequence(
    *,
    harness: AppFlowHarness,
    runtime_audio: TimelineRuntimeAudioController,
    output_root: Path,
    segments: list[PlaybackSegment],
    fps: int,
    frame_interval_samples: int,
    render_samples,
    set_route,
    snap,
    captured_audio: list[np.ndarray],
    total_samples_ref,
    sample_rate: int,
) -> PlaybackCaptureResult:
    harness.dispatch(Play())

    for segment in segments:
        set_route(segment.layer_id)
        frame_count = max(1, int(round(segment.duration_seconds * fps)))
        for _ in range(frame_count):
            render_samples(frame_interval_samples)
            snap()

    audio_path = output_root / "simulated-playback.wav"
    audio_data = np.concatenate(captured_audio) if captured_audio else np.zeros(0, dtype=np.float32)
    sf.write(audio_path, audio_data, sample_rate)

    video_path = output_root / "simulated-playback.mp4"
    _write_video_with_audio(
        frames_dir=output_root / "frames",
        audio_path=audio_path,
        output_path=video_path,
        fps=fps,
    )
    return PlaybackCaptureResult(
        video_path=video_path.resolve(),
        audio_path=audio_path.resolve(),
        frame_count=len(list((output_root / "frames").glob("*.png"))),
        duration_seconds=float(total_samples_ref()) / float(sample_rate),
    )


def _write_video_with_audio(*, frames_dir: Path, audio_path: Path, output_path: Path, fps: int) -> None:
    ffmpeg = _ffmpeg_exe()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write playback video")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(max(1, fps)),
        "-i",
        str(frames_dir / "%04d.png"),
        "-i",
        str(audio_path),
        # libx264 requires even frame dimensions; Qt grabs can land on odd sizes.
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _ffmpeg_exe() -> str | None:
    for candidate in ("ffmpeg", "/opt/homebrew/bin/ffmpeg", "C:/ffmpeg/bin/ffmpeg.exe"):
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
        if Path(candidate).exists():
            return candidate
    return None
