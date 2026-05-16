"""
Audio Engine v2 real-project smoke command.
Exists so developers can validate private .ez projects without CI or speaker dependencies.
Connects ProjectStorage open, runtime presentation assembly, and v2 fake-output callbacks.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import numpy.typing as npt

from echozero.application.audio_engine_v2.live_engine import V2LiveAudioEngine
from echozero.application.playback.engine_selection import (
    ENGINE_BACKEND_ENV,
    build_runtime_audio_engine,
    selected_audio_engine_backend,
)
from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.shared.enums import PlaybackMode
from echozero.application.timeline.intents import (
    Pause,
    Play,
    Seek,
    SetGain,
    SetLayerMute,
    SetLayerOutputBus,
    Stop,
)
from echozero.persistence.session import ProjectStorage
from echozero.testing.fake_output_backend import (
    DEFAULT_FAKE_BLOCK_FRAMES,
    FakeOutputBackend,
    FakeOutputStream,
)
from echozero.ui.qt.app_shell_runtime_services import build_runtime_timeline_application
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController

_DEFAULT_BLOCK_FRAMES = DEFAULT_FAKE_BLOCK_FRAMES


@dataclass(slots=True, frozen=True)
class SmokeResult:
    """Result summary for one real-project smoke run."""

    project_path: Path
    active_song_title: str
    layer_count: int
    playable_layer_count: int
    track_count: int
    graph_hash: str
    peak: float
    paused_peak: float
    preview_exercised: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the smoke runner."""

    parser = argparse.ArgumentParser(
        description="Run Audio Engine v2 fake-output smoke against local .ez project files."
    )
    parser.add_argument(
        "project",
        nargs="*",
        type=Path,
        help="Project .ez path. Defaults to ECHOZERO_REAL_PROJECT_SMOKE.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        help="Fake hardware output channel count.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=3,
        help="Initial callback block count to render after play.",
    )
    return parser.parse_args()


def resolve_project_paths(cli_paths: list[Path]) -> list[Path]:
    """Resolve CLI or env-provided .ez project paths."""

    if cli_paths:
        return [path.expanduser().resolve() for path in cli_paths]
    raw_value = os.environ.get("ECHOZERO_REAL_PROJECT_SMOKE", "").strip()
    if not raw_value:
        raise SystemExit(
            "Set ECHOZERO_REAL_PROJECT_SMOKE=/path/to/project.ez or pass a path argument."
        )
    paths = [Path(item).expanduser().resolve() for item in raw_value.split(os.pathsep) if item]
    if not paths:
        raise SystemExit("ECHOZERO_REAL_PROJECT_SMOKE did not contain any project paths.")
    return paths


def run_real_project_smoke(
    project_path: Path,
    *,
    channels: int = 2,
    blocks: int = 3,
) -> SmokeResult:
    """Open one .ez project and exercise the v2 fake-output runtime callback path."""

    backend = FakeOutputBackend()
    engine = build_runtime_audio_engine(
        channels=max(1, int(channels)),
        stream_blocksize=_DEFAULT_BLOCK_FRAMES,
        backend=backend,
    )
    if not isinstance(engine, V2LiveAudioEngine):
        raise RuntimeError(f"{ENGINE_BACKEND_ENV}=v2 is required for this smoke command.")

    runtime_audio = TimelineRuntimeAudioController(engine=engine)
    with TemporaryDirectory(prefix="ez-v2-real-project-smoke-") as working_root:
        storage = ProjectStorage.open(project_path, working_dir_root=Path(working_root))
        try:
            app = build_runtime_timeline_application(
                project_storage=storage,
                sync_bridge=None,
                sync_service=None,
                runtime_audio=runtime_audio,
            )
            presentation = app.presentation()
            playable_layers = _playable_layers(presentation)
            if not playable_layers:
                raise RuntimeError(f"{project_path} loaded, but no playable layers were found.")

            app.dispatch(Play())
            stream = backend.streams[-1]
            peak = _render_blocks(stream, channels=channels, block_count=max(1, int(blocks)))

            app.dispatch(Seek(0.05))
            _render_blocks(stream, channels=channels, block_count=1)

            target_layer = playable_layers[0]
            app.dispatch(SetGain(layer_id=target_layer.layer_id, gain_db=-6.0))
            _render_blocks(stream, channels=channels, block_count=1)
            app.dispatch(SetLayerMute(layer_id=target_layer.layer_id, muted=True))
            _render_blocks(stream, channels=channels, block_count=1)
            app.dispatch(SetLayerMute(layer_id=target_layer.layer_id, muted=False))
            app.dispatch(
                SetLayerOutputBus(layer_id=target_layer.layer_id, output_bus="outputs_1_1")
            )
            _render_blocks(stream, channels=channels, block_count=1)

            preview_exercised = _exercise_preview(runtime_audio, playable_layers)
            if preview_exercised:
                _render_blocks(stream, channels=channels, block_count=2)

            app.dispatch(Pause())
            _render_prefilled_block(stream, channels=channels)
            paused_peak = _render_prefilled_block(stream, channels=channels)
            app.dispatch(Stop())

            return SmokeResult(
                project_path=project_path,
                active_song_title=presentation.active_song_title,
                layer_count=len(presentation.layers),
                playable_layer_count=len(playable_layers),
                track_count=len(engine.tracks),
                graph_hash=engine.rt_graph_identity_full_hash,
                peak=peak,
                paused_peak=paused_peak,
                preview_exercised=preview_exercised,
            )
        finally:
            runtime_audio.shutdown()
            storage.close()


def _playable_layers(presentation: TimelinePresentation) -> list[LayerPresentation]:
    return [
        layer
        for layer in presentation.layers
        if layer.source_audio_path or layer.playback_source_ref
    ]


def _render_blocks(
    stream: FakeOutputStream,
    *,
    channels: int,
    block_count: int,
) -> float:
    peak = 0.0
    for _index in range(max(1, int(block_count))):
        outdata: npt.NDArray[np.float32] = np.zeros(
            (_DEFAULT_BLOCK_FRAMES, max(1, int(channels))),
            dtype=np.float32,
        )
        stream.callback(outdata, _DEFAULT_BLOCK_FRAMES, None, None)
        peak = max(peak, float(np.max(np.abs(outdata))))
    return peak


def _render_prefilled_block(stream: FakeOutputStream, *, channels: int) -> float:
    outdata: npt.NDArray[np.float32] = np.ones(
        (_DEFAULT_BLOCK_FRAMES, max(1, int(channels))),
        dtype=np.float32,
    )
    stream.callback(outdata, _DEFAULT_BLOCK_FRAMES, None, None)
    return float(np.max(np.abs(outdata)))


def _exercise_preview(
    runtime_audio: TimelineRuntimeAudioController,
    playable_layers: list[LayerPresentation],
) -> bool:
    for layer in playable_layers:
        if layer.playback_mode not in {PlaybackMode.CONTINUOUS_AUDIO, PlaybackMode.EVENT_SLICE}:
            continue
        source_ref = layer.source_audio_path or layer.playback_source_ref
        if not source_ref or not layer.events:
            continue
        event = layer.events[0]
        if float(event.end) <= float(event.start):
            continue
        return bool(
            runtime_audio.preview_clip(
                source_ref,
                start_seconds=float(event.start),
                end_seconds=min(float(event.end), float(event.start) + 0.25),
                gain_db=-12.0,
            )
        )
    return False


def _print_result(result: SmokeResult) -> None:
    print(
        f"{result.project_path.name}: loaded active_song={result.active_song_title!r} "
        f"layers={result.layer_count} playable={result.playable_layer_count} "
        f"tracks={result.track_count} graph={result.graph_hash[:12]} "
        f"peak={result.peak:.6f} paused_peak={result.paused_peak:.6f} "
        f"preview={'yes' if result.preview_exercised else 'no'}"
    )


def main() -> int:
    """Run the smoke command and return a process status code."""

    if selected_audio_engine_backend() != "v2":
        raise SystemExit(f"Set {ENGINE_BACKEND_ENV}=v2 before running this smoke command.")
    args = parse_args()
    for project_path in resolve_project_paths(args.project):
        result = run_real_project_smoke(
            project_path,
            channels=max(1, int(args.channels)),
            blocks=max(1, int(args.blocks)),
        )
        _print_result(result)
    print("speaker_audio=no fake_output=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
