"""
Event preview app-path diagnostics command.
Exists to inspect real project preview playback without opening speaker hardware.
Connects ProjectStorage, runtime presentation, preview_event_clip, and fake output callbacks.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from echozero.application.playback.engine_selection import build_runtime_audio_engine
from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.audio.file_cache import load_audio_file
from echozero.persistence.session import ProjectStorage
from echozero.testing.fake_output_backend import (
    DEFAULT_FAKE_BLOCK_FRAMES,
    FakeOutputBackend,
    render_fake_output_blocks,
)
from echozero.ui.qt.app_shell_runtime_services import build_runtime_timeline_application
from echozero.ui.qt.app_shell_runtime_support import preview_event_clip
from echozero.ui.qt.app_shell_timeline_state import resolve_event_clip_preview
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController


@dataclass(slots=True, frozen=True)
class EventPreviewTarget:
    """One real presentation event that can be previewed through the app path."""

    layer_id: LayerId
    take_id: TakeId | None
    event_id: EventId
    layer_title: str
    playback_mode: str
    start_seconds: float
    end_seconds: float


@dataclass(slots=True, frozen=True)
class EventPreviewDiagnosticResult:
    """One app-path event preview diagnostic summary."""

    project_path: str
    active_song_title: str
    target: dict[str, object]
    source_ref: str
    source_sample_rate: int
    source_channels: int
    source_frame_count: int
    output_sample_rate: int
    output_channels: int
    stream_blocksize: int
    stream_latency: str | float | None
    output_device: str | None
    resolved_output_device: str | None
    clip_start_seconds: float
    clip_end_seconds: float
    clip_duration_seconds: float
    clip_frame_count: int
    resampled: bool
    resample_source_rate: int
    resample_target_rate: int
    output_bus: str
    backend_name: str
    glitch_count: int
    last_audio_status: str | None
    rendered_blocks: list[dict[str, object]]
    runtime_events: list[dict[str, object]]
    diagnostics_bundle_path: str | None
    speaker_audio: bool
    fake_output: bool


class _PreviewShell:
    """Small app-shell adapter for invoking preview_event_clip with real app state."""

    def __init__(self, app, runtime_audio: TimelineRuntimeAudioController) -> None:
        self._app = app
        self.runtime_audio = runtime_audio

    def presentation(self) -> TimelinePresentation:
        return self._app.presentation()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the diagnostic runner."""

    parser = argparse.ArgumentParser(
        description="Run app-path event preview diagnostics with fake audio output."
    )
    parser.add_argument(
        "project",
        nargs="*",
        type=Path,
        help="Project .ez path. Defaults to ECHOZERO_REAL_PROJECT_SMOKE.",
    )
    parser.add_argument("--sample-rate", type=int, default=44100, help="Fake output sample rate.")
    parser.add_argument("--channels", type=int, default=2, help="Fake output channel count.")
    parser.add_argument(
        "--blocks",
        type=int,
        default=8,
        help="Callback blocks to render after preview starts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diagnostics bundle directory. Defaults to artifacts/audio-diagnostics.",
    )
    return parser.parse_args()


def resolve_project_paths(cli_paths: list[Path]) -> list[Path]:
    """Resolve CLI or env-provided .ez project paths."""

    if cli_paths:
        return [path.expanduser().resolve() for path in cli_paths]
    raw_value = os.environ.get("ECHOZERO_REAL_PROJECT_SMOKE", "").strip()
    if not raw_value:
        return []
    return [Path(item).expanduser().resolve() for item in raw_value.split(os.pathsep) if item]


def run_event_preview_diagnostics(
    project_path: Path,
    *,
    sample_rate: int = 44100,
    channels: int = 2,
    blocks: int = 8,
    output_dir: Path | None = None,
) -> EventPreviewDiagnosticResult:
    """Open one real .ez project and preview one event through the app runtime path."""

    original_stat = project_path.stat()
    backend = FakeOutputBackend(
        default_sample_rate=max(1, int(sample_rate)),
        default_channels=max(1, int(channels)),
    )
    engine = build_runtime_audio_engine(
        sample_rate=max(1, int(sample_rate)),
        channels=max(1, int(channels)),
        stream_blocksize=DEFAULT_FAKE_BLOCK_FRAMES,
        backend=backend,
    )
    runtime_audio = TimelineRuntimeAudioController(engine=engine)
    with TemporaryDirectory(prefix="ez-event-preview-diagnostics-") as working_root:
        storage = ProjectStorage.open(project_path, working_dir_root=Path(working_root))
        try:
            app = build_runtime_timeline_application(
                project_storage=storage,
                sync_bridge=None,
                sync_service=None,
                runtime_audio=runtime_audio,
            )
            presentation = app.presentation()
            runtime_audio.sync_structure_state(presentation)
            target = _choose_event_preview_target(presentation)
            if target is None:
                raise RuntimeError(f"{project_path} loaded, but no previewable events were found.")

            clip = resolve_event_clip_preview(
                presentation,
                layer_id=target.layer_id,
                take_id=target.take_id,
                event_id=target.event_id,
            )
            source_buffer, source_sample_rate = load_audio_file(clip.source_ref)
            source_channels = _buffer_channel_count(source_buffer)
            source_frame_count = int(source_buffer.shape[0])
            clip_frame_count = _clip_frame_count(
                start_seconds=clip.start_seconds,
                end_seconds=clip.end_seconds,
                sample_rate=int(source_sample_rate),
                source_frame_count=source_frame_count,
            )

            runtime_audio.start_audio_diagnostics_capture(
                output_dir=output_dir,
                include_audio_buffers=True,
                max_audio_blocks=max(1, int(blocks)),
            )
            shell = _PreviewShell(app, runtime_audio)
            preview_event_clip(
                shell,
                layer_id=target.layer_id,
                take_id=target.take_id,
                event_id=target.event_id,
            )
            if not backend.streams:
                raise RuntimeError("Preview did not open a fake output stream.")
            stream = backend.streams[-1]
            rendered = render_fake_output_blocks(
                stream,
                channels=max(1, int(channels)),
                block_frames=DEFAULT_FAKE_BLOCK_FRAMES,
                block_count=max(1, int(blocks)),
            )
            state = runtime_audio.snapshot_state(presentation)
            stopped = runtime_audio.stop_audio_diagnostics_capture()
            events = list(state.diagnostics.recent_audio_runtime_events)
            preview_event = _last_event(events, "preview-start", "preview-replace")

            return EventPreviewDiagnosticResult(
                project_path=str(project_path),
                active_song_title=str(presentation.active_song_title),
                target=asdict(target),
                source_ref=str(clip.source_ref),
                source_sample_rate=int(source_sample_rate),
                source_channels=int(source_channels),
                source_frame_count=int(source_frame_count),
                output_sample_rate=int(state.output_sample_rate),
                output_channels=int(state.output_channels),
                stream_blocksize=int(state.diagnostics.stream_blocksize),
                stream_latency=state.diagnostics.stream_latency,
                output_device=state.diagnostics.output_device,
                resolved_output_device=state.diagnostics.resolved_output_device,
                clip_start_seconds=float(clip.start_seconds),
                clip_end_seconds=float(clip.end_seconds),
                clip_duration_seconds=max(
                    0.0, float(clip.end_seconds) - float(clip.start_seconds)
                ),
                clip_frame_count=int(clip_frame_count),
                resampled=bool(preview_event.get("resampled", False)),
                resample_source_rate=int(
                    preview_event.get("resample_source_rate", source_sample_rate)
                ),
                resample_target_rate=int(
                    preview_event.get("resample_target_rate", state.output_sample_rate)
                ),
                output_bus=str(preview_event.get("output_bus", "") or ""),
                backend_name=str(state.backend_name),
                glitch_count=int(state.diagnostics.glitch_count),
                last_audio_status=state.diagnostics.last_audio_status,
                rendered_blocks=[asdict(item) for item in rendered],
                runtime_events=events,
                diagnostics_bundle_path=_optional_string(stopped.get("bundle_path")),
                speaker_audio=False,
                fake_output=True,
            )
        finally:
            runtime_audio.shutdown()
            storage.close()
            after_stat = project_path.stat()
            if (
                after_stat.st_size != original_stat.st_size
                or after_stat.st_mtime_ns != original_stat.st_mtime_ns
            ):
                raise RuntimeError(f"Diagnostic unexpectedly modified {project_path}.")


def _choose_event_preview_target(
    presentation: TimelinePresentation,
) -> EventPreviewTarget | None:
    for layer in presentation.layers:
        for target in _layer_event_targets(presentation, layer):
            return target
    return None


def _layer_event_targets(
    presentation: TimelinePresentation,
    layer: LayerPresentation,
) -> list[EventPreviewTarget]:
    targets: list[EventPreviewTarget] = []
    targets.extend(
        _events_for_lane(
            presentation,
            layer=layer,
            take=None,
            events=layer.events,
        )
    )
    for take in layer.takes:
        targets.extend(
            _events_for_lane(
                presentation,
                layer=layer,
                take=take,
                events=take.events,
            )
        )
    return targets


def _events_for_lane(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    take: TakeLanePresentation | None,
    events: list[EventPresentation],
) -> list[EventPreviewTarget]:
    targets: list[EventPreviewTarget] = []
    for event in events:
        if float(event.end) <= float(event.start):
            continue
        take_id = take.take_id if take is not None else layer.main_take_id
        try:
            resolve_event_clip_preview(
                presentation,
                layer_id=layer.layer_id,
                take_id=take_id,
                event_id=event.event_id,
            )
        except Exception:
            continue
        targets.append(
            EventPreviewTarget(
                layer_id=layer.layer_id,
                take_id=take_id,
                event_id=event.event_id,
                layer_title=str(layer.title),
                playback_mode=str(layer.playback_mode.value),
                start_seconds=float(event.start),
                end_seconds=float(event.end),
            )
        )
    return targets


def _clip_frame_count(
    *,
    start_seconds: float,
    end_seconds: float,
    sample_rate: int,
    source_frame_count: int,
) -> int:
    start_sample = max(0, int(round(float(start_seconds) * int(sample_rate))))
    end_sample = max(start_sample, int(round(float(end_seconds) * int(sample_rate))))
    end_sample = min(end_sample, int(source_frame_count))
    return max(0, end_sample - start_sample)


def _buffer_channel_count(buffer: np.ndarray) -> int:
    if buffer.ndim <= 1:
        return 1
    return int(buffer.shape[1])


def _last_event(events: list[dict[str, object]], *kinds: str) -> dict[str, object]:
    wanted = set(kinds)
    for event in reversed(events):
        if str(event.get("kind", "")) in wanted:
            return event
    return {}


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def main() -> int:
    """Run diagnostics for every configured project path."""

    args = parse_args()
    paths = resolve_project_paths(args.project)
    if not paths:
        raise SystemExit(
            "Set ECHOZERO_REAL_PROJECT_SMOKE=/path/to/project.ez or pass a path. "
            "Automated diagnostics use fake_output=yes speaker_audio=no. "
            "Manual hardware confirmation, if needed, must be run separately at low volume."
        )
    for project_path in paths:
        result = run_event_preview_diagnostics(
            project_path,
            sample_rate=max(1, int(args.sample_rate)),
            channels=max(1, int(args.channels)),
            blocks=max(1, int(args.blocks)),
            output_dir=args.output_dir,
        )
        print(json.dumps(asdict(result), sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
