"""Support-only helpers for the simulated gui-lane-b runner.
Exists to keep sync fixtures, video capture, and hit-test helpers out of the lane runner entrypoint.
Never treat these helpers as canonical app runtime or human-path proof logic.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from PyQt6.QtCore import QPoint, QRectF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from echozero.application.presentation.inspector_contract import (
    InspectorAction,
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
)
from echozero.application.session.models import (
    ManualPullEventOption,
    ManualPullTrackOption,
    ManualPushTrackOption,
)
from echozero.application.shared.enums import SyncMode
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.transport.models import TransportState


@dataclass(slots=True)
class StepTrace:
    index: int
    action: str
    label: str
    status: str
    snapshot: dict[str, object]
    error: str | None = None


@dataclass(slots=True)
class _GuiLaneBVideoRecorder:
    output_dir: Path
    fps: int
    frame_paths: list[Path]

    @classmethod
    def create(cls, output_dir: Path, *, fps: int) -> "_GuiLaneBVideoRecorder":
        return cls(output_dir=output_dir, fps=max(1, fps), frame_paths=[])

    def capture(self, harness, name: str) -> Path:
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frames_dir / f"{len(self.frame_paths):03d}-{_slugify(name)}.png"
        if not harness.widget.grab().save(str(frame_path)):
            raise RuntimeError(f"Failed to save frame: {frame_path}")
        self.frame_paths.append(frame_path)
        return frame_path

    def finalize(self) -> Path | None:
        if not self.frame_paths:
            return None
        return _write_frame_video(
            self.frame_paths,
            self.output_dir / "gui-lane-b-simulated.mp4",
            fps=self.fps,
        )


class _LaneBSyncService(SyncService):
    def __init__(self, state: SyncState | None = None):
        self._state = state or SyncState(mode=SyncMode.NONE, connected=False, target_ref="lane_b")
        self._push_tracks = [
            ManualPushTrackOption(coord="tc1_tg2_tr3", name="Track 3", note="Bass", event_count=8),
            ManualPushTrackOption(coord="tc1_tg2_tr4", name="Track 4", note="Lead", event_count=4),
        ]
        self._pull_tracks = [
            ManualPullTrackOption(coord="tc1_tg2_tr3", name="Track 3", note="Bass", event_count=2),
            ManualPullTrackOption(coord="tc1_tg2_tr4", name="Track 4", note="Lead", event_count=1),
        ]
        self._pull_events = {
            "tc1_tg2_tr3": [
                ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5),
                ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2", start=2.0, end=2.5),
            ],
            "tc1_tg2_tr4": [
                ManualPullEventOption(event_id="ma3_evt_9", label="Cue 9", start=9.0, end=9.5),
            ],
        }

    def get_state(self) -> SyncState:
        return self._state

    def set_mode(self, mode: SyncMode) -> SyncState:
        self._state.mode = mode
        return self._state

    def connect(self) -> SyncState:
        if self._state.mode == SyncMode.NONE:
            self._state.mode = SyncMode.MA3
        self._state.connected = True
        self._state.health = "healthy"
        return self._state

    def disconnect(self) -> SyncState:
        self._state.connected = False
        self._state.mode = SyncMode.NONE
        self._state.health = "offline"
        return self._state

    def align_transport(self, transport: TransportState) -> TransportState:
        return transport

    def list_push_track_options(self) -> list[ManualPushTrackOption]:
        return list(self._push_tracks)

    def list_pull_track_options(self) -> list[ManualPullTrackOption]:
        return list(self._pull_tracks)

    def list_pull_source_events(self, source_track_coord: str) -> list[ManualPullEventOption]:
        return list(self._pull_events.get(source_track_coord, []))


def render_for_hit_testing(harness) -> None:
    harness.widget.resize(1200, 720)
    harness.widget.show()
    harness.widget.activateWindow()
    harness.widget.setFocus()
    harness.widget.repaint()
    QApplication.processEvents()
    harness.widget._canvas.repaint()
    QApplication.processEvents()


def click_first_event(harness, *, layer_id: str | None) -> None:
    for rect, candidate_layer_id, _take_id, _event_id in harness.widget._canvas._event_rects:
        if layer_id is None or str(candidate_layer_id) == layer_id:
            _click_rect(harness.widget, QRectF(rect))
            return
    if layer_id is None:
        raise AssertionError("No event specs are available")
    raise AssertionError(f"No event rect found for layer_id={layer_id}")


def resolve_layer(harness, params: dict[str, object]):
    presentation = harness.presentation()
    layer_id = params.get("layer_id")
    layer_title = params.get("layer_title")
    if isinstance(layer_id, str) and layer_id.strip():
        for layer in presentation.layers:
            if str(layer.layer_id) == layer_id.strip():
                return layer
        raise RuntimeError(f"Lane B could not find layer_id '{layer_id}'.")
    if isinstance(layer_title, str) and layer_title.strip():
        wanted = layer_title.strip().casefold()
        matches = [
            layer for layer in presentation.layers if layer.title.strip().casefold() == wanted
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise RuntimeError(f"Lane B could not find layer_title '{layer_title}'.")
        raise RuntimeError(f"Lane B found multiple layers matching layer_title '{layer_title}'.")
    raise RuntimeError("Lane B action requires params.layer_id or params.layer_title.")


def trigger_global_contract_action(
    harness,
    action_id: str,
    *,
    params: dict[str, object] | None = None,
) -> None:
    contract = build_timeline_inspector_contract(harness.presentation())
    try:
        action = find_contract_action(contract.context_sections, action_id)
    except RuntimeError:
        if action_id != "song.add":
            raise
        action = InspectorAction(action_id="song.add", label="Add Song")
    if params:
        action = replace(action, params={**action.params, **params})
    harness.widget._trigger_contract_action(action)
    QApplication.processEvents()


def trigger_layer_contract_action(harness, layer_id, action_id: str) -> None:
    contract = build_timeline_inspector_contract(
        harness.presentation(),
        hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=layer_id),
    )
    harness.widget._trigger_contract_action(find_contract_action(contract.context_sections, action_id))
    QApplication.processEvents()


def find_contract_action(context_sections, action_id: str) -> InspectorAction:
    for section in context_sections:
        for action in section.actions:
            if action.action_id == action_id:
                return action
    raise RuntimeError(f"Lane B could not find inspector action '{action_id}'.")


def resolve_song(
    harness,
    params: dict[str, object],
    *,
    required: bool = True,
):
    songs = harness.runtime.project_storage.songs.list_by_project(
        harness.runtime.project_storage.project.id
    )
    song_id = params.get("song_id")
    if isinstance(song_id, str) and song_id.strip():
        for song in songs:
            if song.id == song_id.strip():
                return song
        raise RuntimeError(f"Lane B could not find song_id '{song_id}'.")
    song_title = params.get("song_title")
    if isinstance(song_title, str) and song_title.strip():
        wanted = song_title.strip().casefold()
        matches = [song for song in songs if song.title.strip().casefold() == wanted]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise RuntimeError(f"Lane B could not find song_title '{song_title}'.")
        raise RuntimeError(f"Lane B found multiple songs matching song_title '{song_title}'.")
    if required:
        raise RuntimeError("Lane B action requires params.song_id or params.song_title.")
    return None


def resolve_song_version(harness, params: dict[str, object]):
    version_id = params.get("song_version_id")
    if isinstance(version_id, str) and version_id.strip():
        version = harness.runtime.project_storage.song_versions.get(version_id.strip())
        if version is None:
            raise RuntimeError(f"Lane B could not find song_version_id '{version_id}'.")
        return version
    version_label = params.get("version_label")
    if isinstance(version_label, str) and version_label.strip():
        song = resolve_song(harness, params, required=False)
        song_id = (
            song.id
            if song is not None
            else str(harness.runtime.session.active_song_id or "").strip()
        )
        if not song_id:
            raise RuntimeError(
                "Lane B requires an active song or params.song_id/song_title for version_label."
            )
        wanted = version_label.strip().casefold()
        matches = [
            version
            for version in harness.runtime.project_storage.song_versions.list_by_song(song_id)
            if version.label.strip().casefold() == wanted
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise RuntimeError(f"Lane B could not find version_label '{version_label}'.")
        raise RuntimeError(
            f"Lane B found multiple song versions matching version_label '{version_label}'."
        )
    raise RuntimeError(
        "Lane B action requires params.song_version_id or params.version_label."
    )


def snapshot_harness(harness) -> dict[str, object]:
    presentation = harness.presentation()
    return {
        "selected_layer_id": (
            None if presentation.selected_layer_id is None else str(presentation.selected_layer_id)
        ),
        "selected_take_id": (
            None if presentation.selected_take_id is None else str(presentation.selected_take_id)
        ),
        "selected_event_ids": [str(event_id) for event_id in presentation.selected_event_ids],
        "push_mode_active": bool(presentation.manual_push_flow.push_mode_active),
        "pull_workspace_active": bool(presentation.manual_pull_flow.workspace_active),
        "batch_transfer_plan_id": (
            None
            if presentation.batch_transfer_plan is None
            else presentation.batch_transfer_plan.plan_id
        ),
        "sync_connected": bool(harness.runtime.session.sync_state.connected),
        "sync_mode": harness.runtime.session.sync_state.mode.value,
        "layers": [
            {
                "layer_id": str(layer.layer_id),
                "title": layer.title,
                "kind": layer.kind.value,
                "event_count": len(layer.events),
                "push_target_label": layer.push_target_label,
                "push_row_status": layer.push_row_status,
                "pull_target_label": layer.pull_target_label,
                "pull_row_status": layer.pull_row_status,
                "sync_target_label": layer.sync_target_label,
                "sync_connected": bool(layer.sync_connected),
            }
            for layer in presentation.layers
        ],
    }


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "frame"


def _ffmpeg_exe() -> str | None:
    for candidate in ("ffmpeg", "/opt/homebrew/bin/ffmpeg", "C:/ffmpeg/bin/ffmpeg.exe"):
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
        if Path(candidate).exists():
            return candidate
    return None


def _write_frame_video(frame_paths: list[Path], output_path: Path, *, fps: int) -> Path | None:
    ffmpeg = _ffmpeg_exe()
    if ffmpeg is None or not frame_paths:
        return None

    concat_path = output_path.with_suffix(".ffconcat")
    concat_lines: list[str] = ["ffconcat version 1.0"]
    frame_duration = 0.75
    for frame_path in frame_paths:
        concat_lines.append(f"file '{frame_path.resolve().as_posix()}'")
        concat_lines.append(f"duration {frame_duration:.2f}")
    concat_lines.append(f"file '{frame_paths[-1].resolve().as_posix()}'")
    concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-vf",
        f"fps={max(1, fps)},format=yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return output_path.resolve()


def _click_rect(widget, rect) -> None:
    center = rect.center().toPoint()
    QTest.mouseClick(
        widget._canvas,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(center.x(), center.y()),
    )
    QApplication.processEvents()


__all__ = [
    "StepTrace",
    "_GuiLaneBVideoRecorder",
    "_LaneBSyncService",
    "click_first_event",
    "find_contract_action",
    "render_for_hit_testing",
    "resolve_layer",
    "resolve_song",
    "resolve_song_version",
    "snapshot_harness",
    "trigger_global_contract_action",
    "trigger_layer_contract_action",
]
