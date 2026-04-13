from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")

from PyQt6.QtWidgets import QApplication

from echozero.application.presentation.models import (
    LayerPresentation,
    ManualPullEventOptionPresentation,
    ManualPullFlowPresentation,
    ManualPullTargetOptionPresentation,
    ManualPullTrackOptionPresentation,
    ManualPushFlowPresentation,
    ManualPushTrackOptionPresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import SyncMode
from echozero.application.sync.models import LiveSyncState
from echozero.application.timeline.intents import Pause, Play, Seek, SelectEvent, SelectLayer, Stop, ToggleLayerExpanded
from echozero.ui.qt.app_shell import AppRuntimeProfile, AppShellRuntime, build_app_shell
from echozero.ui.qt.timeline.demo_app import build_demo_app, build_real_data_demo_app
from echozero.ui.qt.timeline.test_harness import estimate_full_window_height
from echozero.ui.qt.timeline.widget import TimelineWidget


DEFAULT_REFERENCE_ROOT = Path("C:/Users/griff/.openclaw/workspace/tmp/timeline-demo")
DEFAULT_RELEASES_ROOT = Path("artifacts/releases/test")
@dataclass(slots=True)
class ScenarioResult:
    group: str
    name: str
    status: str
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_OPENGL", "software")
    return QApplication.instance() or QApplication([])


def _relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _save_widget_png(widget: TimelineWidget, output_path: Path) -> Path:
    _app()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    widget.show()
    _app().processEvents()
    widget.grab().save(str(output_path))
    _app().processEvents()
    return output_path.resolve()


def _ffmpeg_exe() -> str | None:
    for candidate in ("ffmpeg", "C:/ffmpeg/bin/ffmpeg.exe"):
        if shutil.which(candidate) or Path(candidate).exists():
            return shutil.which(candidate) or candidate
    return None


def _write_looped_mp4(image_path: Path, output_path: Path, fps: int) -> Path | None:
    ffmpeg = _ffmpeg_exe()
    if ffmpeg is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(max(1, fps)),
        "-t",
        "0.75",
        "-i",
        str(image_path),
        "-vf",
        "format=yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return output_path.resolve()


def _snapshot(
    *,
    widget: TimelineWidget,
    run_folder: Path,
    group: str,
    name: str,
    record: bool,
    fps: int,
    notes: list[str] | None = None,
) -> ScenarioResult:
    group_root = run_folder / group
    screenshot_path = _save_widget_png(widget, group_root / f"{name}.png")
    artifacts = {"screenshot": _relative_path(screenshot_path, run_folder)}
    scenario_notes = list(notes or [])
    if record:
        video_path = _write_looped_mp4(screenshot_path, group_root / f"{name}.mp4", fps)
        if video_path is not None:
            artifacts["video"] = _relative_path(video_path, run_folder)
        else:
            scenario_notes.append("ffmpeg not available; skipped video capture")
    return ScenarioResult(group=group, name=name, status="passed", artifacts=artifacts, notes=scenario_notes)


def _first_layer_with_events(presentation: TimelinePresentation) -> LayerPresentation:
    for layer in presentation.layers:
        if layer.events:
            return layer
    raise ValueError("No layer with events found in demo presentation")


def _first_layer_with_takes(presentation: TimelinePresentation) -> LayerPresentation:
    for layer in presentation.layers:
        if layer.takes:
            return layer
    raise ValueError("No layer with takes found in demo presentation")


def _move_selected_events_presentation(presentation: TimelinePresentation, delta_seconds: float) -> TimelinePresentation:
    updated_layers: list[LayerPresentation] = []
    for layer in presentation.layers:
        updated_events = []
        for event in layer.events:
            if event.is_selected:
                updated_events.append(replace(event, start=max(0.0, event.start + delta_seconds), end=max(0.1, event.end + delta_seconds)))
            else:
                updated_events.append(event)
        updated_layers.append(replace(layer, events=updated_events))
    return replace(presentation, layers=updated_layers)


def _duplicate_selected_events_presentation(presentation: TimelinePresentation, offset_seconds: float = 0.5) -> TimelinePresentation:
    updated_layers: list[LayerPresentation] = []
    for layer in presentation.layers:
        next_events = list(layer.events)
        for event in layer.events:
            if event.is_selected:
                next_events.append(
                    replace(
                        event,
                        event_id=f"{event.event_id}_dup",
                        start=event.start + offset_seconds,
                        end=event.end + offset_seconds,
                        is_selected=False,
                        label=f"{event.label} Copy",
                    )
                )
        next_events.sort(key=lambda item: (item.start, item.end, str(item.event_id)))
        updated_layers.append(replace(layer, events=next_events))
    return replace(presentation, layers=updated_layers)


def _canonical_push_surface(presentation: TimelinePresentation) -> TimelinePresentation:
    layer_id = presentation.layers[0].layer_id
    return replace(
        presentation,
        selected_layer_id=layer_id,
        selected_layer_ids=[layer_id],
        manual_push_flow=ManualPushFlowPresentation(
            push_mode_active=True,
            selected_layer_ids=[layer_id],
            available_tracks=[
                ManualPushTrackOptionPresentation(coord="tc1_tg2_tr3", name="Track 3", note="Bass", event_count=8),
            ],
        ),
    )


def _canonical_pull_surface(presentation: TimelinePresentation) -> TimelinePresentation:
    layer_id = presentation.layers[0].layer_id
    return replace(
        presentation,
        selected_layer_id=layer_id,
        selected_layer_ids=[layer_id],
        manual_pull_flow=ManualPullFlowPresentation(
            workspace_active=True,
            available_tracks=[
                ManualPullTrackOptionPresentation(coord="tc1_tg2_tr3", name="MA3 Track", note="Lead", event_count=2),
            ],
            selected_source_track_coords=["tc1_tg2_tr3"],
            active_source_track_coord="tc1_tg2_tr3",
            source_track_coord="tc1_tg2_tr3",
            available_events=[
                ManualPullEventOptionPresentation(event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5),
                ManualPullEventOptionPresentation(event_id="ma3_evt_2", label="Cue 2", start=2.0, end=2.5),
            ],
            selected_ma3_event_ids=["ma3_evt_1"],
            available_target_layers=[
                ManualPullTargetOptionPresentation(layer_id=layer_id, name=presentation.layers[0].title),
            ],
            target_layer_id=layer_id,
        ),
    )


def _canonical_sync_presentation(presentation: TimelinePresentation, enabled: bool) -> TimelinePresentation:
    updated_layers = [
        replace(
            layer,
            sync_mode=SyncMode.MA3 if enabled else SyncMode.NONE,
            sync_connected=enabled,
            live_sync_state=LiveSyncState.OBSERVE if enabled else LiveSyncState.OFF,
            sync_target_label="show_manager" if enabled else "",
        )
        for layer in presentation.layers
    ]
    return replace(presentation, layers=updated_layers, experimental_live_sync_enabled=enabled)


def _copy_file_into_bundle(source_path: Path, target_root: Path) -> Path:
    target_root.mkdir(parents=True, exist_ok=True)
    target_path = target_root / source_path.name
    shutil.copy2(source_path, target_path)
    return target_path.resolve()


def find_latest_smoke_report(releases_root: Path) -> Path | None:
    if not releases_root.exists():
        return None

    candidates = [
        path / "smoke-report.json"
        for path in releases_root.iterdir()
        if path.is_dir() and (path / "smoke-report.json").exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda value: value.parent.name, reverse=True)[0]


def collect_reference_artifacts(
    *,
    run_folder: Path,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    **_: object,
) -> list[ScenarioResult]:
    specs = [
        (
            "reference_inclusion",
            "layer_centric_push_video",
            reference_root / "layer-centric-v1-full-flow" / "push_full_flow_20260412.mp4",
        ),
        (
            "reference_inclusion",
            "layer_centric_pull_video",
            reference_root / "layer-centric-v1-full-flow" / "pull_full_flow_20260412.mp4",
        ),
        (
            "reference_inclusion",
            "f6_pull_popup_video",
            reference_root / "f6-pull-popup" / "f6_pull_popup_zoom_scroll_demo_20260412.mp4",
        ),
        (
            "reference_inclusion",
            "live_sync_guardrail_video",
            reference_root / "live_sync_guardrail_states_20260411.mp4",
        ),
    ]
    results: list[ScenarioResult] = []
    target_root = run_folder / "references"
    for group, name, source_path in specs:
        if source_path.exists():
            copied = _copy_file_into_bundle(source_path, target_root)
            results.append(
                ScenarioResult(
                    group=group,
                    name=name,
                    status="passed",
                    artifacts={"reference": _relative_path(copied, run_folder)},
                )
            )
        else:
            results.append(
                ScenarioResult(
                    group=group,
                    name=name,
                    status="missing",
                    notes=[f"missing reference artifact: {source_path}"],
                )
            )
    return results


def include_packaged_app_evidence(
    *,
    run_folder: Path,
    releases_root: Path = DEFAULT_RELEASES_ROOT,
    **_: object,
) -> list[ScenarioResult]:
    smoke_report = find_latest_smoke_report(releases_root)
    if smoke_report is None:
        return [
            ScenarioResult(
                group="packaged_app_evidence",
                name="latest_smoke_report",
                status="skipped",
                notes=[f"no smoke-report.json found under {releases_root}"],
            )
        ]

    copied = _copy_file_into_bundle(smoke_report, run_folder / "packaged-evidence")
    return [
        ScenarioResult(
            group="packaged_app_evidence",
            name="latest_smoke_report",
            status="passed",
            artifacts={"smoke_report": _relative_path(copied, run_folder)},
            notes=[f"source={smoke_report}"],
        )
    ]


def canonical_app_lifecycle(*, run_folder: Path, record: bool, fps: int, **_: object) -> list[ScenarioResult]:
    app = _app()
    scenario_root = run_folder / "canonical_app_lifecycle"
    working_root = scenario_root / "working"
    project_path = scenario_root / "real-app-demo.ez"
    runtime = build_app_shell(
        profile=AppRuntimeProfile.PRODUCTION,
        use_demo_fixture=False,
        working_dir_root=working_root,
        initial_project_name="EchoZero Demo Suite",
    )
    if not isinstance(runtime, AppShellRuntime):
        raise TypeError("Expected canonical AppShellRuntime for demo suite")

    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    results: list[ScenarioResult] = []

    def capture_current(name: str, presentation: TimelinePresentation | None = None, notes: list[str] | None = None) -> None:
        current = runtime.presentation() if presentation is None else presentation
        widget.resize(1440, estimate_full_window_height(current))
        widget.set_presentation(current)
        widget.show()
        app.processEvents()
        results.append(
            _snapshot(
                widget=widget,
                run_folder=run_folder,
                group="canonical_app_lifecycle",
                name=name,
                record=record,
                fps=fps,
                notes=notes,
            )
        )

    try:
        capture_current("baseline")

        runtime.save_project_as(project_path)
        capture_current("save_as")

        layer_id = runtime.presentation().layers[0].layer_id
        runtime.dispatch(ToggleLayerExpanded(layer_id))
        capture_current("dirty_change")

        runtime.save_project()
        capture_current("save")

        runtime.new_project("EchoZero Demo Suite New")
        capture_current("new")

        runtime.open_project(project_path)
        capture_current("open")

        push_presentation = _canonical_push_surface(runtime.presentation())
        capture_current("push_surface_visible", push_presentation)

        pull_presentation = _canonical_pull_surface(runtime.presentation())
        capture_current("pull_surface_visible", pull_presentation)

        runtime.enable_sync(SyncMode.MA3)
        sync_enabled = _canonical_sync_presentation(runtime.presentation(), True)
        capture_current("sync_enabled", sync_enabled)

        runtime.disable_sync()
        sync_disabled = _canonical_sync_presentation(runtime.presentation(), False)
        capture_current("sync_disabled", sync_disabled)
    finally:
        widget.close()
        runtime.shutdown()
        app.processEvents()
    return results


def fixture_rich_editing_flow(*, run_folder: Path, record: bool, fps: int, **_: object) -> list[ScenarioResult]:
    app = _app()
    demo = build_demo_app()
    widget = TimelineWidget(demo.presentation(), on_intent=demo.dispatch, runtime_audio=demo.runtime_audio)
    results: list[ScenarioResult] = []

    def capture_current(name: str, presentation: TimelinePresentation | None = None, notes: list[str] | None = None) -> None:
        current = demo.presentation() if presentation is None else presentation
        widget.resize(1440, estimate_full_window_height(current))
        widget.set_presentation(current)
        widget.show()
        app.processEvents()
        results.append(
            _snapshot(
                widget=widget,
                run_folder=run_folder,
                group="fixture_rich_editing_flow",
                name=name,
                record=record,
                fps=fps,
                notes=notes,
            )
        )

    try:
        baseline = demo.presentation()
        capture_current("baseline", baseline)

        event_layer = _first_layer_with_events(baseline)
        demo.dispatch(SelectLayer(event_layer.layer_id))
        capture_current("layer_selection")

        selected_layer = demo.presentation()
        event_layer = _first_layer_with_events(selected_layer)
        demo.dispatch(SelectEvent(event_layer.layer_id, event_layer.main_take_id, event_layer.events[0].event_id))
        capture_current("event_selection")

        nudged = _move_selected_events_presentation(demo.presentation(), 1.0 / 30.0)
        capture_current("nudge_selected", nudged)

        duplicated = _duplicate_selected_events_presentation(demo.presentation())
        capture_current("duplicate_selected", duplicated)

        moved = _move_selected_events_presentation(demo.presentation(), 1.0)
        capture_current("move_selected_events", moved)

        take_layer = _first_layer_with_takes(demo.presentation())
        if not take_layer.is_expanded:
            demo.dispatch(ToggleLayerExpanded(take_layer.layer_id))
        capture_current("take_lane_expand")

        take_menu = demo.presentation()
        take_layer = _first_layer_with_takes(take_menu)
        if take_layer.takes:
            take_menu = replace(
                take_menu,
                selected_layer_id=take_layer.layer_id,
                selected_layer_ids=[take_layer.layer_id],
                selected_take_id=take_layer.takes[0].take_id,
            )
        capture_current("take_action_menu", take_menu)

        demo.dispatch(Play())
        capture_current("transport_play")

        demo.dispatch(Pause())
        capture_current("transport_pause")

        demo.dispatch(Stop())
        capture_current("transport_stop")
    finally:
        widget.close()
        if demo.runtime_audio is not None:
            demo.runtime_audio.shutdown()
        app.processEvents()
    return results


def real_data_scenario(
    *,
    run_folder: Path,
    audio_path: Path | None,
    record: bool,
    fps: int,
    **_: object,
) -> list[ScenarioResult]:
    if audio_path is None or not audio_path.exists():
        return [
            ScenarioResult(
                group="real_data_scenario",
                name="real_audio_flow",
                status="skipped",
                notes=[f"audio path missing or not provided: {audio_path}"],
            )
        ]

    app = _app()
    working_root = run_folder / "real-data-working"
    demo, summary = build_real_data_demo_app(audio_path=audio_path, working_root=working_root)
    widget = TimelineWidget(demo.presentation(), on_intent=demo.dispatch, runtime_audio=demo.runtime_audio)
    results: list[ScenarioResult] = []

    def capture_current(name: str, presentation: TimelinePresentation | None = None, notes: list[str] | None = None) -> None:
        current = demo.presentation() if presentation is None else presentation
        widget.resize(1440, estimate_full_window_height(current))
        widget.set_presentation(current)
        widget.show()
        app.processEvents()
        results.append(
            _snapshot(
                widget=widget,
                run_folder=run_folder,
                group="real_data_scenario",
                name=name,
                record=record,
                fps=fps,
                notes=notes,
            )
        )

    try:
        capture_current("baseline", notes=[f"summary_type={type(summary).__name__}"])

        demo.dispatch(Seek(8.0))
        capture_current("seek")

        demo.dispatch(Play())
        capture_current("transport_play")

        demo.dispatch(Pause())
        capture_current("transport_pause")
    finally:
        widget.close()
        if demo.runtime_audio is not None:
            demo.runtime_audio.shutdown()
        app.processEvents()
    return results


ScenarioRunner = Callable[..., list[ScenarioResult]]

SCENARIO_ORDER = [
    "canonical_app_lifecycle",
    "fixture_rich_editing_flow",
    "reference_inclusion",
    "packaged_app_evidence",
    "real_data_scenario",
]

SCENARIO_RUNNERS: dict[str, ScenarioRunner] = {
    "canonical_app_lifecycle": canonical_app_lifecycle,
    "fixture_rich_editing_flow": fixture_rich_editing_flow,
    "reference_inclusion": collect_reference_artifacts,
    "packaged_app_evidence": include_packaged_app_evidence,
    "real_data_scenario": real_data_scenario,
}
