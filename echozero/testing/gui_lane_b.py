"""gui-lane-b: Simulated GUI regression lane for deterministic timeline review.
Exists to exercise app and widget flows for coverage and artifact generation.
Never treat this module as the canonical app runtime or human-path proof surface.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from echozero.application.timeline.intents import (
    ApplyTransferPlan,
    ConfirmPullFromMA3,
    ExitPushToMA3Mode,
    OpenPullFromMA3Dialog,
    SelectLayer,
    SelectPullSourceEvents,
    SelectPullSourceTrack,
    SelectPullSourceTracks,
    SelectPullTargetLayer,
    SetPullImportMode,
)
from echozero.services.orchestrator import Orchestrator
from echozero.testing.analysis_mocks import (
    build_mock_orchestrator,
    write_test_model,
    write_test_wav,
)
from echozero.testing.app_flow import AppFlowHarness
from echozero.testing.gui_dsl import GuiScenario, load_scenario
from echozero.testing.gui_lane_b_support import (
    StepTrace,
    _LaneBSyncService,
    _slugify,
    _write_frame_video,
    click_first_event,
    resolve_layer,
    resolve_song,
    resolve_song_version,
    snapshot_harness,
    trigger_global_contract_action,
    trigger_layer_contract_action,
)


def _render_for_hit_testing(harness) -> None:
    from echozero.testing.gui_lane_b_support import render_for_hit_testing

    render_for_hit_testing(harness)


class _GuiLaneBVideoRecorder:
    def __init__(self, output_dir: Path, *, fps: int):
        self.output_dir = output_dir
        self.fps = max(1, fps)
        self.frame_paths: list[Path] = []

    @classmethod
    def create(cls, output_dir: Path, *, fps: int) -> "_GuiLaneBVideoRecorder":
        return cls(output_dir, fps=fps)

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


class GuiLaneBRunner:
    def __init__(
        self,
        *,
        scenario: GuiScenario,
        output_dir: Path | None = None,
        record_video: bool = False,
        fps: int = 8,
        analysis_service: Orchestrator | None = None,
        use_mock_analysis: bool = True,
    ):
        self._scenario = scenario
        self._output_dir = output_dir
        self._record_video = record_video
        self._fps = max(1, fps)
        self._analysis_service = analysis_service
        self._use_mock_analysis = use_mock_analysis
        base_root = (
            output_dir.parent
            if output_dir is not None
            else Path(tempfile.gettempdir()) / "EchoZero"
        )
        self._run_temp_root = base_root.resolve()

    def run(self) -> list[dict[str, object]]:
        trace: list[dict[str, object]] = []
        self._run_temp_root.mkdir(parents=True, exist_ok=True)
        harness = AppFlowHarness(
            initial_project_name=self._scenario.name,
            working_dir_root=self._run_temp_root / "gui-lane-b-working",
            analysis_service=self._resolved_orchestrator(),
            sync_service=_LaneBSyncService(),
        )
        _render_for_hit_testing(harness)
        recorder = (
            None
            if self._output_dir is None or not self._record_video
            else _GuiLaneBVideoRecorder.create(
                self._output_dir,
                fps=self._fps,
            )
        )

        try:
            if self._output_dir is not None:
                self._output_dir.mkdir(parents=True, exist_ok=True)
            if recorder is not None:
                recorder.capture(harness, "initial")

            for index, step in enumerate(self._scenario.steps):
                label = step.label or step.action
                try:
                    self._execute_step(
                        harness, action=step.action, params=self._resolve_step_params(step.params)
                    )
                    status = "passed"
                    error = None
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
                trace.append(
                    asdict(
                        StepTrace(
                            index=index,
                            action=step.action,
                            label=label,
                            status=status,
                            error=error,
                            snapshot=snapshot_harness(harness),
                        )
                    )
                )
                if recorder is not None:
                    recorder.capture(harness, f"{index:03d}-{label}")
                if status == "failed":
                    break

            if self._output_dir is not None:
                (self._output_dir / "trace.json").write_text(
                    json.dumps(trace, indent=2), encoding="utf-8"
                )
                if recorder is not None:
                    video_path = recorder.finalize()
                    if video_path is not None:
                        (self._output_dir / "artifacts.json").write_text(
                            json.dumps(
                                {
                                    "scenario": self._scenario.name,
                                    "trace": "trace.json",
                                    "video": video_path.name,
                                    "frame_count": len(recorder.frame_paths),
                                    "proof_classification": "simulated_gui_capture",
                                    "operator_demo_valid": False,
                                    "analysis_mode": (
                                        "mock" if self._use_mock_analysis else "runtime"
                                    ),
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
            return trace
        finally:
            harness.runtime._is_dirty = False  # type: ignore[attr-defined]
            harness.launcher.confirm_close = lambda: True  # type: ignore[method-assign]
            harness.shutdown()

    def _resolved_orchestrator(self) -> Orchestrator | None:
        if self._analysis_service is not None:
            return self._analysis_service
        if self._use_mock_analysis:
            return build_mock_orchestrator()
        return None

    def _execute_step(
        self, harness: AppFlowHarness, *, action: str, params: dict[str, object]
    ) -> None:
        if action == "song.add":
            self._ensure_test_asset(Path(str(params["audio_path"])))
            with (
                patch(
                    "echozero.ui.qt.timeline.widget.QInputDialog.getText",
                    return_value=(str(params["title"]), True),
                ),
                patch(
                    "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
                    return_value=(str(params["audio_path"]), "Audio Files"),
                ),
            ):
                trigger_global_contract_action(harness, "song.add")
        elif action == "song.select":
            song = resolve_song(harness, params)
            trigger_global_contract_action(
                harness,
                "song.select",
                params={"song_id": song.id},
            )
        elif action == "song.version.switch":
            version = resolve_song_version(harness, params)
            trigger_global_contract_action(
                harness,
                "song.version.switch",
                params={"song_version_id": version.id},
            )
        elif action == "song.version.add":
            self._ensure_test_asset(Path(str(params["audio_path"])))
            resolved_params = {
                "audio_path": str(params["audio_path"]),
            }
            song = resolve_song(harness, params, required=False)
            if song is not None:
                resolved_params["song_id"] = song.id
            label = params.get("label")
            if isinstance(label, str):
                resolved_params["label"] = label
            trigger_global_contract_action(
                harness,
                "song.version.add",
                params=resolved_params,
            )
        elif action == "timeline.extract_stems":
            layer = resolve_layer(harness, params)
            trigger_layer_contract_action(harness, layer.layer_id, "timeline.extract_stems")
        elif action == "timeline.extract_drum_events":
            layer = resolve_layer(harness, params)
            trigger_layer_contract_action(harness, layer.layer_id, "timeline.extract_drum_events")
        elif action == "timeline.extract_song_sections":
            layer = resolve_layer(harness, params)
            trigger_layer_contract_action(harness, layer.layer_id, "timeline.extract_song_sections")
        elif action == "timeline.extract_classified_drums":
            layer = resolve_layer(harness, params)
            trigger_layer_contract_action(
                harness, layer.layer_id, "timeline.extract_classified_drums"
            )
        elif action == "timeline.classify_drum_events":
            layer = resolve_layer(harness, params)
            model_path = Path(str(params["model_path"]))
            self._ensure_test_asset(model_path)
            with patch(
                "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
                return_value=(str(model_path), "PyTorch Models"),
            ):
                trigger_layer_contract_action(
                    harness, layer.layer_id, "timeline.classify_drum_events"
                )
        elif action == "selection.first_event":
            layer = (
                resolve_layer(harness, params)
                if "layer_id" in params or "layer_title" in params
                else None
            )
            click_first_event(harness, layer_id=None if layer is None else str(layer.layer_id))
        elif action == "timeline.nudge_selection":
            direction = str(params["direction"])
            steps = int(params.get("steps", 1))
            modifiers = (
                Qt.KeyboardModifier.ShiftModifier
                if steps >= 10
                else Qt.KeyboardModifier.NoModifier
            )
            key = Qt.Key.Key_Left if direction == "left" else Qt.Key.Key_Right
            QTest.keyClick(harness.widget._canvas, key, modifiers)
            QApplication.processEvents()
        elif action == "timeline.duplicate_selection":
            steps = int(params.get("steps", 1))
            modifiers = Qt.KeyboardModifier.ControlModifier
            if steps >= 10:
                modifiers |= Qt.KeyboardModifier.ShiftModifier
            QTest.keyClick(harness.widget._canvas, Qt.Key.Key_D, modifiers)
            QApplication.processEvents()
        elif action == "transfer.workspace_open":
            layer = (
                resolve_layer(harness, params)
                if "layer_id" in params or "layer_title" in params
                else None
            )
            if layer is None:
                raise RuntimeError(
                    "transfer.workspace_open requires params.layer_id or params.layer_title."
                )
            direction = str(params.get("direction", "")).lower()
            if direction == "push":
                self._open_push_workspace(harness, layer_id=str(layer.layer_id))
            elif direction == "pull":
                if harness.presentation().manual_push_flow.push_mode_active:
                    harness.widget._dispatch(ExitPushToMA3Mode())
                harness.widget._dispatch(SelectLayer(layer.layer_id))
                harness.widget._dispatch(OpenPullFromMA3Dialog())
                flow = harness.presentation().manual_pull_flow
                if flow.available_tracks:
                    track = flow.available_tracks[0]
                    harness.widget._dispatch(
                        SelectPullSourceTracks(source_track_coords=[track.coord])
                    )
                    harness.widget._dispatch(SelectPullSourceTrack(source_track_coord=track.coord))
                    flow = harness.presentation().manual_pull_flow
                    if flow.available_events:
                        selected_event_ids = [str(flow.available_events[0].event_id)]
                        harness.widget._dispatch(
                            SelectPullSourceEvents(selected_ma3_event_ids=selected_event_ids)
                        )
                    if flow.available_target_layers:
                        harness.widget._dispatch(
                            SelectPullTargetLayer(
                                target_layer_id=flow.available_target_layers[0].layer_id
                            )
                        )
                    harness.widget._dispatch(SetPullImportMode(import_mode="main"))
                    flow = harness.presentation().manual_pull_flow
                    if (
                        flow.active_source_track_coord
                        and flow.target_layer_id
                        and flow.selected_ma3_event_ids
                    ):
                        harness.widget._dispatch(
                            ConfirmPullFromMA3(
                                source_track_coord=flow.active_source_track_coord,
                                selected_ma3_event_ids=list(flow.selected_ma3_event_ids),
                                target_layer_id=flow.target_layer_id,
                                import_mode=flow.import_mode,
                            )
                        )
            else:
                raise RuntimeError(
                    "transfer.workspace_open requires params.direction in push/pull."
                )
        elif action == "transfer.plan_apply":
            plan = harness.presentation().batch_transfer_plan
            if plan is None:
                raise RuntimeError(
                    "Lane B transfer.plan_apply requires an active batch transfer plan."
                )
            harness.widget._dispatch(ApplyTransferPlan(plan_id=plan.plan_id))
            QApplication.processEvents()
        elif action == "sync.enable":
            harness.enable_sync()
        elif action == "sync.disable":
            harness.disable_sync()
        elif action == "capture.screenshot":
            if self._output_dir is None:
                raise ValueError("capture.screenshot action requires an output_dir")
            screenshot_path = self._output_dir / str(params["filename"])
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            if not harness.widget.grab().save(str(screenshot_path)):
                raise RuntimeError(f"Failed to save screenshot: {screenshot_path}")
        else:
            raise ValueError(f"Unsupported action: {action}")
        _render_for_hit_testing(harness)

    @staticmethod
    def _open_push_workspace(harness: AppFlowHarness, *, layer_id: str) -> None:
        last_error: Exception | None = None
        for action_id in ("send_layer_to_ma3", "push_to_ma3", "send_to_ma3"):
            try:
                trigger_layer_contract_action(harness, layer_id, action_id)
                return
            except RuntimeError as exc:
                last_error = exc
                if "could not find inspector action" not in str(exc).lower():
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Lane B could not open push workspace.")

    def _resolve_step_params(self, params: dict[str, object]) -> dict[str, object]:
        resolved: dict[str, object] = {}
        for key, value in params.items():
            if isinstance(value, str):
                resolved[key] = value.replace("__RUN_TEMP__", self._run_temp_root.as_posix())
            else:
                resolved[key] = value
        return resolved

    @staticmethod
    def _ensure_test_asset(path: Path) -> None:
        if path.exists():
            return
        if path.suffix.lower() == ".wav":
            write_test_wav(path)
        elif path.suffix.lower() == ".pth":
            write_test_model(path)


def run_scenario_file(
    *,
    scenario_path: str | Path,
    output_dir: str | Path | None = None,
    record_video: bool = False,
    fps: int = 8,
    analysis_service: Orchestrator | None = None,
    use_mock_analysis: bool = True,
) -> list[dict[str, object]]:
    scenario = load_scenario(scenario_path)
    resolved_output = None if output_dir is None else Path(output_dir)
    return GuiLaneBRunner(
        scenario=scenario,
        output_dir=resolved_output,
        record_video=record_video,
        fps=fps,
        analysis_service=analysis_service,
        use_mock_analysis=use_mock_analysis,
    ).run()


__all__ = ["GuiLaneBRunner", "run_scenario_file"]
