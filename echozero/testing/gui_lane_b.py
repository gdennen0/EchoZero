from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from echozero.application.timeline.intents import OpenPullFromMA3Dialog, OpenPushToMA3Dialog, SelectLayer
from echozero.application.session.models import ManualPullEventOption, ManualPullTrackOption, ManualPushTrackOption
from echozero.application.presentation.inspector_contract import (
    InspectorAction,
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
)
from echozero.application.shared.enums import SyncMode
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.transport.models import TransportState
from echozero.testing.analysis_mocks import build_mock_analysis_service, write_test_model, write_test_wav
from echozero.testing.app_flow import AppFlowHarness
from echozero.testing.gui_dsl import GuiScenario, load_scenario


@dataclass(slots=True)
class StepTrace:
    index: int
    action: str
    label: str
    status: str
    snapshot: dict[str, object]
    error: str | None = None


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


class GuiLaneBRunner:
    def __init__(self, *, scenario: GuiScenario, output_dir: Path | None = None):
        self._scenario = scenario
        self._output_dir = output_dir
        base_root = output_dir.parent if output_dir is not None else Path(tempfile.gettempdir()) / "EchoZero"
        self._run_temp_root = base_root.resolve()

    def run(self) -> list[dict[str, object]]:
        trace: list[dict[str, object]] = []
        self._run_temp_root.mkdir(parents=True, exist_ok=True)
        harness = AppFlowHarness(
            initial_project_name=self._scenario.name,
            working_dir_root=self._run_temp_root / "gui-lane-b-working",
            analysis_service=build_mock_analysis_service(),
            sync_service=_LaneBSyncService(),
        )
        _render_for_hit_testing(harness)

        try:
            if self._output_dir is not None:
                self._output_dir.mkdir(parents=True, exist_ok=True)

            for index, step in enumerate(self._scenario.steps):
                label = step.label or step.action
                try:
                    self._execute_step(harness, action=step.action, params=self._resolve_step_params(step.params))
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
                            snapshot=_snapshot_harness(harness),
                        )
                    )
                )
                if status == "failed":
                    break

            if self._output_dir is not None:
                (self._output_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
            return trace
        finally:
            harness.runtime._is_dirty = False  # type: ignore[attr-defined]
            harness.launcher.confirm_close = lambda: True  # type: ignore[method-assign]
            harness.shutdown()

    def _execute_step(self, harness: AppFlowHarness, *, action: str, params: dict[str, object]) -> None:
        if action == "add_song_from_path":
            self._ensure_test_asset(Path(str(params["audio_path"])))
            with patch(
                "echozero.ui.qt.timeline.widget.QInputDialog.getText",
                return_value=(str(params["title"]), True),
            ), patch(
                "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
                return_value=(str(params["audio_path"]), "Audio Files"),
            ):
                _trigger_global_contract_action(harness, "add_song_from_path")
        elif action == "extract_stems":
            layer = _resolve_layer(harness, params)
            _trigger_layer_contract_action(harness, layer.layer_id, "extract_stems")
        elif action == "extract_drum_events":
            layer = _resolve_layer(harness, params)
            _trigger_layer_contract_action(harness, layer.layer_id, "extract_drum_events")
        elif action == "classify_drum_events":
            layer = _resolve_layer(harness, params)
            model_path = Path(str(params["model_path"]))
            self._ensure_test_asset(model_path)
            with patch(
                "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
                return_value=(str(model_path), "PyTorch Models"),
            ):
                _trigger_layer_contract_action(harness, layer.layer_id, "classify_drum_events")
        elif action == "trigger_action":
            harness.trigger_action(str(params["action_id"]))
        elif action == "select_first_event":
            layer = _resolve_layer(harness, params) if "layer_id" in params or "layer_title" in params else None
            _click_first_event(harness, layer_id=None if layer is None else str(layer.layer_id))
        elif action in {"nudge", "nudge_selected_events"}:
            direction = str(params["direction"])
            steps = int(params.get("steps", 1))
            modifiers = Qt.KeyboardModifier.ShiftModifier if steps >= 10 else Qt.KeyboardModifier.NoModifier
            key = Qt.Key.Key_Left if direction == "left" else Qt.Key.Key_Right
            QTest.keyClick(harness.widget._canvas, key, modifiers)
            QApplication.processEvents()
        elif action in {"duplicate", "duplicate_selected_events"}:
            steps = int(params.get("steps", 1))
            modifiers = Qt.KeyboardModifier.ControlModifier
            if steps >= 10:
                modifiers |= Qt.KeyboardModifier.ShiftModifier
            QTest.keyClick(harness.widget._canvas, Qt.Key.Key_D, modifiers)
            QApplication.processEvents()
        elif action == "open_push_surface":
            layer = _resolve_layer(harness, params) if "layer_id" in params or "layer_title" in params else None
            if layer is None:
                raise RuntimeError("open_push_surface requires params.layer_id or params.layer_title.")
            harness.widget._dispatch(SelectLayer(layer.layer_id))
            harness.widget._dispatch(
                OpenPushToMA3Dialog(selection_event_ids=harness.widget._selected_event_ids_for_selected_layers())
            )
        elif action == "open_pull_surface":
            layer = _resolve_layer(harness, params) if "layer_id" in params or "layer_title" in params else None
            if layer is None:
                raise RuntimeError("open_pull_surface requires params.layer_id or params.layer_title.")
            harness.widget._dispatch(SelectLayer(layer.layer_id))
            harness.widget._dispatch(OpenPullFromMA3Dialog())
        elif action == "enable_sync":
            harness.enable_sync()
        elif action == "disable_sync":
            harness.disable_sync()
        elif action == "screenshot":
            if self._output_dir is None:
                raise ValueError("screenshot action requires an output_dir")
            screenshot_path = self._output_dir / str(params["filename"])
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            if not harness.widget.grab().save(str(screenshot_path)):
                raise RuntimeError(f"Failed to save screenshot: {screenshot_path}")
        else:
            raise ValueError(f"Unsupported action: {action}")
        _render_for_hit_testing(harness)

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


def run_scenario_file(*, scenario_path: str | Path, output_dir: str | Path | None = None) -> list[dict[str, object]]:
    scenario = load_scenario(scenario_path)
    resolved_output = None if output_dir is None else Path(output_dir)
    return GuiLaneBRunner(scenario=scenario, output_dir=resolved_output).run()


def _render_for_hit_testing(harness: AppFlowHarness) -> None:
    harness.widget.resize(1200, 720)
    harness.widget.show()
    harness.widget.activateWindow()
    harness.widget.setFocus()
    harness.widget.repaint()
    QApplication.processEvents()
    harness.widget._canvas.repaint()
    QApplication.processEvents()


def _click_rect(widget, rect) -> None:
    center = rect.center().toPoint()
    QTest.mouseClick(widget._canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(center.x(), center.y()))
    QApplication.processEvents()


def _click_first_event(harness: AppFlowHarness, *, layer_id: str | None) -> None:
    for rect, candidate_layer_id, _take_id, _event_id in harness.widget._canvas._event_rects:
        if layer_id is None or str(candidate_layer_id) == layer_id:
            _click_rect(harness.widget, rect)
            return
    if layer_id is None:
        raise AssertionError("No event rects are available")
    raise AssertionError(f"No event rect found for layer_id={layer_id}")


def _click_layer_surface(harness: AppFlowHarness, *, surface: str, layer_id: str | None) -> None:
    rects = harness.widget._canvas._push_rects if surface == "push" else harness.widget._canvas._pull_rects
    for rect, candidate_layer_id in rects:
        if layer_id is None or str(candidate_layer_id) == layer_id:
            _click_rect(harness.widget, rect)
            return
    if layer_id is None:
        raise AssertionError(f"No {surface} rects are available")
    raise AssertionError(f"No {surface} rect found for layer_id={layer_id}")


def _resolve_layer(harness: AppFlowHarness, params: dict[str, object]):
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
        matches = [layer for layer in presentation.layers if layer.title.strip().casefold() == wanted]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise RuntimeError(f"Lane B could not find layer_title '{layer_title}'.")
        raise RuntimeError(f"Lane B found multiple layers matching layer_title '{layer_title}'.")
    raise RuntimeError("Lane B action requires params.layer_id or params.layer_title.")


def _trigger_global_contract_action(harness: AppFlowHarness, action_id: str) -> None:
    contract = build_timeline_inspector_contract(harness.presentation())
    harness.widget._trigger_contract_action(_find_contract_action(contract.context_sections, action_id))
    QApplication.processEvents()


def _trigger_layer_contract_action(harness: AppFlowHarness, layer_id, action_id: str) -> None:
    contract = build_timeline_inspector_contract(
        harness.presentation(),
        hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=layer_id),
    )
    harness.widget._trigger_contract_action(_find_contract_action(contract.context_sections, action_id))
    QApplication.processEvents()


def _find_contract_action(context_sections, action_id: str) -> InspectorAction:
    for section in context_sections:
        for action in section.actions:
            if action.action_id == action_id:
                return action
    raise RuntimeError(f"Lane B could not find inspector action '{action_id}'.")


def _snapshot_harness(harness: AppFlowHarness) -> dict[str, object]:
    presentation = harness.presentation()
    return {
        "selected_layer_id": None if presentation.selected_layer_id is None else str(presentation.selected_layer_id),
        "selected_take_id": None if presentation.selected_take_id is None else str(presentation.selected_take_id),
        "selected_event_ids": [str(event_id) for event_id in presentation.selected_event_ids],
        "push_mode_active": bool(presentation.manual_push_flow.push_mode_active),
        "pull_workspace_active": bool(presentation.manual_pull_flow.workspace_active),
        "batch_transfer_plan_id": None if presentation.batch_transfer_plan is None else presentation.batch_transfer_plan.plan_id,
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
