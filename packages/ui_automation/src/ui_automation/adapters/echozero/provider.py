from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QByteArray, QBuffer, QIODevice, QPoint, QPointF, QRect, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog, QInputDialog, QWidget

from echozero.application.presentation.inspector_contract import InspectorAction, build_timeline_inspector_contract
from echozero.application.shared.enums import SyncMode
from echozero.application.timeline.intents import Pause, Play, Stop
from echozero.testing.app_flow import AppFlowHarness

from ...core.models import (
    AutomationAction,
    AutomationBounds,
    AutomationHitTarget,
    AutomationObject,
    AutomationObjectFact,
    AutomationSnapshot,
    AutomationTarget,
)


@dataclass(slots=True)
class HarnessEchoZeroAutomationProvider:
    """In-process EchoZero automation provider for internal test support only."""

    working_dir_root: Path | None = None
    initial_project_name: str = "EchoZero Automation"
    window_width: int = 1440
    window_height: int = 720
    analysis_service: Any | None = None
    sync_service: Any | None = None
    simulate_ma3: bool = False
    simulate_ma3_osc: bool = False

    def attach(self) -> "EchoZeroAutomationBackend":
        harness = AppFlowHarness(
            simulate_ma3=self.simulate_ma3,
            simulate_ma3_osc=self.simulate_ma3_osc,
            working_dir_root=self.working_dir_root,
            initial_project_name=self.initial_project_name,
            analysis_service=self.analysis_service,
            sync_service=self.sync_service,
        )
        harness.widget.resize(self.window_width, self.window_height)
        harness.widget.show()
        harness._app.processEvents()
        return EchoZeroAutomationBackend(harness)


class _SurfaceAutomationHarness:
    def __init__(self, *, runtime, widget, launcher, app) -> None:
        self.runtime = runtime
        self.widget = widget
        self.launcher = launcher
        self._app = app

    def presentation(self):
        return self.runtime.presentation()

    def trigger_action(self, action_id: str):
        action_map = {
            "new": "new_project",
            "open": "open_project",
            "save": "save_project",
            "save_as": "save_project_as",
        }
        action_key = action_map.get(action_id, action_id)
        action = self.launcher.actions[action_key]
        action.trigger()
        self._app.processEvents()
        return self.runtime.presentation()

    def queue_open_path(self, path: str | Path) -> Path:
        queued = Path(path)
        original = self.launcher._choose_open_path
        self.launcher._choose_open_path = lambda: queued  # type: ignore[method-assign]
        self._queued_open_restore = original
        return queued

    def queue_save_path(self, path: str | Path) -> Path:
        queued = Path(path)
        original = self.launcher._choose_save_path
        self.launcher._choose_save_path = lambda: queued  # type: ignore[method-assign]
        self._queued_save_restore = original
        return queued

    def restore_dialog_paths(self) -> None:
        restore_open = getattr(self, "_queued_open_restore", None)
        if restore_open is not None:
            self.launcher._choose_open_path = restore_open  # type: ignore[method-assign]
            self._queued_open_restore = None
        restore_save = getattr(self, "_queued_save_restore", None)
        if restore_save is not None:
            self.launcher._choose_save_path = restore_save  # type: ignore[method-assign]
            self._queued_save_restore = None

    def enable_sync(self, mode: SyncMode = SyncMode.MA3):
        state = self.runtime.enable_sync(mode)
        self._app.processEvents()
        return state

    def disable_sync(self):
        state = self.runtime.disable_sync()
        self._app.processEvents()
        return state


def create_backend_for_surface(*, runtime, widget, launcher, app) -> "EchoZeroAutomationBackend":
    """Create the EchoZero automation backend against a live launcher surface."""

    return EchoZeroAutomationBackend(
        _SurfaceAutomationHarness(runtime=runtime, widget=widget, launcher=launcher, app=app)
    )


class EchoZeroAutomationBackend:
    def __init__(self, harness: AppFlowHarness):
        self._harness = harness
        self._root = harness.widget
        self._pointer_target_id: str | None = None
        self._pointer_position: tuple[float, float] | None = None

    def snapshot(self) -> AutomationSnapshot:
        self._render()
        presentation = self._harness.presentation()
        viewport = self._harness.widget.presentation
        targets = [
            AutomationTarget(
                kind="window",
                target_id="shell.root",
                label=self._root.windowTitle() or "EchoZero",
                bounds=self._widget_bounds(self._root),
                metadata={"project_title": presentation.title},
            ),
            AutomationTarget(
                kind="toolbar",
                target_id="shell.transport",
                parent_id="shell.root",
                label="Transport",
                bounds=self._widget_bounds(self._harness.widget._transport),
            ),
            AutomationTarget(
                kind="canvas",
                target_id="shell.timeline",
                parent_id="shell.root",
                label="Timeline",
                bounds=self._widget_bounds(self._harness.widget._canvas),
                metadata={"scroll_x": viewport.scroll_x},
            ),
            AutomationTarget(
                kind="ruler",
                target_id="shell.ruler",
                parent_id="shell.root",
                label="Ruler",
                bounds=self._widget_bounds(self._harness.widget._ruler),
            ),
        ]
        hit_targets: list[AutomationHitTarget] = []

        for layer in presentation.layers:
            header_rect = self._find_layer_rect(str(layer.layer_id))
            targets.append(
                AutomationTarget(
                    kind="layer",
                    target_id=self._layer_target_id(str(layer.layer_id)),
                    parent_id="shell.timeline",
                    label=layer.title,
                    bounds=None if header_rect is None else self._canvas_bounds(header_rect),
                    metadata={
                        "layer_id": str(layer.layer_id),
                        "kind": layer.kind.value,
                        "event_count": len(layer.events),
                        "selected": layer.layer_id == presentation.selected_layer_id,
                    },
                )
            )
            if header_rect is not None:
                hit_targets.append(
                    AutomationHitTarget(
                        target_id=self._layer_target_id(str(layer.layer_id)),
                        kind="layer",
                        bounds=self._canvas_bounds(header_rect),
                        metadata={"layer_id": str(layer.layer_id)},
                    )
                )
            push_rect = self._find_layer_surface_rect("push", str(layer.layer_id))
            if push_rect is not None:
                targets.append(
                    AutomationTarget(
                        kind="control",
                        target_id=f"timeline.push:{layer.layer_id}",
                        parent_id=self._layer_target_id(str(layer.layer_id)),
                        label=f"Push {layer.title}",
                        bounds=self._canvas_bounds(push_rect),
                    )
                )
            pull_rect = self._find_layer_surface_rect("pull", str(layer.layer_id))
            if pull_rect is not None:
                targets.append(
                    AutomationTarget(
                        kind="control",
                        target_id=f"timeline.pull:{layer.layer_id}",
                        parent_id=self._layer_target_id(str(layer.layer_id)),
                        label=f"Pull {layer.title}",
                        bounds=self._canvas_bounds(pull_rect),
                    )
                )

        for rect, layer_id, _take_id, event_id in self._harness.widget._canvas._event_rects:
            layer = next((item for item in presentation.layers if str(item.layer_id) == str(layer_id)), None)
            event_label = str(event_id)
            if layer is not None:
                for candidate in layer.events:
                    if str(candidate.event_id) == str(event_id):
                        event_label = candidate.label
                        break
            target_id = self._event_target_id(str(event_id))
            targets.append(
                AutomationTarget(
                    kind="event",
                    target_id=target_id,
                    parent_id=self._layer_target_id(str(layer_id)),
                    label=event_label,
                    bounds=self._canvas_bounds(rect),
                    time_seconds=self._event_time_seconds(str(layer_id), str(event_id)),
                    metadata={"layer_id": str(layer_id), "event_id": str(event_id)},
                )
            )
            hit_targets.append(
                AutomationHitTarget(
                    target_id=target_id,
                    kind="event",
                    bounds=self._canvas_bounds(rect),
                    metadata={"layer_id": str(layer_id), "event_id": str(event_id)},
                )
            )

        actions = [
            AutomationAction(action_id="transport.play", label="Play", group="transport"),
            AutomationAction(action_id="transport.pause", label="Pause", group="transport"),
            AutomationAction(action_id="transport.stop", label="Stop", group="transport"),
            AutomationAction(action_id="sync.enable", label="Enable Sync", group="sync"),
            AutomationAction(action_id="sync.disable", label="Disable Sync", group="sync"),
            AutomationAction(action_id="app.new", label="New Project", group="app"),
            AutomationAction(action_id="app.save", label="Save Project", group="app"),
            AutomationAction(action_id="app.save_as", label="Save Project As", group="app"),
            AutomationAction(action_id="add_song_from_path", label="Add Song From Path", group="app"),
        ]
        contract = build_timeline_inspector_contract(presentation)
        contract_actions = self._automation_actions_from_contract(contract)
        actions.extend(contract_actions)

        selection = []
        if presentation.selected_layer_id is not None:
            selection.append(self._layer_target_id(str(presentation.selected_layer_id)))
        for event_id in presentation.selected_event_ids:
            selection.append(self._event_target_id(str(event_id)))

        transport_bounds = self._transport_control_bounds()
        for control_name, bounds in transport_bounds.items():
            targets.append(
                AutomationTarget(
                    kind="control",
                    target_id=f"shell.transport.{control_name}",
                    parent_id="shell.transport",
                    label=control_name.replace("_", " ").title(),
                    bounds=bounds,
                )
            )

        automation_objects = self._automation_objects_from_contract(contract, contract_actions)
        focused_target_id = self._focused_target_id(selection)
        focused_object_id = automation_objects[0].object_id if automation_objects else None

        return AutomationSnapshot(
            app="EchoZero",
            selection=tuple(selection),
            focused_target_id=focused_target_id,
            focused_object_id=focused_object_id,
            sync={
                "connected": bool(self._harness.runtime.session.sync_state.connected),
                "mode": self._harness.runtime.session.sync_state.mode.value,
            },
            targets=tuple(targets),
            actions=tuple(actions),
            objects=automation_objects,
            hit_targets=tuple(hit_targets),
            artifacts={
                "project_title": presentation.title,
                "pointer_target_id": self._pointer_target_id,
                "pointer_position": self._pointer_position,
            },
        )

    def screenshot(self, *, target_id: str | None = None) -> bytes:
        self._render()
        if target_id is None:
            return self._pixmap_to_png_bytes(self._root.grab())
        target = self._target_lookup(target_id)
        if target is None or target.bounds is None:
            raise ValueError(f"Unknown or non-visual target_id: {target_id}")
        rect = QRect(
            int(target.bounds.x),
            int(target.bounds.y),
            max(1, int(target.bounds.width)),
            max(1, int(target.bounds.height)),
        )
        return self._pixmap_to_png_bytes(self._root.grab(rect))

    def click(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        _ = args
        self._render()
        if target_id.startswith("shell.transport."):
            control = target_id.removeprefix("shell.transport.")
            rect = self._harness.widget._transport._control_rects.get(control)
            if rect is None:
                raise ValueError(f"Unknown transport control: {target_id}")
            self._move_pointer_to_widget_rect(self._harness.widget._transport, rect, target_id=target_id)
            self._click_widget_rect(self._harness.widget._transport, rect)
            return self.snapshot()

        if target_id == "shell.timeline":
            rect = self._harness.widget._canvas.rect()
            self._move_pointer_to_widget_rect(self._harness.widget._canvas, rect, target_id=target_id)
            self._click_widget_rect(self._harness.widget._canvas, rect)
            return self.snapshot()

        rect = self._target_canvas_rect(target_id)
        if rect is None:
            raise ValueError(f"Unsupported click target: {target_id}")
        self._move_pointer_to_widget_rect(self._harness.widget._canvas, rect, target_id=target_id)
        self._click_widget_rect(self._harness.widget._canvas, rect)
        return self.snapshot()

    def move_pointer(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        _ = args
        self._render()
        self._move_pointer_to_target(target_id)
        return self.snapshot()

    def double_click(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        _ = args
        self._render()
        if target_id.startswith("shell.transport."):
            control = target_id.removeprefix("shell.transport.")
            rect = self._harness.widget._transport._control_rects.get(control)
            if rect is None:
                raise ValueError(f"Unknown transport control: {target_id}")
            self._move_pointer_to_widget_rect(self._harness.widget._transport, rect, target_id=target_id)
            self._double_click_widget_rect(self._harness.widget._transport, rect)
            return self.snapshot()
        rect = self._target_canvas_rect(target_id)
        if rect is None:
            raise ValueError(f"Unsupported double_click target: {target_id}")
        self._move_pointer_to_widget_rect(self._harness.widget._canvas, rect, target_id=target_id)
        self._double_click_widget_rect(self._harness.widget._canvas, rect)
        return self.snapshot()

    def hover(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        _ = args
        self._render()
        self._move_pointer_to_target(target_id)
        return self.snapshot()

    def type_text(
        self,
        target_id: str,
        text: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        _ = target_id, args
        self._root.setFocus()
        QTest.keyClicks(self._root, text)
        QApplication.processEvents()
        return self.snapshot()

    def press_key(
        self,
        key: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        _ = args
        key_mapping = {
            "space": Qt.Key.Key_Space,
            "left": Qt.Key.Key_Left,
            "right": Qt.Key.Key_Right,
            "up": Qt.Key.Key_Up,
            "down": Qt.Key.Key_Down,
            "escape": Qt.Key.Key_Escape,
            "d": Qt.Key.Key_D,
        }
        qt_key = key_mapping.get(key.strip().lower())
        if qt_key is None:
            raise ValueError(f"Unsupported key: {key}")
        QTest.keyClick(self._harness.widget._canvas, qt_key)
        QApplication.processEvents()
        return self.snapshot()

    def drag(
        self,
        target_id: str,
        destination: Any,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        _ = args
        self._render()
        source_rect = self._target_canvas_rect(target_id)
        if source_rect is None:
            raise ValueError(f"Unsupported drag target: {target_id}")
        source_point = source_rect.center().toPoint()
        destination_point = self._resolve_drag_destination(destination, source_point)
        self._drag_canvas(self._harness.widget._canvas, source_point, destination_point)
        return self.snapshot()

    def scroll(
        self,
        target_id: str,
        *,
        dx: int = 0,
        dy: int = 0,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        _ = dy, args
        if target_id not in {"shell.timeline", "shell.ruler", "shell.root"}:
            raise ValueError(f"Unsupported scroll target: {target_id}")
        bar = self._harness.widget._hscroll
        next_value = max(bar.minimum(), min(bar.maximum(), bar.value() + int(dx)))
        bar.setValue(next_value)
        QApplication.processEvents()
        return self.snapshot()

    def invoke(
        self,
        action_id: str,
        *,
        target_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        payload = dict(params or {})
        if target_id is not None:
            self._select_target(target_id)

        if action_id == "transport.play":
            self._harness.runtime.dispatch(Play())
        elif action_id == "transport.pause":
            self._harness.runtime.dispatch(Pause())
        elif action_id == "transport.stop":
            self._harness.runtime.dispatch(Stop())
        elif action_id == "sync.enable":
            self._harness.enable_sync(SyncMode.MA3)
        elif action_id == "sync.disable":
            self._harness.disable_sync()
        elif action_id == "app.new":
            project_name = str(payload.get("name", "EchoZero Project"))
            if project_name == "EchoZero Project":
                self._harness.trigger_action("new_project")
            else:
                self._harness.runtime.new_project(project_name)
                self._harness.widget.set_presentation(self._harness.runtime.presentation())
                QApplication.processEvents()
        elif action_id == "app.save":
            project_path = payload.get("path")
            if project_path is not None:
                target_path = Path(str(project_path))
                target_path.parent.mkdir(parents=True, exist_ok=True)
                self._harness.queue_save_path(target_path)
                try:
                    self._harness.trigger_action("save_project")
                finally:
                    self._restore_dialog_paths()
            else:
                self._harness.trigger_action("save_project")
        elif action_id == "app.save_as":
            if "path" not in payload:
                raise ValueError("app.save_as requires params.path")
            target_path = Path(str(payload["path"]))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            self._harness.queue_save_path(target_path)
            try:
                self._harness.trigger_action("save_project_as")
            finally:
                self._restore_dialog_paths()
        elif action_id == "app.open":
            if "path" not in payload:
                raise ValueError("app.open requires params.path")
            self._harness.queue_open_path(Path(str(payload["path"])))
            try:
                self._harness.trigger_action("open_project")
            finally:
                self._restore_dialog_paths()
        elif action_id == "add_song_from_path":
            contract_action = self._require_contract_action("add_song_from_path")
            with self._dialog_overrides(
                text_responses=[(str(payload["title"]), True)],
                open_file_responses=[(str(payload["audio_path"]), "")],
            ):
                self._harness.widget._trigger_contract_action(contract_action)
            QApplication.processEvents()
        elif action_id in {"extract_stems", "extract_drum_events", "extract_classified_drums"}:
            self._harness.widget._trigger_contract_action(self._require_contract_action(action_id))
            QApplication.processEvents()
        elif action_id == "classify_drum_events":
            with self._dialog_overrides(
                open_file_responses=[(str(payload["model_path"]), "")],
            ):
                self._harness.widget._trigger_contract_action(self._require_contract_action(action_id))
            QApplication.processEvents()
        else:
            self._harness.widget._trigger_contract_action(self._require_contract_action(action_id))
            QApplication.processEvents()
        return self.snapshot()

    def close(self) -> None:
        self._harness.runtime._is_dirty = False  # type: ignore[attr-defined]
        self._harness.launcher.confirm_close = lambda: True  # type: ignore[method-assign]
        self._harness.shutdown()

    def _select_target(self, target_id: str) -> None:
        if target_id in {"shell.root", "shell.timeline", "shell.transport", "shell.ruler"}:
            return
        self.click(target_id)

    def _move_pointer_to_target(self, target_id: str) -> None:
        if target_id.startswith("shell.transport."):
            control = target_id.removeprefix("shell.transport.")
            rect = self._harness.widget._transport._control_rects.get(control)
            if rect is None:
                raise ValueError(f"Unknown transport control: {target_id}")
            self._move_pointer_to_widget_rect(self._harness.widget._transport, rect, target_id=target_id)
            return
        if target_id == "shell.timeline":
            self._move_pointer_to_widget_rect(self._harness.widget._canvas, self._harness.widget._canvas.rect(), target_id=target_id)
            return
        if target_id == "shell.ruler":
            self._move_pointer_to_widget_rect(self._harness.widget._ruler, self._harness.widget._ruler.rect(), target_id=target_id)
            return
        rect = self._target_canvas_rect(target_id)
        if rect is None:
            raise ValueError(f"Unsupported pointer target: {target_id}")
        self._move_pointer_to_widget_rect(self._harness.widget._canvas, rect, target_id=target_id)

    def _find_contract_action(self, action_id: str) -> InspectorAction | None:
        contract = build_timeline_inspector_contract(self._harness.presentation())
        for section in contract.context_sections:
            for action in section.actions:
                if action.action_id == action_id:
                    return action
        return None

    def _require_contract_action(self, action_id: str) -> InspectorAction:
        action = self._find_contract_action(action_id)
        if action is None:
            raise ValueError(f"Unsupported action: {action_id}")
        return action

    def _restore_dialog_paths(self) -> None:
        restore = getattr(self._harness, "restore_dialog_paths", None)
        if callable(restore):
            restore()

    @contextmanager
    def _dialog_overrides(
        self,
        *,
        text_responses: list[tuple[str, bool]] | None = None,
        open_file_responses: list[tuple[str, str]] | None = None,
    ):
        original_get_text = QInputDialog.getText
        original_get_open = QFileDialog.getOpenFileName
        queued_text = list(text_responses or [])
        queued_open = list(open_file_responses or [])

        def _get_text(*_args, **_kwargs):
            if queued_text:
                return queued_text.pop(0)
            return ("", False)

        def _get_open(*_args, **_kwargs):
            if queued_open:
                return queued_open.pop(0)
            return ("", "")

        QInputDialog.getText = staticmethod(_get_text)  # type: ignore[method-assign]
        QFileDialog.getOpenFileName = staticmethod(_get_open)  # type: ignore[method-assign]
        try:
            yield
        finally:
            QInputDialog.getText = original_get_text  # type: ignore[method-assign]
            QFileDialog.getOpenFileName = original_get_open  # type: ignore[method-assign]

    def _automation_actions_from_contract(self, contract) -> list[AutomationAction]:
        actions: list[AutomationAction] = []
        for section in contract.context_sections:
            for action in section.actions:
                actions.append(
                    AutomationAction(
                        action_id=action.action_id,
                        label=action.label,
                        enabled=action.enabled,
                        group=action.group,
                        params=dict(action.params),
                    )
                )
        return actions

    def _automation_objects_from_contract(
        self,
        contract,
        contract_actions: list[AutomationAction],
    ) -> tuple[AutomationObject, ...]:
        if contract.identity is None:
            return ()
        fact_rows: list[AutomationObjectFact] = []
        for section in contract.sections:
            for row in section.rows:
                fact_rows.append(AutomationObjectFact(label=row.label, value=row.value))
        target_id = self._contract_target_id(contract.identity.object_type, contract.identity.object_id)
        return (
            AutomationObject(
                object_id=contract.identity.object_id,
                object_type=contract.identity.object_type,
                label=contract.identity.label,
                target_id=target_id,
                facts=tuple(fact_rows),
                actions=tuple(contract_actions),
                metadata={"title": contract.title},
            ),
        )

    def _contract_target_id(self, object_type: str, object_id: str) -> str | None:
        if object_type == "layer":
            return self._layer_target_id(object_id)
        if object_type == "event":
            return self._event_target_id(object_id)
        return None

    def _focused_target_id(self, selection: list[str]) -> str | None:
        if selection:
            return selection[0]
        return self._pointer_target_id

    def _target_lookup(self, target_id: str) -> AutomationTarget | None:
        for target in self.snapshot().targets:
            if target.target_id == target_id:
                return target
        return None

    def _target_canvas_rect(self, target_id: str):
        canvas = self._harness.widget._canvas
        if target_id.startswith("timeline.layer:"):
            return self._find_layer_rect(target_id.removeprefix("timeline.layer:"))
        if target_id.startswith("timeline.event:"):
            event_id = target_id.removeprefix("timeline.event:")
            for rect, _layer_id, _take_id, candidate_event_id in canvas._event_rects:
                if str(candidate_event_id) == event_id:
                    return rect
        if target_id.startswith("timeline.push:"):
            return self._find_layer_surface_rect("push", target_id.removeprefix("timeline.push:"))
        if target_id.startswith("timeline.pull:"):
            return self._find_layer_surface_rect("pull", target_id.removeprefix("timeline.pull:"))
        return None

    def _resolve_drag_destination(self, destination: Any, source_point: QPoint) -> QPoint:
        if isinstance(destination, str):
            rect = self._target_canvas_rect(destination)
            if rect is None:
                raise ValueError(f"Unsupported drag destination: {destination}")
            return rect.center().toPoint()
        if isinstance(destination, dict):
            if "target_id" in destination:
                return self._resolve_drag_destination(str(destination["target_id"]), source_point)
            dx = int(destination.get("dx", 0))
            dy = int(destination.get("dy", 0))
            return QPoint(source_point.x() + dx, source_point.y() + dy)
        if isinstance(destination, (tuple, list)) and len(destination) == 2:
            return QPoint(int(destination[0]), int(destination[1]))
        raise ValueError(f"Unsupported drag destination payload: {destination!r}")

    def _event_time_seconds(self, layer_id: str, event_id: str) -> float | None:
        for layer in self._harness.presentation().layers:
            if str(layer.layer_id) != layer_id:
                continue
            for event in layer.events:
                if str(event.event_id) == event_id:
                    return event.start
            for take in layer.takes:
                for event in take.events:
                    if str(event.event_id) == event_id:
                        return event.start
        return None

    def _find_layer_rect(self, layer_id: str):
        for rect, candidate_layer_id in self._harness.widget._canvas._header_select_rects:
            if str(candidate_layer_id) == layer_id:
                return rect
        for rect, candidate_layer_id in self._harness.widget._canvas._row_body_select_rects:
            if str(candidate_layer_id) == layer_id:
                return rect
        return None

    def _find_layer_surface_rect(self, surface: str, layer_id: str):
        rects = (
            self._harness.widget._canvas._push_rects
            if surface == "push"
            else self._harness.widget._canvas._pull_rects
        )
        for rect, candidate_layer_id in rects:
            if str(candidate_layer_id) == layer_id:
                return rect
        return None

    def _transport_control_bounds(self) -> dict[str, AutomationBounds]:
        bounds: dict[str, AutomationBounds] = {}
        origin = self._harness.widget._transport.mapTo(self._root, QPoint(0, 0))
        for name, rect in self._harness.widget._transport._control_rects.items():
            bounds[name] = AutomationBounds(
                x=float(origin.x()) + rect.x(),
                y=float(origin.y()) + rect.y(),
                width=rect.width(),
                height=rect.height(),
            )
        return bounds

    def _render(self) -> None:
        self._harness.widget.show()
        self._harness.widget.activateWindow()
        self._harness.widget.setFocus()
        self._harness.widget.repaint()
        QApplication.processEvents()
        self._harness.widget._canvas.repaint()
        self._harness.widget._transport.repaint()
        self._harness.widget._ruler.repaint()
        QApplication.processEvents()

    def _widget_bounds(self, widget: QWidget) -> AutomationBounds:
        origin = widget.mapTo(self._root, QPoint(0, 0))
        return AutomationBounds(
            x=float(origin.x()),
            y=float(origin.y()),
            width=float(widget.width()),
            height=float(widget.height()),
        )

    def _canvas_bounds(self, rect) -> AutomationBounds:
        origin = self._harness.widget._canvas.mapTo(self._root, QPoint(0, 0))
        return AutomationBounds(
            x=float(origin.x()) + rect.x(),
            y=float(origin.y()) + rect.y(),
            width=rect.width(),
            height=rect.height(),
        )

    @staticmethod
    def _pixmap_to_png_bytes(pixmap) -> bytes:
        payload = QByteArray()
        buffer = QBuffer(payload)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "PNG")
        buffer.close()
        return bytes(payload)

    @staticmethod
    def _click_widget_rect(widget: QWidget, rect) -> None:
        center = EchoZeroAutomationBackend._rect_center_point(rect)
        QTest.mouseClick(
            widget,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            center,
        )
        QApplication.processEvents()

    @staticmethod
    def _double_click_widget_rect(widget: QWidget, rect) -> None:
        center = EchoZeroAutomationBackend._rect_center_point(rect)
        QTest.mouseDClick(
            widget,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            center,
        )
        QApplication.processEvents()

    def _move_pointer_to_widget_rect(self, widget: QWidget, rect, *, target_id: str) -> None:
        center = self._rect_center_point(rect)
        QTest.mouseMove(widget, center)
        root_point = widget.mapTo(self._root, center)
        self._pointer_target_id = target_id
        self._pointer_position = (float(root_point.x()), float(root_point.y()))
        QApplication.processEvents()

    @staticmethod
    def _rect_center_point(rect) -> QPoint:
        center = rect.center()
        if isinstance(center, QPoint):
            return center
        return center.toPoint()

    @staticmethod
    def _drag_canvas(widget: QWidget, start: QPoint, end: QPoint) -> None:
        QApplication.sendEvent(
            widget,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress,
                QPointF(start),
                QPointF(start),
                QPointF(start),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        midpoint = QPoint(int((start.x() + end.x()) / 2), int((start.y() + end.y()) / 2))
        for point in (midpoint, end):
            QApplication.sendEvent(
                widget,
                QMouseEvent(
                    QMouseEvent.Type.MouseMove,
                    QPointF(point),
                    QPointF(point),
                    QPointF(point),
                    Qt.MouseButton.NoButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )
        QApplication.sendEvent(
            widget,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease,
                QPointF(end),
                QPointF(end),
                QPointF(end),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.processEvents()

    @staticmethod
    def _layer_target_id(layer_id: str) -> str:
        return f"timeline.layer:{layer_id}"

    @staticmethod
    def _event_target_id(event_id: str) -> str:
        return f"timeline.event:{event_id}"
