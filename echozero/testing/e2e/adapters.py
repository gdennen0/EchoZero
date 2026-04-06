"""Thin E2E adapters for Stage Zero and Foundry."""

from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QApplication

from echozero.application.timeline.intents import (
    Pause,
    Play,
    Seek,
    SelectEvent,
    SelectLayer,
    SelectTake,
    Stop,
    TimelineIntent,
    ToggleLayerExpanded,
    ToggleMute,
    ToggleSolo,
    TriggerTakeAction,
)
from echozero.ui.qt.timeline.demo_app import DemoTimelineApp, build_demo_app
from echozero.ui.qt.timeline.widget import TimelineWidget


class StageZeroDriver:
    def __init__(
        self,
        app: QApplication,
        timeline_app: DemoTimelineApp,
        widget: TimelineWidget,
    ) -> None:
        self._app = app
        self._timeline_app = timeline_app
        self._widget = widget

    def click(self, target: str, *, args: dict[str, Any] | None = None) -> Any:
        payload = args or {}
        if target == "transport.play":
            return self.dispatch_intent("Play", payload)
        if target == "transport.pause":
            return self.dispatch_intent("Pause", payload)
        if target == "transport.stop":
            return self.dispatch_intent("Stop", payload)
        if target == "layer.toggle_expanded":
            return self.dispatch_intent("ToggleLayerExpanded", payload)
        raise ValueError(f"Unsupported click target: {target}")

    def type_text(self, target: str, text: str, *, args: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError(f"Stage Zero driver does not support text input: {target}")

    def press_key(self, key: str, *, args: dict[str, Any] | None = None) -> Any:
        mapping = {"space": "Play", "media_stop": "Stop"}
        intent_name = mapping.get(key.lower())
        if intent_name is None:
            raise ValueError(f"Unsupported key: {key}")
        return self.dispatch_intent(intent_name, args)

    def drag(self, target: str, destination: Any, *, args: dict[str, Any] | None = None) -> Any:
        if target == "playhead":
            return self.dispatch_intent("Seek", {"position": float(destination)})
        raise ValueError(f"Unsupported drag target: {target}")

    def dispatch_intent(self, intent_name: str, payload: dict[str, Any] | None = None) -> Any:
        intent = _build_intent(intent_name, payload or {})
        presentation = self._timeline_app.dispatch(intent)
        self._widget.set_presentation(presentation)
        self._app.processEvents()
        return presentation

    def query_state(self, query: str) -> Any:
        if query == "presentation":
            return _normalize(self._timeline_app.presentation())
        return _resolve_query(self._timeline_app.presentation(), query)

    def capture_screenshot(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self._widget.show()
        self._app.processEvents()
        self._widget.grab().save(str(output))
        return output.resolve()

    def wait(self, duration_ms: int) -> None:
        deadline = time.perf_counter() + max(0, duration_ms) / 1000.0
        while time.perf_counter() < deadline:
            self._app.processEvents()
            time.sleep(0.01)

    def close(self) -> None:
        self._widget.close()
        self._app.processEvents()


class FoundryDriverPlaceholder:
    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError(
            "Foundry E2E adapter is a placeholder. Wire a concrete Foundry surface before use."
        )


def create_stage_zero_driver(*, width: int = 1440, height: int = 720) -> StageZeroDriver:
    app = QApplication.instance() or QApplication([])
    timeline_app = build_demo_app()
    widget = TimelineWidget(timeline_app.presentation(), on_intent=timeline_app.dispatch)
    widget.resize(width, height)
    widget.show()
    app.processEvents()
    return StageZeroDriver(app=app, timeline_app=timeline_app, widget=widget)


def _build_intent(intent_name: str, payload: dict[str, Any]) -> TimelineIntent:
    normalized = intent_name.strip()
    if normalized == "Play":
        return Play()
    if normalized == "Pause":
        return Pause()
    if normalized == "Stop":
        return Stop()
    if normalized == "Seek":
        return Seek(position=float(payload["position"]))
    if normalized == "SelectLayer":
        return SelectLayer(layer_id=payload.get("layer_id"))
    if normalized == "SelectTake":
        return SelectTake(layer_id=payload["layer_id"], take_id=payload.get("take_id"))
    if normalized == "SelectEvent":
        return SelectEvent(
            layer_id=payload["layer_id"],
            take_id=payload.get("take_id"),
            event_id=payload.get("event_id"),
            mode=str(payload.get("mode", "replace")),
        )
    if normalized == "ToggleLayerExpanded":
        return ToggleLayerExpanded(layer_id=payload["layer_id"])
    if normalized == "ToggleMute":
        return ToggleMute(layer_id=payload["layer_id"])
    if normalized == "ToggleSolo":
        return ToggleSolo(layer_id=payload["layer_id"])
    if normalized == "TriggerTakeAction":
        return TriggerTakeAction(
            layer_id=payload["layer_id"],
            take_id=payload["take_id"],
            action_id=str(payload["action_id"]),
        )
    raise ValueError(f"Unsupported intent: {intent_name}")


def _resolve_query(source: Any, query: str) -> Any:
    current = source
    remaining = query
    while remaining:
        if "[" in remaining:
            head, rest = remaining.split("[", 1)
            if head:
                current = getattr(current, head)
            index_txt, remaining = rest.split("]", 1)
            current = current[int(index_txt)]
            if remaining.startswith("."):
                remaining = remaining[1:]
            continue
        if "." not in remaining:
            current = getattr(current, remaining)
            break
        head, remaining = remaining.split(".", 1)
        current = getattr(current, head)
    return _normalize(current)


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _normalize(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    return value
