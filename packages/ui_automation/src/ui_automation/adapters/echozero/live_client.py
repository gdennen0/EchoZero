from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any
from urllib import request

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
class LiveEchoZeroAutomationProvider:
    base_url: str

    def attach(self) -> "LiveEchoZeroAutomationBackend":
        return LiveEchoZeroAutomationBackend(self.base_url)


class LiveEchoZeroAutomationBackend:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def snapshot(self) -> AutomationSnapshot:
        payload = self._request_json("GET", "/snapshot")
        return _snapshot_from_payload(payload)

    def screenshot(self, *, target_id: str | None = None) -> bytes:
        payload = self._request_json("POST", "/screenshot", {"target_id": target_id})
        return base64.b64decode(str(payload["png_base64"]))

    def click(self, target_id: str, *, args: dict[str, Any] | None = None) -> AutomationSnapshot:
        return self._action("click", target_id=target_id, args=args)

    def move_pointer(self, target_id: str, *, args: dict[str, Any] | None = None) -> AutomationSnapshot:
        return self._action("move_pointer", target_id=target_id, args=args)

    def double_click(self, target_id: str, *, args: dict[str, Any] | None = None) -> AutomationSnapshot:
        return self._action("double_click", target_id=target_id, args=args)

    def hover(self, target_id: str, *, args: dict[str, Any] | None = None) -> AutomationSnapshot:
        return self._action("hover", target_id=target_id, args=args)

    def type_text(
        self,
        target_id: str,
        text: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        return self._action("type_text", target_id=target_id, text=text, args=args)

    def press_key(self, key: str, *, args: dict[str, Any] | None = None) -> AutomationSnapshot:
        return self._action("press_key", key=key, args=args)

    def drag(
        self,
        target_id: str,
        destination: Any,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        return self._action("drag", target_id=target_id, destination=destination, args=args)

    def scroll(
        self,
        target_id: str,
        *,
        dx: int = 0,
        dy: int = 0,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        return self._action("scroll", target_id=target_id, dx=dx, dy=dy, args=args)

    def invoke(
        self,
        action_id: str,
        *,
        target_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        return self._action("invoke", action_id=action_id, target_id=target_id, params=params)

    def close(self) -> None:
        return None

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/health")

    def _action(self, action: str, **payload: Any) -> AutomationSnapshot:
        response = self._request_json("POST", "/action", {"action": action, **payload})
        return _snapshot_from_payload(response)

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self._base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=5.0) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object from {path}")
        return parsed


def _snapshot_from_payload(payload: dict[str, Any]) -> AutomationSnapshot:
    return AutomationSnapshot(
        app=str(payload["app"]),
        selection=tuple(str(item) for item in payload.get("selection", [])),
        focused_target_id=None
        if payload.get("focused_target_id") is None
        else str(payload["focused_target_id"]),
        focused_object_id=None
        if payload.get("focused_object_id") is None
        else str(payload["focused_object_id"]),
        sync=dict(payload.get("sync", {})),
        targets=tuple(_target_from_payload(item) for item in payload.get("targets", [])),
        actions=tuple(_action_from_payload(item) for item in payload.get("actions", [])),
        objects=tuple(_object_from_payload(item) for item in payload.get("objects", [])),
        hit_targets=tuple(_hit_target_from_payload(item) for item in payload.get("hit_targets", [])),
        artifacts=dict(payload.get("artifacts", {})),
    )


def _target_from_payload(payload: dict[str, Any]) -> AutomationTarget:
    bounds_payload = payload.get("bounds")
    return AutomationTarget(
        kind=str(payload["kind"]),
        target_id=str(payload["target_id"]),
        parent_id=None if payload.get("parent_id") is None else str(payload["parent_id"]),
        label=None if payload.get("label") is None else str(payload["label"]),
        bounds=None if bounds_payload is None else _bounds_from_payload(bounds_payload),
        time_seconds=None if payload.get("time_seconds") is None else float(payload["time_seconds"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _action_from_payload(payload: dict[str, Any]) -> AutomationAction:
    return AutomationAction(
        action_id=str(payload["action_id"]),
        label=str(payload["label"]),
        enabled=bool(payload.get("enabled", True)),
        group=None if payload.get("group") is None else str(payload["group"]),
        params=dict(payload.get("params", {})),
        target_id=None if payload.get("target_id") is None else str(payload["target_id"]),
    )


def _hit_target_from_payload(payload: dict[str, Any]) -> AutomationHitTarget:
    return AutomationHitTarget(
        target_id=str(payload["target_id"]),
        kind=str(payload["kind"]),
        bounds=_bounds_from_payload(payload["bounds"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _object_from_payload(payload: dict[str, Any]) -> AutomationObject:
    return AutomationObject(
        object_id=str(payload["object_id"]),
        object_type=str(payload["object_type"]),
        label=str(payload["label"]),
        target_id=None if payload.get("target_id") is None else str(payload["target_id"]),
        facts=tuple(_object_fact_from_payload(item) for item in payload.get("facts", [])),
        actions=tuple(_action_from_payload(item) for item in payload.get("actions", [])),
        metadata=dict(payload.get("metadata", {})),
    )


def _object_fact_from_payload(payload: dict[str, Any]) -> AutomationObjectFact:
    return AutomationObjectFact(
        label=str(payload["label"]),
        value=str(payload["value"]),
    )


def _bounds_from_payload(payload: dict[str, Any]) -> AutomationBounds:
    return AutomationBounds(
        x=float(payload["x"]),
        y=float(payload["y"]),
        width=float(payload["width"]),
        height=float(payload["height"]),
    )
