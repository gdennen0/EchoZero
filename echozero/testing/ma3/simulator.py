from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SimulatedMA3Bridge:
    connected: bool = False
    connect_calls: int = 0
    disconnect_calls: int = 0
    emitted_events: list[dict[str, Any]] = field(default_factory=list)
    _pending_events: deque[dict[str, Any]] = field(default_factory=deque)

    def on_ma3_connected(self) -> None:
        self.connect_calls += 1
        self.connected = True

    def on_ma3_disconnected(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    def emit(self, kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {"kind": kind, "payload": dict(payload or {})}
        self.emitted_events.append(event)
        return event

    def push_event(self, kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {"kind": kind, "payload": dict(payload or {})}
        self._pending_events.append(event)
        return event

    def pop_event(self) -> dict[str, Any] | None:
        if not self._pending_events:
            return None
        return self._pending_events.popleft()

    def pending_events(self) -> list[dict[str, Any]]:
        return list(self._pending_events)
