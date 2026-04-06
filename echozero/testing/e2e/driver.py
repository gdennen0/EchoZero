"""Driver protocol for E2E adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class E2EDriver(Protocol):
    def click(self, target: str, *, args: dict[str, Any] | None = None) -> Any:
        ...

    def type_text(self, target: str, text: str, *, args: dict[str, Any] | None = None) -> Any:
        ...

    def press_key(self, key: str, *, args: dict[str, Any] | None = None) -> Any:
        ...

    def drag(self, target: str, destination: Any, *, args: dict[str, Any] | None = None) -> Any:
        ...

    def dispatch_intent(self, intent_name: str, payload: dict[str, Any] | None = None) -> Any:
        ...

    def query_state(self, query: str) -> Any:
        ...

    def capture_screenshot(self, path: str | Path) -> Path:
        ...

    def wait(self, duration_ms: int) -> None:
        ...
