from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .models import AutomationSnapshot


@runtime_checkable
class AutomationSessionBackend(Protocol):
    def snapshot(self) -> AutomationSnapshot:
        ...

    def screenshot(self, *, target_id: str | None = None) -> bytes:
        ...

    def click(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        ...

    def move_pointer(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        ...

    def double_click(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        ...

    def hover(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        ...

    def type_text(
        self,
        target_id: str,
        text: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        ...

    def press_key(
        self,
        key: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        ...

    def drag(
        self,
        target_id: str,
        destination: Any,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        ...

    def scroll(
        self,
        target_id: str,
        *,
        dx: int = 0,
        dy: int = 0,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        ...

    def invoke(
        self,
        action_id: str,
        *,
        target_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class AutomationProvider(Protocol):
    def attach(self) -> AutomationSessionBackend:
        ...
