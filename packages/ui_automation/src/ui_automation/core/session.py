from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import AutomationSnapshot, AutomationTarget
from .provider import AutomationProvider, AutomationSessionBackend


@dataclass(slots=True)
class AutomationSession:
    backend: AutomationSessionBackend

    @classmethod
    def attach(cls, provider: AutomationProvider | AutomationSessionBackend) -> "AutomationSession":
        if hasattr(provider, "attach"):
            return cls(backend=provider.attach())  # type: ignore[union-attr]
        return cls(backend=provider)

    def snapshot(self) -> AutomationSnapshot:
        return self.backend.snapshot()

    def screenshot(self, *, target_id: str | None = None) -> bytes:
        return self.backend.screenshot(target_id=target_id)

    def click(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        return self.backend.click(target_id, args=args)

    def move_pointer(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        return self.backend.move_pointer(target_id, args=args)

    def double_click(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        return self.backend.double_click(target_id, args=args)

    def hover(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        return self.backend.hover(target_id, args=args)

    def type_text(
        self,
        target_id: str,
        text: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        return self.backend.type_text(target_id, text, args=args)

    def press_key(
        self,
        key: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        return self.backend.press_key(key, args=args)

    def drag(
        self,
        target_id: str,
        destination: Any,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        return self.backend.drag(target_id, destination, args=args)

    def scroll(
        self,
        target_id: str,
        *,
        dx: int = 0,
        dy: int = 0,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        return self.backend.scroll(target_id, dx=dx, dy=dy, args=args)

    def invoke(
        self,
        action_id: str,
        *,
        target_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        return self.backend.invoke(action_id, target_id=target_id, params=params)

    def find_target(self, query: str) -> AutomationTarget | None:
        needle = query.strip().lower()
        if not needle:
            return None
        snapshot = self.snapshot()
        for target in snapshot.targets:
            if target.target_id.lower() == needle:
                return target
            label = target.label or ""
            if label.lower() == needle:
                return target
        for target in snapshot.targets:
            label = target.label or ""
            if needle in target.target_id.lower() or needle in label.lower():
                return target
        return None

    def close(self) -> None:
        self.backend.close()
