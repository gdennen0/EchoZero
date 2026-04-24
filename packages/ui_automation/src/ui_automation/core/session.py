"""
AutomationSession: Thin semantic client over an automation backend.
Exists so callers use one control surface for lookup and user-equivalent actions.
Connects backend snapshots to OpenClaw/Codex-friendly query helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, TypeVar

from .models import (
    AutomationAction,
    AutomationHitTarget,
    AutomationObject,
    AutomationSnapshot,
    AutomationTarget,
)
from .provider import AutomationProvider, AutomationSessionBackend

ItemT = TypeVar("ItemT")


@dataclass(slots=True)
class AutomationSession:
    """Wrap an automation backend with semantic lookup and action helpers."""

    backend: AutomationSessionBackend

    @classmethod
    def attach(cls, provider: AutomationProvider | AutomationSessionBackend) -> "AutomationSession":
        """Attach to a provider or backend and return a session facade."""
        if hasattr(provider, "attach"):
            return cls(backend=provider.attach())  # type: ignore[union-attr]
        return cls(backend=provider)

    def snapshot(self) -> AutomationSnapshot:
        """Return the current semantic snapshot from the backend."""
        return self.backend.snapshot()

    def screenshot(self, *, target_id: str | None = None) -> bytes:
        """Capture a screenshot for the full app or one target."""
        return self.backend.screenshot(target_id=target_id)

    def click(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        """Click a semantic target."""
        return self.backend.click(target_id, args=args)

    def move_pointer(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        """Move the pointer to a semantic target."""
        return self.backend.move_pointer(target_id, args=args)

    def double_click(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        """Double-click a semantic target."""
        return self.backend.double_click(target_id, args=args)

    def hover(
        self,
        target_id: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        """Hover a semantic target."""
        return self.backend.hover(target_id, args=args)

    def type_text(
        self,
        target_id: str,
        text: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        """Type text into a semantic target."""
        return self.backend.type_text(target_id, text, args=args)

    def press_key(
        self,
        key: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        """Press one key through the backend."""
        return self.backend.press_key(key, args=args)

    def drag(
        self,
        target_id: str,
        destination: Any,
        *,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        """Drag one semantic target to a destination."""
        return self.backend.drag(target_id, destination, args=args)

    def scroll(
        self,
        target_id: str,
        *,
        dx: int = 0,
        dy: int = 0,
        args: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        """Scroll inside one semantic target."""
        return self.backend.scroll(target_id, dx=dx, dy=dy, args=args)

    def invoke(
        self,
        action_id: str,
        *,
        target_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> AutomationSnapshot:
        """Invoke one semantic action exposed by the backend."""
        return self.backend.invoke(action_id, target_id=target_id, params=params)

    def find_target(self, query: str) -> AutomationTarget | None:
        """Return the first target whose id or label matches the query."""
        return _find_first_match(
            query,
            self.snapshot().targets,
            exact_values=lambda target: (target.target_id, target.label),
            partial_values=lambda target: (target.target_id, target.label),
        )

    def find_object(
        self,
        query: str,
        *,
        object_type: str | None = None,
    ) -> AutomationObject | None:
        """Return the first object whose identity, label, or target matches the query."""
        expected_object_type = _normalize_query(object_type)
        return _find_first_match(
            query,
            self.snapshot().objects,
            exact_values=lambda item: (item.object_id, item.label, item.target_id),
            partial_values=lambda item: (
                item.object_id,
                item.label,
                item.target_id,
                item.object_type,
                *[fact.label for fact in item.facts],
                *[fact.value for fact in item.facts],
            ),
            include=lambda item: expected_object_type is None or item.object_type.lower() == expected_object_type,
        )

    def find_action(
        self,
        query: str,
        *,
        target_id: str | None = None,
        group: str | None = None,
    ) -> AutomationAction | None:
        """Return the first visible action whose id or label matches the query."""
        expected_target_id = _normalize_query(target_id)
        expected_group = _normalize_query(group)
        return _find_first_match(
            query,
            _visible_actions(self.snapshot()),
            exact_values=lambda item: (item.action_id, item.label),
            partial_values=lambda item: (item.action_id, item.label, item.group, item.target_id),
            include=lambda item: _action_matches_filters(
                item,
                expected_target_id=expected_target_id,
                expected_group=expected_group,
            ),
        )

    def find_hit_target(
        self,
        query: str,
        *,
        kind: str | None = None,
    ) -> AutomationHitTarget | None:
        """Return the first hit target whose id or kind matches the query."""
        expected_kind = _normalize_query(kind)
        return _find_first_match(
            query,
            self.snapshot().hit_targets,
            exact_values=lambda item: (item.target_id,),
            partial_values=lambda item: (item.target_id, item.kind),
            include=lambda item: expected_kind is None or item.kind.lower() == expected_kind,
        )

    def close(self) -> None:
        """Close the underlying backend session."""
        self.backend.close()


def _find_first_match(
    query: str,
    items: Iterable[ItemT],
    *,
    exact_values: Callable[[ItemT], tuple[str | None, ...]],
    partial_values: Callable[[ItemT], tuple[str | None, ...]],
    include: Callable[[ItemT], bool] | None = None,
) -> ItemT | None:
    needle = _normalize_query(query)
    if needle is None:
        return None

    filtered_items = tuple(item for item in items if include is None or include(item))
    for item in filtered_items:
        if any(_normalize_query(value) == needle for value in exact_values(item)):
            return item
    for item in filtered_items:
        if any(needle in value for value in _normalized_values(partial_values(item))):
            return item
    return None


def _normalized_values(values: Iterable[str | None]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        cleaned = _normalize_query(value)
        if cleaned is not None:
            normalized.append(cleaned)
    return tuple(normalized)


def _normalize_query(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    return cleaned or None


def _visible_actions(snapshot: AutomationSnapshot) -> tuple[AutomationAction, ...]:
    actions: list[AutomationAction] = []
    seen_keys: set[tuple[str, str | None, str | None, str]] = set()

    for item in snapshot.objects:
        for action in item.actions:
            key = (action.action_id, action.target_id, action.group, action.label)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            actions.append(action)

    for action in snapshot.actions:
        key = (action.action_id, action.target_id, action.group, action.label)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        actions.append(action)

    return tuple(actions)


def _action_matches_filters(
    action: AutomationAction,
    *,
    expected_target_id: str | None,
    expected_group: str | None,
) -> bool:
    normalized_target_id = _normalize_query(action.target_id)
    normalized_group = _normalize_query(action.group)
    if expected_target_id is not None and normalized_target_id != expected_target_id:
        return False
    if expected_group is not None and normalized_group != expected_group:
        return False
    return True
