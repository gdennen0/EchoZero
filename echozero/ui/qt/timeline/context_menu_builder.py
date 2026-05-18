"""Categorized Qt context-menu builders for timeline actions.
Exists to keep right-click menu structure consistent without changing app contracts.
Connects inspector context actions to ergonomic Qt menus and stable action payloads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from PyQt6.QtWidgets import QMenu, QWidget

from echozero.application.presentation.inspector_contract import (
    InspectorAction,
    InspectorContextSection,
)


class _PipelinePlan(Protocol):
    run_label: str
    requires_settings_confirmation: bool


PipelinePlanResolver = Callable[[InspectorAction], _PipelinePlan | None]


@dataclass(slots=True)
class _ContextMenuGroup:
    label: str
    actions: list[InspectorAction]


@dataclass(slots=True, frozen=True)
class _FallbackPipelinePlan:
    run_label: str = "Run"
    requires_settings_confirmation: bool = False


_GROUP_ORDER = (
    "clip",
    "song",
    "add",
    "view",
    "transport",
    "selection",
    "batch",
    "sync_transfer",
    "layer",
    "mix",
    "pipeline",
    "live_sync",
    "take",
    "other",
)
_GROUP_LABELS = {
    "clip": "Clip",
    "song": "Song",
    "add": "Add",
    "view": "View",
    "transport": "Transport",
    "selection": "Selection",
    "batch": "Batch Edit",
    "sync_transfer": "Sync && Transfer",
    "layer": "Layer",
    "mix": "Mix",
    "pipeline": "Pipeline",
    "live_sync": "Live Sync",
    "take": "Take",
    "other": "More",
}
_DESTRUCTIVE_ACTION_IDS = frozenset(
    {
        "delete_layer",
        "delete_take",
        "song.delete",
        "song.version.delete",
        "video.remove",
    }
)
_VIEW_ACTION_IDS = frozenset(
    {
        "layer.set_expanded",
        "timeline.collapse_all_layers",
        "timeline.expand_all_layers",
    }
)
_TRANSFER_ACTION_IDS = frozenset(
    {
        "transfer.route_layer_track",
        "transfer.workspace_open",
        "transfer.send_selection",
        "transfer.match_ma3_cues",
        "transfer.send_to_track_once",
        "transfer.plan_apply",
        "transfer.plan_cancel",
        "transfer.plan_preview",
    }
)
_CLIP_ACTION_IDS = frozenset({"preview_event_clip", "selection.compare_events"})


def build_inspector_context_menu(
    parent: QWidget,
    sections: Iterable[InspectorContextSection],
    *,
    pipeline_plan_resolver: PipelinePlanResolver | None = None,
) -> QMenu:
    """Build a categorized context menu from presentation-owned inspector actions."""

    menu = QMenu(parent)
    groups = _categorized_groups(sections)
    for index, group_key in enumerate(_GROUP_ORDER):
        group = groups.get(group_key)
        if group is None or not group.actions:
            continue
        if index and menu.actions():
            menu.addSeparator()
        submenu = menu.addMenu(group.label)
        if submenu is None:
            continue
        _populate_action_group(
            submenu,
            group.actions,
            pipeline_plan_resolver=pipeline_plan_resolver if group_key == "pipeline" else None,
        )
    return menu


def build_layer_pipeline_context_menu(
    parent: QWidget,
    actions: Iterable[InspectorAction],
    *,
    pipeline_plan_resolver: PipelinePlanResolver | None = None,
) -> QMenu:
    """Build the layer-header pipeline context menu with settings/run pairs."""

    menu = QMenu(parent)
    _populate_pipeline_actions(
        menu,
        tuple(actions),
        pipeline_plan_resolver=pipeline_plan_resolver,
        use_action_submenus=True,
    )
    return menu


def _categorized_groups(
    sections: Iterable[InspectorContextSection],
) -> dict[str, _ContextMenuGroup]:
    groups = {
        group_key: _ContextMenuGroup(label=_GROUP_LABELS[group_key], actions=[])
        for group_key in _GROUP_ORDER
    }
    for section in sections:
        for action in section.actions:
            groups[_category_key(section, action)].actions.append(action)
    for group in groups.values():
        group.actions[:] = _with_destructive_actions_last(group.actions)
    return groups


def _category_key(section: InspectorContextSection, action: InspectorAction) -> str:
    action_id = action.action_id.strip()
    group = (action.group or "").strip().lower()
    section_id = section.section_id.strip().lower()
    if section_id == "event-preview" or action_id in _CLIP_ACTION_IDS:
        return "clip"
    if action_id in _TRANSFER_ACTION_IDS:
        return "sync_transfer"
    if group == "song":
        return "song"
    if group == "tools":
        return "view" if action_id in _VIEW_ACTION_IDS else "add"
    if group == "transport":
        return "transport"
    if group == "selection":
        return "selection"
    if group == "batch":
        return "batch"
    if group == "transfer":
        return "sync_transfer"
    if group == "routing":
        return "layer"
    if group == "layer":
        return "view" if action_id in _VIEW_ACTION_IDS else "layer"
    if group in {"mix", "gain"}:
        return "mix"
    if group == "pipeline":
        return "pipeline"
    if group == "live_sync":
        return "live_sync"
    if group == "take":
        return "take"
    return "other"


def _populate_action_group(
    menu: QMenu,
    actions: Iterable[InspectorAction],
    *,
    pipeline_plan_resolver: PipelinePlanResolver | None,
) -> None:
    action_tuple = tuple(actions)
    if pipeline_plan_resolver is not None:
        _populate_pipeline_actions(
            menu,
            action_tuple,
            pipeline_plan_resolver=pipeline_plan_resolver,
            use_action_submenus=True,
        )
        return
    for index, action in enumerate(action_tuple):
        if index and _starts_destructive_block(action_tuple, index):
            menu.addSeparator()
        _add_inspector_action(menu, action)


def _populate_pipeline_actions(
    menu: QMenu,
    actions: tuple[InspectorAction, ...],
    *,
    pipeline_plan_resolver: PipelinePlanResolver | None,
    use_action_submenus: bool,
) -> None:
    for index, action in enumerate(actions):
        if index:
            menu.addSeparator()
        plan = pipeline_plan_resolver(action) if pipeline_plan_resolver is not None else None
        if plan is None:
            if pipeline_plan_resolver is not None:
                plan = _FallbackPipelinePlan()
            else:
                _add_inspector_action(menu, action)
                continue
        if use_action_submenus:
            action_menu = menu.addMenu(action.label)
            if action_menu is None:
                continue
            target_menu = action_menu
        else:
            target_menu = menu
        _add_pipeline_settings_action(target_menu, action)
        _add_pipeline_run_action(target_menu, action, plan)


def _add_pipeline_settings_action(menu: QMenu, action: InspectorAction) -> None:
    settings_entry = menu.addAction(f"Open {action.label} Settings")
    if settings_entry is None:
        return
    settings_entry.setEnabled(action.enabled)
    settings_entry.setData(
        InspectorAction(
            action_id=action.action_id,
            label=action.label,
            enabled=action.enabled,
            kind="settings",
            params=dict(action.params),
        )
    )


def _add_pipeline_run_action(
    menu: QMenu,
    action: InspectorAction,
    plan: _PipelinePlan,
) -> None:
    run_prefix = "Review" if plan.requires_settings_confirmation else plan.run_label
    run_entry = menu.addAction(f"{run_prefix} {action.label}")
    if run_entry is None:
        return
    run_entry.setEnabled(action.enabled)
    run_entry.setData(
        InspectorAction(
            action_id=action.action_id,
            label=action.label,
            enabled=action.enabled,
            params=dict(action.params),
        )
    )


def _add_inspector_action(menu: QMenu, action: InspectorAction) -> None:
    menu_action = menu.addAction(action.label)
    if menu_action is None:
        return
    menu_action.setEnabled(action.enabled)
    menu_action.setData(action)
    if _is_destructive_action(action):
        menu_action.setProperty("destructive", True)


def _with_destructive_actions_last(actions: list[InspectorAction]) -> list[InspectorAction]:
    regular_actions = [action for action in actions if not _is_destructive_action(action)]
    destructive_actions = [action for action in actions if _is_destructive_action(action)]
    return [*regular_actions, *destructive_actions]


def _starts_destructive_block(actions: tuple[InspectorAction, ...], index: int) -> bool:
    return _is_destructive_action(actions[index]) and not _is_destructive_action(actions[index - 1])


def _is_destructive_action(action: InspectorAction) -> bool:
    action_id = action.action_id.strip()
    label = action.label.strip().lower()
    return action_id in _DESTRUCTIVE_ACTION_IDS or label.startswith(("delete ", "remove "))
