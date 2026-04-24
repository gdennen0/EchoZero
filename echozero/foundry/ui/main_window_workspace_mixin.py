"""Workspace, queue, and results helpers for the Foundry window.
Exists to keep overview, selection, queue, and result-list concerns out of the main window shell.
Connects Foundry query results to typed Qt row models and status/activity surfaces.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import (
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QWidget,
)

from echozero.foundry import FoundryApp
from echozero.foundry.domain import Dataset, EvalReport, TrainRun
from echozero.foundry.ui.main_window_workspace_build_mixin import (
    FoundryWindowWorkspaceBuildMixin,
)
from echozero.foundry.ui.main_window_workspace_state_mixin import (
    FoundryWindowWorkspaceStateMixin,
)


class FoundryWindowWorkspaceMixin(
    FoundryWindowWorkspaceBuildMixin,
    FoundryWindowWorkspaceStateMixin,
):
    _root: Path
    _app: FoundryApp
    _show_error_dialogs: bool
    _dataset_id: str | None
    _version_id: str | None
    _run_id: str | None
    _artifact_id: str | None
    _selected_artifact_id: str | None
    _run_thread: object | None
    _ACTIVE_RUN_STATUSES: set[str]

    workspace_path: QLineEdit
    workspace_summary: QPlainTextEdit
    run_overview: QTableWidget
    past_runs_overview: QTableWidget
    queue_list: QListWidget
    artifact_list: QListWidget
    eval_list: QListWidget
    eval_summary: QPlainTextEdit
    run_summary: QPlainTextEdit
    artifact_summary: QPlainTextEdit
    dataset_summary: QPlainTextEdit
    status_line: QLabel
    activity: QPlainTextEdit
    cancel_queue_run_btn: QPushButton
    retry_queue_run_btn: QPushButton
    class_names: QLineEdit
    activity_item_received: Any

    _build_dataset_box: Callable[[], QWidget]
    _build_training_box: Callable[[], QWidget]
    _build_artifact_box: Callable[[], QWidget]
    _format_dataset_summary: Callable[[], str]
    _format_run_summary: Callable[[TrainRun], str]
    _format_queue_entry: Callable[..., str]
    _format_artifact_summary: Callable[[str | None], str]
    _format_eval_summary: Callable[[EvalReport], str]
    _update_queue_action_buttons: Callable[[], None]
    _resolve_active_run_id: Callable[[list[TrainRun]], str | None]
    _cancel_selected_queue_run: Callable[[], None]
    _retry_selected_queue_run: Callable[[], None]
    _populate_dataset_selectors: Callable[[list[Dataset]], None]
    _selected_queue_run_id: Callable[[], str | None]


__all__ = ["FoundryWindowWorkspaceMixin"]
