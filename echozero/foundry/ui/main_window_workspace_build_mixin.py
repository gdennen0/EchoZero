"""Workspace layout and root-switch helpers for the Foundry window.
Exists to keep widget construction and workspace root switching out of the workspace state mixin.
Connects Foundry workspace UI chrome to the main window shell.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from echozero.foundry import FoundryApp
from echozero.ui.style import SHELL_TOKENS


class _WorkspaceBuildHost(Protocol):
    _root: Path
    _app: FoundryApp
    _dataset_id: str | None
    _version_id: str | None
    _run_id: str | None
    _artifact_id: str | None
    _selected_artifact_id: str | None
    _run_thread: object | None

    workspace_path: QLineEdit
    workspace_picker: QPushButton
    refresh_btn: QPushButton
    workspace_summary: QPlainTextEdit
    run_overview: QTableWidget
    past_runs_overview: QTableWidget
    queue_list: QListWidget
    artifact_list: QListWidget
    eval_list: QListWidget
    eval_summary: QPlainTextEdit
    status_line: QLabel
    activity: QPlainTextEdit
    cancel_queue_run_btn: QPushButton
    retry_queue_run_btn: QPushButton

    _build_dataset_box: Callable[[], QWidget]
    _build_training_box: Callable[[], QWidget]
    _build_artifact_box: Callable[[], QWidget]
    _build_past_runs_box: Callable[[], QWidget]
    _build_activity_box: Callable[[], QWidget]
    _build_run_table: Callable[..., QTableWidget]
    _pick_workspace_root: Callable[[], None]
    _refresh_workspace_state: Callable[..., None]
    _queue_workspace_refresh: Callable[[], None]
    _select_run_from_past_overview: Callable[[], None]
    _select_run_from_overview: Callable[[], None]
    _select_run_from_queue: Callable[[str], None]
    _select_artifact_from_list: Callable[[str], None]
    _cancel_selected_queue_run: Callable[[], None]
    _retry_selected_queue_run: Callable[[], None]
    _on_activity: Callable[[object], None]
    _set_status: Callable[[str], None]
    _error: Callable[[Exception], None]


class FoundryWindowWorkspaceBuildMixin:
    def _build_header(self) -> QWidget:
        host = cast(_WorkspaceBuildHost, self)
        widget = QWidget()
        widget.setObjectName("foundryHeader")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SHELL_TOKENS.scales.layout_gap)

        host.workspace_path = QLineEdit(str(host._root))
        host.workspace_path.setReadOnly(True)
        host.workspace_picker = QPushButton("Workspace...")
        host.workspace_picker.clicked.connect(host._pick_workspace_root)
        host.refresh_btn = QPushButton("Refresh")
        host.refresh_btn.clicked.connect(host._queue_workspace_refresh)

        layout.addWidget(QLabel("Workspace"))
        layout.addWidget(host.workspace_path, stretch=1)
        layout.addWidget(host.workspace_picker)
        layout.addWidget(host.refresh_btn)
        return widget

    def _build_workflow_tabs(self) -> QWidget:
        host = cast(_WorkspaceBuildHost, self)
        tabs = QTabWidget()
        tabs.addTab(host._build_dataset_box(), "Dataset")
        tabs.addTab(host._build_training_box(), "Run")
        tabs.addTab(host._build_past_runs_box(), "Past Runs")
        tabs.addTab(host._build_artifact_box(), "Artifacts")
        tabs.addTab(host._build_activity_box(), "Activity")
        return tabs

    def _build_activity_box(self) -> QWidget:
        host = cast(_WorkspaceBuildHost, self)
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setSpacing(SHELL_TOKENS.scales.compact_gap)
        host.status_line = QLabel("Ready")
        host.status_line.setObjectName("foundryStatusLine")
        host.activity = QPlainTextEdit()
        host.activity.setReadOnly(True)
        host.activity.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(host.status_line)
        layout.addWidget(host.activity, stretch=1)
        return box

    def _build_run_table(self, *, for_run_selection: bool = True) -> QTableWidget:
        from echozero.foundry.ui.main_window_types import require_horizontal_header

        table = QTableWidget()
        table.setColumnCount(8)
        table.setHorizontalHeaderLabels(
            [
                "Run ID",
                "Status",
                "Dataset Version",
                "Model",
                "Epochs",
                "LR",
                "Updated",
                "Artifacts",
            ]
        )
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setMinimumHeight(220 if for_run_selection else 420)
        header = require_horizontal_header(table)
        header.setSectionResizeMode(0, header.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, header.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, header.ResizeMode.Stretch)
        header.setSectionResizeMode(3, header.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, header.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, header.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, header.ResizeMode.Stretch)
        header.setSectionResizeMode(7, header.ResizeMode.ResizeToContents)
        return table

    def _build_past_runs_box(self) -> QWidget:
        host = cast(_WorkspaceBuildHost, self)
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setSpacing(SHELL_TOKENS.scales.layout_gap)

        host.past_runs_overview = host._build_run_table(for_run_selection=False)
        host.past_runs_overview.itemSelectionChanged.connect(host._select_run_from_past_overview)
        layout.addWidget(host.past_runs_overview)
        return box

    def _build_workspace_panel(self) -> QWidget:
        host = cast(_WorkspaceBuildHost, self)
        widget = QWidget()
        widget.setObjectName("foundryWorkspacePanel")
        layout = QVBoxLayout(widget)
        layout.setSpacing(SHELL_TOKENS.scales.layout_gap)

        overview_box = QGroupBox("Workspace Overview")
        overview_layout = QVBoxLayout(overview_box)
        host.workspace_summary = QPlainTextEdit()
        host.workspace_summary.setReadOnly(True)
        overview_layout.addWidget(host.workspace_summary)

        runs_box = QGroupBox("Runs")
        runs_layout = QVBoxLayout(runs_box)
        host.run_overview = host._build_run_table()
        host.run_overview.itemSelectionChanged.connect(host._select_run_from_overview)
        runs_layout.addWidget(host.run_overview)

        queue_box = QGroupBox("Background Queue")
        queue_layout = QVBoxLayout(queue_box)
        host.queue_list = QListWidget()
        host.queue_list.currentTextChanged.connect(host._select_run_from_queue)
        host.queue_list.setMinimumHeight(120)
        queue_layout.addWidget(host.queue_list)
        queue_actions = QHBoxLayout()
        queue_actions.setSpacing(SHELL_TOKENS.scales.inline_gap)
        host.cancel_queue_run_btn = QPushButton("Cancel Run")
        host.cancel_queue_run_btn.clicked.connect(host._cancel_selected_queue_run)
        host.retry_queue_run_btn = QPushButton("Retry / Requeue")
        host.retry_queue_run_btn.clicked.connect(host._retry_selected_queue_run)
        queue_actions.addWidget(host.cancel_queue_run_btn)
        queue_actions.addWidget(host.retry_queue_run_btn)
        queue_layout.addLayout(queue_actions)

        artifacts_box = QGroupBox("Artifacts")
        artifacts_layout = QVBoxLayout(artifacts_box)
        host.artifact_list = QListWidget()
        host.artifact_list.currentTextChanged.connect(host._select_artifact_from_list)
        host.artifact_list.setMinimumHeight(110)
        artifacts_layout.addWidget(host.artifact_list)

        evals_box = QGroupBox("Eval Reports")
        evals_layout = QVBoxLayout(evals_box)
        host.eval_list = QListWidget()
        host.eval_list.setMinimumHeight(110)
        evals_layout.addWidget(host.eval_list)
        host.eval_summary = QPlainTextEdit()
        host.eval_summary.setReadOnly(True)
        evals_layout.addWidget(host.eval_summary, stretch=1)

        layout.addWidget(overview_box)
        layout.addWidget(runs_box)
        layout.addWidget(queue_box)
        layout.addWidget(artifacts_box)
        layout.addWidget(evals_box, stretch=1)
        return widget

    def _queue_workspace_refresh(self) -> None:
        host = cast(_WorkspaceBuildHost, self)
        host._set_status("Refreshing workspace state...")
        QTimer.singleShot(0, host._refresh_workspace_state)

    def _pick_workspace_root(self) -> None:
        host = cast(_WorkspaceBuildHost, self)
        if host._run_thread is not None:
            host._error(Exception("Stop the active run before switching workspace roots."))
            return

        candidate = QFileDialog.getExistingDirectory(
            cast(QWidget, host),
            "Select Foundry workspace folder",
            str(host._root),
        )
        if not candidate:
            return

        host._run_id = None
        host._artifact_id = None
        host._selected_artifact_id = None
        host._dataset_id = None
        host._version_id = None
        host._app = FoundryApp(Path(candidate))
        host._app.activity.set_listener(host._on_activity)
        host._root = Path(candidate)
        host.workspace_path.setText(str(host._root))
        host._refresh_workspace_state()
        host._set_status(f"Workspace switched to: {host._root}")


__all__ = ["FoundryWindowWorkspaceBuildMixin"]
