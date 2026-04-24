"""Workspace state, selection, and status helpers for the Foundry window.
Exists to keep run/artifact/eval selection logic out of the workspace layout mixin.
Connects Foundry query results to typed row models and status surfaces.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from echozero.foundry import FoundryApp
from echozero.foundry.domain import Dataset, EvalReport, ModelArtifact, TrainRun
from echozero.foundry.ui.main_window_types import (
    ArtifactListRow,
    EvalListRow,
    QueueRunRow,
    RunTableRow,
    list_item_user_role,
    selected_list_run_id,
    selected_table_row,
    table_item_or_none,
)


class _WorkspaceActivitySignal(Protocol):
    def emit(self, kind: str, message: str) -> None: ...


class _WorkspaceStateHost(Protocol):
    _root: Path
    _app: FoundryApp
    _show_error_dialogs: bool
    _dataset_id: str | None
    _version_id: str | None
    _run_id: str | None
    _artifact_id: str | None
    _selected_artifact_id: str | None
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
    activity_item_received: _WorkspaceActivitySignal

    _populate_dataset_selectors: Callable[[list[Dataset]], None]
    _format_dataset_summary: Callable[[], str]
    _format_workspace_summary: Callable[[list[Dataset], list[TrainRun]], str]
    _format_run_summary: Callable[[TrainRun], str]
    _format_queue_entry: Callable[..., str]
    _format_artifact_summary: Callable[[str | None], str]
    _format_eval_summary: Callable[[EvalReport], str]
    _update_queue_action_buttons: Callable[[], None]
    _resolve_active_run_id: Callable[[list[TrainRun]], str | None]
    _selected_queue_run_id: Callable[[], str | None]
    _populate_run_overview: Callable[..., None]
    _populate_past_runs_overview: Callable[..., None]
    _populate_run_rows: Callable[..., None]
    _populate_queue_list: Callable[[list[TrainRun]], None]
    _update_selection_details: Callable[..., None]
    _populate_artifact_list: Callable[..., None]
    _populate_eval_list: Callable[[list[EvalReport]], None]
    _sync_run_selection: Callable[[str | None], None]
    _sync_table_selection: Callable[[QTableWidget, str | None], None]
    _sync_queue_selection: Callable[[str | None], None]
    _set_status: Callable[[str], None]


class FoundryWindowWorkspaceStateMixin:
    def _refresh_workspace_state(
        self,
        *,
        select_run_id: str | None = None,
        select_artifact_id: str | None = None,
    ) -> None:
        host = cast(_WorkspaceStateHost, self)
        datasets = host._app.list_datasets()
        runs = sorted(host._app.list_runs(), key=lambda item: item.created_at)

        latest_dataset = datasets[-1] if datasets else None
        if host._dataset_id is None and latest_dataset is not None:
            host._dataset_id = latest_dataset.id

        host._populate_dataset_selectors(datasets)
        host.workspace_summary.setPlainText(host._format_workspace_summary(datasets, runs))
        host.dataset_summary.setPlainText(host._format_dataset_summary())
        host._populate_run_overview(runs, select_run_id=select_run_id)
        host._populate_past_runs_overview(runs, select_run_id=select_run_id)
        host._populate_queue_list(runs)
        host._update_selection_details(
            select_run_id=select_run_id,
            select_artifact_id=select_artifact_id,
        )
        host._update_queue_action_buttons()

    def _populate_run_overview(
        self,
        runs: list[TrainRun],
        *,
        select_run_id: str | None = None,
    ) -> None:
        host = cast(_WorkspaceStateHost, self)
        host._populate_run_rows(host.run_overview, runs, select_run_id=select_run_id)
        if host.run_overview.rowCount() == 0:
            host.run_summary.setPlainText("No runs yet.")
            host.queue_list.clear()
            host.artifact_list.clear()
            host.artifact_summary.clear()
            host.eval_list.clear()
            host.eval_summary.clear()

    def _populate_past_runs_overview(
        self,
        runs: list[TrainRun],
        *,
        select_run_id: str | None = None,
    ) -> None:
        host = cast(_WorkspaceStateHost, self)
        host._populate_run_rows(host.past_runs_overview, runs, select_run_id=select_run_id)

    def _populate_run_rows(
        self,
        table: QTableWidget,
        runs: list[TrainRun],
        *,
        select_run_id: str | None = None,
        blank_message: str = "No runs yet.",
    ) -> None:
        host = cast(_WorkspaceStateHost, self)
        current_run_id = select_run_id or host._run_id
        if not current_run_id and runs:
            current_run_id = runs[-1].id

        table_rows = [
            RunTableRow(
                run=run,
                model_type=str(run.spec.get("model", {}).get("type", "n/a")),
                epochs_label=str(run.spec.get("training", {}).get("epochs", "n/a")),
                learning_rate_label=str(run.spec.get("training", {}).get("learningRate", "n/a")),
                updated_label=run.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
                artifact_summary=(
                    f"{len(host._app.list_artifacts_for_run(run.id))} artifacts, "
                    f"{len(sorted(run.checkpoints_dir(host._root).glob('epoch_*.json')))} ckpt"
                ),
            )
            for run in reversed(runs)
        ]

        table.blockSignals(True)
        table.setRowCount(0)
        selected_row = -1

        for index, row_model in enumerate(table_rows):
            row = table.rowCount()
            table.insertRow(row)
            for column, value in enumerate(row_model.cells):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, row_model.run.id)
                    item.setToolTip(str(row_model.run.id))
                table.setItem(row, column, item)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if row_model.run.id == current_run_id:
                selected_row = index
            status_item = table_item_or_none(table, row, 1)
            if status_item is not None:
                status_item.setToolTip(row_model.run.status.value)
            version_item = table_item_or_none(table, row, 2)
            if version_item is not None:
                version_item.setToolTip(str(row_model.run.dataset_version_id))

        if table.rowCount() == 0:
            table.clearContents()
            table.setRowCount(1)
            blank = QTableWidgetItem(blank_message)
            table.setSpan(0, 0, 1, table.columnCount())
            blank.setFlags(blank.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            blank.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(0, 0, blank)

        table.blockSignals(False)
        if 0 <= selected_row < table.rowCount():
            table.selectRow(selected_row)

    def _select_run_from_overview(self) -> None:
        self._select_run_from_table(cast(_WorkspaceStateHost, self).run_overview)

    def _select_run_from_past_overview(self) -> None:
        self._select_run_from_table(cast(_WorkspaceStateHost, self).past_runs_overview)

    def _select_run_from_table(self, table: QTableWidget) -> None:
        host = cast(_WorkspaceStateHost, self)
        row = selected_table_row(table)
        if row is None:
            return
        item = table_item_or_none(table, row, 0)
        if item is None:
            return
        run_id = item.data(Qt.ItemDataRole.UserRole)
        host._run_id = str(run_id) if run_id else None
        host._update_selection_details(select_run_id=host._run_id)
        host._sync_run_selection(host._run_id)
        host._sync_queue_selection(host._run_id)
        host._update_queue_action_buttons()

    def _select_run_from_queue(self, _: str) -> None:
        host = cast(_WorkspaceStateHost, self)
        run_id = selected_list_run_id(host.queue_list)
        if run_id is None:
            host._update_queue_action_buttons()
            return
        host._run_id = run_id
        host._sync_run_selection(host._run_id)
        host._update_selection_details(select_run_id=host._run_id)
        host._update_queue_action_buttons()

    def _select_artifact_from_list(self, _: str) -> None:
        host = cast(_WorkspaceStateHost, self)
        artifact_id = list_item_user_role(host.artifact_list.currentItem())
        if artifact_id is None:
            return
        host._selected_artifact_id = artifact_id
        host._artifact_id = artifact_id
        host.artifact_summary.setPlainText(host._format_artifact_summary(artifact_id))

    def _update_selection_details(
        self,
        *,
        select_run_id: str | None = None,
        select_artifact_id: str | None = None,
    ) -> None:
        host = cast(_WorkspaceStateHost, self)
        run_id = select_run_id or host._run_id
        if not run_id:
            return

        run = host._app.runs.get_run(run_id)
        if run is None:
            host.run_summary.setPlainText(f"Run not found: {run_id}")
            return

        host._run_id = run.id
        host._version_id = run.dataset_version_id
        version = host._app.datasets.get_version(run.dataset_version_id)
        if version is not None:
            host._dataset_id = version.dataset_id
            host.class_names.setText(",".join(version.class_map))

        host.run_summary.setPlainText(host._format_run_summary(run))

        artifacts = sorted(host._app.list_artifacts_for_run(run.id), key=lambda item: item.created_at)
        evals = sorted(host._app.list_eval_reports_for_run(run.id), key=lambda item: item.created_at)
        host._populate_artifact_list(artifacts, select_artifact_id=select_artifact_id)
        host._populate_eval_list(evals)

    def _populate_artifact_list(
        self,
        artifacts: list[ModelArtifact],
        *,
        select_artifact_id: str | None = None,
    ) -> None:
        host = cast(_WorkspaceStateHost, self)
        artifact_id = select_artifact_id or host._artifact_id
        if not artifact_id and artifacts:
            artifact_id = artifacts[-1].id

        rows = [
            ArtifactListRow(
                artifact=artifact,
                label=f"{artifact.id} [{artifact.consumer_hints.get('consumer', 'consumer?')}]",
            )
            for artifact in reversed(artifacts)
        ]

        host.artifact_list.blockSignals(True)
        host.artifact_list.clear()
        selected_row = -1
        for index, row_model in enumerate(rows):
            host.artifact_list.addItem(row_model.label)
            item = host.artifact_list.item(index)
            if item is not None:
                item.setData(Qt.ItemDataRole.UserRole, row_model.artifact.id)
            if row_model.artifact.id == artifact_id:
                selected_row = index
        host.artifact_list.blockSignals(False)

        if selected_row >= 0:
            host.artifact_list.setCurrentRow(selected_row)
            host._selected_artifact_id = artifact_id
            host._artifact_id = artifact_id
            host.artifact_summary.setPlainText(host._format_artifact_summary(artifact_id))
        else:
            host._selected_artifact_id = None
            host.artifact_summary.setPlainText("No artifacts yet.")

    def _populate_eval_list(self, evals: list[EvalReport]) -> None:
        host = cast(_WorkspaceStateHost, self)
        host.eval_list.clear()
        if not evals:
            host.eval_summary.setPlainText("No eval reports yet.")
            return

        rows = [
            EvalListRow(
                report=report,
                label=(
                    f"{report.id} [{report.split_name}] "
                    f"macro_f1={report.metrics.get('macro_f1', report.aggregate_metrics.get('macro_f1', 'n/a'))} "
                    f"accuracy={report.metrics.get('accuracy', report.aggregate_metrics.get('accuracy', 'n/a'))}"
                ),
            )
            for report in reversed(evals)
        ]
        for row_model in rows:
            host.eval_list.addItem(row_model.label)
        host.eval_summary.setPlainText(host._format_eval_summary(evals[-1]))

    def _populate_queue_list(self, runs: list[TrainRun]) -> None:
        host = cast(_WorkspaceStateHost, self)
        selected_run_id = host._selected_queue_run_id()
        host.queue_list.blockSignals(True)
        host.queue_list.clear()
        if not runs:
            host.queue_list.addItem("No queued or recent runs yet.")
            host.queue_list.blockSignals(False)
            return

        active_runs = [
            run for run in runs if str(run.status.value).lower() in host._ACTIVE_RUN_STATUSES
        ]
        recent_terminal_runs = [
            run
            for run in sorted(runs, key=lambda item: (item.updated_at, item.id), reverse=True)
            if str(run.status.value).lower() not in host._ACTIVE_RUN_STATUSES
        ][:5]
        visible_runs = sorted(
            {run.id: run for run in [*active_runs, *recent_terminal_runs]}.values(),
            key=lambda item: (item.updated_at, item.id),
            reverse=True,
        )
        active_run_id = host._resolve_active_run_id(active_runs)
        if not visible_runs:
            host.queue_list.addItem("No queued or recent runs yet.")
            host.queue_list.blockSignals(False)
            return

        rows = [
            QueueRunRow(
                run=run,
                label=host._format_queue_entry(run, is_active=run.id == active_run_id),
            )
            for run in visible_runs
        ]
        selected_row = -1
        for index, row_model in enumerate(rows):
            host.queue_list.addItem(row_model.label)
            item = host.queue_list.item(index)
            if item is not None:
                item.setData(Qt.ItemDataRole.UserRole, row_model.run.id)
            if row_model.run.id == selected_run_id or (
                selected_run_id is None and row_model.run.id == host._run_id
            ):
                selected_row = index
        host.queue_list.blockSignals(False)
        if selected_row >= 0:
            host.queue_list.setCurrentRow(selected_row)

    def _sync_run_selection(self, run_id: str | None) -> None:
        host = cast(_WorkspaceStateHost, self)
        host._sync_table_selection(host.run_overview, run_id)
        host._sync_table_selection(host.past_runs_overview, run_id)

    def _sync_table_selection(self, table: QTableWidget, run_id: str | None) -> None:
        if not run_id:
            return
        for row in range(table.rowCount()):
            item = table_item_or_none(table, row, 0)
            if item is None or item.data(Qt.ItemDataRole.UserRole) != run_id:
                continue
            if table.currentRow() != row:
                table.blockSignals(True)
                table.selectRow(row)
                table.blockSignals(False)
            return

    def _sync_queue_selection(self, run_id: str | None) -> None:
        host = cast(_WorkspaceStateHost, self)
        if not run_id:
            return
        for row in range(host.queue_list.count()):
            item = host.queue_list.item(row)
            if item is None or item.data(Qt.ItemDataRole.UserRole) != run_id:
                continue
            if host.queue_list.currentRow() != row:
                host.queue_list.blockSignals(True)
                host.queue_list.setCurrentRow(row)
                host.queue_list.blockSignals(False)
            return

    def _format_workspace_summary(
        self,
        datasets: list[Dataset],
        runs: list[TrainRun],
    ) -> str:
        host = cast(_WorkspaceStateHost, self)
        latest_run = runs[-1] if runs else None
        latest_run_line = "none"
        if latest_run is not None:
            latest_run_line = f"{latest_run.id} ({latest_run.status.value})"
        return "\n".join(
            [
                f"Root: {host._root}",
                f"Datasets: {len(datasets)}",
                f"Runs: {len(runs)}",
                f"Latest run: {latest_run_line}",
                f"State dir: {host._root / 'foundry' / 'state'}",
            ]
        )

    def _on_activity(self, item: object) -> None:
        host = cast(_WorkspaceStateHost, self)
        kind = str(getattr(item, "kind", "activity"))
        message = str(getattr(item, "message", ""))
        host.activity_item_received.emit(kind, message)

    def _append_activity_item(self, kind: str, message: str) -> None:
        cast(_WorkspaceStateHost, self).activity.appendPlainText(f"[{kind}] {message}")

    def _set_status(self, text: str) -> None:
        host = cast(_WorkspaceStateHost, self)
        host.status_line.setText(text)
        host.activity.appendPlainText(f"[status] {text}")

    def _error(self, exc: Exception) -> None:
        host = cast(_WorkspaceStateHost, self)
        message = str(exc)
        host._set_status(f"Error: {message}")
        host.activity.appendPlainText(f"[error] {message}")
        if host._show_error_dialogs:
            QMessageBox.critical(cast(QWidget, host), "Foundry Error", message)


__all__ = ["FoundryWindowWorkspaceStateMixin"]
