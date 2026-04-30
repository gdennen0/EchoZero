"""Run, artifact, and worker-lifecycle actions for the Foundry window.
Exists to keep background execution and artifact/export commands out of the run mixin root.
Connects Foundry application services to run-state updates and desktop actions.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import cast

from PyQt6.QtCore import QObject, QThread, QTimer, QUrl, Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
)

from echozero.foundry import FoundryApp
from echozero.foundry.domain import TrainRun
from echozero.foundry.ui.main_window_types import selected_list_run_id
from echozero.foundry.ui.main_window_worker import _RunWorker


class _FoundryWindowRunActionsMixin:
    _root: Path
    _app: FoundryApp
    _run_id: str | None
    _version_id: str | None
    _artifact_id: str | None
    _selected_artifact_id: str | None
    _run_thread: QThread | None
    _run_worker: _RunWorker | None
    _run_poll_timer: QTimer
    _running_action_label: str | None
    _run_cancel_event: Event | None
    _last_event_count_by_run: dict[str, int]
    _ACTIVE_RUN_STATUSES: set[str]

    epochs: QSpinBox
    batch_size: QSpinBox
    learning_rate: QDoubleSpinBox
    class_names: QLineEdit
    run_summary: QPlainTextEdit
    status_line: QLabel
    activity: QPlainTextEdit
    queue_list: QListWidget
    cancel_queue_run_btn: QPushButton
    retry_queue_run_btn: QPushButton
    _run_action_buttons: list[QPushButton]

    _set_status: Callable[[str], None]
    _error: Callable[[Exception], None]
    _refresh_workspace_state: Callable[..., None]
    _populate_run_overview: Callable[..., None]
    _populate_past_runs_overview: Callable[..., None]
    _populate_queue_list: Callable[[list[TrainRun]], None]
    _populate_artifact_list: Callable[..., None]
    _populate_eval_list: Callable[..., None]
    _format_run_summary: Callable[[TrainRun], str]

    def _create_run(self) -> None:
        try:
            if not self._version_id:
                raise ValueError("Create and plan a dataset before creating a run")
            run = self._app.create_run(self._version_id, self._build_run_spec())
            self._run_id = run.id
            self._set_status(f"Run created: {run.id}")
            self._refresh_workspace_state(select_run_id=run.id)
        except Exception as exc:
            self._error(exc)

    def _create_and_start_run(self) -> None:
        if self._run_thread is not None:
            self._set_status("A run is already active.")
            return

        version_id = self._version_id
        if not version_id:
            self._error(ValueError("Create and plan a dataset before creating a run"))
            return

        run_spec = self._build_run_spec()

        def action() -> str:
            run = self._app.create_run(version_id, run_spec)
            self._run_id = run.id
            return self._app.start_run(run.id, cancel_event=self._run_cancel_event).id

        self._start_background_run(action, action_label="Create + Start")

    def _start_run(self) -> None:
        try:
            run_id = self._require_run_id()
        except Exception as exc:
            self._error(exc)
            return

        if self._run_thread is not None:
            self._set_status("A run is already active.")
            return

        self._start_background_run(
            lambda: self._app.start_run(run_id, cancel_event=self._run_cancel_event).id,
            action_label="Start Run",
        )

    def _checkpoint_run(self) -> None:
        try:
            path = self._app.runs.save_checkpoint(
                self._require_run_id(), epoch=1, metric_snapshot={"loss": 0.123}
            )
            self._set_status(f"Checkpoint saved: {path.name}")
            self._refresh_workspace_state(select_run_id=self._run_id)
        except Exception as exc:
            self._error(exc)

    def _complete_run(self) -> None:
        try:
            run = self._app.runs.complete_run(self._require_run_id(), metrics={"f1": 0.91})
            self._set_status(f"Run marked completed: {run.id}")
            self._refresh_workspace_state(select_run_id=run.id)
        except Exception as exc:
            self._error(exc)

    def _fail_run(self) -> None:
        try:
            run = self._app.runs.fail_run(self._require_run_id(), error="manual-failure")
            self._set_status(f"Run marked failed: {run.id}")
            self._refresh_workspace_state(select_run_id=run.id)
        except Exception as exc:
            self._error(exc)

    def _cancel_selected_queue_run(self) -> None:
        try:
            run = self._require_selected_queue_run()
            if str(run.status.value).lower() not in self._ACTIVE_RUN_STATUSES:
                raise ValueError("Only queued or active runs can be canceled")
            if (
                self._run_thread is not None
                and self._run_id == run.id
                and self._run_cancel_event is not None
            ):
                self._run_cancel_event.set()
            run = self._app.runs.cancel_run(run.id, reason="user")
            self._set_status(f"Canceled run: {run.id}")
            self._refresh_workspace_state(select_run_id=run.id)
        except Exception as exc:
            self._error(exc)

    def _retry_selected_queue_run(self) -> None:
        try:
            run = self._require_selected_queue_run()
            if run.status.value not in {"failed", "canceled"}:
                raise ValueError("Only failed or canceled runs can be requeued")
            run = self._app.runs.resume_run(run.id)
            self._set_status(f"Requeued run: {run.id}")
            self._refresh_workspace_state(select_run_id=run.id)
        except Exception as exc:
            self._error(exc)

    def _finalize_artifact(self) -> None:
        try:
            artifact = self._app.finalize_artifact(
                self._require_run_id(), self._build_artifact_manifest()
            )
            self._artifact_id = artifact.id
            self._set_status(f"Artifact finalized: {artifact.id}")
            self._refresh_workspace_state(
                select_run_id=self._run_id,
                select_artifact_id=artifact.id,
            )
        except Exception as exc:
            self._error(exc)

    def _validate_artifact(self) -> None:
        try:
            artifact_id = self._selected_artifact_id or self._artifact_id
            if not artifact_id:
                raise ValueError("Select or create an artifact first")
            report = self._app.validate_artifact(artifact_id)
            self._set_status(
                f"Validation for {artifact_id}: ok={report.ok}, "
                f"errors={len(report.errors)}, warnings={len(report.warnings)}"
            )
            self._refresh_workspace_state(
                select_run_id=self._run_id,
                select_artifact_id=artifact_id,
            )
        except Exception as exc:
            self._error(exc)

    def _open_exports_dir(self) -> None:
        try:
            run = self._require_selected_run()
            self._open_existing_path(run.exports_dir(self._root), label="exports dir")
        except Exception as exc:
            self._error(exc)

    def _open_metrics_json(self) -> None:
        try:
            run = self._require_selected_run()
            self._open_existing_path(
                run.exports_dir(self._root) / "metrics.json", label="metrics.json"
            )
        except Exception as exc:
            self._error(exc)

    def _open_run_summary_json(self) -> None:
        try:
            run = self._require_selected_run()
            self._open_existing_path(
                run.exports_dir(self._root) / "run_summary.json", label="run_summary.json"
            )
        except Exception as exc:
            self._error(exc)

    def _open_artifact_manifest(self) -> None:
        try:
            artifact_id = self._selected_artifact_id or self._artifact_id
            if not artifact_id and self._run_id:
                artifacts = sorted(
                    self._app.list_artifacts_for_run(self._run_id),
                    key=lambda item: item.created_at,
                )
                artifact_id = artifacts[-1].id if artifacts else None
            if not artifact_id:
                raise ValueError("Select or create an artifact first")
            artifact = self._app.get_artifact(artifact_id)
            if artifact is None:
                raise ValueError(f"Artifact not found: {artifact_id}")
            self._open_existing_path(artifact.path, label="artifact manifest")
        except Exception as exc:
            self._error(exc)

    def _open_latest_artifact_package(self) -> None:
        path = self._resolve_latest_artifact_package_path()
        if path is None:
            self._set_status(
                "No artifact package available yet. Complete a run to generate exports."
            )
            return
        if not path.exists():
            self._set_status(f"Latest artifact package is missing on disk: {path}")
            return
        try:
            self._open_existing_path(path, label="latest artifact package")
        except Exception as exc:
            self._error(exc)

    def _build_run_spec(self) -> dict[str, object]:
        if not self._version_id:
            raise ValueError("No dataset version selected")
        return {
            "schema": "foundry.train_run_spec.v1",
            "classificationMode": "multiclass",
            "data": {
                "datasetVersionId": self._version_id,
                "sampleRate": 22050,
                "maxLength": 22050,
                "nFft": 2048,
                "hopLength": 512,
                "nMels": 128,
                "fmax": 8000,
            },
            "training": {
                "epochs": int(self.epochs.value()),
                "batchSize": int(self.batch_size.value()),
                "learningRate": float(self.learning_rate.value()),
                "seed": 42,
            },
        }

    def _build_artifact_manifest(self) -> dict[str, object]:
        run = self._app.runs.get_run(self._require_run_id())
        if run is None:
            raise ValueError("Run not found")

        version = self._app.datasets.get_version(run.dataset_version_id)
        classes = [c.strip() for c in self.class_names.text().split(",") if c.strip()]
        if version is not None and not classes:
            classes = list(version.class_map)
        if not classes:
            raise ValueError("At least one class is required")

        return {
            "weightsPath": "model.pth",
            "classes": classes,
            "classificationMode": "multiclass",
            "inferencePreprocessing": dict(run.spec.get("data", {})),
        }

    def _resolve_latest_artifact_package_path(self) -> Path | None:
        artifacts = sorted(self._app.list_artifacts(), key=lambda item: (item.created_at, item.id))
        if artifacts:
            return artifacts[-1].path.parent

        runs = sorted(self._app.list_runs(), key=lambda item: (item.updated_at, item.id))
        for run in reversed(runs):
            exports_dir = run.exports_dir(self._root)
            if exports_dir.exists() and any(exports_dir.iterdir()):
                return exports_dir
        return None

    def _resolve_active_run_id(self, active_runs: list[TrainRun]) -> str | None:
        if self._run_id:
            current_run = self._app.runs.get_run(self._run_id)
            if (
                current_run is not None
                and str(current_run.status.value).lower() in self._ACTIVE_RUN_STATUSES
            ):
                return current_run.id
        if not active_runs:
            return None
        return max(active_runs, key=lambda item: (item.updated_at, item.id)).id

    def _format_queue_entry(self, run: TrainRun, *, is_active: bool) -> str:
        marker = "ACTIVE " if is_active else ""
        return (
            f"{marker}{run.id} [{run.status.value}] "
            f"created {run.created_at.strftime('%Y-%m-%d %H:%M:%S')} "
            f"updated {run.updated_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def _require_run_id(self) -> str:
        if not self._run_id:
            raise ValueError("No run selected")
        return self._run_id

    def _require_selected_run(self) -> TrainRun:
        run_id = self._require_run_id()
        run = self._app.runs.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        return run

    def _start_background_run(self, action: Callable[[], str], *, action_label: str) -> None:
        self._running_action_label = action_label
        self._run_cancel_event = Event()
        self._set_run_controls_enabled(False)
        self._set_status(f"{action_label} running in background...")
        self._run_thread = QThread(cast(QObject, self))
        self._run_worker = _RunWorker(action)
        self._run_worker.moveToThread(self._run_thread)
        # Queue the work after the thread event loop is running so quit() is processed reliably.
        self._run_thread.started.connect(
            self._run_worker.run,
            Qt.ConnectionType.QueuedConnection,
        )
        self._run_worker.finished.connect(self._on_background_run_finished)
        self._run_worker.failed.connect(self._on_background_run_failed)
        self._run_worker.finished.connect(self._run_thread.quit)
        self._run_worker.failed.connect(self._run_thread.quit)
        self._run_thread.start()
        self._run_poll_timer.start()

    def _poll_active_run(self) -> None:
        if not self._run_id:
            return
        run = self._app.runs.get_run(self._run_id)
        runs = sorted(self._app.list_runs(), key=lambda item: item.created_at)
        if run is not None:
            self.run_summary.setPlainText(self._format_run_summary(run))
            self.status_line.setText(
                f"{self._running_action_label or 'Run'}: {run.id} [{run.status.value}]"
            )
            if (
                self._run_thread is not None
                and str(run.status.value).lower() not in self._ACTIVE_RUN_STATUSES
            ):
                self._append_new_run_events(run.id)
                self._refresh_workspace_state(
                    select_run_id=run.id,
                    select_artifact_id=self._artifact_id,
                )
                self._finalize_background_run_thread()
        self._populate_run_overview(runs, select_run_id=self._run_id)
        self._populate_past_runs_overview(runs, select_run_id=self._run_id)
        self._append_new_run_events(self._run_id)
        self._populate_queue_list(runs)
        self._populate_artifact_list(
            sorted(
                self._app.list_artifacts_for_run(self._run_id),
                key=lambda item: item.created_at,
            ),
            select_artifact_id=self._artifact_id,
        )
        self._populate_eval_list(
            sorted(
                self._app.list_eval_reports_for_run(self._run_id),
                key=lambda item: item.created_at,
            )
        )

    def _append_new_run_events(self, run_id: str) -> None:
        run = self._app.runs.get_run(run_id)
        if run is None:
            return
        path = run.event_log_path(self._root)
        if not path.exists():
            return
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        previous_count = self._last_event_count_by_run.get(run_id, 0)
        for raw_line in lines[previous_count:]:
            payload = json.loads(raw_line)
            event_type = str(payload.get("type", "RUN_EVENT"))
            event_payload = payload.get("payload", {})
            details = ", ".join(f"{key}={value}" for key, value in sorted(event_payload.items()))
            message = event_type if not details else f"{event_type}: {details}"
            self.activity.appendPlainText(f"[run] {message}")
        self._last_event_count_by_run[run_id] = len(lines)

    def _on_background_run_finished(self, run_id: str) -> None:
        run = self._app.runs.get_run(run_id)
        artifacts = sorted(
            self._app.list_artifacts_for_run(run_id),
            key=lambda item: item.created_at,
        )
        evals = sorted(
            self._app.list_eval_reports_for_run(run_id),
            key=lambda item: item.created_at,
        )
        self._artifact_id = artifacts[-1].id if artifacts else None
        self._append_new_run_events(run_id)
        if run is not None:
            self._run_id = run.id
            self._set_status(
                f"Run {run.id} finished with status {run.status.value} "
                f"({len(evals)} eval, {len(artifacts)} artifact)"
            )
        self._refresh_workspace_state(
            select_run_id=run_id,
            select_artifact_id=self._artifact_id,
        )
        self._run_poll_timer.stop()
        self._finalize_background_run_thread()

    def _on_background_run_failed(self, message: str) -> None:
        self._run_poll_timer.stop()
        if self._run_id is not None:
            self._append_new_run_events(self._run_id)
        self._refresh_workspace_state(
            select_run_id=self._run_id,
            select_artifact_id=self._artifact_id,
        )
        self._error(RuntimeError(message))
        self._finalize_background_run_thread()

    def _finalize_background_run_thread(self) -> None:
        thread = self._run_thread
        if thread is None:
            return
        if thread.isRunning():
            thread.quit()
            thread.wait(1000)
        if thread.isRunning():
            return
        self._cleanup_run_worker()

    def _cleanup_run_worker(self) -> None:
        self._set_run_controls_enabled(True)
        self._run_cancel_event = None
        if self._run_worker is not None:
            self._run_worker.deleteLater()
            self._run_worker = None
        if self._run_thread is not None:
            self._run_thread.deleteLater()
            self._run_thread = None
        self._running_action_label = None

    def _set_run_controls_enabled(self, enabled: bool) -> None:
        for button in self._run_action_buttons:
            button.setEnabled(enabled)
        self._update_queue_action_buttons()

    def _selected_queue_run_id(self) -> str | None:
        return selected_list_run_id(self.queue_list)

    def _require_selected_queue_run(self) -> TrainRun:
        run_id = self._selected_queue_run_id() or self._run_id
        if not run_id:
            raise ValueError("Select a queue run first")
        run = self._app.runs.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        return run

    def _update_queue_action_buttons(self) -> None:
        run_id = self._selected_queue_run_id() or self._run_id
        run = self._app.runs.get_run(run_id) if run_id else None
        can_cancel = False
        can_retry = False
        if run is not None:
            status = str(run.status.value).lower()
            can_cancel = status in self._ACTIVE_RUN_STATUSES
            can_retry = status in {"failed", "canceled"}
        self.cancel_queue_run_btn.setEnabled(can_cancel)
        self.retry_queue_run_btn.setEnabled(can_retry)

    def _open_existing_path(self, path: Path, *, label: str) -> None:
        from echozero.foundry.ui import main_window as main_window_module

        if not path.exists():
            raise ValueError(f"{label} not found: {path}")
        if not main_window_module.QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            raise ValueError(f"Could not open {label}: {path}")
        self._set_status(f"Opened {label}: {path}")
