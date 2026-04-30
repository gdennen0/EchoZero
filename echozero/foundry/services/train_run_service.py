from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from uuid import uuid4

from echozero.foundry.domain import TrainRun, TrainRunStatus
from echozero.foundry.persistence import DatasetVersionRepository, EvalReportRepository, TrainRunRepository
from echozero.foundry.services.artifact_service import ArtifactService
from echozero.foundry.services.baseline_trainer import BaselineTrainer, RunCanceledError
from echozero.foundry.services.eval_service import EvalService
from echozero.foundry.services.run_notification_service import RunNotificationService
from echozero.foundry.services.run_spec_validator import RunSpecValidator
from echozero.foundry.services.run_telemetry_service import RunTelemetryService
from echozero.foundry.services.trainer_backend_factory import TrainerBackendFactory


_ALLOWED_TRANSITIONS: dict[TrainRunStatus, set[TrainRunStatus]] = {
    TrainRunStatus.QUEUED: {TrainRunStatus.PREPARING, TrainRunStatus.RUNNING, TrainRunStatus.CANCELED},
    TrainRunStatus.PREPARING: {TrainRunStatus.RUNNING, TrainRunStatus.FAILED, TrainRunStatus.CANCELED},
    TrainRunStatus.RUNNING: {
        TrainRunStatus.EVALUATING,
        TrainRunStatus.EXPORTING,
        TrainRunStatus.COMPLETED,
        TrainRunStatus.FAILED,
        TrainRunStatus.CANCELED,
    },
    TrainRunStatus.EVALUATING: {TrainRunStatus.EXPORTING, TrainRunStatus.COMPLETED, TrainRunStatus.FAILED, TrainRunStatus.CANCELED},
    TrainRunStatus.EXPORTING: {TrainRunStatus.COMPLETED, TrainRunStatus.FAILED, TrainRunStatus.CANCELED},
    TrainRunStatus.COMPLETED: set(),
    TrainRunStatus.FAILED: {TrainRunStatus.QUEUED},
    TrainRunStatus.CANCELED: {TrainRunStatus.QUEUED},
}


_TERMINAL_STATUSES = {TrainRunStatus.COMPLETED, TrainRunStatus.FAILED, TrainRunStatus.CANCELED}


class TrainRunService:
    def __init__(
        self,
        root: Path,
        repository: TrainRunRepository | None = None,
        dataset_version_repository: DatasetVersionRepository | None = None,
        eval_service: EvalService | None = None,
        artifact_service: ArtifactService | None = None,
        baseline_trainer: BaselineTrainer | None = None,
        trainer_factory: TrainerBackendFactory | None = None,
    ):
        self._root = root
        self._repo = repository or TrainRunRepository(root)
        self._dataset_versions = dataset_version_repository or DatasetVersionRepository(root)
        self._eval = eval_service or EvalService(EvalReportRepository(root))
        self._artifacts = artifact_service or ArtifactService(root)
        self._legacy_trainer = baseline_trainer or BaselineTrainer(root)
        # Backward-compatible alias used by existing UI tests that monkeypatch trainer.train.
        self._trainer = self._legacy_trainer
        self._trainer_factory = trainer_factory or TrainerBackendFactory()
        self._validator = RunSpecValidator(self._dataset_versions)
        self._telemetry = RunTelemetryService(root)
        self._notifications = RunNotificationService(root)

    def create_run(self, dataset_version_id: str, run_spec: dict, backend: str = "pytorch", device: str = "cpu") -> TrainRun:
        self._validate_run_spec(dataset_version_id, run_spec)
        spec_json = json.dumps(run_spec, sort_keys=True)
        spec_hash = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()
        run = TrainRun(
            id=f"run_{uuid4().hex[:12]}",
            dataset_version_id=dataset_version_id,
            status=TrainRunStatus.QUEUED,
            spec=run_spec,
            spec_hash=spec_hash,
            backend=backend,
            device=device,
        )
        run_dir = run.run_dir(self._root)
        run.checkpoints_dir(self._root).mkdir(parents=True, exist_ok=True)
        run.exports_dir(self._root).mkdir(parents=True, exist_ok=True)
        run.logs_dir(self._root).mkdir(parents=True, exist_ok=True)
        (run_dir / "spec.json").write_text(json.dumps(run_spec, indent=2), encoding="utf-8")
        self._append_event(run, "RUN_CREATED", {"status": run.status.value})
        return self._repo.save(run)

    def start_run(self, run_id: str, cancel_event: Event | None = None) -> TrainRun:
        run = self._transition(run_id, TrainRunStatus.PREPARING, "RUN_PREPARING")
        try:
            self._raise_if_canceled(cancel_event)
            dataset_version = self._dataset_versions.get(run.dataset_version_id)
            if dataset_version is None:
                raise ValueError(f"DatasetVersion not found: {run.dataset_version_id}")

            run = self._transition(run.id, TrainRunStatus.RUNNING, "RUN_STARTED")
            trainer = self._trainer_factory.resolve(run.spec, legacy_backend=self._trainer)
            progress_callback = lambda payload: self.save_checkpoint(
                run.id,
                epoch=int(payload.get("epoch", 0)),
                metric_snapshot=dict(payload.get("checkpoint", {})),
            )
            try:
                result = trainer.train(
                    run,
                    dataset_version,
                    cancel_event=cancel_event,
                    progress_callback=progress_callback,
                )
            except TypeError as exc:
                if "progress_callback" not in str(exc):
                    raise
                result = trainer.train(
                    run,
                    dataset_version,
                    cancel_event=cancel_event,
                )

            saved_epochs = {
                int(path.stem.split("_")[-1])
                for path in run.checkpoints_dir(self._root).glob("epoch_*.json")
                if path.stem.split("_")[-1].isdigit()
            }
            for checkpoint in result.checkpoint_metrics:
                epoch = int(checkpoint["epoch"])
                if epoch in saved_epochs:
                    continue
                self._raise_if_canceled(cancel_event)
                self.save_checkpoint(
                    run.id,
                    epoch=epoch,
                    metric_snapshot={key: value for key, value in checkpoint.items() if key != "epoch"},
                )

            run = self._transition(run.id, TrainRunStatus.EVALUATING, "RUN_EVALUATING")
            self._raise_if_canceled(cancel_event)
            self._eval.record_eval(
                run.id,
                classification_mode=run.spec["classificationMode"],
                metrics=result.final_metrics,
                dataset_version_id=dataset_version.id,
                split_name=result.eval_split_name,
                aggregate_metrics=result.aggregate_metrics,
                per_class_metrics=result.per_class_metrics,
                baseline=result.baseline,
                confusion=result.confusion,
                summary=result.summary,
            )

            run = self._transition(run.id, TrainRunStatus.EXPORTING, "RUN_EXPORTING")
            self._raise_if_canceled(cancel_event)
            self._artifacts.finalize_artifact(run.id, result.artifact_manifest)

            return self.complete_run(run.id, metrics=result.final_metrics)
        except RunCanceledError:
            current = self.get_run(run.id)
            if current is not None and current.status == TrainRunStatus.CANCELED:
                return current
            return self.cancel_run(run.id, "user")
        except Exception as exc:
            return self.fail_run(run.id, str(exc))

    def cancel_run(self, run_id: str, reason: str = "user") -> TrainRun:
        return self._transition(run_id, TrainRunStatus.CANCELED, "RUN_CANCELED", {"reason": reason})

    def complete_run(self, run_id: str, metrics: dict | None = None) -> TrainRun:
        return self._transition(
            run_id,
            TrainRunStatus.COMPLETED,
            "RUN_COMPLETED",
            {"metrics": metrics or {}},
        )

    def fail_run(self, run_id: str, error: str) -> TrainRun:
        return self._transition(run_id, TrainRunStatus.FAILED, "RUN_FAILED", {"error": error})

    def resume_run(self, run_id: str) -> TrainRun:
        return self._transition(run_id, TrainRunStatus.QUEUED, "RUN_RESUMED")

    def set_stage(self, run_id: str, stage: TrainRunStatus) -> TrainRun:
        if stage not in {TrainRunStatus.PREPARING, TrainRunStatus.EVALUATING, TrainRunStatus.EXPORTING}:
            raise ValueError(f"Unsupported stage transition: {stage.value}")
        return self._transition(run_id, stage, f"RUN_{stage.value.upper()}")

    def save_checkpoint(self, run_id: str, epoch: int, metric_snapshot: dict | None = None) -> Path:
        if epoch < 1:
            raise ValueError("checkpoint epoch must be >= 1")
        run = self._require(run_id)
        ckpt_path = run.checkpoints_dir(self._root) / f"epoch_{epoch:04d}.json"
        payload = {
            "run_id": run.id,
            "epoch": epoch,
            "metric_snapshot": metric_snapshot or {},
            "at": datetime.now(UTC).isoformat(),
        }
        temp_path = ckpt_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(ckpt_path)
        self._append_event(
            run,
            "CHECKPOINT_SAVED",
            {"epoch": epoch, "path": str(ckpt_path), "metric_snapshot": metric_snapshot or {}},
        )
        snapshot = metric_snapshot or {}
        self._write_progress_snapshot(run.id, epoch=epoch, metric_snapshot=snapshot)
        self._append_run_telemetry(run, epoch=epoch, metric_snapshot=snapshot)
        return ckpt_path

    def get_run(self, run_id: str) -> TrainRun | None:
        run = self._repo.get(run_id)
        if run is not None:
            return run
        return self._repo.read_run_from_disk(run_id)

    def list_runs(self) -> list[TrainRun]:
        return self._repo.list_all()

    @staticmethod
    def _raise_if_canceled(cancel_event: Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RunCanceledError("run canceled")

    def _require(self, run_id: str) -> TrainRun:
        run = self._repo.get(run_id)
        if not run:
            raise ValueError(f"TrainRun not found: {run_id}")
        return run

    def _transition(
        self,
        run_id: str,
        new_status: TrainRunStatus,
        event_type: str,
        extra_payload: dict | None = None,
    ) -> TrainRun:
        run = self._require(run_id)
        allowed = _ALLOWED_TRANSITIONS.get(run.status, set())
        if new_status not in allowed:
            raise ValueError(f"Invalid run transition: {run.status.value} -> {new_status.value}")

        run.status = new_status
        run.updated_at = datetime.now(UTC)
        payload = {"status": run.status.value}
        if extra_payload:
            payload.update(extra_payload)
        self._append_event(run, event_type, payload)
        saved = self._repo.save(run)

        self._write_status_snapshot(saved.id, status=saved.status.value, event_type=event_type)
        if new_status in _TERMINAL_STATUSES:
            self._refresh_tracking_artifacts(saved.id, status=saved.status.value)
        self._emit_notification_cadence(saved, event_type=event_type)

        return saved

    def _append_event(self, run: TrainRun, event_type: str, payload: dict) -> None:
        event = {
            "at": datetime.now(UTC).isoformat(),
            "type": event_type,
            "payload": payload,
        }
        path = run.event_log_path(self._root)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")

    def _write_status_snapshot(self, run_id: str, *, status: str, event_type: str) -> None:
        self._telemetry.write_status_snapshot(run_id, status=status, event_type=event_type)

    def _write_progress_snapshot(self, run_id: str, *, epoch: int, metric_snapshot: dict) -> None:
        self._telemetry.write_progress_snapshot(run_id, epoch=epoch, metric_snapshot=metric_snapshot)

    def _append_run_telemetry(self, run: TrainRun, *, epoch: int, metric_snapshot: dict[str, object]) -> None:
        self._telemetry.append_run_telemetry(run, epoch=epoch, metric_snapshot=metric_snapshot)

    def _collect_system_stats(self) -> dict[str, object]:
        return self._telemetry.collect_system_stats()

    def _refresh_tracking_artifacts(self, run_id: str, *, status: str) -> None:
        self._telemetry.refresh_tracking_artifacts(run_id, status=status)

    def _emit_notification_cadence(self, run: TrainRun, *, event_type: str) -> None:
        self._notifications.emit_notification_cadence(
            run,
            event_type=event_type,
            list_runs=self.list_runs,
            notify=self._notify_openclaw,
        )

    def _notification_state_path(self) -> Path:
        return self._notifications.notification_state_path()

    def _read_notification_state(self) -> dict[str, object]:
        return self._notifications.read_notification_state()

    def _write_notification_state(self, state: dict[str, object]) -> None:
        self._notifications.write_notification_state(state)

    def _notify_openclaw_deduped(
        self,
        key: str,
        text: str,
        *,
        cooldown_seconds: int,
        state: dict[str, object],
    ) -> None:
        self._notifications.notify_openclaw_deduped(
            key,
            text,
            cooldown_seconds=cooldown_seconds,
            state=state,
            notify=self._notify_openclaw,
        )

    def _notify_openclaw(self, text: str) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        self._notifications.notify_openclaw(text)

    def _validate_run_spec(self, dataset_version_id: str, run_spec: dict) -> None:
        self._validator.validate(dataset_version_id, run_spec)
