from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from echozero.foundry.domain import TrainRun, TrainRunStatus
from echozero.foundry.persistence import DatasetVersionRepository, EvalReportRepository, TrainRunRepository
from echozero.foundry.services.artifact_service import ArtifactService
from echozero.foundry.services.baseline_trainer import BaselineTrainer
from echozero.foundry.services.eval_service import EvalService


_REQUIRED_DATA_KEYS = {"datasetVersionId", "sampleRate", "maxLength", "nFft", "hopLength", "nMels", "fmax"}
_REQUIRED_TRAINING_KEYS = {"epochs", "batchSize", "learningRate"}
_SUPPORTED_CLASSIFICATION_MODES = {"multiclass", "binary", "positive_vs_other"}


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
    TrainRunStatus.EVALUATING: {TrainRunStatus.EXPORTING, TrainRunStatus.COMPLETED, TrainRunStatus.FAILED},
    TrainRunStatus.EXPORTING: {TrainRunStatus.COMPLETED, TrainRunStatus.FAILED},
    TrainRunStatus.COMPLETED: set(),
    TrainRunStatus.FAILED: {TrainRunStatus.QUEUED},
    TrainRunStatus.CANCELED: {TrainRunStatus.QUEUED},
}


class TrainRunService:
    def __init__(
        self,
        root: Path,
        repository: TrainRunRepository | None = None,
        dataset_version_repository: DatasetVersionRepository | None = None,
        eval_service: EvalService | None = None,
        artifact_service: ArtifactService | None = None,
        baseline_trainer: BaselineTrainer | None = None,
    ):
        self._root = root
        self._repo = repository or TrainRunRepository(root)
        self._dataset_versions = dataset_version_repository or DatasetVersionRepository(root)
        self._eval = eval_service or EvalService(EvalReportRepository(root))
        self._artifacts = artifact_service or ArtifactService(root)
        self._trainer = baseline_trainer or BaselineTrainer(root)

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

    def start_run(self, run_id: str) -> TrainRun:
        run = self._transition(run_id, TrainRunStatus.PREPARING, "RUN_PREPARING")
        try:
            dataset_version = self._dataset_versions.get(run.dataset_version_id)
            if dataset_version is None:
                raise ValueError(f"DatasetVersion not found: {run.dataset_version_id}")

            run = self._transition(run.id, TrainRunStatus.RUNNING, "RUN_STARTED")
            result = self._trainer.train(run, dataset_version)

            for checkpoint in result.checkpoint_metrics:
                self.save_checkpoint(
                    run.id,
                    epoch=int(checkpoint["epoch"]),
                    metric_snapshot={key: value for key, value in checkpoint.items() if key != "epoch"},
                )

            run = self._transition(run.id, TrainRunStatus.EVALUATING, "RUN_EVALUATING")
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
            self._artifacts.finalize_artifact(run.id, result.artifact_manifest)

            return self.complete_run(run.id, metrics=result.final_metrics)
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
        run = self._require(run_id)
        ckpt_path = run.checkpoints_dir(self._root) / f"epoch_{epoch:04d}.json"
        payload = {
            "run_id": run.id,
            "epoch": epoch,
            "metric_snapshot": metric_snapshot or {},
            "at": datetime.now(UTC).isoformat(),
        }
        ckpt_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._append_event(
            run,
            "CHECKPOINT_SAVED",
            {"epoch": epoch, "path": str(ckpt_path), "metric_snapshot": metric_snapshot or {}},
        )
        return ckpt_path

    def get_run(self, run_id: str) -> TrainRun | None:
        return self._repo.get(run_id)

    def list_runs(self) -> list[TrainRun]:
        return self._repo.list()

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
        return self._repo.save(run)

    def _append_event(self, run: TrainRun, event_type: str, payload: dict) -> None:
        event = {
            "at": datetime.now(UTC).isoformat(),
            "type": event_type,
            "payload": payload,
        }
        path = run.event_log_path(self._root)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")

    def _validate_run_spec(self, dataset_version_id: str, run_spec: dict) -> None:
        if run_spec.get("schema") != "foundry.train_run_spec.v1":
            raise ValueError("run_spec.schema must be foundry.train_run_spec.v1")

        classification_mode = run_spec.get("classificationMode")
        if classification_mode not in _SUPPORTED_CLASSIFICATION_MODES:
            raise ValueError("run_spec.classificationMode is unsupported")

        data = run_spec.get("data")
        if not isinstance(data, dict):
            raise ValueError("run_spec.data must be an object")
        missing_data = sorted(_REQUIRED_DATA_KEYS - set(data.keys()))
        if missing_data:
            raise ValueError(f"run_spec.data missing keys: {', '.join(missing_data)}")

        training = run_spec.get("training")
        if not isinstance(training, dict):
            raise ValueError("run_spec.training must be an object")
        missing_training = sorted(_REQUIRED_TRAINING_KEYS - set(training.keys()))
        if missing_training:
            raise ValueError(f"run_spec.training missing keys: {', '.join(missing_training)}")

        dataset_version = self._dataset_versions.get(dataset_version_id)
        if dataset_version is None:
            raise ValueError(f"DatasetVersion not found: {dataset_version_id}")

        if data.get("datasetVersionId") != dataset_version_id:
            raise ValueError("run_spec.data.datasetVersionId must match the requested dataset version")
        if int(data["sampleRate"]) != dataset_version.sample_rate:
            raise ValueError("run_spec.data.sampleRate must match dataset version sample_rate")
        if dataset_version.label_policy.get("classification_mode") not in {None, classification_mode}:
            raise ValueError("run_spec.classificationMode must match dataset label policy")

        split_plan = dataset_version.split_plan or {}
        assignments = split_plan.get("assignments", {})
        if not assignments:
            raise ValueError("dataset version must have split assignments before training")
        if not split_plan.get("train_ids"):
            raise ValueError("dataset version split plan must contain train samples")
        if not split_plan.get("val_ids"):
            raise ValueError("dataset version split plan must contain validation samples")
        if split_plan.get("leakage", {}).get("duplicate_hashes_across_splits"):
            raise ValueError("dataset version split plan has duplicate content hashes across splits")

        if len(dataset_version.class_map) < 2 and classification_mode == "multiclass":
            raise ValueError("multiclass training requires at least two classes")
        if dataset_version.taxonomy.get("namespace") != "percussion.one_shot":
            raise ValueError("dataset taxonomy must target percussion.one_shot for the v1 baseline")

        if int(training["epochs"]) < 1:
            raise ValueError("run_spec.training.epochs must be >= 1")
        if int(training["batchSize"]) < 1:
            raise ValueError("run_spec.training.batchSize must be >= 1")
        if float(training["learningRate"]) <= 0:
            raise ValueError("run_spec.training.learningRate must be > 0")

        if int(data["maxLength"]) < int(data["sampleRate"]) // 10:
            raise ValueError("run_spec.data.maxLength is too small for one-shot training")
        if int(data["hopLength"]) >= int(data["nFft"]):
            raise ValueError("run_spec.data.hopLength must be smaller than nFft")
        if int(data["fmax"]) > int(data["sampleRate"]) // 2:
            raise ValueError("run_spec.data.fmax must not exceed the Nyquist limit")
