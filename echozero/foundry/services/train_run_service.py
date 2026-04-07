from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None
from pathlib import Path
from threading import Event
from uuid import uuid4

from echozero.foundry.domain import TrainRun, TrainRunStatus
from echozero.foundry.persistence import DatasetVersionRepository, EvalReportRepository, TrainRunRepository
from echozero.foundry.services.artifact_service import ArtifactService
from echozero.foundry.services.baseline_trainer import BaselineTrainer, RunCanceledError
from echozero.foundry.services.eval_service import EvalService
from echozero.foundry.services.trainer_backend_factory import TrainerBackendFactory


_REQUIRED_DATA_KEYS = {"datasetVersionId", "sampleRate", "maxLength", "nFft", "hopLength", "nMels", "fmax"}
_REQUIRED_TRAINING_KEYS = {"epochs", "batchSize", "learningRate"}
_SUPPORTED_CLASSIFICATION_MODES = {"multiclass", "binary", "positive_vs_other"}
_SYNTHETIC_MIX_KEYS = {"enabled", "ratio", "cap"}
_SUPPORTED_TRAINER_PROFILES = {"baseline_v1", "stronger_v1"}
_SUPPORTED_OPTIMIZERS = {"sgd_constant", "sgd_optimal"}
_PROMOTION_KEYS = {"gate_policy", "reference_run_id", "reference_artifact_id"}
_GATE_POLICY_KEYS = {
    "macro_f1_floor",
    "max_regression_vs_reference",
    "max_real_vs_synth_gap",
    "per_class_recall_floors",
}


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
            trainer = self._trainer_factory.resolve(run.spec, legacy_backend=self._legacy_trainer)
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
        return self._repo.get(run_id)

    def list_runs(self) -> list[TrainRun]:
        return self._repo.list()

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
        tracking = self._root / "foundry" / "tracking" / "snapshots"
        tracking.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "status": status,
            "event_type": event_type,
            "at": datetime.now(UTC).isoformat(),
        }
        (tracking / f"{run_id}_latest_status.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _write_progress_snapshot(self, run_id: str, *, epoch: int, metric_snapshot: dict) -> None:
        tracking = self._root / "foundry" / "tracking" / "snapshots"
        tracking.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "epoch": epoch,
            "metric_snapshot": metric_snapshot,
            "at": datetime.now(UTC).isoformat(),
        }
        (tracking / f"{run_id}_latest_progress.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _append_run_telemetry(self, run: TrainRun, *, epoch: int, metric_snapshot: dict[str, Any]) -> None:
        run_dir = run.run_dir(self._root)
        telemetry_path = run_dir / "telemetry.jsonl"
        latest_path = run_dir / "telemetry.latest.json"

        loss = metric_snapshot.get("val_loss")
        if loss is None:
            loss = metric_snapshot.get("train_loss")

        payload: dict[str, Any] = {
            "at": datetime.now(UTC).isoformat(),
            "run_id": run.id,
            "status": run.status.value,
            "epoch": int(epoch),
            "loss": loss,
            "eta_seconds": metric_snapshot.get("eta_seconds"),
            "metrics": metric_snapshot,
        }

        system_stats = self._collect_system_stats()
        payload.update(system_stats)

        with telemetry_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
        latest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _collect_system_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "cpu_percent": None,
            "ram_percent": None,
            "ram_used_mb": None,
            "gpu_vram_used_mb": None,
            "gpu_vram_total_mb": None,
        }

        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                stats["cpu_percent"] = float(psutil.cpu_percent(interval=None))
                stats["ram_percent"] = float(vm.percent)
                stats["ram_used_mb"] = round(float(vm.used) / (1024 * 1024), 2)
            except Exception:
                pass

        if torch is not None:
            try:
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    used = float(torch.cuda.memory_allocated(device)) / (1024 * 1024)
                    total = float(torch.cuda.get_device_properties(device).total_memory) / (1024 * 1024)
                    stats["gpu_vram_used_mb"] = round(used, 2)
                    stats["gpu_vram_total_mb"] = round(total, 2)
            except Exception:
                pass

        return stats

    def _refresh_tracking_artifacts(self, run_id: str, *, status: str) -> None:
        scripts = [
            self._root / "scripts" / "refresh_foundry_tracking.py",
            self._root / "scripts" / "build_foundry_dashboard.py",
        ]
        for script in scripts:
            if not script.exists():
                continue
            try:
                subprocess.run(
                    [sys.executable, str(script)],
                    cwd=str(self._root),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except Exception:
                pass

        marker = self._root / "foundry" / "tracking" / "snapshots"
        marker.mkdir(parents=True, exist_ok=True)
        (marker / f"{run_id}_terminal.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": status,
                    "at": datetime.now(UTC).isoformat(),
                    "dashboard": str(self._root / "foundry" / "tracking" / "dashboard.html"),
                    "brief": str(self._root / "foundry" / "tracking" / "training_brief.md"),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _emit_notification_cadence(self, run: TrainRun, *, event_type: str) -> None:
        state = self._read_notification_state()

        if event_type == "RUN_STARTED":
            self._notify_openclaw_deduped(
                f"start:{run.id}",
                f"Foundry run started: {run.id} ({run.spec.get('model', {}).get('type', 'baseline_sgd')})",
                cooldown_seconds=45,
                state=state,
            )

        if run.status in _TERMINAL_STATUSES:
            terminal_runs = [item for item in self.list_runs() if item.status in _TERMINAL_STATUSES]
            active_runs = [item for item in self.list_runs() if item.status in {TrainRunStatus.QUEUED, TrainRunStatus.PREPARING, TrainRunStatus.RUNNING, TrainRunStatus.EVALUATING, TrainRunStatus.EXPORTING}]

            if run.status == TrainRunStatus.FAILED and not state.get("first_failure_sent", False):
                self._notify_openclaw_deduped(
                    "first_failure",
                    f"Foundry first failure detected: {run.id}. Check dashboard for details.",
                    cooldown_seconds=60,
                    state=state,
                )
                state["first_failure_sent"] = True

            terminal_count = len(terminal_runs)
            last_milestone = int(state.get("last_milestone_notified", 0))
            if terminal_count >= 3 and terminal_count % 3 == 0 and terminal_count != last_milestone:
                self._notify_openclaw_deduped(
                    f"milestone:{terminal_count}",
                    f"Foundry milestone: {terminal_count} runs reached terminal state.",
                    cooldown_seconds=120,
                    state=state,
                )
                state["last_milestone_notified"] = terminal_count

            if not active_runs:
                completed = sum(1 for item in terminal_runs if item.status == TrainRunStatus.COMPLETED)
                failed = sum(1 for item in terminal_runs if item.status == TrainRunStatus.FAILED)
                canceled = sum(1 for item in terminal_runs if item.status == TrainRunStatus.CANCELED)
                digest_signature = f"{terminal_count}:{completed}:{failed}:{canceled}"
                if digest_signature != state.get("last_digest_signature"):
                    self._notify_openclaw_deduped(
                        f"digest:{digest_signature}",
                        "Foundry final digest: "
                        f"terminal={terminal_count}, completed={completed}, failed={failed}, canceled={canceled}.",
                        cooldown_seconds=180,
                        state=state,
                    )
                    state["last_digest_signature"] = digest_signature

        self._write_notification_state(state)

    def _notification_state_path(self) -> Path:
        path = self._root / "foundry" / "tracking" / "notification_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _read_notification_state(self) -> dict[str, Any]:
        path = self._notification_state_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_notification_state(self, state: dict[str, Any]) -> None:
        path = self._notification_state_path()
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _notify_openclaw_deduped(
        self,
        key: str,
        text: str,
        *,
        cooldown_seconds: int,
        state: dict[str, Any],
    ) -> None:
        now = datetime.now(UTC).timestamp()
        sent_at = float((state.get("sent", {}) or {}).get(key, 0.0))
        if now - sent_at < max(0, cooldown_seconds):
            return
        self._notify_openclaw(text)
        sent = dict(state.get("sent", {}) or {})
        sent[key] = now
        state["sent"] = sent

    def _notify_openclaw(self, text: str) -> None:
        try:
            subprocess.run(
                ["openclaw", "system", "event", "--text", text, "--mode", "now"],
                cwd=str(self._root),
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception:
            pass

    def _validate_run_spec(self, dataset_version_id: str, run_spec: dict) -> None:
        if run_spec.get("schema") != "foundry.train_run_spec.v1":
            raise ValueError("run_spec.schema must be foundry.train_run_spec.v1")

        classification_mode = run_spec.get("classificationMode")
        if classification_mode not in _SUPPORTED_CLASSIFICATION_MODES:
            raise ValueError("run_spec.classificationMode is unsupported")

        model = run_spec.get("model")
        model_type = "baseline_sgd"
        if model is not None:
            if not isinstance(model, dict):
                raise ValueError("run_spec.model must be an object")
            model_type = str(model.get("type", "baseline_sgd")).lower()
            if model_type not in {"baseline_sgd", "cnn", "crnn"}:
                raise ValueError("run_spec.model.type must be one of: baseline_sgd, cnn, crnn")

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
        trainer_profile = str(training.get("trainerProfile", "baseline_v1")).lower()
        if trainer_profile not in _SUPPORTED_TRAINER_PROFILES:
            raise ValueError("run_spec.training.trainerProfile must be one of: baseline_v1, stronger_v1")
        optimizer = str(training.get("optimizer", "sgd_constant")).lower()
        if model_type == "baseline_sgd":
            if optimizer not in _SUPPORTED_OPTIMIZERS:
                raise ValueError("run_spec.training.optimizer must be one of: sgd_constant, sgd_optimal")
        elif optimizer not in {"sgd_constant", "sgd_optimal", "adam", "adamw"}:
            raise ValueError("run_spec.training.optimizer must be one of: sgd_constant, sgd_optimal, adam, adamw")
        if float(training.get("regularizationAlpha", 0.0001)) <= 0:
            raise ValueError("run_spec.training.regularizationAlpha must be > 0")
        if float(training.get("gradientClipNorm", 1.0)) < 0:
            raise ValueError("run_spec.training.gradientClipNorm must be >= 0")
        if float(training.get("weightDecay", 0.0001)) < 0:
            raise ValueError("run_spec.training.weightDecay must be >= 0")
        early_stopping_patience = training.get("earlyStoppingPatience")
        if early_stopping_patience is not None and int(early_stopping_patience) < 1:
            raise ValueError("run_spec.training.earlyStoppingPatience must be >= 1")
        min_epochs = training.get("minEpochs")
        if min_epochs is not None and int(min_epochs) < 1:
            raise ValueError("run_spec.training.minEpochs must be >= 1")
        if min_epochs is not None and int(min_epochs) > int(training["epochs"]):
            raise ValueError("run_spec.training.minEpochs must be <= epochs")

        class_weighting = str(training.get("classWeighting", "none")).lower()
        if class_weighting not in {"none", "balanced"}:
            raise ValueError("run_spec.training.classWeighting must be one of: none, balanced")
        rebalance_strategy = str(training.get("rebalanceStrategy", "none")).lower()
        if rebalance_strategy not in {"none", "oversample"}:
            raise ValueError("run_spec.training.rebalanceStrategy must be one of: none, oversample")
        if float(training.get("augmentNoiseStd", 0.02)) < 0:
            raise ValueError("run_spec.training.augmentNoiseStd must be >= 0")
        if float(training.get("augmentGainJitter", 0.10)) < 0:
            raise ValueError("run_spec.training.augmentGainJitter must be >= 0")
        if int(training.get("augmentCopies", 1)) < 0:
            raise ValueError("run_spec.training.augmentCopies must be >= 0")

        synthetic_mix = training.get("syntheticMix")
        if synthetic_mix is not None:
            if not isinstance(synthetic_mix, dict):
                raise ValueError("run_spec.training.syntheticMix must be an object")
            unknown_keys = sorted(set(synthetic_mix.keys()) - _SYNTHETIC_MIX_KEYS)
            if unknown_keys:
                raise ValueError(
                    f"run_spec.training.syntheticMix contains unsupported keys: {', '.join(unknown_keys)}"
                )
            enabled = bool(synthetic_mix.get("enabled", False))
            ratio = float(synthetic_mix.get("ratio", 0.0))
            cap = synthetic_mix.get("cap")
            if ratio < 0 or ratio > 1:
                raise ValueError("run_spec.training.syntheticMix.ratio must be between 0 and 1")
            if cap is not None and int(cap) < 0:
                raise ValueError("run_spec.training.syntheticMix.cap must be >= 0")
            if enabled and ratio <= 0 and int(cap or 0) <= 0:
                raise ValueError("enabled syntheticMix requires a positive ratio or cap")

        promotion = run_spec.get("promotion")
        if promotion is not None:
            if not isinstance(promotion, dict):
                raise ValueError("run_spec.promotion must be an object")
            unknown_keys = sorted(set(promotion.keys()) - _PROMOTION_KEYS)
            if unknown_keys:
                raise ValueError(f"run_spec.promotion contains unsupported keys: {', '.join(unknown_keys)}")

            reference_run_id = promotion.get("reference_run_id")
            reference_artifact_id = promotion.get("reference_artifact_id")
            if reference_run_id and reference_artifact_id:
                raise ValueError("run_spec.promotion cannot specify both reference_run_id and reference_artifact_id")

            gate_policy = promotion.get("gate_policy")
            if gate_policy is not None:
                if not isinstance(gate_policy, dict):
                    raise ValueError("run_spec.promotion.gate_policy must be an object")
                unknown_gate_keys = sorted(set(gate_policy.keys()) - _GATE_POLICY_KEYS)
                if unknown_gate_keys:
                    raise ValueError(
                        "run_spec.promotion.gate_policy contains unsupported keys: "
                        + ", ".join(unknown_gate_keys)
                    )
                for key in ("macro_f1_floor", "max_regression_vs_reference", "max_real_vs_synth_gap"):
                    value = gate_policy.get(key)
                    if value is None:
                        continue
                    value = float(value)
                    if value < 0:
                        raise ValueError(f"run_spec.promotion.gate_policy.{key} must be >= 0")
                recall_floors = gate_policy.get("per_class_recall_floors")
                if recall_floors is not None:
                    if not isinstance(recall_floors, dict):
                        raise ValueError(
                            "run_spec.promotion.gate_policy.per_class_recall_floors must be an object"
                        )
                    unknown_labels = sorted(set(recall_floors.keys()) - set(dataset_version.class_map))
                    if unknown_labels:
                        raise ValueError(
                            "run_spec.promotion.gate_policy.per_class_recall_floors contains unknown classes: "
                            + ", ".join(unknown_labels)
                        )
                    for label, floor in recall_floors.items():
                        value = float(floor)
                        if value < 0 or value > 1:
                            raise ValueError(
                                "run_spec.promotion.gate_policy.per_class_recall_floors."
                                f"{label} must be between 0 and 1"
                            )

        if int(data["maxLength"]) < int(data["sampleRate"]) // 10:
            raise ValueError("run_spec.data.maxLength is too small for one-shot training")
        if int(data["hopLength"]) >= int(data["nFft"]):
            raise ValueError("run_spec.data.hopLength must be smaller than nFft")
        if int(data["fmax"]) > int(data["sampleRate"]) // 2:
            raise ValueError("run_spec.data.fmax must not exceed the Nyquist limit")
