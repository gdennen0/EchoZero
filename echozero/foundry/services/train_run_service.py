from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from echozero.foundry.domain import TrainRun, TrainRunStatus
from echozero.foundry.persistence import TrainRunRepository


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
    def __init__(self, root: Path, repository: TrainRunRepository | None = None):
        self._root = root
        self._repo = repository or TrainRunRepository(root)

    def create_run(self, dataset_version_id: str, run_spec: dict, backend: str = "pytorch", device: str = "cpu") -> TrainRun:
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
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "exports").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "spec.json").write_text(json.dumps(run_spec, indent=2), encoding="utf-8")
        self._append_event(run, "RUN_CREATED", {"status": run.status.value})
        return self._repo.save(run)

    def start_run(self, run_id: str) -> TrainRun:
        return self._transition(run_id, TrainRunStatus.RUNNING, "RUN_STARTED")

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
        ckpt_path = run.run_dir(self._root) / "checkpoints" / f"epoch_{epoch:04d}.json"
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
        path = run.run_dir(self._root) / "events.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")
