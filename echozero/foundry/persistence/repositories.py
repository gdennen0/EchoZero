from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from echozero.foundry.domain import ModelArtifact, TrainRun, TrainRunStatus


class TrainRunRepository:
    def __init__(self, root: Path):
        self._root = root
        self._path = root / "foundry" / "state" / "train_runs.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, run: TrainRun) -> TrainRun:
        rows = self._read_all()
        rows[run.id] = {
            "id": run.id,
            "dataset_version_id": run.dataset_version_id,
            "status": run.status.value,
            "spec": run.spec,
            "spec_hash": run.spec_hash,
            "backend": run.backend,
            "device": run.device,
            "created_at": run.created_at.isoformat(),
            "updated_at": run.updated_at.isoformat(),
        }
        self._write_all(rows)
        return run

    def get(self, run_id: str) -> TrainRun | None:
        row = self._read_all().get(run_id)
        if not row:
            return None
        return TrainRun(
            id=row["id"],
            dataset_version_id=row["dataset_version_id"],
            status=TrainRunStatus(row["status"]),
            spec=row["spec"],
            spec_hash=row["spec_hash"],
            backend=row.get("backend", "pytorch"),
            device=row.get("device", "cpu"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def list(self) -> list[TrainRun]:
        return [self.get(run_id) for run_id in self._read_all().keys() if self.get(run_id)]

    def _read_all(self) -> dict:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write_all(self, rows: dict) -> None:
        self._path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


class ModelArtifactRepository:
    def __init__(self, root: Path):
        self._root = root
        self._path = root / "foundry" / "state" / "artifacts.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, artifact: ModelArtifact) -> ModelArtifact:
        rows = self._read_all()
        rows[artifact.id] = {
            "id": artifact.id,
            "run_id": artifact.run_id,
            "artifact_version": artifact.artifact_version,
            "path": str(artifact.path),
            "sha256": artifact.sha256,
            "manifest": artifact.manifest,
            "created_at": artifact.created_at.isoformat(),
        }
        self._write_all(rows)
        return artifact

    def get(self, artifact_id: str) -> ModelArtifact | None:
        row = self._read_all().get(artifact_id)
        if not row:
            return None
        return ModelArtifact(
            id=row["id"],
            run_id=row["run_id"],
            artifact_version=row["artifact_version"],
            path=Path(row["path"]),
            sha256=row["sha256"],
            manifest=row["manifest"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _read_all(self) -> dict:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write_all(self, rows: dict) -> None:
        self._path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
