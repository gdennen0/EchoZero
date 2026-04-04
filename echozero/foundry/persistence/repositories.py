from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from echozero.foundry.domain import (
    CurationState,
    Dataset,
    DatasetSample,
    DatasetVersion,
    EvalReport,
    ModelArtifact,
    TrainRun,
    TrainRunStatus,
)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class TrainRunRepository:
    def __init__(self, root: Path):
        self._path = root / "foundry" / "state" / "train_runs.json"

    def save(self, run: TrainRun) -> TrainRun:
        rows = _read_json(self._path)
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
        _write_json(self._path, rows)
        return run

    def get(self, run_id: str) -> TrainRun | None:
        row = _read_json(self._path).get(run_id)
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
        runs: list[TrainRun] = []
        for run_id in _read_json(self._path).keys():
            run = self.get(run_id)
            if run is not None:
                runs.append(run)
        return runs


class DatasetRepository:
    def __init__(self, root: Path):
        self._path = root / "foundry" / "state" / "datasets.json"

    def save(self, dataset: Dataset) -> Dataset:
        rows = _read_json(self._path)
        rows[dataset.id] = {
            "id": dataset.id,
            "name": dataset.name,
            "source_kind": dataset.source_kind,
            "created_at": dataset.created_at.isoformat(),
        }
        _write_json(self._path, rows)
        return dataset

    def get(self, dataset_id: str) -> Dataset | None:
        row = _read_json(self._path).get(dataset_id)
        if not row:
            return None
        return Dataset(
            id=row["id"],
            name=row["name"],
            source_kind=row["source_kind"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def list(self) -> list[Dataset]:
        items: list[Dataset] = []
        for dataset_id in _read_json(self._path).keys():
            item = self.get(dataset_id)
            if item is not None:
                items.append(item)
        return items


class DatasetVersionRepository:
    def __init__(self, root: Path):
        self._path = root / "foundry" / "state" / "dataset_versions.json"

    def save(self, version: DatasetVersion) -> DatasetVersion:
        rows = _read_json(self._path)
        rows[version.id] = {
            "id": version.id,
            "dataset_id": version.dataset_id,
            "version": version.version,
            "manifest_hash": version.manifest_hash,
            "sample_rate": version.sample_rate,
            "audio_standard": version.audio_standard,
            "class_map": version.class_map,
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "audio_ref": s.audio_ref,
                    "label": s.label,
                    "duration_ms": s.duration_ms,
                    "source_provenance": s.source_provenance,
                    "quality_flags": s.quality_flags,
                    "curation_state": s.curation_state.value,
                }
                for s in version.samples
            ],
            "split_plan": version.split_plan,
            "balance_plan": version.balance_plan,
            "stats": version.stats,
            "created_at": version.created_at.isoformat(),
        }
        _write_json(self._path, rows)
        return version

    def get(self, version_id: str) -> DatasetVersion | None:
        row = _read_json(self._path).get(version_id)
        if not row:
            return None
        return DatasetVersion(
            id=row["id"],
            dataset_id=row["dataset_id"],
            version=int(row["version"]),
            manifest_hash=row["manifest_hash"],
            sample_rate=int(row["sample_rate"]),
            audio_standard=row["audio_standard"],
            class_map=list(row.get("class_map", [])),
            samples=[
                DatasetSample(
                    sample_id=s["sample_id"],
                    audio_ref=s["audio_ref"],
                    label=s["label"],
                    duration_ms=s.get("duration_ms"),
                    source_provenance=s.get("source_provenance", {}),
                    quality_flags=list(s.get("quality_flags", [])),
                    curation_state=CurationState(s.get("curation_state", CurationState.UNKNOWN.value)),
                )
                for s in row.get("samples", [])
            ],
            split_plan=row.get("split_plan", {}),
            balance_plan=row.get("balance_plan", {}),
            stats=row.get("stats", {}),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def list_for_dataset(self, dataset_id: str) -> list[DatasetVersion]:
        results: list[DatasetVersion] = []
        rows = _read_json(self._path)
        for version_id, row in rows.items():
            if row.get("dataset_id") != dataset_id:
                continue
            item = self.get(version_id)
            if item is not None:
                results.append(item)
        return sorted(results, key=lambda v: v.version)


class EvalReportRepository:
    def __init__(self, root: Path):
        self._path = root / "foundry" / "state" / "eval_reports.json"

    def save(self, report: EvalReport) -> EvalReport:
        rows = _read_json(self._path)
        rows[report.id] = {
            "id": report.id,
            "run_id": report.run_id,
            "classification_mode": report.classification_mode,
            "metrics": report.metrics,
            "threshold_policy": report.threshold_policy,
            "confusion": report.confusion,
            "created_at": report.created_at.isoformat(),
        }
        _write_json(self._path, rows)
        return report

    def get(self, report_id: str) -> EvalReport | None:
        row = _read_json(self._path).get(report_id)
        if not row:
            return None
        return EvalReport(
            id=row["id"],
            run_id=row["run_id"],
            classification_mode=row["classification_mode"],
            metrics=row.get("metrics", {}),
            threshold_policy=row.get("threshold_policy"),
            confusion=row.get("confusion"),
            created_at=datetime.fromisoformat(row["created_at"]),
        )


class ModelArtifactRepository:
    def __init__(self, root: Path):
        self._path = root / "foundry" / "state" / "artifacts.json"

    def save(self, artifact: ModelArtifact) -> ModelArtifact:
        rows = _read_json(self._path)
        rows[artifact.id] = {
            "id": artifact.id,
            "run_id": artifact.run_id,
            "artifact_version": artifact.artifact_version,
            "path": str(artifact.path),
            "sha256": artifact.sha256,
            "manifest": artifact.manifest,
            "created_at": artifact.created_at.isoformat(),
        }
        _write_json(self._path, rows)
        return artifact

    def get(self, artifact_id: str) -> ModelArtifact | None:
        row = _read_json(self._path).get(artifact_id)
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
