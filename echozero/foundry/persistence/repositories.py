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


_STATE_VERSION = 1
_STATE_ITEMS_KEY = "items"


class StateFormatError(RuntimeError):
    """Raised when foundry state files are corrupt/unsupported/unmigrated."""


def _state_envelope(schema: str, items: dict) -> dict:
    return {
        "schema": schema,
        "version": _STATE_VERSION,
        _STATE_ITEMS_KEY: items,
    }


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_state(path: Path, expected_schema: str) -> dict:
    """Read state envelope with strict version/schema checks.

    Strict mode rules:
    - Missing file / empty object => empty state
    - Proper envelope required for non-empty state
    - Legacy flat maps are rejected until explicit migrate-state is run
    - Unknown schema/version are hard failures
    """

    payload = _read_json(path)
    if not payload:
        return {}

    if not isinstance(payload, dict):
        raise StateFormatError(f"Invalid state format in {path}: expected JSON object")

    has_envelope_keys = any(k in payload for k in ("schema", "version", _STATE_ITEMS_KEY))
    if not has_envelope_keys:
        raise StateFormatError(
            f"Legacy state detected in {path}. Run `ez-foundry --root <path> migrate-state` before continuing."
        )

    schema = payload.get("schema")
    version = payload.get("version")
    items = payload.get(_STATE_ITEMS_KEY)

    if schema != expected_schema:
        raise StateFormatError(
            f"Unsupported schema in {path}: expected {expected_schema}, got {schema!r}"
        )
    if version != _STATE_VERSION:
        raise StateFormatError(
            f"Unsupported state version in {path}: expected {_STATE_VERSION}, got {version!r}"
        )
    if not isinstance(items, dict):
        raise StateFormatError(f"Invalid state payload in {path}: items must be an object")

    return items


def _write_state(path: Path, schema: str, items: dict) -> None:
    _write_json(path, _state_envelope(schema=schema, items=items))


def migrate_state_file(path: Path, schema: str) -> bool:
    """Explicitly migrate a legacy flat state file into v1 envelope format.

    Returns True when a migration write occurred, False when no change was needed.
    """

    payload = _read_json(path)
    if not payload:
        _write_state(path, schema, {})
        return True

    if not isinstance(payload, dict):
        raise StateFormatError(f"Invalid state format in {path}: expected JSON object")

    has_envelope_keys = any(k in payload for k in ("schema", "version", _STATE_ITEMS_KEY))
    if has_envelope_keys:
        # Validate known-good envelope and no-op.
        _read_state(path, schema)
        return False

    # Legacy flat map -> wrap.
    _write_state(path, schema, payload)
    return True


def migrate_foundry_state(root: Path) -> dict[str, bool]:
    state_dir = root / "foundry" / "state"
    mappings = {
        "train_runs.json": "foundry.state.train_runs.v1",
        "datasets.json": "foundry.state.datasets.v1",
        "dataset_versions.json": "foundry.state.dataset_versions.v1",
        "eval_reports.json": "foundry.state.eval_reports.v1",
        "artifacts.json": "foundry.state.artifacts.v1",
    }
    results: dict[str, bool] = {}
    for filename, schema in mappings.items():
        path = state_dir / filename
        results[filename] = migrate_state_file(path, schema)
    return results


class TrainRunRepository:
    def __init__(self, root: Path):
        self._path = root / "foundry" / "state" / "train_runs.json"
        self._schema = "foundry.state.train_runs.v1"

    def save(self, run: TrainRun) -> TrainRun:
        rows = _read_state(self._path, self._schema)
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
        _write_state(self._path, self._schema, rows)
        return run

    def get(self, run_id: str) -> TrainRun | None:
        row = _read_state(self._path, self._schema).get(run_id)
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
        for run_id in _read_state(self._path, self._schema).keys():
            run = self.get(run_id)
            if run is not None:
                runs.append(run)
        return runs


class DatasetRepository:
    def __init__(self, root: Path):
        self._path = root / "foundry" / "state" / "datasets.json"
        self._schema = "foundry.state.datasets.v1"

    def save(self, dataset: Dataset) -> Dataset:
        rows = _read_state(self._path, self._schema)
        rows[dataset.id] = {
            "id": dataset.id,
            "name": dataset.name,
            "source_kind": dataset.source_kind,
            "source_ref": dataset.source_ref,
            "metadata": dataset.metadata,
            "created_at": dataset.created_at.isoformat(),
        }
        _write_state(self._path, self._schema, rows)
        return dataset

    def get(self, dataset_id: str) -> Dataset | None:
        row = _read_state(self._path, self._schema).get(dataset_id)
        if not row:
            return None
        return Dataset(
            id=row["id"],
            name=row["name"],
            source_kind=row["source_kind"],
            source_ref=row.get("source_ref"),
            metadata=row.get("metadata", {}),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def list(self) -> list[Dataset]:
        items: list[Dataset] = []
        for dataset_id in _read_state(self._path, self._schema).keys():
            item = self.get(dataset_id)
            if item is not None:
                items.append(item)
        return items


class DatasetVersionRepository:
    def __init__(self, root: Path):
        self._path = root / "foundry" / "state" / "dataset_versions.json"
        self._schema = "foundry.state.dataset_versions.v1"

    def save(self, version: DatasetVersion) -> DatasetVersion:
        rows = _read_state(self._path, self._schema)
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
                    "content_hash": s.content_hash,
                    "source_provenance": s.source_provenance,
                    "is_synthetic": s.is_synthetic,
                    "synthetic_provenance": s.synthetic_provenance,
                    "quality_flags": s.quality_flags,
                    "split_assignment": s.split_assignment,
                    "curation_state": s.curation_state.value,
                }
                for s in version.samples
            ],
            "taxonomy": version.taxonomy,
            "label_policy": version.label_policy,
            "manifest": version.manifest,
            "split_plan": version.split_plan,
            "balance_plan": version.balance_plan,
            "stats": version.stats,
            "lineage": version.lineage,
            "created_at": version.created_at.isoformat(),
        }
        _write_state(self._path, self._schema, rows)
        return version

    def get(self, version_id: str) -> DatasetVersion | None:
        row = _read_state(self._path, self._schema).get(version_id)
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
                    content_hash=s.get("content_hash", ""),
                    source_provenance=s.get("source_provenance", {}),
                    is_synthetic=bool(s.get("is_synthetic", False)),
                    synthetic_provenance=s.get("synthetic_provenance", {}),
                    quality_flags=list(s.get("quality_flags", [])),
                    split_assignment=s.get("split_assignment"),
                    curation_state=CurationState(s.get("curation_state", CurationState.UNKNOWN.value)),
                )
                for s in row.get("samples", [])
            ],
            taxonomy=row.get("taxonomy", {}),
            label_policy=row.get("label_policy", {}),
            manifest=row.get("manifest", {}),
            split_plan=row.get("split_plan", {}),
            balance_plan=row.get("balance_plan", {}),
            stats=row.get("stats", {}),
            lineage=row.get("lineage", {}),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def list_for_dataset(self, dataset_id: str) -> list[DatasetVersion]:
        results: list[DatasetVersion] = []
        rows = _read_state(self._path, self._schema)
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
        self._schema = "foundry.state.eval_reports.v1"

    def save(self, report: EvalReport) -> EvalReport:
        rows = _read_state(self._path, self._schema)
        rows[report.id] = {
            "id": report.id,
            "run_id": report.run_id,
            "classification_mode": report.classification_mode,
            "metrics": report.metrics,
            "dataset_version_id": report.dataset_version_id,
            "split_name": report.split_name,
            "aggregate_metrics": report.aggregate_metrics,
            "per_class_metrics": report.per_class_metrics,
            "baseline": report.baseline,
            "threshold_policy": report.threshold_policy,
            "confusion": report.confusion,
            "summary": report.summary,
            "created_at": report.created_at.isoformat(),
        }
        _write_state(self._path, self._schema, rows)
        return report

    def get(self, report_id: str) -> EvalReport | None:
        row = _read_state(self._path, self._schema).get(report_id)
        if not row:
            return None
        return EvalReport(
            id=row["id"],
            run_id=row["run_id"],
            classification_mode=row["classification_mode"],
            metrics=row.get("metrics", {}),
            dataset_version_id=row.get("dataset_version_id"),
            split_name=row.get("split_name", "test"),
            aggregate_metrics=row.get("aggregate_metrics", {}),
            per_class_metrics=row.get("per_class_metrics", {}),
            baseline=row.get("baseline", {}),
            threshold_policy=row.get("threshold_policy"),
            confusion=row.get("confusion"),
            summary=row.get("summary", {}),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def list_for_run(self, run_id: str) -> list[EvalReport]:
        reports: list[EvalReport] = []
        for report_id, row in _read_state(self._path, self._schema).items():
            if row.get("run_id") != run_id:
                continue
            report = self.get(report_id)
            if report is not None:
                reports.append(report)
        return reports


class ModelArtifactRepository:
    def __init__(self, root: Path):
        self._path = root / "foundry" / "state" / "artifacts.json"
        self._schema = "foundry.state.artifacts.v1"

    def save(self, artifact: ModelArtifact) -> ModelArtifact:
        rows = _read_state(self._path, self._schema)
        rows[artifact.id] = {
            "id": artifact.id,
            "run_id": artifact.run_id,
            "artifact_version": artifact.artifact_version,
            "path": str(artifact.path),
            "sha256": artifact.sha256,
            "manifest": artifact.manifest,
            "consumer_hints": artifact.consumer_hints,
            "created_at": artifact.created_at.isoformat(),
        }
        _write_state(self._path, self._schema, rows)
        return artifact

    def get(self, artifact_id: str) -> ModelArtifact | None:
        row = _read_state(self._path, self._schema).get(artifact_id)
        if not row:
            return None
        return ModelArtifact(
            id=row["id"],
            run_id=row["run_id"],
            artifact_version=row["artifact_version"],
            path=Path(row["path"]),
            sha256=row["sha256"],
            manifest=row["manifest"],
            consumer_hints=row.get("consumer_hints", {}),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def list(self) -> list[ModelArtifact]:
        artifacts: list[ModelArtifact] = []
        for artifact_id in _read_state(self._path, self._schema).keys():
            artifact = self.get(artifact_id)
            if artifact is not None:
                artifacts.append(artifact)
        return artifacts

    def list_for_run(self, run_id: str) -> list[ModelArtifact]:
        artifacts: list[ModelArtifact] = []
        for artifact_id, row in _read_state(self._path, self._schema).items():
            if row.get("run_id") != run_id:
                continue
            artifact = self.get(artifact_id)
            if artifact is not None:
                artifacts.append(artifact)
        return artifacts
