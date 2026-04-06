from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from echozero.foundry.domain import CompatibilityReport, ModelArtifact
from echozero.foundry.persistence import DatasetVersionRepository, ModelArtifactRepository, TrainRunRepository


_REQUIRED_PREPROCESSING_KEYS = {
    "sampleRate",
    "maxLength",
    "nFft",
    "hopLength",
    "nMels",
    "fmax",
}


class ArtifactService:
    def __init__(
        self,
        root: Path,
        train_run_repository: TrainRunRepository | None = None,
        dataset_version_repository: DatasetVersionRepository | None = None,
        artifact_repository: ModelArtifactRepository | None = None,
    ):
        self._root = root
        self._run_repo = train_run_repository or TrainRunRepository(root)
        self._dataset_version_repo = dataset_version_repository or DatasetVersionRepository(root)
        self._artifact_repo = artifact_repository or ModelArtifactRepository(root)

    def finalize_artifact(self, run_id: str, manifest: dict) -> ModelArtifact:
        run = self._run_repo.get(run_id)
        if not run:
            raise ValueError(f"TrainRun not found: {run_id}")
        if run.status.value not in {"exporting", "completed"}:
            raise ValueError("Artifacts can only be finalized from exporting or completed runs")

        dataset_version = self._dataset_version_repo.get(run.dataset_version_id)
        if dataset_version is None:
            raise ValueError(f"DatasetVersion not found: {run.dataset_version_id}")

        artifact_id = f"art_{uuid4().hex[:12]}"
        run_export_dir = run.run_dir(self._root) / "exports"
        run_export_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_export_dir / f"{artifact_id}.manifest.json"
        manifest = {
            "schema": "foundry.artifact_manifest.v1",
            "artifactId": artifact_id,
            "runId": run_id,
            "createdAt": datetime.now(UTC).isoformat(),
            "datasetVersionId": dataset_version.id,
            "specHash": run.spec_hash,
            "taxonomy": dataset_version.taxonomy,
            "labelPolicy": dataset_version.label_policy,
            "syntheticProvenance": {
                "syntheticSampleIds": list(dataset_version.manifest.get("synthetic_sample_ids", [])),
                "realSampleIds": list(dataset_version.manifest.get("real_sample_ids", [])),
                "syntheticSampleCount": int(dataset_version.stats.get("synthetic_sample_count", 0)),
                "realSampleCount": int(dataset_version.stats.get("real_sample_count", 0)),
            },
            "runtime": {
                "consumer": "PyTorchAudioClassify",
                "backend": run.backend,
                "device": run.device,
            },
            **manifest,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        artifact = ModelArtifact(
            id=artifact_id,
            run_id=run_id,
            artifact_version="v1",
            path=manifest_path,
            sha256=digest,
            manifest=manifest,
            consumer_hints={"consumer": "PyTorchAudioClassify"},
        )
        return self._artifact_repo.save(artifact)

    def validate_compatibility(self, artifact_id: str, consumer: str = "PyTorchAudioClassify") -> CompatibilityReport:
        artifact = self._artifact_repo.get(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact not found: {artifact_id}")

        errors: list[str] = []
        warnings: list[str] = []
        m = artifact.manifest
        run = self._run_repo.get(artifact.run_id)
        dataset_version = None
        run_data = {}
        if run is None:
            errors.append(f"originating run not found: {artifact.run_id}")
        else:
            run_data = run.spec.get("data", {})
            dataset_version = self._dataset_version_repo.get(run.dataset_version_id)
            if dataset_version is None:
                errors.append(f"dataset version not found: {run.dataset_version_id}")

        if m.get("schema") != "foundry.artifact_manifest.v1":
            errors.append("manifest.schema must be foundry.artifact_manifest.v1")

        classes = m.get("classes")
        if not isinstance(classes, list) or not classes:
            errors.append("manifest.classes must be a non-empty list")

        preprocessing = m.get("inferencePreprocessing") or {}
        missing = sorted(_REQUIRED_PREPROCESSING_KEYS - set(preprocessing.keys()))
        if missing:
            errors.append(f"manifest.inferencePreprocessing missing keys: {', '.join(missing)}")

        if run is not None:
            if m.get("runId") != run.id:
                errors.append("manifest.runId must match the originating run id")
            if m.get("datasetVersionId") != run.dataset_version_id:
                errors.append("manifest.datasetVersionId must match the originating dataset version id")
            if m.get("specHash") != run.spec_hash:
                errors.append("manifest.specHash must match the originating run spec hash")

            if m.get("classificationMode") != run.spec.get("classificationMode"):
                errors.append("manifest.classificationMode must match run spec classificationMode")

            for key in sorted(_REQUIRED_PREPROCESSING_KEYS):
                expected = run_data.get(key)
                actual = preprocessing.get(key)
                if expected is None or actual is None:
                    continue
                if actual != expected:
                    errors.append(f"manifest.inferencePreprocessing.{key} must match run spec")

        if dataset_version is not None and classes:
            expected_classes = list(dataset_version.class_map)
            if classes != expected_classes:
                errors.append("manifest.classes must match dataset version class_map order")
            if m.get("taxonomy") != dataset_version.taxonomy:
                errors.append("manifest.taxonomy must match dataset version taxonomy")
            if m.get("labelPolicy") != dataset_version.label_policy:
                errors.append("manifest.labelPolicy must match dataset version label_policy")

        if consumer == "PyTorchAudioClassify":
            weights_path = m.get("weightsPath")
            if not weights_path:
                errors.append("manifest.weightsPath is required for PyTorchAudioClassify")
            elif not str(weights_path).endswith(".pth"):
                errors.append("manifest.weightsPath must point to a .pth file for PyTorchAudioClassify")
            if Path(str(weights_path)).is_absolute():
                errors.append("manifest.weightsPath must be relative for portable runtime use")
            runtime = m.get("runtime") or {}
            if runtime.get("consumer") != consumer:
                errors.append("manifest.runtime.consumer must match the validated consumer")

        if m.get("classificationMode") == "binary" and "thresholdPolicy" not in m:
            warnings.append("binary classifier missing thresholdPolicy")

        return CompatibilityReport(
            artifact_id=artifact.id,
            consumer=consumer,
            ok=not errors,
            errors=errors,
            warnings=warnings,
        )
