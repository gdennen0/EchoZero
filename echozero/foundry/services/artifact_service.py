from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from echozero.foundry.domain import CompatibilityReport, ModelArtifact
from echozero.foundry.persistence import ModelArtifactRepository, TrainRunRepository


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
        artifact_repository: ModelArtifactRepository | None = None,
    ):
        self._root = root
        self._run_repo = train_run_repository or TrainRunRepository(root)
        self._artifact_repo = artifact_repository or ModelArtifactRepository(root)

    def finalize_artifact(self, run_id: str, manifest: dict) -> ModelArtifact:
        run = self._run_repo.get(run_id)
        if not run:
            raise ValueError(f"TrainRun not found: {run_id}")

        artifact_id = f"art_{uuid4().hex[:12]}"
        run_export_dir = run.run_dir(self._root) / "exports"
        run_export_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_export_dir / f"{artifact_id}.manifest.json"
        manifest = {
            "schema": "foundry.artifact_manifest.v1",
            "artifactId": artifact_id,
            "runId": run_id,
            "createdAt": datetime.now(UTC).isoformat(),
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
        )
        return self._artifact_repo.save(artifact)

    def validate_compatibility(self, artifact_id: str, consumer: str = "PyTorchAudioClassify") -> CompatibilityReport:
        artifact = self._artifact_repo.get(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact not found: {artifact_id}")

        errors: list[str] = []
        warnings: list[str] = []
        m = artifact.manifest

        if m.get("schema") != "foundry.artifact_manifest.v1":
            errors.append("manifest.schema must be foundry.artifact_manifest.v1")

        classes = m.get("classes")
        if not isinstance(classes, list) or not classes:
            errors.append("manifest.classes must be a non-empty list")

        preprocessing = m.get("inferencePreprocessing") or {}
        missing = sorted(_REQUIRED_PREPROCESSING_KEYS - set(preprocessing.keys()))
        if missing:
            errors.append(f"manifest.inferencePreprocessing missing keys: {', '.join(missing)}")

        if consumer == "PyTorchAudioClassify" and not m.get("weightsPath"):
            errors.append("manifest.weightsPath is required for PyTorchAudioClassify")

        if m.get("classificationMode") == "binary" and "thresholdPolicy" not in m:
            warnings.append("binary classifier missing thresholdPolicy")

        return CompatibilityReport(
            artifact_id=artifact.id,
            consumer=consumer,
            ok=not errors,
            errors=errors,
            warnings=warnings,
        )
