from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from echozero.foundry.domain import CompatibilityReport, ModelArtifact
from echozero.foundry.persistence import (
    DatasetVersionRepository,
    ModelArtifactRepository,
    TrainRunRepository,
)


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
        metrics_payload = self._load_json(run_export_dir / "metrics.json")
        run_summary_payload = self._load_json(run_export_dir / "run_summary.json")
        comparison_summary = self._build_comparison_summary(run, metrics_payload)
        promotion_result = self._evaluate_promotion_gate(run, metrics_payload, comparison_summary)
        if metrics_payload:
            metrics_payload["promotionGate"] = promotion_result
            if comparison_summary is not None:
                metrics_payload["referenceComparison"] = comparison_summary
            (run_export_dir / "metrics.json").write_text(
                json.dumps(metrics_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if run_summary_payload:
            run_summary_payload["promotionGate"] = promotion_result
            if comparison_summary is not None:
                run_summary_payload["referenceComparison"] = comparison_summary
            (run_export_dir / "run_summary.json").write_text(
                json.dumps(run_summary_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
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
            "promotionGate": promotion_result,
            **manifest,
        }
        if comparison_summary is not None:
            manifest["referenceComparison"] = comparison_summary
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

    @staticmethod
    def _load_json(path: Path) -> dict:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_comparison_summary(self, run: object, metrics_payload: dict) -> dict | None:
        promotion = run.spec.get("promotion") or {}
        reference_run_id = promotion.get("reference_run_id")
        reference_artifact_id = promotion.get("reference_artifact_id")
        if not reference_run_id and not reference_artifact_id:
            return None

        reference_artifact = None
        if reference_artifact_id:
            reference_artifact = self._artifact_repo.get(str(reference_artifact_id))
            if reference_artifact is None:
                raise ValueError(f"Reference artifact not found: {reference_artifact_id}")
        else:
            artifacts = self._artifact_repo.list_for_run(str(reference_run_id))
            if not artifacts:
                raise ValueError(f"Reference run has no artifacts: {reference_run_id}")
            artifacts.sort(key=lambda artifact: artifact.created_at)
            reference_artifact = artifacts[-1]

        reference_metrics = self._load_reference_metrics(reference_artifact)
        current_eval = metrics_payload.get("finalEval", {})
        current_metrics = current_eval.get("metrics", {})
        current_per_class = current_eval.get("per_class_metrics", {})
        reference_eval = reference_metrics.get("finalEval", {})
        reference_eval_metrics = reference_eval.get("metrics", {})
        reference_per_class = reference_eval.get("per_class_metrics", {})
        per_class_recall_delta = {
            label: float(current_per_class.get(label, {}).get("recall", 0.0))
            - float(reference_per_class.get(label, {}).get("recall", 0.0))
            for label in sorted(set(current_per_class) | set(reference_per_class))
        }
        return {
            "referenceRunId": reference_artifact.run_id,
            "referenceArtifactId": reference_artifact.id,
            "current": {
                "macroF1": float(current_metrics.get("macro_f1", 0.0)),
                "accuracy": float(current_metrics.get("accuracy", 0.0)),
            },
            "reference": {
                "macroF1": float(reference_eval_metrics.get("macro_f1", 0.0)),
                "accuracy": float(reference_eval_metrics.get("accuracy", 0.0)),
            },
            "delta": {
                "macroF1": float(current_metrics.get("macro_f1", 0.0))
                - float(reference_eval_metrics.get("macro_f1", 0.0)),
                "accuracy": float(current_metrics.get("accuracy", 0.0))
                - float(reference_eval_metrics.get("accuracy", 0.0)),
                "perClassRecall": per_class_recall_delta,
            },
        }

    def _load_reference_metrics(self, artifact: ModelArtifact) -> dict:
        metrics_path_name = (artifact.manifest.get("trainingSummary") or {}).get("metricsPath", "metrics.json")
        metrics_path = artifact.path.parent / str(metrics_path_name)
        metrics_payload = self._load_json(metrics_path)
        if metrics_payload:
            return metrics_payload

        eval_summary = artifact.manifest.get("evalSummary") or {}
        return {
            "finalEval": {
                "metrics": {
                    "macro_f1": float(eval_summary.get("macroF1", 0.0)),
                    "accuracy": float(eval_summary.get("accuracy", 0.0)),
                },
                "per_class_metrics": {},
            }
        }

    def _evaluate_promotion_gate(
        self,
        run: object,
        metrics_payload: dict,
        comparison_summary: dict | None,
    ) -> dict[str, object]:
        promotion = run.spec.get("promotion") or {}
        gate_policy = promotion.get("gate_policy") or {}
        reasons: list[str] = []
        final_eval = metrics_payload.get("finalEval", {})
        final_metrics = final_eval.get("metrics", {})
        per_class_metrics = final_eval.get("per_class_metrics", {})
        synthetic_eval = metrics_payload.get("syntheticEval") or {}
        trainer_options = metrics_payload.get("trainerOptions") or {}
        synthetic_mix = trainer_options.get("syntheticMix") or {}

        macro_f1_floor = gate_policy.get("macro_f1_floor")
        if macro_f1_floor is not None and float(final_metrics.get("macro_f1", 0.0)) < float(macro_f1_floor):
            reasons.append(
                f"macro_f1 {float(final_metrics.get('macro_f1', 0.0)):.4f} below floor {float(macro_f1_floor):.4f}"
            )

        max_regression = gate_policy.get("max_regression_vs_reference")
        if max_regression is not None:
            if comparison_summary is None:
                reasons.append("reference comparison required for max_regression_vs_reference gate")
            else:
                regression = -float((comparison_summary.get("delta") or {}).get("macroF1", 0.0))
                if regression > float(max_regression):
                    reasons.append(
                        f"macro_f1 regression {regression:.4f} exceeds max {float(max_regression):.4f}"
                    )

        max_gap = gate_policy.get("max_real_vs_synth_gap")
        if max_gap is not None and bool(synthetic_mix.get("enabled")):
            synthetic_metrics = synthetic_eval.get("metrics") or {}
            if not synthetic_metrics:
                reasons.append("synthetic evaluation required for max_real_vs_synth_gap gate")
            else:
                gap = abs(float(final_metrics.get("macro_f1", 0.0)) - float(synthetic_metrics.get("macro_f1", 0.0)))
                if gap > float(max_gap):
                    reasons.append(f"real_vs_synth macro_f1 gap {gap:.4f} exceeds max {float(max_gap):.4f}")

        per_class_recall_floors = gate_policy.get("per_class_recall_floors") or {}
        for label, floor in sorted(per_class_recall_floors.items()):
            recall = float((per_class_metrics.get(label) or {}).get("recall", 0.0))
            if recall < float(floor):
                reasons.append(f"{label} recall {recall:.4f} below floor {float(floor):.4f}")

        return {
            "enabled": bool(gate_policy),
            "passed": not reasons,
            "reasons": reasons,
            "policy": gate_policy,
        }

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
