from __future__ import annotations

from echozero.foundry.persistence import DatasetVersionRepository
from echozero.foundry.services.dataset_service import DatasetService


_REQUIRED_DATA_KEYS = {"datasetVersionId", "sampleRate", "maxLength", "nFft", "hopLength", "nMels", "fmax"}
_REQUIRED_TRAINING_KEYS = {"epochs", "batchSize", "learningRate", "seed"}
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


class RunSpecValidator:
    def __init__(self, dataset_version_repository: DatasetVersionRepository):
        self._dataset_versions = dataset_version_repository

    def validate(self, dataset_version_id: str, run_spec: dict) -> None:
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

        integrity = DatasetService.validate_version_integrity(dataset_version)
        if not integrity["ok"]:
            raise ValueError(f"dataset version integrity check failed: {integrity['errors'][0]}")

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
        if split_plan.get("leakage", {}).get("duplicate_groups_across_splits"):
            raise ValueError("dataset version split plan has cross-group contamination across splits")

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
        if int(training["seed"]) < 0:
            raise ValueError("run_spec.training.seed must be >= 0")
        if "deterministic" in training and not isinstance(training.get("deterministic"), bool):
            raise ValueError("run_spec.training.deterministic must be a boolean when provided")
        if "backend" in training and training.get("backend") is not None:
            backend_name = str(training.get("backend")).strip()
            if not backend_name:
                raise ValueError("run_spec.training.backend must be a non-empty string when provided")
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
