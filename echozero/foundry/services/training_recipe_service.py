"""Training recipe presets for the library-first Foundry flow.
Exists to expose simple operator choices instead of raw run-spec knob soup.
Connects named recipes to concrete Foundry training specs.
"""

from __future__ import annotations

from typing import Any

from echozero.foundry.domain import TrainingRecipeName


class TrainingRecipeService:
    """Build concrete run specs from a small set of named recipes."""

    def build_run_spec(
        self,
        dataset_version_id: str,
        *,
        recipe_name: TrainingRecipeName,
        sample_rate: int = 22050,
    ) -> dict[str, Any]:
        """Return one Foundry run spec for the requested recipe."""
        training: dict[str, Any] = {
            "seed": 42,
            "batchSize": 4,
            "learningRate": 0.01,
            "trainerProfile": "baseline_v1",
            "optimizer": "sgd_constant",
            "classWeighting": "balanced",
            "rebalanceStrategy": "oversample",
            "augmentTrain": True,
            "augmentNoiseStd": 0.03,
            "augmentGainJitter": 0.15,
            "augmentCopies": 2,
        }
        if recipe_name is TrainingRecipeName.QUICK:
            training.update({"epochs": 2})
        elif recipe_name is TrainingRecipeName.BALANCED:
            training.update({"epochs": 4})
        else:
            training.update(
                {
                    "epochs": 8,
                    "trainerProfile": "stronger_v1",
                    "optimizer": "sgd_optimal",
                    "averageWeights": True,
                    "earlyStoppingPatience": 3,
                    "minEpochs": 3,
                }
            )
        return {
            "schema": "foundry.train_run_spec.v1",
            "classificationMode": "multiclass",
            "data": {
                "datasetVersionId": dataset_version_id,
                "sampleRate": sample_rate,
                "maxLength": sample_rate,
                "nFft": 2048,
                "hopLength": 512,
                "nMels": 128,
                "fmax": 8000,
            },
            "training": training,
        }
