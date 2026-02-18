"""
Strict-validation tests for TrainingConfig.
"""
import pytest

from src.application.blocks.training.config import TrainingConfig


def test_from_block_metadata_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown training configuration keys"):
        TrainingConfig.from_block_metadata(
            {
                "epochs": 10,
                "unknown_knob": 123,
            }
        )


def test_from_block_metadata_allows_last_training_metadata_key():
    cfg = TrainingConfig.from_block_metadata(
        {
            "epochs": 5,
            "last_training": {"timestamp": "2026-01-01T00:00:00"},
        }
    )
    assert cfg.epochs == 5


def test_from_block_metadata_rejects_wrong_field_type():
    with pytest.raises(ValueError, match="Invalid value for 'epochs'"):
        TrainingConfig.from_block_metadata(
            {
                "epochs": "100",
            }
        )


def test_validate_rejects_cross_validation_for_unimplemented_runtime():
    with pytest.raises(ValueError, match="cross-validation is not implemented"):
        TrainingConfig(
            classification_mode="multiclass",
            use_cross_validation=True,
        )


def test_validate_rejects_hyperopt_for_unimplemented_runtime():
    with pytest.raises(ValueError, match="hyperparameter optimization is not implemented"):
        TrainingConfig(
            classification_mode="multiclass",
            use_hyperopt=True,
        )


def test_validate_rejects_positive_filter_outside_binary_mode():
    with pytest.raises(ValueError, match="positive_filter_type can only be used"):
        TrainingConfig(
            classification_mode="multiclass",
            positive_filter_type="lowpass",
        )
