"""Unit tests for beginner-friendly model coach feedback."""

from pathlib import Path

import src.application.blocks.training.model_coach as model_coach
from src.application.blocks.training.model_coach import (
    build_inference_feedback,
    build_training_feedback,
)


def _patch_history_path(monkeypatch, tmp_path: Path) -> None:
    history_file = tmp_path / "coach_history.jsonl"
    monkeypatch.setattr(model_coach, "_history_path", lambda: history_file)


def test_training_feedback_flags_overfitting_and_imbalance(monkeypatch, tmp_path: Path):
    _patch_history_path(monkeypatch, tmp_path)
    feedback = build_training_feedback(
        block_id="block-1",
        model_path="/tmp/model.pth",
        config={"epochs": 40, "classification_mode": "multiclass"},
        training_stats={
            "train_accuracy": [96.0],
            "val_accuracy": [76.0],
            "epochs": [1],
        },
        dataset_stats={
            "total_samples": 300,
            "class_distribution": {"kick": 240, "snare": 60},
        },
        validation_metrics={"accuracy": 76.0},
        test_metrics=None,
        threshold_metrics=None,
        excluded_bad_file_count=3,
    )

    assert feedback["verdict"] in {"needs_work", "unreliable"}
    assert any("Overfitting" in f or "overfitting" in f for f in feedback["findings"])
    assert any("imbalance" in f.lower() for f in feedback["findings"])
    assert feedback["key_metrics"]["excluded_bad_file_count"] == 3
    assert len(feedback["next_actions"]) >= 1


def test_training_feedback_good_path_has_green_style_verdict(monkeypatch, tmp_path: Path):
    _patch_history_path(monkeypatch, tmp_path)
    feedback = build_training_feedback(
        block_id="block-2",
        model_path="/tmp/model_good.pth",
        config={"epochs": 30, "classification_mode": "multiclass"},
        training_stats={
            "train_accuracy": [90.0],
            "val_accuracy": [87.5],
            "epochs": list(range(1, 11)),
        },
        dataset_stats={
            "total_samples": 1200,
            "class_distribution": {"kick": 400, "snare": 400, "hihat": 400},
        },
        validation_metrics={"accuracy": 87.5},
        test_metrics={"accuracy": 88.2},
        threshold_metrics=None,
        excluded_bad_file_count=0,
    )

    assert feedback["verdict"] == "good"
    assert feedback["score"] >= 80
    assert feedback["trend"]["label"] in {"baseline", "improving", "stable", "regressing"}


def test_inference_feedback_flags_low_confidence():
    feedback = build_inference_feedback(
        execution_summary={
            "total_events_input": 200,
            "total_classified": 190,
            "total_skipped": 10,
            "events_per_second": 45.0,
            "confidence_stats": {"avg": 0.54, "min": 0.20, "max": 0.94},
        },
        confidence_threshold=0.7,
    )

    assert feedback["verdict"] in {"needs_work", "unreliable"}
    assert any("confidence" in f.lower() for f in feedback["findings"])
    assert len(feedback["next_actions"]) >= 1
