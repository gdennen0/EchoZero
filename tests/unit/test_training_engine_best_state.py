"""
Regression test for best-model state restoration in TrainingEngine.
"""
import pytest

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.application.blocks.training.engine import TrainingEngine


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_training_engine_restores_true_best_state(monkeypatch):
    config = {
        "epochs": 2,
        "early_stopping_patience": 10,
        "use_early_stopping": True,
        "optimizer": "sgd",
        "learning_rate": 0.1,
        "lr_scheduler": "none",
    }
    engine = TrainingEngine(config, device="cpu")

    model = nn.Linear(2, 2, bias=False)
    criterion = nn.CrossEntropyLoss()

    # Deterministic epoch mutations: epoch1 -> weight 1.0 (best), epoch2 -> weight 2.0 (worse)
    call_idx = {"train": 0, "val": 0}

    def fake_train_epoch(*args, **kwargs):
        call_idx["train"] += 1
        with torch.no_grad():
            model.weight.fill_(float(call_idx["train"]))
        return 0.1, 100.0, {"total": 0.01, "data": 0.001, "compute": 0.009}

    val_scores = [90.0, 80.0]

    def fake_validate_epoch(*args, **kwargs):
        score = val_scores[call_idx["val"]]
        call_idx["val"] += 1
        return 0.1, score

    monkeypatch.setattr(engine, "_train_epoch", fake_train_epoch)
    monkeypatch.setattr(engine, "_validate_epoch", fake_validate_epoch)

    # Minimal loaders only need __len__ for scheduler setup and epoch loops.
    train_loader = [("x", "y")]
    val_loader = [("x", "y")]

    result = engine.train(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        classes=["a", "b"],
        progress_tracker=None,
    )

    assert result.best_epoch == 1
    assert result.best_val_metric == 90.0
    assert torch.allclose(model.weight, torch.ones_like(model.weight))
