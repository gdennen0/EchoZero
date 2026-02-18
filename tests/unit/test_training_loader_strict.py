"""
Strict failure-mode tests for DataLoader creation.
"""
import threading

import pytest

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.application.blocks.training.datasets import create_data_loaders


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class _DummyDataset(Dataset):
    def __init__(self):
        # labels used by create_data_splits
        self.samples = [(None, 0), (None, 1), (None, 0), (None, 1), (None, 0), (None, 1)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, label = self.samples[idx]
        return torch.zeros(1, 8, 8), torch.tensor(label, dtype=torch.long)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_create_data_loaders_fails_when_augmentation_and_zero_workers():
    ds = _DummyDataset()
    cfg = {
        "use_augmentation": True,
        "num_workers": 0,
        "batch_size": 2,
        "validation_split": 0.2,
        "test_split": 0.2,
        "stratified_split": False,
    }
    with pytest.raises(RuntimeError, match="use_augmentation=true with num_workers=0"):
        create_data_loaders(ds, cfg)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_create_data_loaders_fails_on_macos_background_thread_with_workers(monkeypatch):
    ds = _DummyDataset()
    cfg = {
        "use_augmentation": False,
        "num_workers": 1,
        "batch_size": 2,
        "validation_split": 0.2,
        "test_split": 0.2,
        "stratified_split": False,
    }

    main_thread = threading.main_thread()

    class _FakeThread:
        pass

    monkeypatch.setattr("src.application.blocks.training.datasets.sys.platform", "darwin")
    monkeypatch.setattr(
        "src.application.blocks.training.datasets.threading.current_thread",
        lambda: _FakeThread(),
    )
    monkeypatch.setattr(
        "src.application.blocks.training.datasets.threading.main_thread",
        lambda: main_thread,
    )

    with pytest.raises(RuntimeError, match="background thread on macOS"):
        create_data_loaders(ds, cfg)
