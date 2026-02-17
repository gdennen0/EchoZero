"""
Unit tests for training augmentation module.

Covers audio transforms, AudioAugmentationPipeline, SpectrogramAugmentationPipeline,
and batch-level Mixup/CutMix. Uses fixed seeds for reproducibility.
"""
import numpy as np
import pytest

from src.application.blocks.training.augmentation import (
    augment_add_noise,
    augment_frequency_mask,
    augment_polarity_inversion,
    augment_time_mask,
    augment_time_shift,
    augment_volume,
    AudioAugmentationPipeline,
    SpectrogramAugmentationPipeline,
    MIN_LEN_FOR_LIBROSA_STFT,
)

try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# ---------------------------------------------------------------------------
# Audio transforms (no STFT): shape, dtype, and where applicable, output differs
# ---------------------------------------------------------------------------

def test_augment_add_noise_shape_dtype():
    np.random.seed(42)
    audio = np.random.randn(1000).astype(np.float64)
    out = augment_add_noise(audio, noise_factor=0.01)
    assert out.shape == audio.shape
    assert out.dtype == audio.dtype
    assert not np.allclose(out, audio)


def test_augment_add_noise_zero_factor_returns_unchanged():
    audio = np.random.randn(100).astype(np.float32)
    out = augment_add_noise(audio, noise_factor=0)
    assert out is audio or np.allclose(out, audio)


def test_augment_volume_shape_dtype():
    np.random.seed(42)
    audio = np.random.randn(500).astype(np.float32)
    out = augment_volume(audio, max_factor=0.1)
    assert out.shape == audio.shape
    assert out.dtype == audio.dtype


def test_augment_volume_zero_factor_returns_unchanged():
    audio = np.random.randn(100).astype(np.float32)
    out = augment_volume(audio, max_factor=0)
    assert out is audio or np.allclose(out, audio)


def test_augment_polarity_inversion():
    audio = np.array([1.0, -0.5, 0.0], dtype=np.float32)
    out = augment_polarity_inversion(audio)
    assert out.shape == audio.shape
    assert np.allclose(out, [-1.0, 0.5, 0.0])


def test_augment_time_shift_shape():
    np.random.seed(42)
    audio = np.random.randn(22050).astype(np.float32)
    out = augment_time_shift(audio, sr=22050, max_fraction=0.1)
    assert out.shape == audio.shape


def test_augment_time_shift_zero_fraction_returns_unchanged():
    audio = np.random.randn(100).astype(np.float32)
    out = augment_time_shift(audio, sr=22050, max_fraction=0)
    assert np.allclose(out, audio)


# ---------------------------------------------------------------------------
# SpecAugment-style masking
# ---------------------------------------------------------------------------

def test_augment_frequency_mask_shape_and_zeros():
    np.random.seed(42)
    spec = np.random.randn(64, 100).astype(np.float32)
    out = augment_frequency_mask(spec, max_width=10)
    assert out.shape == spec.shape
    # At least one horizontal band should be zeroed
    assert np.any(out == 0)
    assert np.any(out != 0)


def test_augment_frequency_mask_zero_width_returns_unchanged():
    spec = np.random.randn(32, 50).astype(np.float32)
    out = augment_frequency_mask(spec, max_width=0)
    assert np.allclose(out, spec)


def test_augment_time_mask_shape_and_zeros():
    np.random.seed(42)
    spec = np.random.randn(64, 100).astype(np.float32)
    out = augment_time_mask(spec, max_width=15)
    assert out.shape == spec.shape
    assert np.any(out == 0)
    assert np.any(out != 0)


def test_augment_time_mask_zero_width_returns_unchanged():
    spec = np.random.randn(32, 50).astype(np.float32)
    out = augment_time_mask(spec, max_width=0)
    assert np.allclose(out, spec)


# ---------------------------------------------------------------------------
# AudioAugmentationPipeline
# ---------------------------------------------------------------------------

def test_audio_augmentation_pipeline_disabled():
    config = {"use_augmentation": False}
    pipeline = AudioAugmentationPipeline(config)
    np.random.seed(42)
    audio = np.random.randn(3000).astype(np.float32)
    out = pipeline(audio, sr=22050)
    assert out is audio or np.allclose(out, audio)


def test_audio_augmentation_pipeline_enabled_shape():
    """With augmentation on, short input is padded to min_len_stft then processed; output length can vary (time_stretch)."""
    config = {
        "use_augmentation": True,
        "n_fft": 2048,
        "pitch_shift_range": 0,  # avoid librosa if not installed
        "time_stretch_range": 0,
        "noise_factor": 0.01,
        "volume_factor": 0.1,
        "time_shift_max": 0.05,
    }
    pipeline = AudioAugmentationPipeline(config)
    np.random.seed(123)
    audio = np.random.randn(1500).astype(np.float32)  # shorter than 2048
    out = pipeline(audio, sr=22050)
    # Pipeline pads to 2048 then may apply time_stretch; output is at least 2048 or whatever length after transforms
    assert out.ndim == 1
    assert len(out) >= len(audio)
    assert out.dtype == audio.dtype


def test_audio_augmentation_pipeline_long_input():
    config = {
        "use_augmentation": True,
        "n_fft": 2048,
        "pitch_shift_range": 0,
        "time_stretch_range": 0,
        "noise_factor": 0,
        "volume_factor": 0,
        "time_shift_max": 0,
    }
    pipeline = AudioAugmentationPipeline(config)
    np.random.seed(99)
    audio = np.random.randn(5000).astype(np.float32)
    out = pipeline(audio, sr=22050)
    assert out.ndim == 1
    assert out.dtype == audio.dtype


# ---------------------------------------------------------------------------
# SpectrogramAugmentationPipeline
# ---------------------------------------------------------------------------

def test_spectrogram_augmentation_pipeline_disabled():
    config = {"use_augmentation": False, "frequency_mask": 5, "time_mask": 5}
    pipeline = SpectrogramAugmentationPipeline(config)
    spec = np.random.randn(64, 100).astype(np.float32)
    out = pipeline(spec)
    assert np.allclose(out, spec)


def test_spectrogram_augmentation_pipeline_enabled_applies_masking():
    config = {"use_augmentation": True, "frequency_mask": 8, "time_mask": 10}
    pipeline = SpectrogramAugmentationPipeline(config)
    np.random.seed(77)
    spec = np.random.randn(64, 100).astype(np.float32)
    out = pipeline(spec)
    assert out.shape == spec.shape
    assert np.any(out == 0)


# ---------------------------------------------------------------------------
# Mixup / CutMix (require torch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not installed")
def test_apply_mixup_binary_shapes_and_targets_in_range():
    from src.application.blocks.training.augmentation import apply_mixup

    torch.manual_seed(42)
    np.random.seed(42)
    inputs = torch.randn(4, 1, 64, 100)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])  # binary
    mixed_in, mixed_t = apply_mixup(inputs, targets, alpha=0.2, num_classes=1)
    assert mixed_in.shape == inputs.shape
    assert mixed_t.shape == targets.shape
    assert mixed_t.min() >= 0 and mixed_t.max() <= 1


@pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not installed")
def test_apply_mixup_multiclass_targets_sum_to_one():
    from src.application.blocks.training.augmentation import apply_mixup

    torch.manual_seed(42)
    np.random.seed(42)
    inputs = torch.randn(4, 1, 64, 100)
    targets = torch.tensor([0, 1, 2, 0])  # 3 classes
    mixed_in, mixed_t = apply_mixup(inputs, targets, alpha=0.2, num_classes=3)
    assert mixed_in.shape == inputs.shape
    assert mixed_t.shape == (4, 3)
    row_sums = mixed_t.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))


@pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not installed")
def test_apply_cutmix_binary_shapes_and_targets_in_range():
    from src.application.blocks.training.augmentation import apply_cutmix

    torch.manual_seed(42)
    np.random.seed(42)
    inputs = torch.randn(4, 1, 64, 100)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    mixed_in, mixed_t = apply_cutmix(inputs, targets, alpha=1.0, num_classes=1)
    assert mixed_in.shape == inputs.shape
    assert mixed_t.shape == targets.shape
    assert mixed_t.min() >= 0 and mixed_t.max() <= 1


@pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not installed")
def test_apply_cutmix_multiclass_targets_sum_to_one():
    from src.application.blocks.training.augmentation import apply_cutmix

    torch.manual_seed(42)
    np.random.seed(42)
    inputs = torch.randn(4, 1, 64, 100)
    targets = torch.tensor([0, 1, 2, 0])
    mixed_in, mixed_t = apply_cutmix(inputs, targets, alpha=1.0, num_classes=3)
    assert mixed_in.shape == inputs.shape
    assert mixed_t.shape == (4, 3)
    row_sums = mixed_t.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))


def test_min_len_constant():
    """MIN_LEN_FOR_LIBROSA_STFT is used for padding; should be 2048 (librosa default)."""
    assert MIN_LEN_FOR_LIBROSA_STFT == 2048
