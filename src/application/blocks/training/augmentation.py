"""
Audio Data Augmentation Pipeline

Provides composable audio and spectrogram augmentation for training.
Includes standard techniques (pitch shift, time stretch, noise, SpecAugment)
and advanced methods (Mixup, CutMix, Random EQ, polarity inversion).

Mixup and CutMix are batch-level operations applied inside the training loop,
not at the dataset level. Use apply_mixup() and apply_cutmix() directly.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.utils.message import Log

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

try:
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Librosa's pitch_shift and time_stretch use STFT with n_fft (default 2048).
# Signals shorter than this trigger "n_fft too large for input signal" warnings.
MIN_LEN_FOR_LIBROSA_STFT = 2048

# ---------------------------------------------------------------------------
# Audio-Level Augmentations (applied to raw waveform before spectrogram)
# ---------------------------------------------------------------------------

def augment_pitch_shift(audio: np.ndarray, sr: int, max_semitones: float = 2.0) -> np.ndarray:
    """Random pitch shift within +/- max_semitones."""
    if not HAS_LIBROSA or max_semitones <= 0:
        return audio
    shift = np.random.uniform(-max_semitones, max_semitones)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift)


def augment_time_stretch(audio: np.ndarray, max_deviation: float = 0.2) -> np.ndarray:
    """Random time stretch within +/- max_deviation of original speed."""
    if not HAS_LIBROSA or max_deviation <= 0:
        return audio
    rate = 1.0 + np.random.uniform(-max_deviation, max_deviation)
    rate = max(rate, 0.5)  # Safety clamp
    return librosa.effects.time_stretch(audio, rate=rate)


def augment_add_noise(audio: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
    """Add Gaussian noise."""
    if noise_factor <= 0:
        return audio
    noise = np.random.normal(0, noise_factor, len(audio))
    return audio + noise


def augment_volume(audio: np.ndarray, max_factor: float = 0.1) -> np.ndarray:
    """Random volume change."""
    if max_factor <= 0:
        return audio
    factor = 1.0 + np.random.uniform(-max_factor, max_factor)
    return audio * factor


def augment_time_shift(audio: np.ndarray, sr: int, max_fraction: float = 0.1) -> np.ndarray:
    """Random time shift (circular roll)."""
    if max_fraction <= 0:
        return audio
    shift_samples = int(np.random.uniform(-max_fraction, max_fraction) * sr)
    return np.roll(audio, shift_samples)


def augment_polarity_inversion(audio: np.ndarray) -> np.ndarray:
    """Flip polarity (multiply by -1)."""
    return -audio


def augment_random_eq(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply a random EQ filter (bandpass/highpass/lowpass) to simulate
    different recording conditions.
    """
    if not HAS_SCIPY:
        return audio

    filter_type = np.random.choice(["lowpass", "highpass", "bandpass"])
    nyquist = sr / 2.0

    try:
        if filter_type == "lowpass":
            cutoff = np.random.uniform(2000, nyquist * 0.95)
            b, a = scipy_signal.butter(2, cutoff / nyquist, btype="low")
        elif filter_type == "highpass":
            cutoff = np.random.uniform(50, 1000)
            b, a = scipy_signal.butter(2, cutoff / nyquist, btype="high")
        else:  # bandpass
            low = np.random.uniform(50, 1000)
            high = np.random.uniform(low + 500, min(low + 8000, nyquist * 0.95))
            b, a = scipy_signal.butter(2, [low / nyquist, high / nyquist], btype="band")

        filtered = scipy_signal.filtfilt(b, a, audio)
        # Blend: 50-100% filtered to keep some original character
        blend = np.random.uniform(0.5, 1.0)
        return blend * filtered + (1 - blend) * audio
    except Exception:
        return audio


def apply_positive_class_filter(
    audio: np.ndarray,
    sr: int,
    filter_type: str,
    cutoff_hz: float,
    cutoff_high_hz: float = 4000.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a deterministic lowpass, highpass, or bandpass filter to audio.

    Used to preprocess all positive-class samples in binary mode (e.g. to focus
    on a frequency band). Uses Butterworth design and second-order sections
    for numerical stability.

    Args:
        audio: 1D float32 waveform.
        sr: Sample rate in Hz.
        filter_type: "lowpass", "highpass", or "bandpass".
        cutoff_hz: Cutoff frequency for lowpass/highpass; low edge for bandpass.
        cutoff_high_hz: High edge for bandpass only (ignored for lowpass/highpass).
        order: Butterworth filter order (1-8).

    Returns:
        Filtered audio, same shape as input.
    """
    if not HAS_SCIPY or not filter_type or filter_type not in ("lowpass", "highpass", "bandpass"):
        return audio

    from scipy.signal import butter, sosfilt

    nyquist = sr / 2.0
    cutoff_hz = max(20.0, min(cutoff_hz, nyquist - 1.0))

    try:
        if filter_type == "lowpass":
            wn = max(0.001, min(cutoff_hz / nyquist, 0.999))
            sos = butter(order, wn, btype="low", output="sos")
        elif filter_type == "highpass":
            wn = max(0.001, min(cutoff_hz / nyquist, 0.999))
            sos = butter(order, wn, btype="high", output="sos")
        else:  # bandpass
            cutoff_high_hz = max(cutoff_hz + 1.0, min(cutoff_high_hz, nyquist - 1.0))
            wn_low = max(0.001, min(cutoff_hz / nyquist, 0.998))
            wn_high = max(wn_low + 0.001, min(cutoff_high_hz / nyquist, 0.999))
            sos = butter(order, [wn_low, wn_high], btype="band", output="sos")
        return np.ascontiguousarray(sosfilt(sos, audio).astype(np.float32))
    except Exception:
        return audio


class AudioAugmentationPipeline:
    """
    Composable audio augmentation pipeline configured from TrainingConfig.

    Each augmentation is applied with a random probability to create variety.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("use_augmentation", False)
        # Skip pitch/time STFT-based ops when signal is shorter than this (avoids librosa warning)
        self._min_len_stft = config.get("n_fft", MIN_LEN_FOR_LIBROSA_STFT)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply augmentation pipeline to audio waveform."""
        if not self.enabled:
            return audio

        augmented = audio.copy()

        # Pad to at least n_fft so pitch_shift/time_stretch (STFT-based) never see too-short signals
        if len(augmented) < self._min_len_stft:
            augmented = np.pad(
                augmented,
                (0, self._min_len_stft - len(augmented)),
                mode="constant",
                constant_values=0,
            )

        # Pitch shift (50% chance)
        if np.random.random() < 0.5:
            augmented = augment_pitch_shift(
                augmented, sr, self.config.get("pitch_shift_range", 2.0)
            )

        # Time stretch (30% chance)
        if np.random.random() < 0.3:
            augmented = augment_time_stretch(
                augmented, self.config.get("time_stretch_range", 0.2)
            )

        # Add noise (20% chance)
        if np.random.random() < 0.2:
            augmented = augment_add_noise(
                augmented, self.config.get("noise_factor", 0.01)
            )

        # Volume change (30% chance)
        if np.random.random() < 0.3:
            augmented = augment_volume(
                augmented, self.config.get("volume_factor", 0.1)
            )

        # Time shift (20% chance)
        if np.random.random() < 0.2:
            augmented = augment_time_shift(
                augmented, sr, self.config.get("time_shift_max", 0.1)
            )

        # Polarity inversion
        prob = self.config.get("polarity_inversion_prob", 0.0)
        if prob > 0 and np.random.random() < prob:
            augmented = augment_polarity_inversion(augmented)

        # Random EQ
        if self.config.get("use_random_eq", False) and np.random.random() < 0.3:
            augmented = augment_random_eq(augmented, sr)

        return augmented


# ---------------------------------------------------------------------------
# Spectrogram-Level Augmentations (SpecAugment-style)
# ---------------------------------------------------------------------------

def augment_frequency_mask(spectrogram: np.ndarray, max_width: int) -> np.ndarray:
    """Apply frequency masking (SpecAugment). Masks a horizontal band."""
    if max_width <= 0:
        return spectrogram
    aug = spectrogram.copy()
    width = np.random.randint(1, max_width + 1)
    f0 = np.random.randint(0, max(1, spectrogram.shape[0] - width))
    aug[f0:f0 + width, :] = 0
    return aug


def augment_time_mask(spectrogram: np.ndarray, max_width: int) -> np.ndarray:
    """Apply time masking (SpecAugment). Masks a vertical band."""
    if max_width <= 0:
        return spectrogram
    aug = spectrogram.copy()
    width = np.random.randint(1, max_width + 1)
    t0 = np.random.randint(0, max(1, spectrogram.shape[1] - width))
    aug[:, t0:t0 + width] = 0
    return aug


class SpectrogramAugmentationPipeline:
    """
    Spectrogram-level augmentations (applied after mel spectrogram computation).
    """

    def __init__(self, config: Dict[str, Any]):
        self.freq_mask = config.get("frequency_mask", 0)
        self.time_mask = config.get("time_mask", 0)
        self.enabled = config.get("use_augmentation", False) and (
            self.freq_mask > 0 or self.time_mask > 0
        )

    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return spectrogram

        aug = spectrogram
        if self.freq_mask > 0:
            aug = augment_frequency_mask(aug, self.freq_mask)
        if self.time_mask > 0:
            aug = augment_time_mask(aug, self.time_mask)
        return aug


# ---------------------------------------------------------------------------
# Batch-Level Augmentations (Mixup / CutMix)
# ---------------------------------------------------------------------------

def apply_mixup(
    inputs: "torch.Tensor",
    targets: "torch.Tensor",
    alpha: float = 0.2,
    num_classes: int = 1,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Apply Mixup augmentation at the batch level.

    Blends pairs of samples and their labels using a Beta distribution.
    For binary mode (num_classes=1), targets are float probabilities.
    For multiclass, targets are converted to one-hot then blended.

    Args:
        inputs: Batch of input tensors (B, C, H, W)
        targets: Batch of target labels (B,) for multiclass or (B, 1) for binary
        alpha: Beta distribution parameter (higher = more mixing)
        num_classes: Number of classes (1 for binary)

    Returns:
        Tuple of (mixed_inputs, mixed_targets)
    """
    if not HAS_PYTORCH:
        return inputs, targets

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)

    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]

    if num_classes == 1:
        # Binary: targets are already float-like
        targets_float = targets.float()
        mixed_targets = lam * targets_float + (1 - lam) * targets_float[index]
    else:
        # Multiclass: convert to one-hot then blend
        targets_onehot = torch.zeros(batch_size, num_classes, device=targets.device)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
        shuffled_onehot = targets_onehot[index]
        mixed_targets = lam * targets_onehot + (1 - lam) * shuffled_onehot

    return mixed_inputs, mixed_targets


def apply_cutmix(
    inputs: "torch.Tensor",
    targets: "torch.Tensor",
    alpha: float = 1.0,
    num_classes: int = 1,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Apply CutMix augmentation at the batch level.

    Cuts a rectangular region from one spectrogram and pastes it onto another.
    Labels are blended proportionally to the area of the cut region.

    Args:
        inputs: Batch of input tensors (B, C, H, W)
        targets: Batch of target labels
        alpha: Beta distribution parameter
        num_classes: Number of classes (1 for binary)

    Returns:
        Tuple of (mixed_inputs, mixed_targets)
    """
    if not HAS_PYTORCH:
        return inputs, targets

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)

    _, _, h, w = inputs.shape

    # Generate random bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)

    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)

    # Apply cut
    mixed_inputs = inputs.clone()
    mixed_inputs[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]

    # Adjust lambda based on actual cut area
    actual_lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (h * w)

    if num_classes == 1:
        targets_float = targets.float()
        mixed_targets = actual_lam * targets_float + (1 - actual_lam) * targets_float[index]
    else:
        targets_onehot = torch.zeros(batch_size, num_classes, device=targets.device)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
        shuffled_onehot = targets_onehot[index]
        mixed_targets = actual_lam * targets_onehot + (1 - actual_lam) * shuffled_onehot

    return mixed_inputs, mixed_targets
