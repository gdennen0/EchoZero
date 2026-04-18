"""
Runtime model loading and feature extraction for installed inference bundles.
Exists because EchoZero runtime inference must consume a stable shared contract, not Foundry internals.
Used by app processors and future model-selection/install services.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from echozero.errors import ExecutionError, ValidationError
from echozero.inference_eval.runtime_preflight import resolve_runtime_model_path, run_runtime_preflight

from .architectures import CrnnRuntimeModel, SimpleCnnRuntimeModel


@dataclass(frozen=True, slots=True)
class LoadedRuntimeModel:
    """Loaded runtime bundle ready for feature extraction and inference."""

    model: Any
    classes: tuple[str, ...]
    sample_rate: int
    max_length: int
    n_fft: int
    hop_length: int
    n_mels: int
    fmax: int
    device: str
    source_path: Path


def resolve_device(device: str) -> str:
    """Resolve runtime device selection against local torch availability."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError as exc:
        raise ExecutionError("PyTorch is required for runtime model inference.") from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_runtime_model(model_path: str | Path, *, device: str = "auto") -> LoadedRuntimeModel:
    """Load a runtime bundle and reconstruct its shared inference model."""
    try:
        import torch
    except ImportError as exc:
        raise ExecutionError("PyTorch is required for runtime model inference.") from exc

    resolved_device = resolve_device(device)
    resolved_model_path = resolve_runtime_model_path(model_path)
    checkpoint = torch.load(resolved_model_path, map_location=resolved_device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ExecutionError(f"Unexpected checkpoint format from {resolved_model_path}")
    run_runtime_preflight(resolved_model_path, checkpoint)

    classes = tuple(str(value).strip().lower() for value in checkpoint.get("classes", ()))
    if not classes:
        raise ValidationError(f"Checkpoint {resolved_model_path} is missing classes metadata.")
    preprocessing = checkpoint.get("preprocessing") or checkpoint.get("inference_preprocessing") or {}
    sample_rate = int(preprocessing["sampleRate"])
    max_length = int(preprocessing["maxLength"])
    n_fft = int(preprocessing["nFft"])
    hop_length = int(preprocessing["hopLength"])
    n_mels = int(preprocessing["nMels"])
    fmax = int(preprocessing["fmax"])

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ExecutionError(f"Checkpoint {resolved_model_path} is missing model_state_dict.")
    model = instantiate_runtime_model(state_dict=state_dict, n_mels=n_mels, num_classes=len(classes))
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()

    return LoadedRuntimeModel(
        model=model,
        classes=classes,
        sample_rate=sample_rate,
        max_length=max_length,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=fmax,
        device=resolved_device,
        source_path=resolved_model_path,
    )


def instantiate_runtime_model(*, state_dict: dict[str, Any], n_mels: int, num_classes: int) -> Any:
    """Instantiate a shared runtime architecture from a checkpoint state dict."""
    keys = set(state_dict.keys())
    if any(key.startswith("rnn.") for key in keys):
        return CrnnRuntimeModel(num_classes=num_classes, mel_bins=n_mels)
    if any(key.startswith("features.") for key in keys):
        return SimpleCnnRuntimeModel(num_classes=num_classes)
    raise ValidationError("Unsupported runtime checkpoint architecture.")


def build_feature_tensor(
    *,
    audio: np.ndarray,
    event_time: float,
    sample_rate: int,
    max_length: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmax: int,
) -> np.ndarray:
    """Build a normalized mel-spectrogram window for one onset event."""
    try:
        import librosa
    except ImportError as exc:
        raise ExecutionError("librosa is required for runtime model inference.") from exc

    start_index = max(0, int(round(event_time * sample_rate)))
    segment = audio[start_index : start_index + max_length]
    if len(segment) < max_length:
        segment = np.pad(segment, (0, max_length - len(segment)))
    peak = float(np.max(np.abs(segment))) if len(segment) else 0.0
    if peak > 0:
        segment = segment / peak

    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel + 1e-10, ref=np.max).astype(np.float32)
    mel_mean = float(np.mean(mel_db))
    mel_std = float(np.std(mel_db))
    if mel_std > 0:
        mel_db = (mel_db - mel_mean) / mel_std
    return mel_db[np.newaxis, np.newaxis, :, :].astype(np.float32)


def predict_probabilities(runtime_model: LoadedRuntimeModel, feature: np.ndarray) -> np.ndarray:
    """Return class probabilities for one preprocessed runtime feature tensor."""
    try:
        import torch
    except ImportError as exc:
        raise ExecutionError("PyTorch is required for runtime model inference.") from exc

    with torch.no_grad():
        logits = runtime_model.model(torch.from_numpy(feature).to(runtime_model.device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probabilities.astype(np.float32)
