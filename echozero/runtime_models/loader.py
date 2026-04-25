"""
Runtime model loading and feature extraction for installed inference bundles.
Exists because EchoZero runtime inference must consume a stable shared contract, not Foundry internals.
Used by app processors and future model-selection/install services.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
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
    manifest_path: Path | None = None
    artifact_manifest: dict[str, Any] | None = None


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
    manifest_path, artifact_manifest = _resolve_runtime_artifact_manifest(resolved_model_path)

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
        manifest_path=manifest_path,
        artifact_manifest=artifact_manifest,
    )


def build_model_artifact_reference(runtime_model: LoadedRuntimeModel) -> dict[str, Any]:
    """Build structured model-artifact provenance for classified runtime events."""
    manifest = runtime_model.artifact_manifest or {}
    artifact_identity = _artifact_identity_from_manifest(manifest)
    display_identity = _display_identity_from_manifest(
        manifest,
        artifact_identity=artifact_identity,
        source_path=runtime_model.source_path,
    )
    reference: dict[str, Any] = {
        "schema": "echozero.model_artifact_ref.v1",
    }
    if artifact_identity:
        reference["artifactIdentity"] = artifact_identity
    if display_identity:
        reference["displayIdentity"] = display_identity
    return reference


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


def predict_probabilities_batch(
    runtime_model: LoadedRuntimeModel,
    features: np.ndarray,
) -> np.ndarray:
    """Return class probabilities for a batch of runtime feature tensors."""
    try:
        import torch
    except ImportError as exc:
        raise ExecutionError("PyTorch is required for runtime model inference.") from exc

    with torch.no_grad():
        logits = runtime_model.model(torch.from_numpy(features).to(runtime_model.device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return probabilities.astype(np.float32)


def _resolve_runtime_artifact_manifest(model_path: Path) -> tuple[Path | None, dict[str, Any] | None]:
    resolved_model_path = model_path.resolve()
    matches: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in sorted(resolved_model_path.parent.glob("*.manifest.json")):
        manifest = _load_manifest(manifest_path)
        if manifest is None:
            continue
        target_path = _resolve_manifest_weights_path(manifest_path, manifest)
        if target_path == resolved_model_path:
            matches.append((manifest_path.resolve(), manifest))
    if len(matches) != 1:
        return None, None
    return matches[0]


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def _resolve_manifest_weights_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path | None:
    raw_weights_path = manifest.get("weightsPath")
    if not isinstance(raw_weights_path, str) or not raw_weights_path.strip():
        return None
    weights_path = Path(raw_weights_path)
    if weights_path.is_absolute():
        return weights_path.resolve()
    return (manifest_path.parent / weights_path).resolve()


def _artifact_identity_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    nested = manifest.get("artifactIdentity")
    if isinstance(nested, Mapping):
        return dict(nested)

    identity: dict[str, Any] = {}
    for source_key in (
        "artifactId",
        "runId",
        "datasetVersionId",
        "specHash",
        "sharedContractFingerprint",
    ):
        value = manifest.get(source_key)
        if value is None:
            continue
        identity[source_key] = value
    return identity


def _display_identity_from_manifest(
    manifest: Mapping[str, Any],
    *,
    artifact_identity: Mapping[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    nested = manifest.get("displayIdentity")
    if isinstance(nested, Mapping):
        return dict(nested)

    raw_runtime = manifest.get("runtime")
    runtime = dict(raw_runtime) if isinstance(raw_runtime, Mapping) else {}
    raw_classes = manifest.get("classes")
    classes = [str(value) for value in raw_classes] if isinstance(raw_classes, list) else []
    raw_weights_path = manifest.get("weightsPath")
    weights_path = str(raw_weights_path) if isinstance(raw_weights_path, str) and raw_weights_path.strip() else None
    display_identity: dict[str, Any] = {
        "artifactId": artifact_identity.get("artifactId"),
        "runId": artifact_identity.get("runId"),
        "datasetVersionId": artifact_identity.get("datasetVersionId"),
        "weightsFile": Path(weights_path).name if weights_path else source_path.name,
        "classes": classes,
        "classificationMode": manifest.get("classificationMode"),
        "consumer": runtime.get("consumer"),
    }
    return {
        key: value
        for key, value in display_identity.items()
        if value is not None and value != []
    }
