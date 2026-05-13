"""Runtime-audio diagnostics bundle writer.
Exists so dev capture can persist bounded playback evidence without coupling UI automation to engine internals.
Connects playback runtime state, sensor events, and optional callback blocks to portable artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import wave
from typing import Any

import numpy as np

_DEFAULT_OUTPUT_DIR = Path("artifacts") / "audio-diagnostics"


def timestamped_capture_id(prefix: str = "audio-diagnostics") -> str:
    """Return a filesystem-safe UTC timestamp capture id."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}"


def write_audio_diagnostics_bundle(
    *,
    capture_id: str,
    capture: dict[str, Any],
    playback_state: Any,
    device_config: dict[str, Any],
    runtime_sensor_events: tuple[dict[str, object], ...] | list[dict[str, object]],
    audio_blocks: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write one diagnostics bundle and return a JSON-safe summary."""

    bundle_root = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    bundle_dir = bundle_root / _safe_name(capture_id)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    block_payloads, block_arrays = _prepare_audio_blocks(audio_blocks)
    npy_path: Path | None = None
    wav_path: Path | None = None
    if block_arrays:
        concatenated = _concatenate_blocks(block_arrays)
        if concatenated.size:
            npy_path = bundle_dir / "output_callback_blocks.npy"
            np.save(npy_path, concatenated.astype(np.float32, copy=False))
            wav_path = bundle_dir / "output_callback_blocks.wav"
            _write_wav(wav_path, concatenated, int(device_config.get("sample_rate", 0) or 0))

    payload = {
        "schema_version": 1,
        "capture": _jsonable(capture),
        "device_config": _jsonable(device_config),
        "playback_state": _jsonable(playback_state),
        "runtime_sensor_events": _jsonable(list(runtime_sensor_events)),
        "audio_blocks": _jsonable(block_payloads),
        "files": {
            "json": "diagnostics.json",
            "npy": npy_path.name if npy_path is not None else None,
            "wav": wav_path.name if wav_path is not None else None,
        },
    }
    json_path = bundle_dir / "diagnostics.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "capture_id": str(capture_id),
        "active": False,
        "bundle_path": str(bundle_dir),
        "json_path": str(json_path),
        "npy_path": str(npy_path) if npy_path is not None else None,
        "wav_path": str(wav_path) if wav_path is not None else None,
        "event_count": len(list(runtime_sensor_events)),
        "audio_block_count": len(block_arrays),
    }


def _prepare_audio_blocks(
    audio_blocks: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    payloads: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    for index, block in enumerate(audio_blocks):
        if not isinstance(block, dict):
            continue
        samples = block.get("samples")
        if samples is None:
            continue
        array = np.asarray(samples, dtype=np.float32)
        if array.size == 0:
            continue
        arrays.append(array.copy())
        payload = {key: value for key, value in block.items() if key != "samples"}
        payload.setdefault("index", index)
        payload["shape"] = list(array.shape)
        payloads.append(payload)
    return payloads, arrays


def _concatenate_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    normalized: list[np.ndarray] = []
    max_channels = 1
    for block in blocks:
        array = np.asarray(block, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        elif array.ndim > 2:
            array = array.reshape((array.shape[0], -1))
        max_channels = max(max_channels, int(array.shape[1]))
        normalized.append(array)
    padded: list[np.ndarray] = []
    for array in normalized:
        if int(array.shape[1]) < max_channels:
            pad = np.zeros((array.shape[0], max_channels - array.shape[1]), dtype=np.float32)
            array = np.concatenate((array, pad), axis=1)
        padded.append(array)
    return np.concatenate(padded, axis=0) if padded else np.zeros((0, 1), dtype=np.float32)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    sample_rate = max(1, int(sample_rate or 44100))
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    array = np.clip(array, -1.0, 1.0)
    pcm = (array * np.float32(32767.0)).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(pcm.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value") and isinstance(getattr(value, "value"), (str, int, float, bool)):
        return getattr(value, "value")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)
    return safe.strip(".-") or "audio-diagnostics"


__all__ = ["timestamped_capture_id", "write_audio_diagnostics_bundle"]
