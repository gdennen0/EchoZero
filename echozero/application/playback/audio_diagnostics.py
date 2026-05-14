"""Runtime audio diagnostics bundle helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import wave
from typing import Any

import numpy as np


def timestamped_capture_id() -> str:
    return datetime.now(timezone.utc).strftime("audio-diagnostics-%Y%m%dT%H%M%S%fZ")


def write_audio_diagnostics_bundle(
    *,
    capture_id: str,
    capture: dict[str, object],
    playback_state: object,
    device_config: dict[str, object],
    runtime_sensor_events: tuple[object, ...] | list[object],
    audio_blocks: tuple[object, ...] | list[object],
    output_dir: object | None = None,
) -> dict[str, object]:
    """Write a compact JSON + NumPy + WAV diagnostics bundle."""

    root = Path(str(output_dir)) if output_dir else Path("artifacts") / "audio-diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in str(capture_id))
    json_path = root / f"{safe_id}.json"
    npy_path = root / f"{safe_id}.npy"
    wav_path = root / f"{safe_id}.wav"

    blocks = [_as_float_array(block) for block in audio_blocks]
    if blocks:
        audio = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    else:
        channels = max(1, int(device_config.get("channels", 1) or 1))
        audio = np.zeros((0, channels), dtype=np.float32)
    np.save(npy_path, audio)
    _write_wav(wav_path, audio, sample_rate=int(device_config.get("sample_rate", 44100) or 44100))

    payload = {
        "capture": _jsonable(capture),
        "playback_state": _jsonable(playback_state),
        "device_config": _jsonable(device_config),
        "runtime_sensor_events": _jsonable(tuple(runtime_sensor_events)),
        "audio_block_count": len(blocks),
        "sample_count": int(audio.shape[0]),
        "channels": int(audio.shape[1]) if audio.ndim == 2 else 1,
        "npy_path": str(npy_path),
        "wav_path": str(wav_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "active": False,
        "capture_id": safe_id,
        "json_path": str(json_path),
        "npy_path": str(npy_path),
        "wav_path": str(wav_path),
        "bundle_path": str(json_path),
        "audio_block_count": len(blocks),
    }


def _as_float_array(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        arr = arr.reshape((-1, 1))
    return arr


def _write_wav(path: Path, audio: np.ndarray, *, sample_rate: int) -> None:
    arr = _as_float_array(audio)
    clipped = np.clip(arr, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(pcm.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["timestamped_capture_id", "write_audio_diagnostics_bundle"]
