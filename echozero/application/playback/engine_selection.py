"""
Runtime audio engine selection for local app playback.
Exists so dev/test runs can opt into the v2 live backend while v1 remains default.
Connects playback controller construction to reversible backend selection.
"""

from __future__ import annotations

import os
from typing import Any, Literal, TypeAlias, cast

from echozero.application.audio_engine_v2.live_engine import V2LiveAudioEngine
from echozero.audio.engine import AudioEngine

ENGINE_BACKEND_ENV = "ECHOZERO_AUDIO_ENGINE"
AudioEngineBackendName = Literal["v1", "v2"]
RuntimeAudioEngine: TypeAlias = AudioEngine | V2LiveAudioEngine


def selected_audio_engine_backend(
    requested: str | None = None,
) -> AudioEngineBackendName:
    """Return the requested runtime audio backend, defaulting safely to v1."""

    value = str(requested if requested is not None else os.environ.get(ENGINE_BACKEND_ENV, ""))
    normalized = value.strip().lower()
    if normalized in {"v2", "audio_engine_v2", "engine_v2"}:
        return "v2"
    return "v1"


def build_runtime_audio_engine(
    *,
    backend_name: str | None = None,
    **kwargs: Any,
) -> RuntimeAudioEngine:
    """Build the selected live runtime audio engine."""

    if selected_audio_engine_backend(backend_name) == "v2":
        return V2LiveAudioEngine(**kwargs)
    return cast(RuntimeAudioEngine, AudioEngine(**kwargs))


__all__ = [
    "ENGINE_BACKEND_ENV",
    "AudioEngineBackendName",
    "RuntimeAudioEngine",
    "build_runtime_audio_engine",
    "selected_audio_engine_backend",
]
