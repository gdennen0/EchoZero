"""
Infrastructure playback backend compatibility import.
Exists because older wiring points still reference one infrastructure-level backend protocol.
Connects those callers to the canonical audio output backend contract under `echozero.audio`.
"""

from echozero.audio.output_backend import AudioOutputBackend as AudioBackend

__all__ = ["AudioBackend"]
