"""
Block specifications for pipeline construction.

Each spec declares ports, types, and default settings. Callable spec functions
(LoadAudio(), Separator(), etc.) return BlockSpec instances with default ports
pre-configured. Settings passed as kwargs override defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from echozero.domain.enums import BlockCategory, Direction, PortType


# ---------------------------------------------------------------------------
# PortSpec / BlockSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortSpec:
    """Declares a port on a block spec."""

    name: str
    port_type: PortType
    direction: Direction


@dataclass
class BlockSpec:
    """Declarative specification of a block type for pipeline construction.

    Defines what ports a block has and what processor it maps to.
    """

    block_type: str
    name: str = ""
    category: BlockCategory = BlockCategory.PROCESSOR
    input_ports: tuple[PortSpec, ...] = ()
    output_ports: tuple[PortSpec, ...] = ()
    settings: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Callable spec factories — one per processor type
# ---------------------------------------------------------------------------


def LoadAudio(**settings: Any) -> BlockSpec:
    """Load an audio file. Source block — no inputs."""
    return BlockSpec(
        block_type="LoadAudio",
        name="Load Audio",
        input_ports=(),
        output_ports=(
            PortSpec("audio_out", PortType.AUDIO, Direction.OUTPUT),
        ),
        settings=settings,
    )


def Separator(**settings: Any) -> BlockSpec:
    """Separate audio into stems (drums, bass, vocals, other)."""
    return BlockSpec(
        block_type="SeparateAudio",
        name="Separate Audio",
        input_ports=(
            PortSpec("audio_in", PortType.AUDIO, Direction.INPUT),
        ),
        output_ports=(
            PortSpec("drums_out", PortType.AUDIO, Direction.OUTPUT),
            PortSpec("bass_out", PortType.AUDIO, Direction.OUTPUT),
            PortSpec("vocals_out", PortType.AUDIO, Direction.OUTPUT),
            PortSpec("other_out", PortType.AUDIO, Direction.OUTPUT),
        ),
        settings=settings,
    )


def DetectOnsets(**settings: Any) -> BlockSpec:
    """Detect onsets in audio. Returns event data."""
    return BlockSpec(
        block_type="DetectOnsets",
        name="Detect Onsets",
        input_ports=(
            PortSpec("audio_in", PortType.AUDIO, Direction.INPUT),
        ),
        output_ports=(
            PortSpec("events_out", PortType.EVENT, Direction.OUTPUT),
        ),
        settings=settings,
    )


def AudioFilter(**settings: Any) -> BlockSpec:
    """Apply audio filtering (EQ, highpass, lowpass, etc.)."""
    return BlockSpec(
        block_type="AudioFilter",
        name="Audio Filter",
        input_ports=(
            PortSpec("audio_in", PortType.AUDIO, Direction.INPUT),
        ),
        output_ports=(
            PortSpec("audio_out", PortType.AUDIO, Direction.OUTPUT),
        ),
        settings=settings,
    )


def Classify(**settings: Any) -> BlockSpec:
    """Classify events using a PyTorch model."""
    return BlockSpec(
        block_type="PyTorchAudioClassify",
        name="Classify",
        input_ports=(
            PortSpec("events_in", PortType.EVENT, Direction.INPUT),
            PortSpec("audio_in", PortType.AUDIO, Direction.INPUT),
        ),
        output_ports=(
            PortSpec("events_out", PortType.EVENT, Direction.OUTPUT),
        ),
        settings=settings,
    )


def BinaryDrumClassify(**settings: Any) -> BlockSpec:
    """Classify drum onsets into one event layer per target drum class."""
    return BlockSpec(
        block_type="BinaryDrumClassify",
        name="Binary Drum Classify",
        input_ports=(
            PortSpec("audio_in", PortType.AUDIO, Direction.INPUT),
            PortSpec("events_in", PortType.EVENT, Direction.INPUT),
        ),
        output_ports=(
            PortSpec("events_out", PortType.EVENT, Direction.OUTPUT),
        ),
        settings=settings,
    )


def TranscribeNotes(**settings: Any) -> BlockSpec:
    """Transcribe notes from audio (pitch, duration, velocity)."""
    return BlockSpec(
        block_type="TranscribeNotes",
        name="Transcribe Notes",
        input_ports=(
            PortSpec("audio_in", PortType.AUDIO, Direction.INPUT),
        ),
        output_ports=(
            PortSpec("events_out", PortType.EVENT, Direction.OUTPUT),
        ),
        settings=settings,
    )


def ExportMA2(**settings: Any) -> BlockSpec:
    """Export events to grandMA2 timecode format."""
    return BlockSpec(
        block_type="ExportMA2",
        name="Export MA2",
        category=BlockCategory.PROCESSOR,
        input_ports=(
            PortSpec("events_in", PortType.EVENT, Direction.INPUT),
        ),
        output_ports=(),
        settings=settings,
    )


def ExportAudio(**settings: Any) -> BlockSpec:
    """Export audio to a directory in a specified format."""
    return BlockSpec(
        block_type="ExportAudio",
        name="Export Audio",
        category=BlockCategory.PROCESSOR,
        input_ports=(
            PortSpec("audio_in", PortType.AUDIO, Direction.INPUT),
        ),
        output_ports=(),
        settings=settings,
    )


def EQBands(**settings: Any) -> BlockSpec:
    """Multi-band parametric equalizer."""
    return BlockSpec(
        block_type="EQBands",
        name="EQ Bands",
        input_ports=(
            PortSpec("audio_in", PortType.AUDIO, Direction.INPUT),
        ),
        output_ports=(
            PortSpec("audio_out", PortType.AUDIO, Direction.OUTPUT),
        ),
        settings=settings,
    )


def AudioNegate(**settings: Any) -> BlockSpec:
    """Silence, attenuate, or subtract audio at event regions."""
    return BlockSpec(
        block_type="AudioNegate",
        name="Audio Negate",
        input_ports=(
            PortSpec("audio_in", PortType.AUDIO, Direction.INPUT),
            PortSpec("events_in", PortType.EVENT, Direction.INPUT),
        ),
        output_ports=(
            PortSpec("audio_out", PortType.AUDIO, Direction.OUTPUT),
        ),
        settings=settings,
    )


def ExportAudioDataset(**settings: Any) -> BlockSpec:
    """Extract audio clips at event times for ML dataset creation."""
    return BlockSpec(
        block_type="ExportAudioDataset",
        name="Export Audio Dataset",
        input_ports=(
            PortSpec("audio_in", PortType.AUDIO, Direction.INPUT),
            PortSpec("events_in", PortType.EVENT, Direction.INPUT),
        ),
        output_ports=(),
        settings=settings,
    )


def DatasetViewer(**settings: Any) -> BlockSpec:
    """Scan an exported dataset directory and show class distribution."""
    return BlockSpec(
        block_type="DatasetViewer",
        name="Dataset Viewer",
        category=BlockCategory.WORKSPACE,
        input_ports=(),
        output_ports=(),
        settings=settings,
    )
