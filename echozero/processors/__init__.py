"""
Processors: BlockExecutor implementations for each block type.
Exists because execution logic is block-type-specific and must be testable in isolation.
Each processor implements the BlockExecutor protocol and is registered with the ExecutionEngine.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from echozero.processors.audio_filter import AudioFilterProcessor
from echozero.processors.audio_negate import AudioNegateProcessor
from echozero.processors.dataset_viewer import DatasetViewerProcessor
from echozero.processors.detect_note_contour import DetectNoteContourProcessor
from echozero.processors.detect_onsets import DetectOnsetsProcessor
from echozero.processors.eq_bands import EQBandsProcessor
from echozero.processors.export_audio import ExportAudioProcessor
from echozero.processors.export_audio_dataset import ExportAudioDatasetProcessor
from echozero.processors.export_ma2 import ExportMA2Processor
from echozero.processors.generate_waveform import GenerateWaveformProcessor
from echozero.processors.load_audio import LoadAudioProcessor
from echozero.processors.separate_audio import SeparateAudioProcessor
from echozero.processors.song_sections import SongSectionsProcessor
from echozero.processors.transcribe_notes import TranscribeNotesProcessor

if TYPE_CHECKING:
    from echozero.processors.binary_drum_classify import BinaryDrumClassifyProcessor
    from echozero.processors.pytorch_audio_classify import PyTorchAudioClassifyProcessor

__all__ = [
    "AudioFilterProcessor",
    "AudioNegateProcessor",
    "BinaryDrumClassifyProcessor",
    "DatasetViewerProcessor",
    "DetectNoteContourProcessor",
    "DetectOnsetsProcessor",
    "EQBandsProcessor",
    "ExportAudioProcessor",
    "ExportAudioDatasetProcessor",
    "ExportMA2Processor",
    "GenerateWaveformProcessor",
    "LoadAudioProcessor",
    "PyTorchAudioClassifyProcessor",
    "SeparateAudioProcessor",
    "SongSectionsProcessor",
    "TranscribeNotesProcessor",
]

_LAZY_EXPORTS = {
    "BinaryDrumClassifyProcessor": (
        "echozero.processors.binary_drum_classify",
        "BinaryDrumClassifyProcessor",
    ),
    "PyTorchAudioClassifyProcessor": (
        "echozero.processors.pytorch_audio_classify",
        "PyTorchAudioClassifyProcessor",
    ),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name), attr_name)
