"""
main: Creates a fully wired Project with real processor executors.

This is the production entry point. Tests use Project.create() directly
with mock executors. The UI calls these functions to start the app.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from echozero.execution import BlockExecutor
from echozero.persistence.entities import ProjectSettingsRecord
from echozero.pipelines.registry import get_registry
from echozero.project import Project

logger = logging.getLogger(__name__)


def get_default_executors() -> dict[str, BlockExecutor]:
    """Build the default executor map with all real processors.

    Each processor is imported lazily to avoid heavy imports at startup.
    """
    executors: dict[str, BlockExecutor] = {}

    # Core processors — always available
    from echozero.processors.load_audio import LoadAudioProcessor
    from echozero.processors.detect_onsets import DetectOnsetsProcessor

    executors["LoadAudio"] = LoadAudioProcessor()
    executors["DetectOnsets"] = DetectOnsetsProcessor()

    # Optional processors — import errors are non-fatal
    _optional = [
        ("SeparateAudio", "echozero.processors.separate_audio", "SeparateAudioProcessor"),
        ("AudioFilter", "echozero.processors.audio_filter", "AudioFilterProcessor"),
        ("AudioNegate", "echozero.processors.audio_negate", "AudioNegateProcessor"),
        ("EQBands", "echozero.processors.eq_bands", "EQBandsProcessor"),
        ("ExportAudio", "echozero.processors.export_audio", "ExportAudioProcessor"),
        ("ExportMA2", "echozero.processors.export_ma2", "ExportMA2Processor"),
        ("ExportAudioDataset", "echozero.processors.export_audio_dataset", "ExportAudioDatasetProcessor"),
        ("GenerateWaveform", "echozero.processors.generate_waveform", "GenerateWaveformProcessor"),
        ("TranscribeNotes", "echozero.processors.transcribe_notes", "TranscribeNotesProcessor"),
        ("PyTorchAudioClassify", "echozero.processors.pytorch_audio_classify", "PyTorchAudioClassifyProcessor"),
        ("DetectSongSections", "echozero.processors.song_sections", "SongSectionsProcessor"),
        ("DatasetViewer", "echozero.processors.dataset_viewer", "DatasetViewerProcessor"),
    ]

    for block_type, module_path, class_name in _optional:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            executors[block_type] = cls()
        except Exception as exc:
            logger.debug("Optional processor %s not available: %s", block_type, exc)

    return executors


def create_project(
    name: str,
    settings: ProjectSettingsRecord | None = None,
    working_dir_root: Path | None = None,
) -> Project:
    """Create a new project with all real processors wired up."""
    return Project.create(
        name=name,
        settings=settings,
        executors=get_default_executors(),
        registry=get_registry(),
        working_dir_root=working_dir_root,
    )


def open_project(
    ez_path: Path,
    working_dir_root: Path | None = None,
) -> Project:
    """Open an existing .ez project with all real processors."""
    return Project.open(
        ez_path=ez_path,
        executors=get_default_executors(),
        registry=get_registry(),
        working_dir_root=working_dir_root,
    )


def open_project_db(
    working_dir: Path,
) -> Project:
    """Open from working directory with all real processors."""
    return Project.open_db(
        working_dir=working_dir,
        executors=get_default_executors(),
        registry=get_registry(),
    )
