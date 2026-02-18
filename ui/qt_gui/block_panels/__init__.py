"""
Block panel system for EchoZero Qt GUI.

Provides standardized, dockable UI panels for individual blocks.
Each block type can register a custom panel with block-specific settings.
"""

from ui.qt_gui.block_panels.panel_registry import (
    register_block_panel,
    get_panel_class,
    is_panel_registered,
    BLOCK_PANEL_REGISTRY
)
from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase

# Import concrete panels to trigger registration
from ui.qt_gui.block_panels.separator_panel import SeparatorPanel
from ui.qt_gui.block_panels.generic_panel import GenericBlockPanel

# Cloud separator panels
from ui.qt_gui.block_panels.separator_cloud_panel import SeparatorCloudPanel

# Phase 2 custom panels
from ui.qt_gui.block_panels.load_audio_panel import LoadAudioPanel
from ui.qt_gui.block_panels.export_audio_panel import ExportAudioPanel
from ui.qt_gui.block_panels.transcribe_note_panel import TranscribeNotePanel
from ui.qt_gui.block_panels.detect_onsets_panel import DetectOnsetsPanel
from ui.qt_gui.block_panels.learned_onset_detector_panel import LearnedOnsetDetectorPanel
from ui.qt_gui.block_panels.editor_panel import EditorPanel
from ui.qt_gui.block_panels.tensorflow_classify_panel import TensorFlowClassifyPanel
from ui.qt_gui.block_panels.pytorch_audio_trainer_panel import PyTorchAudioTrainerPanel
from ui.qt_gui.block_panels.learned_onset_trainer_panel import LearnedOnsetTrainerPanel
from ui.qt_gui.block_panels.pytorch_audio_classify_panel import PyTorchAudioClassifyPanel
from ui.qt_gui.block_panels.show_manager_panel import ShowManagerPanel
from ui.qt_gui.block_panels.audio_filter_panel import AudioFilterPanel
from ui.qt_gui.block_panels.audio_negate_panel import AudioNegatePanel
from ui.qt_gui.block_panels.eq_bands_panel import EQBandsPanel
from ui.qt_gui.block_panels.export_audio_dataset_panel import ExportAudioDatasetPanel
from ui.qt_gui.block_panels.dataset_viewer_panel import DatasetViewerPanel

__all__ = [
    'BlockPanelBase',
    'register_block_panel',
    'get_panel_class',
    'is_panel_registered',
    'BLOCK_PANEL_REGISTRY',
    'SeparatorPanel',
    'SeparatorCloudPanel',
    'GenericBlockPanel',
    'LoadAudioPanel',
    'ExportAudioPanel',
    'TranscribeNotePanel',
    'DetectOnsetsPanel',
    'LearnedOnsetDetectorPanel',
    'EditorPanel',
    'TensorFlowClassifyPanel',
    'PyTorchAudioTrainerPanel',
    'LearnedOnsetTrainerPanel',
    'PyTorchAudioClassifyPanel',
    'ShowManagerPanel',
    'AudioFilterPanel',
    'AudioNegatePanel',
    'EQBandsPanel',
    'ExportAudioDatasetPanel',
    'DatasetViewerPanel',
]

