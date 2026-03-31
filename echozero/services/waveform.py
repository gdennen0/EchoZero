"""
WaveformService: Auto-generates waveform peaks for newly imported audio.
Exists because waveform generation must happen on every song add/version update
through the real engine (FP1), but ProjectStorage is persistence — it shouldn't
hold an engine reference. This service bridges the gap.

Called by the application layer after import_song or add_song_version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import (
    AudioData,
    Block,
    BlockSettings,
    Connection,
    Port,
    WaveformData,
)
from echozero.execution import ExecutionContext, ExecutionEngine, GraphPlanner
from echozero.persistence.entities import SongVersionRecord
from echozero.processors.generate_waveform import GenerateWaveformProcessor
from echozero.processors.load_audio import AudioFileInfo, LoadAudioProcessor
from echozero.progress import RuntimeBus
from echozero.result import Ok, is_ok, unwrap


def generate_waveform_for_version(
    version: SongVersionRecord,
    audio_dir: Path,
    runtime_bus: RuntimeBus | None = None,
    load_samples_fn: Callable[[str, int], np.ndarray] | None = None,
    audio_info_fn: Callable[[str], AudioFileInfo] | None = None,
) -> WaveformData | None:
    """Run the waveform pipeline for a song version through the real engine.

    Builds a minimal pipeline: LoadAudio → GenerateWaveform.
    Returns WaveformData on success, None on failure.

    Args:
        version: The SongVersionRecord to generate waveforms for.
        audio_dir: Base directory to resolve version.audio_file against.
        runtime_bus: Optional RuntimeBus for progress reporting.
        load_samples_fn: Injectable sample loader for testing.
        audio_info_fn: Injectable audio info reader for testing.
    """
    bus = runtime_bus or RuntimeBus()
    audio_path = str(audio_dir / version.audio_file)

    # Build a one-off graph: LoadAudio → GenerateWaveform
    graph = Graph()
    graph.add_block(Block(
        id="load",
        name="Load Audio",
        block_type="LoadAudio",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
        settings=BlockSettings({"file_path": audio_path}),
    ))
    graph.add_block(Block(
        id="waveform",
        name="Generate Waveform",
        block_type="GenerateWaveform",
        category=BlockCategory.PROCESSOR,
        input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
        output_ports=(Port("waveform_out", PortType.WAVEFORM, Direction.OUTPUT),),
    ))
    graph.add_connection(Connection("load", "audio_out", "waveform", "audio_in"))

    # Register executors
    engine = ExecutionEngine(graph, bus)
    engine.register_executor("LoadAudio", LoadAudioProcessor(audio_info_fn=audio_info_fn))
    engine.register_executor("GenerateWaveform", GenerateWaveformProcessor(
        load_samples_fn=load_samples_fn,
    ))

    # Plan and run
    planner = GraphPlanner()
    plan = planner.plan(graph)
    result = engine.run(plan)

    if is_ok(result):
        outputs = unwrap(result)
        waveform_output = outputs.get("waveform", {})
        if isinstance(waveform_output, dict):
            return waveform_output.get("waveform_out")
    return None
