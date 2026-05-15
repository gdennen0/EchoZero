"""
SongSectionsProcessor tests for detector-mode routing and output metadata.
Exists because extract-song-sections now supports multiple detection methods.
Tests prove method dispatch and generator metadata on emitted section cues.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import AudioData, Block, BlockSettings, Connection, Port
from echozero.execution import ExecutionContext
from echozero.processors.song_sections import SongSectionsProcessor
from echozero.processors.song_structure_mir import segment_song_structure_with_mir
from echozero.progress import RuntimeBus
from echozero.result import Ok


def _audio_out(name: str = "audio_out") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.OUTPUT)


def _audio_in(name: str = "audio_in") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.INPUT)


def _event_out(name: str = "events_out") -> Port:
    return Port(name=name, port_type=PortType.EVENT, direction=Direction.OUTPUT)


@dataclass(frozen=True)
class _StubSection:
    start_seconds: float
    cue_ref: str
    label: str
    confidence: float


def _make_graph(detect_method: str) -> Graph:
    graph = Graph()
    graph.add_block(
        Block(
            id="load",
            name="Load Audio",
            block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(_audio_out(),),
        )
    )
    graph.add_block(
        Block(
            id="detect",
            name="Detect Song Sections",
            block_type="DetectSongSections",
            category=BlockCategory.PROCESSOR,
            input_ports=(_audio_in(),),
            output_ports=(_event_out(),),
            settings=BlockSettings({"detect_method": detect_method}),
        )
    )
    graph.add_connection(Connection("load", "audio_out", "detect", "audio_in"))
    return graph


def _make_context(graph: Graph) -> ExecutionContext:
    return ExecutionContext(
        execution_id="song-sections-test",
        graph=graph,
        progress_bus=RuntimeBus(),
    )


def test_song_sections_processor_uses_default_detector_and_marks_generator() -> None:
    calls: list[dict[str, Any]] = []

    def _default_segment(*args):
        calls.append({"sample_rate": args[1], "n_mfcc": args[2]})
        return (
            _StubSection(start_seconds=0.0, cue_ref="intro_01", label="Intro", confidence=0.9),
        )

    def _determine_segment(*_args):
        raise AssertionError("determine-sections segmenter should not run for default mode")

    graph = _make_graph("mfcc_sequence_pooling")
    context = _make_context(graph)
    context.set_output(
        "load",
        "audio_out",
        AudioData(sample_rate=44100, duration=1.0, file_path="/tmp/song.wav", channel_count=1),
    )

    processor = SongSectionsProcessor(
        segment_song_sections_fn=_default_segment,
        determine_sections_segment_fn=_determine_segment,
    )
    result = processor.execute("detect", context)

    assert isinstance(result, Ok)
    assert len(calls) == 1
    event = result.value.layers[0].events[0]
    assert event.metadata["generator"] == "mfcc_sequence_pooling_v1"
    assert event.metadata["cue_ref"] == "part_01"
    assert event.metadata["section_label"] == "Part 1"
    assert event.classifications["label"] == "Part 1"


def test_song_sections_processor_uses_determine_sections_mode_and_marks_generator() -> None:
    def _default_segment(*_args):
        raise AssertionError("default segmenter should not run for determine-sections mode")

    def _determine_segment(*_args):
        return (
            _StubSection(start_seconds=0.0, cue_ref="intro_01", label="Intro", confidence=0.8),
        )

    graph = _make_graph("determine_sections_style")
    context = _make_context(graph)
    context.set_output(
        "load",
        "audio_out",
        AudioData(sample_rate=44100, duration=1.0, file_path="/tmp/song.wav", channel_count=1),
    )

    processor = SongSectionsProcessor(
        segment_song_sections_fn=_default_segment,
        determine_sections_segment_fn=_determine_segment,
    )
    result = processor.execute("detect", context)

    assert isinstance(result, Ok)
    event = result.value.layers[0].events[0]
    assert event.metadata["generator"] == "determine_sections_style_v1"
    assert event.metadata["cue_ref"] == "part_01"
    assert event.metadata["section_label"] == "Part 1"
    assert event.classifications["label"] == "Part 1"


def test_song_sections_processor_uses_mir_mode_and_marks_generator() -> None:
    def _default_segment(*_args):
        raise AssertionError("default segmenter should not run for mir mode")

    def _determine_segment(*_args):
        raise AssertionError("determine segmenter should not run for mir mode")

    def _mir_segment(*_args):
        return (
            _StubSection(start_seconds=0.0, cue_ref="intro_01", label="Intro", confidence=0.91),
        )

    graph = _make_graph("mir_self_similarity")
    context = _make_context(graph)
    context.set_output(
        "load",
        "audio_out",
        AudioData(sample_rate=44100, duration=1.0, file_path="/tmp/song.wav", channel_count=1),
    )

    processor = SongSectionsProcessor(
        segment_song_sections_fn=_default_segment,
        determine_sections_segment_fn=_determine_segment,
        mir_self_similarity_segment_fn=_mir_segment,
    )
    result = processor.execute("detect", context)

    assert isinstance(result, Ok)
    event = result.value.layers[0].events[0]
    assert event.metadata["generator"] == "mir_self_similarity_v1"
    assert event.metadata["cue_ref"] == "part_01"
    assert event.metadata["section_label"] == "Part 1"
    assert event.classifications["label"] == "Part 1"


def test_segment_song_structure_with_mir_finds_multiple_part_starts(tmp_path: Path) -> None:
    path = tmp_path / "multi-part.wav"
    sample_rate = 22050
    samples: list[int] = []
    for frequency_hz in (220.0, 440.0, 220.0, 660.0):
        frame_count = int(round(sample_rate * 3.0))
        for frame_index in range(frame_count):
            value = math.sin((2.0 * math.pi * frequency_hz * frame_index) / sample_rate) * 0.4
            samples.append(int(round(value * 32767.0)))

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(
            b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples)
        )

    segments = segment_song_structure_with_mir(
        file_path=str(path),
        sample_rate=sample_rate,
        n_mfcc=20,
        n_fft=2048,
        hop_length=512,
        boundary_sensitivity=0.6,
        min_section_seconds=1.0,
        max_sections=12,
        similarity_threshold=0.84,
        intro_tail_seconds=4.0,
        end_tail_seconds=4.0,
    )

    assert len(segments) >= 3
    assert segments[0].start_seconds == 0.0
    assert all(segments[index].start_seconds < segments[index + 1].start_seconds for index in range(len(segments) - 1))
