"""
BinaryDrumClassifyProcessor tests: verify per-class layer output from drum onset classification.
Exists because the app-facing classified-drum pipeline must produce separate kick/snare layers.
Tests assert on layer splitting, validation, and audio/event input requirements.
"""

from __future__ import annotations

from typing import Any

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import AudioData, Block, BlockSettings, Connection, Event, EventData, Layer, Port
from echozero.execution import ExecutionContext
from echozero.processors.binary_drum_classify import BinaryDrumClassifyProcessor
from echozero.progress import RuntimeBus
from echozero.result import Err, Ok


def _event_in(name: str = "events_in") -> Port:
    return Port(name=name, port_type=PortType.EVENT, direction=Direction.INPUT)


def _audio_in(name: str = "audio_in") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.INPUT)


def _event_out(name: str = "events_out") -> Port:
    return Port(name=name, port_type=PortType.EVENT, direction=Direction.OUTPUT)


def _make_graph() -> Graph:
    graph = Graph()
    graph.add_block(
        Block(
            id="source",
            name="Onsets",
            block_type="DetectOnsets",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(_event_out(),),
        )
    )
    graph.add_block(
        Block(
            id="load",
            name="Audio",
            block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
        )
    )
    graph.add_block(
        Block(
            id="binary",
            name="Binary Drum Classify",
            block_type="BinaryDrumClassify",
            category=BlockCategory.PROCESSOR,
            input_ports=(_audio_in(), _event_in()),
            output_ports=(_event_out(),),
            settings=BlockSettings(
                {
                    "kick_model_path": "/models/kick.manifest.json",
                    "snare_model_path": "/models/snare.manifest.json",
                }
            ),
        )
    )
    graph.add_connection(Connection("source", "events_out", "binary", "events_in"))
    graph.add_connection(Connection("load", "audio_out", "binary", "audio_in"))
    return graph


def _make_context(graph: Graph) -> ExecutionContext:
    return ExecutionContext(
        execution_id="binary-test",
        graph=graph,
        progress_bus=RuntimeBus(),
    )


def _mock_binary_classify(
    events: list[Event],
    audio_file: str,
    model_paths: dict[str, str],
    device: str,
    positive_threshold: float,
) -> dict[str, list[tuple[Event, float]]]:
    return {
        "kick": [
            (
                Event(
                    id=f"{events[0].id}_kick",
                    time=events[0].time,
                    duration=events[0].duration,
                    classifications={"class": "kick", "confidence": 0.9},
                    metadata={"classified": True},
                    origin="test:kick",
                ),
                0.9,
            )
        ],
        "snare": [
            (
                Event(
                    id=f"{events[1].id}_snare",
                    time=events[1].time,
                    duration=events[1].duration,
                    classifications={"class": "snare", "confidence": 0.8},
                    metadata={"classified": True},
                    origin="test:snare",
                ),
                0.8,
            )
        ],
    }


def test_binary_drum_classify_processor_returns_kick_and_snare_layers() -> None:
    graph = _make_graph()
    context = _make_context(graph)
    context.set_output(
        "source",
        "events_out",
        EventData(
            layers=(
                Layer(
                    id="onsets",
                    name="Onsets",
                    events=(
                        Event(id="evt1", time=0.25, duration=0.05, classifications={}, metadata={}, origin="src"),
                        Event(id="evt2", time=0.50, duration=0.05, classifications={}, metadata={}, origin="src"),
                    ),
                ),
            )
        ),
    )
    context.set_output(
        "load",
        "audio_out",
        AudioData(sample_rate=44100, duration=1.0, file_path="/tmp/drums.wav", channel_count=1),
    )

    processor = BinaryDrumClassifyProcessor(classify_fn=_mock_binary_classify)
    result = processor.execute("binary", context)

    assert isinstance(result, Ok)
    output = result.value
    assert [layer.name for layer in output.layers] == ["kick", "snare"]
    assert output.layers[0].events[0].classifications["class"] == "kick"
    assert output.layers[1].events[0].classifications["class"] == "snare"


def test_binary_drum_classify_processor_requires_audio_input() -> None:
    graph = _make_graph()
    context = _make_context(graph)
    context.set_output("source", "events_out", EventData(layers=(Layer(id="onsets", name="Onsets", events=()),)))

    processor = BinaryDrumClassifyProcessor(classify_fn=_mock_binary_classify)
    result = processor.execute("binary", context)

    assert isinstance(result, Err)
    assert "audio_in" in str(result.error)
