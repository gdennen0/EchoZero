from __future__ import annotations

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import AudioData, Block, BlockSettings, Connection, Port
from echozero.execution import ExecutionContext
from echozero.processors.detect_note_contour import (
    DetectNoteContourProcessor,
    PitchFrame,
    note_color_hex,
)
from echozero.progress import RuntimeBus
from echozero.result import is_ok, unwrap


def _audio_out(name: str = "audio_out") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.OUTPUT)


def _audio_in(name: str = "audio_in") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.INPUT)


def _event_out(name: str = "event_out") -> Port:
    return Port(name=name, port_type=PortType.EVENT, direction=Direction.OUTPUT)


def _context(*, block_id: str = "detect_note_contour") -> ExecutionContext:
    graph = Graph()
    load_block = Block(
        id="load_audio",
        name="Load Audio",
        block_type="LoadAudio",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(_audio_out(),),
        settings=BlockSettings({"file_path": "/tmp/source.wav"}),
    )
    detect_block = Block(
        id=block_id,
        name="Detect Note Contour",
        block_type="DetectNoteContour",
        category=BlockCategory.PROCESSOR,
        input_ports=(_audio_in(),),
        output_ports=(_event_out("events_out"),),
        settings=BlockSettings(
            {
                "frame_length": 4096,
                "hop_length": 1024,
                "min_note_midi": 36,
                "max_note_midi": 60,
                "min_note_length": 0.05,
            }
        ),
    )
    graph.add_block(load_block)
    graph.add_block(detect_block)
    graph.add_connection(
        Connection(
            source_block_id="load_audio",
            source_output_name="audio_out",
            target_block_id=block_id,
            target_input_name="audio_in",
        )
    )
    context = ExecutionContext(
        execution_id="test-run",
        graph=graph,
        progress_bus=RuntimeBus(),
    )
    context.set_output(
        "load_audio",
        "audio_out",
        AudioData(
            file_path="/tmp/source.wav",
            sample_rate=44100,
            channel_count=1,
            duration=1.0,
        ),
    )
    return context


def test_detect_note_contour_processor_groups_frames_into_note_segments():
    observed: list[tuple[str, int, int, int, float, float]] = []

    def _pitch_track(
        file_path: str,
        sample_rate: int,
        frame_length: int,
        hop_length: int,
        min_frequency_hz: float,
        max_frequency_hz: float,
    ) -> list[PitchFrame]:
        observed.append(
            (
                file_path,
                sample_rate,
                frame_length,
                hop_length,
                round(min_frequency_hz, 3),
                round(max_frequency_hz, 3),
            )
        )
        return [
            PitchFrame(0.00, 65.406),
            PitchFrame(0.05, 65.406),
            PitchFrame(0.10, 82.407),
            PitchFrame(0.15, 82.407),
        ]

    processor = DetectNoteContourProcessor(pitch_track_fn=_pitch_track)
    result = processor.execute("detect_note_contour", _context())

    assert is_ok(result)
    event_data = unwrap(result)
    assert observed
    assert len(event_data.layers) == 1
    layer = event_data.layers[0]
    assert layer.name == "Notes"
    assert [event.classifications["note"] for event in layer.events] == ["C2", "E2"]
    assert layer.events[0].metadata["color"] == note_color_hex("C2")
    assert layer.events[1].metadata["color"] == note_color_hex("E2")
    assert layer.events[0].metadata["detection"]["midi_note"] == 36
    assert layer.events[1].metadata["detection"]["midi_note"] == 40
