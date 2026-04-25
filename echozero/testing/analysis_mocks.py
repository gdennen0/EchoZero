import math
import wave
from pathlib import Path

import echozero.pipelines.templates  # noqa: F401
from echozero.domain.types import AudioData, Event as DomainEvent, EventData, Layer as DomainLayer
from echozero.execution import ExecutionContext
from echozero.pipelines.registry import get_registry
from echozero.result import ok
from echozero.services.orchestrator import Orchestrator


def _write_pcm16_mono_wav(path: Path, samples: list[int], sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))
    return path


def write_test_wav(path: Path, frames: int = 4410, sample_rate: int = 44100) -> Path:
    return _write_pcm16_mono_wav(path, [0] * frames, sample_rate)


def write_test_tone_wav(
    path: Path,
    *,
    frequency_hz: float = 220.0,
    duration_seconds: float = 1.0,
    sample_rate: int = 44100,
    amplitude: float = 0.4,
) -> Path:
    total_frames = max(1, int(round(duration_seconds * sample_rate)))
    clamped_amplitude = max(0.0, min(float(amplitude), 0.95))
    samples = [
        int(
            round(
                math.sin((2.0 * math.pi * float(frequency_hz) * frame_index) / float(sample_rate))
                * clamped_amplitude
                * 32767.0
            )
        )
        for frame_index in range(total_frames)
    ]
    return _write_pcm16_mono_wav(path, samples, sample_rate)


def write_test_model(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-model")
    return path


class _MockLoadAudioExecutor:
    def execute(self, block_id: str, context: ExecutionContext):
        block = context.graph.blocks[block_id]
        return ok(
            AudioData(
                sample_rate=44100,
                duration=0.1,
                file_path=str(block.settings["file_path"]),
                channel_count=1,
            )
        )


class _MockSeparateAudioExecutor:
    def execute(self, block_id: str, context: ExecutionContext):
        audio = context.get_input(block_id, "audio_in", AudioData)
        assert audio is not None
        base = Path(audio.file_path).parent
        stems = {}
        stem_specs = {
            "drums": (110.0, 0.55),
            "bass": (196.0, 0.35),
            "vocals": (329.63, 0.25),
            "other": (523.25, 0.2),
        }
        for name, (frequency_hz, amplitude) in stem_specs.items():
            stem_path = write_test_tone_wav(
                base / f"{name}.wav",
                frequency_hz=frequency_hz,
                amplitude=amplitude,
                duration_seconds=max(float(audio.duration), 1.0),
            )
            stems[f"{name}_out"] = AudioData(
                sample_rate=44100,
                duration=0.1,
                file_path=str(stem_path),
                channel_count=1,
            )
        return ok(stems)


class _MockAudioFilterExecutor:
    def execute(self, block_id: str, context: ExecutionContext):
        audio = context.get_input(block_id, "audio_in", AudioData)
        assert audio is not None
        base = Path(str(audio.file_path)).parent
        filtered_path = write_test_tone_wav(
            base / f"{block_id}.wav",
            frequency_hz=110.0 if "kick" in block_id else 220.0,
            amplitude=0.35,
            duration_seconds=max(float(audio.duration), 0.1),
            sample_rate=int(audio.sample_rate),
        )
        return ok(
            AudioData(
                sample_rate=int(audio.sample_rate),
                duration=float(audio.duration),
                file_path=str(filtered_path),
                channel_count=int(audio.channel_count),
            )
        )


class _MockDetectOnsetsExecutor:
    def execute(self, _block_id: str, _context: ExecutionContext):
        event = DomainEvent(
            id="evt_1",
            time=0.25,
            duration=0.05,
            classifications={"namespace:onset": "hit"},
            metadata={},
            origin="detect_onsets",
        )
        return ok(
            EventData(
                layers=(
                    DomainLayer(
                        id="layer_onsets",
                        name="Onsets",
                        events=(event,),
                    ),
                )
            )
        )


class _MockClassifyExecutor:
    def execute(self, block_id: str, context: ExecutionContext):
        event_data = context.get_input(block_id, "events_in", EventData)
        assert event_data is not None
        classified_layers: list[DomainLayer] = []
        for layer in event_data.layers:
            classified_events: list[DomainEvent] = []
            for event in layer.events:
                classified_events.append(
                    DomainEvent(
                        id=event.id,
                        time=event.time,
                        duration=event.duration,
                        classifications={"class": "kick", "confidence": "0.99"},
                        metadata={**event.metadata, "classified": True},
                        origin="classify",
                    )
                )
            classified_layers.append(DomainLayer(id=layer.id, name="Kick", events=tuple(classified_events)))
        return ok(EventData(layers=tuple(classified_layers)))


class _MockBinaryDrumClassifyExecutor:
    def execute(self, block_id: str, context: ExecutionContext):
        block = context.graph.blocks[block_id]
        target_class = str(block.settings.get("target_class", "")).strip().lower()
        input_events = _merged_binary_drum_input_events(block_id, context)
        kick_events: list[DomainEvent] = []
        snare_events: list[DomainEvent] = []
        for event in input_events:
            kick_events.append(
                DomainEvent(
                    id=f"{event.id}_kick",
                    time=event.time,
                    duration=event.duration,
                    classifications={"class": "kick", "confidence": "0.99"},
                    metadata={**event.metadata, "classified": True},
                    origin="binary_classify:kick",
                )
            )
            snare_events.append(
                DomainEvent(
                    id=f"{event.id}_snare",
                    time=event.time + 0.1,
                    duration=event.duration,
                    classifications={"class": "snare", "confidence": "0.97"},
                    metadata={**event.metadata, "classified": True},
                    origin="binary_classify:snare",
                )
            )
        if target_class == "kick":
            return ok(EventData(layers=(DomainLayer(id="kick", name="kick", events=tuple(kick_events)),)))
        if target_class == "snare":
            return ok(
                EventData(layers=(DomainLayer(id="snare", name="snare", events=tuple(snare_events)),))
            )
        return ok(
            EventData(
                layers=(
                    DomainLayer(id="kick", name="kick", events=tuple(kick_events)),
                    DomainLayer(id="snare", name="snare", events=tuple(snare_events)),
                )
            )
        )


def build_mock_orchestrator() -> Orchestrator:
    return Orchestrator(
        get_registry(),
        {
            "LoadAudio": _MockLoadAudioExecutor(),
            "SeparateAudio": _MockSeparateAudioExecutor(),
            "AudioFilter": _MockAudioFilterExecutor(),
            "DetectOnsets": _MockDetectOnsetsExecutor(),
            "PyTorchAudioClassify": _MockClassifyExecutor(),
            "BinaryDrumClassify": _MockBinaryDrumClassifyExecutor(),
        },
    )


build_mock_analysis_service = build_mock_orchestrator


def _merged_binary_drum_input_events(
    block_id: str,
    context: ExecutionContext,
) -> tuple[DomainEvent, ...]:
    event_batches: list[EventData] = []
    for port_name in ("events_in", "kick_events_in", "snare_events_in"):
        event_data = context.get_input(block_id, port_name, EventData)
        if event_data is not None:
            event_batches.append(event_data)

    assert event_batches
    merged: list[DomainEvent] = []
    seen: set[tuple[str, float, float]] = set()
    for event_data in event_batches:
        for layer in event_data.layers:
            for event in layer.events:
                key = (str(event.id), float(event.time), float(event.duration))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(event)
    return tuple(merged)
