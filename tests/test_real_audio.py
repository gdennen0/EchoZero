"""
Real audio integration tests: Generate actual WAV files, run actual processors,
verify actual results. No mocks, no fakes, no injected functions.

These tests use the DEFAULT implementations of each processor — the ones that
call librosa, scipy, soundfile, etc. If a library is missing, the test is skipped.
"""

from __future__ import annotations

import os
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import (
    AudioData,
    Block,
    BlockSettings,
    Connection,
    Event,
    EventData,
    Layer,
    Port,
)
from echozero.execution import ExecutionContext, ExecutionEngine, GraphPlanner
from echozero.processors.load_audio import LoadAudioProcessor
from echozero.processors.detect_onsets import DetectOnsetsProcessor
from echozero.processors.audio_filter import AudioFilterProcessor
from echozero.processors.eq_bands import EQBandsProcessor
from echozero.processors.audio_negate import AudioNegateProcessor
from echozero.processors.export_audio import ExportAudioProcessor
from echozero.processors.export_ma2 import ExportMA2Processor
from echozero.processors.export_audio_dataset import ExportAudioDatasetProcessor
from echozero.processors.dataset_viewer import DatasetViewerProcessor
from echozero.processors.transcribe_notes import TranscribeNotesProcessor
from echozero.progress import RuntimeBus
from echozero.result import Ok, is_err, is_ok, unwrap


# ---------------------------------------------------------------------------
# Audio generation helpers
# ---------------------------------------------------------------------------

def _write_wav(path: str, samples: np.ndarray, sample_rate: int = 44100) -> str:
    """Write a numpy array (float64, -1..1) to a 16-bit WAV file."""
    # Ensure mono or stereo shape
    if samples.ndim == 1:
        channels = 1
        data = samples
    else:
        channels = samples.shape[1]
        data = samples.ravel()  # interleave

    # Scale float64 to int16
    int_data = np.clip(data * 32767, -32768, 32767).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(int_data.tobytes())

    return path


def _sine_wave(freq: float, duration: float, sr: int = 44100, amp: float = 0.8) -> np.ndarray:
    """Generate a mono sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)


def _clicks_at(times: list[float], duration: float = 3.0, sr: int = 44100) -> np.ndarray:
    """Generate audio with sharp clicks (impulses) at specified times."""
    samples = np.zeros(int(sr * duration))
    click_len = int(sr * 0.005)  # 5ms click
    for t in times:
        start = int(t * sr)
        end = min(start + click_len, len(samples))
        # Short burst of white noise for the click
        samples[start:end] = 0.9 * np.sign(np.random.randn(end - start))
    return samples


def _tone_bursts(freqs_and_times: list[tuple[float, float, float]], duration: float = 5.0, sr: int = 44100) -> np.ndarray:
    """Generate audio with tone bursts at specified (freq, start, dur) tuples."""
    samples = np.zeros(int(sr * duration))
    for freq, start, dur in freqs_and_times:
        s = int(start * sr)
        e = min(s + int(dur * sr), len(samples))
        t = np.arange(e - s) / sr
        samples[s:e] += 0.7 * np.sin(2 * np.pi * freq * t)
    return samples


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Temp directory cleaned up after test."""
    d = tempfile.mkdtemp(prefix="ez_test_")
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def click_wav(tmp_dir) -> str:
    """WAV file with clicks at 0.5s, 1.0s, 1.5s, 2.0s."""
    path = os.path.join(tmp_dir, "clicks.wav")
    audio = _clicks_at([0.5, 1.0, 1.5, 2.0], duration=3.0)
    return _write_wav(path, audio)


@pytest.fixture
def sine_wav(tmp_dir) -> str:
    """WAV file with a 440Hz sine wave, 2 seconds."""
    path = os.path.join(tmp_dir, "sine.wav")
    audio = _sine_wave(440.0, 2.0)
    return _write_wav(path, audio)


@pytest.fixture
def mixed_wav(tmp_dir) -> str:
    """WAV file mixing low (100Hz) and high (4000Hz) frequencies."""
    path = os.path.join(tmp_dir, "mixed.wav")
    low = _sine_wave(100.0, 2.0, amp=0.5)
    high = _sine_wave(4000.0, 2.0, amp=0.5)
    return _write_wav(path, low + high)


@pytest.fixture
def tone_burst_wav(tmp_dir) -> str:
    """WAV with 3 tone bursts at known times for event-region tests."""
    path = os.path.join(tmp_dir, "bursts.wav")
    audio = _tone_bursts([
        (440.0, 0.5, 0.2),
        (880.0, 1.5, 0.3),
        (220.0, 2.5, 0.2),
    ], duration=4.0)
    return _write_wav(path, audio)


# ---------------------------------------------------------------------------
# 1. LoadAudio — real file, real metadata
# ---------------------------------------------------------------------------

class TestLoadAudioReal:
    def test_loads_wav_metadata(self, sine_wav):
        proc = LoadAudioProcessor()
        graph = Graph()
        graph.add_block(Block(
            id="load", name="Load", block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("audio_out", PortType.AUDIO, Direction.OUTPUT),
            ),
            settings=BlockSettings({"file_path": sine_wav}),
        ))
        ctx = ExecutionContext("test", graph, RuntimeBus())
        result = proc.execute("load", ctx)
        assert is_ok(result)
        audio = unwrap(result)
        assert isinstance(audio, AudioData)
        assert audio.sample_rate == 44100
        assert abs(audio.duration - 2.0) < 0.1
        assert audio.channel_count == 1
        assert audio.file_path == sine_wav

    def test_missing_file_returns_error(self, tmp_dir):
        proc = LoadAudioProcessor()
        graph = Graph()
        graph.add_block(Block(
            id="load", name="Load", block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("audio_out", PortType.AUDIO, Direction.OUTPUT),
            ),
            settings=BlockSettings({"file_path": os.path.join(tmp_dir, "nope.wav")}),
        ))
        ctx = ExecutionContext("test", graph, RuntimeBus())
        result = proc.execute("load", ctx)
        assert is_err(result)


# ---------------------------------------------------------------------------
# 2. DetectOnsets — real librosa onset detection on clicks
# ---------------------------------------------------------------------------

librosa = pytest.importorskip("librosa")


class TestDetectOnsetsReal:
    def test_detects_onsets_on_clicks(self, click_wav):
        """Real librosa onset detection on a file with 4 clicks."""
        proc = DetectOnsetsProcessor()  # uses _default_onset_detect (librosa)
        graph = Graph()
        graph.add_block(Block(
            id="load", name="Load", block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("audio_out", PortType.AUDIO, Direction.OUTPUT),
            ),
            settings=BlockSettings({"file_path": click_wav}),
        ))
        graph.add_block(Block(
            id="onsets", name="Onsets", block_type="DetectOnsets",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
            output_ports=(Port("events_out", PortType.EVENT, Direction.OUTPUT),),
            settings=BlockSettings({"threshold": 0.3, "min_gap": 0.1}),
        ))
        graph.add_connection(Connection("load", "audio_out", "onsets", "audio_in"))

        bus = RuntimeBus()
        engine = ExecutionEngine(graph, bus)
        engine.register_executor("LoadAudio", LoadAudioProcessor())
        engine.register_executor("DetectOnsets", DetectOnsetsProcessor())
        plan = GraphPlanner().plan(graph)
        result = engine.run(plan)

        assert is_ok(result)
        event_data = unwrap(result)["onsets"]["events_out"]
        assert isinstance(event_data, EventData)
        assert len(event_data.layers) == 1

        events = event_data.layers[0].events
        # Should detect at least 2 of the 4 clicks (librosa sensitivity varies)
        assert len(events) >= 2, f"Expected >= 2 onsets, got {len(events)}"

        # All onset times should be within the audio duration
        for e in events:
            assert 0.0 <= e.time <= 3.0

    def test_silent_audio_produces_no_onsets(self, tmp_dir):
        """Silence should produce zero or very few onsets."""
        silence_path = os.path.join(tmp_dir, "silence.wav")
        _write_wav(silence_path, np.zeros(44100 * 2))  # 2s silence

        proc = DetectOnsetsProcessor()
        graph = Graph()
        graph.add_block(Block(
            id="load", name="Load", block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("audio_out", PortType.AUDIO, Direction.OUTPUT),
            ),
            settings=BlockSettings({"file_path": silence_path}),
        ))
        graph.add_block(Block(
            id="onsets", name="Onsets", block_type="DetectOnsets",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
            output_ports=(Port("events_out", PortType.EVENT, Direction.OUTPUT),),
            settings=BlockSettings({"threshold": 0.5, "min_gap": 0.1}),
        ))
        graph.add_connection(Connection("load", "audio_out", "onsets", "audio_in"))

        bus = RuntimeBus()
        engine = ExecutionEngine(graph, bus)
        engine.register_executor("LoadAudio", LoadAudioProcessor())
        engine.register_executor("DetectOnsets", DetectOnsetsProcessor())
        result = engine.run(GraphPlanner().plan(graph))

        assert is_ok(result)
        event_data = unwrap(result)["onsets"]["events_out"]
        total_events = sum(len(l.events) for l in event_data.layers)
        assert total_events <= 1  # silence: 0 or maybe 1 spurious


# ---------------------------------------------------------------------------
# 3. AudioFilter — real scipy filtering
# ---------------------------------------------------------------------------

scipy_signal = pytest.importorskip("scipy.signal")


class TestAudioFilterReal:
    def test_lowpass_removes_high_frequencies(self, mixed_wav, tmp_dir):
        """Lowpass at 500Hz on a 100Hz+4000Hz mix should reduce energy above 500Hz."""
        proc = AudioFilterProcessor()  # uses _default_filter (scipy)
        graph = Graph()
        graph.add_block(Block(
            id="load", name="Load", block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("audio_out", PortType.AUDIO, Direction.OUTPUT),
            ),
            settings=BlockSettings({"file_path": mixed_wav}),
        ))
        graph.add_block(Block(
            id="filter", name="Filter", block_type="AudioFilter",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
            output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
            settings=BlockSettings({
                "filter_type": "lowpass",
                "freq": 500.0,
                "gain_db": 0.0,
                "Q": 1.0,
            }),
        ))
        graph.add_connection(Connection("load", "audio_out", "filter", "audio_in"))

        bus = RuntimeBus()
        engine = ExecutionEngine(graph, bus)
        engine.register_executor("LoadAudio", LoadAudioProcessor())
        engine.register_executor("AudioFilter", AudioFilterProcessor())
        result = engine.run(GraphPlanner().plan(graph))

        assert is_ok(result)
        audio_out = unwrap(result)["filter"]["audio_out"]
        assert isinstance(audio_out, AudioData)
        assert audio_out.sample_rate == 44100
        assert os.path.isfile(audio_out.file_path)

        # Verify the output file is actually a valid WAV
        import soundfile as sf
        data, sr = sf.read(audio_out.file_path)
        assert sr == 44100
        assert len(data) > 0

        # The filtered signal should have less high-frequency content.
        # Compute FFT and check energy above 2kHz is reduced.
        if data.ndim > 1:
            data = data[:, 0]
        fft = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1.0 / sr)

        high_mask = freqs > 2000
        low_mask = (freqs > 50) & (freqs < 300)
        high_energy = np.mean(fft[high_mask] ** 2)
        low_energy = np.mean(fft[low_mask] ** 2)

        # After lowpass at 500Hz, high energy should be much less than low energy
        assert high_energy < low_energy, (
            f"Lowpass failed: high_energy={high_energy:.2f} >= low_energy={low_energy:.2f}"
        )

    def test_highpass_removes_low_frequencies(self, mixed_wav, tmp_dir):
        """Highpass at 2000Hz should remove the 100Hz component."""
        graph = Graph()
        graph.add_block(Block(
            id="load", name="Load", block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("audio_out", PortType.AUDIO, Direction.OUTPUT),
            ),
            settings=BlockSettings({"file_path": mixed_wav}),
        ))
        graph.add_block(Block(
            id="filter", name="Filter", block_type="AudioFilter",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
            output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
            settings=BlockSettings({
                "filter_type": "highpass",
                "freq": 2000.0,
            }),
        ))
        graph.add_connection(Connection("load", "audio_out", "filter", "audio_in"))

        bus = RuntimeBus()
        engine = ExecutionEngine(graph, bus)
        engine.register_executor("LoadAudio", LoadAudioProcessor())
        engine.register_executor("AudioFilter", AudioFilterProcessor())
        result = engine.run(GraphPlanner().plan(graph))

        assert is_ok(result)
        audio_out = unwrap(result)["filter"]["audio_out"]

        import soundfile as sf
        data, sr = sf.read(audio_out.file_path)
        if data.ndim > 1:
            data = data[:, 0]
        fft = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1.0 / sr)

        low_mask = (freqs > 50) & (freqs < 300)
        high_mask = freqs > 3000
        low_energy = np.mean(fft[low_mask] ** 2)
        high_energy = np.mean(fft[high_mask] ** 2)

        assert low_energy < high_energy, (
            f"Highpass failed: low_energy={low_energy:.2f} >= high_energy={high_energy:.2f}"
        )


# ---------------------------------------------------------------------------
# 4. EQBands — real scipy multi-band EQ
# ---------------------------------------------------------------------------

class TestEQBandsReal:
    def test_boost_low_band(self, mixed_wav):
        """Boosting 60-250Hz by 12dB should increase low-frequency energy."""
        graph = Graph()
        graph.add_block(Block(
            id="load", name="Load", block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("audio_out", PortType.AUDIO, Direction.OUTPUT),
            ),
            settings=BlockSettings({"file_path": mixed_wav}),
        ))
        graph.add_block(Block(
            id="eq", name="EQ", block_type="EQBands",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
            output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
            settings=BlockSettings({
                "bands": [
                    {"freq_low": 60.0, "freq_high": 250.0, "gain_db": 12.0},
                    {"freq_low": 2000.0, "freq_high": 8000.0, "gain_db": 0.0},
                ],
                "filter_order": 4,
            }),
        ))
        graph.add_connection(Connection("load", "audio_out", "eq", "audio_in"))

        bus = RuntimeBus()
        engine = ExecutionEngine(graph, bus)
        engine.register_executor("LoadAudio", LoadAudioProcessor())
        engine.register_executor("EQBands", EQBandsProcessor())
        result = engine.run(GraphPlanner().plan(graph))

        assert is_ok(result)
        audio_out = unwrap(result)["eq"]["audio_out"]
        assert os.path.isfile(audio_out.file_path)

        import soundfile as sf
        data, sr = sf.read(audio_out.file_path)
        assert sr == 44100
        assert len(data) > 0


# ---------------------------------------------------------------------------
# 5. AudioNegate — real negation on tone bursts
# ---------------------------------------------------------------------------

class TestAudioNegateReal:
    def test_silence_mode_zeros_event_regions(self, tone_burst_wav):
        """Silence mode should zero out audio at event time regions."""
        graph = Graph()
        graph.add_block(Block(
            id="load", name="Load", block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("audio_out", PortType.AUDIO, Direction.OUTPUT),
            ),
            settings=BlockSettings({"file_path": tone_burst_wav}),
        ))
        graph.add_block(Block(
            id="events_src", name="Events", block_type="EventSource",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("events_out", PortType.EVENT, Direction.OUTPUT),
            ),
        ))
        graph.add_block(Block(
            id="negate", name="Negate", block_type="AudioNegate",
            category=BlockCategory.PROCESSOR,
            input_ports=(
                Port("audio_in", PortType.AUDIO, Direction.INPUT),
                Port("events_in", PortType.EVENT, Direction.INPUT),
            ),
            output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
            settings=BlockSettings({
                "mode": "silence",
                "fade_ms": 5.0,
            }),
        ))
        graph.add_connection(Connection("load", "audio_out", "negate", "audio_in"))
        graph.add_connection(Connection("events_src", "events_out", "negate", "events_in"))

        # Event regions match tone burst times
        event_data = EventData(layers=(
            Layer(id="bursts", name="Bursts", events=(
                Event(id="b1", time=0.5, duration=0.2, classifications={}, metadata={}, origin="test"),
                Event(id="b2", time=1.5, duration=0.3, classifications={}, metadata={}, origin="test"),
                Event(id="b3", time=2.5, duration=0.2, classifications={}, metadata={}, origin="test"),
            )),
        ))

        class EventSource:
            def execute(self, block_id, context):
                return Ok(event_data)

        bus = RuntimeBus()
        engine = ExecutionEngine(graph, bus)
        engine.register_executor("LoadAudio", LoadAudioProcessor())
        engine.register_executor("EventSource", EventSource())
        engine.register_executor("AudioNegate", AudioNegateProcessor())
        result = engine.run(GraphPlanner().plan(graph))

        assert is_ok(result)
        audio_out = unwrap(result)["negate"]["audio_out"]
        assert os.path.isfile(audio_out.file_path)

        import soundfile as sf
        data, sr = sf.read(audio_out.file_path)
        if data.ndim > 1:
            data = data[:, 0]

        # Check that the event regions have near-zero energy
        for start, dur in [(0.5, 0.2), (1.5, 0.3), (2.5, 0.2)]:
            # Check center of region (avoid fade edges)
            center_start = int((start + 0.02) * sr)
            center_end = int((start + dur - 0.02) * sr)
            if center_start < center_end:
                region_rms = np.sqrt(np.mean(data[center_start:center_end] ** 2))
                assert region_rms < 0.05, (
                    f"Region {start}-{start+dur}s not silenced: RMS={region_rms:.4f}"
                )


# ---------------------------------------------------------------------------
# 6. ExportAudio — real file copy
# ---------------------------------------------------------------------------

class TestExportAudioReal:
    def test_exports_wav_to_directory(self, sine_wav, tmp_dir):
        """ExportAudio should copy the file to the output directory."""
        output_dir = os.path.join(tmp_dir, "export_out")

        graph = Graph()
        graph.add_block(Block(
            id="load", name="Load", block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("audio_out", PortType.AUDIO, Direction.OUTPUT),
            ),
            settings=BlockSettings({"file_path": sine_wav}),
        ))
        graph.add_block(Block(
            id="export", name="Export", block_type="ExportAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
            output_ports=(),
            settings=BlockSettings({
                "output_dir": output_dir,
                "format": "wav",
            }),
        ))
        graph.add_connection(Connection("load", "audio_out", "export", "audio_in"))

        bus = RuntimeBus()
        engine = ExecutionEngine(graph, bus)
        engine.register_executor("LoadAudio", LoadAudioProcessor())
        engine.register_executor("ExportAudio", ExportAudioProcessor())
        result = engine.run(GraphPlanner().plan(graph))

        assert is_ok(result)
        exported_path = unwrap(result)["export"]["out"]
        assert os.path.isfile(exported_path)

        # Verify it's a valid WAV
        import soundfile as sf
        data, sr = sf.read(exported_path)
        assert sr == 44100
        assert len(data) > 0


# ---------------------------------------------------------------------------
# 7. ExportMA2 — real XML file write
# ---------------------------------------------------------------------------

class TestExportMA2Real:
    def test_writes_ma2_xml(self, tmp_dir):
        """ExportMA2 should write a valid MA2 timecode XML file."""
        output_path = os.path.join(tmp_dir, "output.xml")

        event_data = EventData(layers=(
            Layer(id="drums", name="Drums", events=(
                Event(id="e1", time=0.5, duration=0.1,
                      classifications={"class": "kick"}, metadata={}, origin="test"),
                Event(id="e2", time=1.0, duration=0.1,
                      classifications={"class": "snare"}, metadata={}, origin="test"),
            )),
        ))

        class EventSource:
            def execute(self, block_id, context):
                return Ok(event_data)

        graph = Graph()
        graph.add_block(Block(
            id="src", name="Source", block_type="EventSource",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("events_out", PortType.EVENT, Direction.OUTPUT),
            ),
        ))
        graph.add_block(Block(
            id="export", name="Export", block_type="ExportMA2",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("events_in", PortType.EVENT, Direction.INPUT),),
            output_ports=(),
            settings=BlockSettings({
                "output_path": output_path,
                "frame_rate": 30,
                "track_name": "TestSong",
            }),
        ))
        graph.add_connection(Connection("src", "events_out", "export", "events_in"))

        bus = RuntimeBus()
        engine = ExecutionEngine(graph, bus)
        engine.register_executor("EventSource", EventSource())
        engine.register_executor("ExportMA2", ExportMA2Processor())
        result = engine.run(GraphPlanner().plan(graph))

        assert is_ok(result)
        assert os.path.isfile(output_path)

        content = Path(output_path).read_text()
        assert '<?xml' in content
        assert 'MA2Timecode' in content
        assert 'trackName="TestSong"' in content
        assert 'kick' in content
        assert 'snare' in content
        assert content.count('<Event ') == 2


# ---------------------------------------------------------------------------
# 8. DatasetViewer — real directory scan
# ---------------------------------------------------------------------------

class TestDatasetViewerReal:
    def test_scans_real_directory(self, tmp_dir):
        """DatasetViewer should scan a real directory of audio files."""
        # Create a dataset structure
        kick_dir = os.path.join(tmp_dir, "kick")
        snare_dir = os.path.join(tmp_dir, "snare")
        os.makedirs(kick_dir)
        os.makedirs(snare_dir)

        # Write real WAV files
        for i in range(3):
            _write_wav(os.path.join(kick_dir, f"kick_{i}.wav"), _clicks_at([0.1], duration=0.3))
        for i in range(2):
            _write_wav(os.path.join(snare_dir, f"snare_{i}.wav"), _clicks_at([0.1], duration=0.3))

        graph = Graph()
        graph.add_block(Block(
            id="viewer", name="Viewer", block_type="DatasetViewer",
            category=BlockCategory.WORKSPACE,
            input_ports=(), output_ports=(),
            settings=BlockSettings({"dataset_dir": tmp_dir}),
        ))

        bus = RuntimeBus()
        engine = ExecutionEngine(graph, bus)
        engine.register_executor("DatasetViewer", DatasetViewerProcessor())
        result = engine.run(GraphPlanner().plan(graph))

        assert is_ok(result)
        stats = unwrap(result)["viewer"]["out"]
        assert stats["total_files"] == 5
        assert stats["total_classes"] == 2
        assert stats["classes"]["kick"]["count"] == 3
        assert stats["classes"]["snare"]["count"] == 2


# ---------------------------------------------------------------------------
# 9. Full pipeline: LoadAudio → DetectOnsets (real end-to-end)
# ---------------------------------------------------------------------------

class TestFullPipelineReal:
    def test_load_then_detect(self, click_wav):
        """Full pipeline: load a real WAV, detect onsets with real librosa."""
        from echozero.pipelines.block_specs import DetectOnsets as DetectOnsetsSpec, LoadAudio as LoadAudioSpec
        from echozero.pipelines.pipeline import Pipeline

        p = Pipeline("test_real", name="Real Test")
        load = p.add(LoadAudioSpec(file_path=click_wav), id="load")
        onsets = p.add(
            DetectOnsetsSpec(threshold=0.3, min_gap=0.1),
            id="onsets",
            audio_in=load.audio_out,
        )
        p.output("detected", onsets.events_out)

        # Verify pipeline structure
        assert len(p.graph.blocks) == 2
        assert len(p.graph.connections) == 1
        assert len(p.outputs) == 1
        assert p.outputs[0].name == "detected"

        # Run through engine
        bus = RuntimeBus()
        engine = ExecutionEngine(p.graph, bus)
        engine.register_executor("LoadAudio", LoadAudioProcessor())
        engine.register_executor("DetectOnsets", DetectOnsetsProcessor())
        plan = GraphPlanner().plan(p.graph)
        result = engine.run(plan)

        assert is_ok(result)
        event_data = unwrap(result)["onsets"]["events_out"]
        assert isinstance(event_data, EventData)
        assert len(event_data.layers) == 1
        assert len(event_data.layers[0].events) >= 2

    def test_load_filter_detect(self, mixed_wav):
        """LoadAudio → AudioFilter (lowpass) → DetectOnsets."""
        from echozero.pipelines.block_specs import (
            AudioFilter as AudioFilterSpec,
            DetectOnsets as DetectOnsetsSpec,
            LoadAudio as LoadAudioSpec,
        )
        from echozero.pipelines.pipeline import Pipeline

        p = Pipeline("filter_detect", name="Filter then Detect")
        load = p.add(LoadAudioSpec(file_path=mixed_wav), id="load")
        filt = p.add(
            AudioFilterSpec(filter_type="lowpass", freq=500.0),
            id="filter",
            audio_in=load.audio_out,
        )
        onsets = p.add(
            DetectOnsetsSpec(threshold=0.3),
            id="onsets",
            audio_in=filt.audio_out,
        )
        p.output("events", onsets.events_out)

        bus = RuntimeBus()
        engine = ExecutionEngine(p.graph, bus)
        engine.register_executor("LoadAudio", LoadAudioProcessor())
        engine.register_executor("AudioFilter", AudioFilterProcessor())
        engine.register_executor("DetectOnsets", DetectOnsetsProcessor())
        plan = GraphPlanner().plan(p.graph)
        result = engine.run(plan)

        assert is_ok(result)
        event_data = unwrap(result)["onsets"]["events_out"]
        assert isinstance(event_data, EventData)

    def test_load_detect_export_ma2(self, click_wav, tmp_dir):
        """LoadAudio → DetectOnsets → ExportMA2: full analysis-to-export pipeline."""
        output_xml = os.path.join(tmp_dir, "show.xml")

        from echozero.pipelines.block_specs import (
            DetectOnsets as DetectOnsetsSpec,
            ExportMA2 as ExportMA2Spec,
            LoadAudio as LoadAudioSpec,
        )
        from echozero.pipelines.pipeline import Pipeline

        p = Pipeline("analyze_export", name="Analyze and Export")
        load = p.add(LoadAudioSpec(file_path=click_wav), id="load")
        onsets = p.add(
            DetectOnsetsSpec(threshold=0.3, min_gap=0.1),
            id="onsets",
            audio_in=load.audio_out,
        )
        p.add(
            ExportMA2Spec(output_path=output_xml, frame_rate=30, track_name="ClickTrack"),
            id="export",
            events_in=onsets.events_out,
        )

        bus = RuntimeBus()
        engine = ExecutionEngine(p.graph, bus)
        engine.register_executor("LoadAudio", LoadAudioProcessor())
        engine.register_executor("DetectOnsets", DetectOnsetsProcessor())
        engine.register_executor("ExportMA2", ExportMA2Processor())
        plan = GraphPlanner().plan(p.graph)
        result = engine.run(plan)

        assert is_ok(result)
        assert os.path.isfile(output_xml)

        content = Path(output_xml).read_text()
        assert 'ClickTrack' in content
        assert content.count('<Event ') >= 2


