"""
Tests for SeparateAudioProcessor.
Uses mock separation functions — no Demucs or PyTorch required.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from uuid import uuid4
from typing import Any

import pytest

from echozero.domain.enums import BlockCategory, BlockState, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import AudioData, Block, BlockSettings, Connection, Port
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext, ExecutionEngine, ExecutionPlan, GraphPlanner
from echozero.processors.separate_audio import (
    DEMUCS_MODELS,
    SeparateAudioProcessor,
    StemResult,
    _detect_device,
    resolve_demucs_model_name,
)
from echozero.progress import ProgressReport, RuntimeBus
from echozero.result import Err, Ok, is_err, is_ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEPARATE_AUDIO_TMP_ROOT = Path(__file__).resolve().parent / ".local-separate-audio-tmp"
_SEPARATE_AUDIO_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _next_tmp_dir() -> Path:
    path = _SEPARATE_AUDIO_TMP_ROOT / uuid4().hex
    path.mkdir()
    return path


def _audio_in_port() -> Port:
    return Port(name="audio_in", port_type=PortType.AUDIO, direction=Direction.INPUT)


def _audio_out_port(name: str) -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.OUTPUT)


def _make_load_block(block_id: str = "load", file_path: str = "test.wav") -> Block:
    return Block(
        id=block_id,
        name="Load Audio",
        block_type="LoadAudio",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(_audio_out_port("audio_out"),),
        settings=BlockSettings({"file_path": file_path}),
    )


def _make_separator_block(
    block_id: str = "sep",
    model: str = "htdemucs",
    device: str = "cpu",
    shifts: int = 1,
    two_stems: str | None = None,
    include_drums_stem_layer: bool = False,
    include_bass_stem_layer: bool = False,
    include_vocals_stem_layer: bool = False,
    include_other_stem_layer: bool = False,
    output_format: str = "wav",
    mp3_bitrate: int = 320,
    working_dir: str | None = None,
) -> Block:
    settings_working_dir = working_dir or str(_next_tmp_dir())
    settings: dict[str, Any] = {
        "model": model,
        "device": device,
        "shifts": shifts,
        "output_format": output_format,
        "mp3_bitrate": mp3_bitrate,
        "working_dir": settings_working_dir,
        "include_drums_stem_layer": include_drums_stem_layer,
        "include_bass_stem_layer": include_bass_stem_layer,
        "include_vocals_stem_layer": include_vocals_stem_layer,
        "include_other_stem_layer": include_other_stem_layer,
    }
    if two_stems is not None:
        settings["two_stems"] = two_stems

    return Block(
        id=block_id,
        name="Separator",
        block_type="SeparateAudio",
        category=BlockCategory.PROCESSOR,
        input_ports=(_audio_in_port(),),
        output_ports=(
            _audio_out_port("drums_out"),
            _audio_out_port("bass_out"),
            _audio_out_port("other_out"),
            _audio_out_port("vocals_out"),
        ),
        settings=BlockSettings(settings),
    )


def _mock_separate_fn(
    input_file: str,
    model_name: str,
    device: str,
    shifts: int,
    two_stems: str | None,
    output_dir: str,
    output_format: str,
    mp3_bitrate: int,
) -> list[StemResult]:
    """Mock separation that writes tiny files and returns StemResults."""
    ext = "mp3" if output_format == "mp3" else "wav"
    model_info = DEMUCS_MODELS.get(model_name, DEMUCS_MODELS["htdemucs"])

    if two_stems:
        stem_names = [two_stems, f"no_{two_stems}"]
    else:
        stem_names = list(model_info["stems"])

    results = []
    for name in stem_names:
        stem_path = os.path.join(output_dir, f"{name}.{ext}")
        # Write a tiny fake file
        with open(stem_path, "wb") as f:
            f.write(b"\x00" * 100)
        results.append(StemResult(
            name=name,
            file_path=stem_path,
            sample_rate=44100,
            duration=180.0,
            channel_count=2,
        ))
    return results


def _failing_separate_fn(*args: Any, **kwargs: Any) -> list[StemResult]:
    """Mock separation that always fails."""
    raise RuntimeError("GPU exploded")


def _empty_separate_fn(*args: Any, **kwargs: Any) -> list[StemResult]:
    """Mock separation that returns no stems."""
    return []


def _build_graph(blocks: list[Block], connections: list[Connection]) -> Graph:
    """Build a Graph using the add_block/add_connection API."""
    graph = Graph()
    for block in blocks:
        graph.add_block(block)
    for conn in connections:
        graph.add_connection(conn)
    return graph


def _make_context_with_audio(
    separator_block: Block,
    audio: AudioData | None = None,
) -> tuple[ExecutionContext, RuntimeBus]:
    """Build a minimal execution context with audio pre-loaded from a fake upstream block."""
    load_block = _make_load_block()
    connection = Connection(
        source_block_id=load_block.id,
        source_output_name="audio_out",
        target_block_id=separator_block.id,
        target_input_name="audio_in",
    )
    graph = _build_graph([load_block, separator_block], [connection])
    bus = RuntimeBus()
    ctx = ExecutionContext(
        execution_id="test-exec",
        graph=graph,
        progress_bus=bus,
    )
    # Pre-load the upstream audio output
    if audio is None:
        audio = AudioData(sample_rate=44100, duration=180.0, file_path="test.wav", channel_count=2)
    ctx.set_output(load_block.id, "audio_out", audio)
    return ctx, bus


# ---------------------------------------------------------------------------
# Tests: Basic separation
# ---------------------------------------------------------------------------


class TestSeparateAudioBasic:
    """Core happy-path tests."""

    def test_4_stem_separation(self):
        """Default 4-stem mode returns drums, bass, other, vocals."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block()
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_ok(result)
        outputs = result.value
        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == {"drums_out", "bass_out", "other_out", "vocals_out"}

        for port_name, audio_data in outputs.items():
            assert isinstance(audio_data, AudioData)
            assert audio_data.sample_rate == 44100
            assert audio_data.duration == 180.0
            assert audio_data.channel_count == 2
            assert os.path.exists(audio_data.file_path)

    def test_2_stem_separation(self):
        """two_stems='vocals' returns vocals_out and no_vocals_out."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block(two_stems="vocals")
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_ok(result)
        outputs = result.value
        assert set(outputs.keys()) == {"vocals_out", "no_vocals_out"}

    def test_2_stem_drums(self):
        """two_stems='drums' returns drums_out and no_drums_out."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block(two_stems="drums")
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_ok(result)
        assert set(result.value.keys()) == {"drums_out", "no_drums_out"}

    def test_non_drum_stem_selection_forces_full_stem_separation(self):
        """Requesting bass/vocals/other stems overrides two_stems to full separation."""
        captured: dict[str, Any] = {}

        def capture_fn(
            input_file, model_name, device, shifts, two_stems,
            output_dir, output_format, mp3_bitrate,
        ):
            captured["two_stems"] = two_stems
            return _mock_separate_fn(
                input_file, model_name, device, shifts, two_stems,
                output_dir, output_format, mp3_bitrate,
            )

        proc = SeparateAudioProcessor(separate_fn=capture_fn)
        sep = _make_separator_block(two_stems="drums", include_bass_stem_layer=True)
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_ok(result)
        assert captured["two_stems"] is None
        assert set(result.value.keys()) == {"drums_out", "bass_out", "other_out", "vocals_out"}

    def test_two_stems_none_string_is_treated_as_full_separation(self):
        """two_stems='none' is normalized to full-stem separation."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block(two_stems="none")
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_ok(result)
        assert set(result.value.keys()) == {"drums_out", "bass_out", "other_out", "vocals_out"}

    def test_6_stem_model(self):
        """htdemucs_6s returns 6 stems including guitar and piano."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block(model="htdemucs_6s")
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_ok(result)
        outputs = result.value
        assert "guitar_out" in outputs
        assert "piano_out" in outputs
        assert len(outputs) == 6

    def test_settings_passed_to_fn(self):
        """Verify all settings are forwarded to the separation function."""
        captured: dict[str, Any] = {}

        def capture_fn(
            input_file, model_name, device, shifts, two_stems,
            output_dir, output_format, mp3_bitrate,
        ):
            captured["input_file"] = input_file
            captured["model_name"] = model_name
            captured["device"] = device
            captured["shifts"] = shifts
            captured["two_stems"] = two_stems
            captured["output_format"] = output_format
            captured["mp3_bitrate"] = mp3_bitrate
            return _mock_separate_fn(
                input_file, model_name, device, shifts, two_stems,
                output_dir, output_format, mp3_bitrate,
            )

        proc = SeparateAudioProcessor(separate_fn=capture_fn)
        sep = _make_separator_block(
            model="htdemucs_ft", device="cpu", shifts=5,
            two_stems="bass", output_format="mp3", mp3_bitrate=192,
        )
        ctx, _ = _make_context_with_audio(sep)
        proc.execute(sep.id, ctx)

        assert captured["model_name"] == "htdemucs_ft"
        assert captured["device"] == "cpu"
        assert captured["shifts"] == 5
        assert captured["two_stems"] == "bass"
        assert captured["output_format"] == "mp3"
        assert captured["mp3_bitrate"] == 192

    def test_output_files_exist(self):
        """All output AudioData file_paths point to real files."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block()
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_ok(result)
        for audio_data in result.value.values():
            assert os.path.isfile(audio_data.file_path)

    def test_latest_model_alias_resolves_to_concrete_model(self):
        """latest_model settings are coerced to a concrete Demucs model before execution."""
        captured: dict[str, Any] = {}

        def capture_fn(
            input_file, model_name, device, shifts, two_stems,
            output_dir, output_format, mp3_bitrate,
        ):
            captured["model_name"] = model_name
            return _mock_separate_fn(
                input_file,
                model_name,
                device,
                shifts,
                two_stems,
                output_dir,
                output_format,
                mp3_bitrate,
            )

        proc = SeparateAudioProcessor(separate_fn=capture_fn)
        sep = _make_separator_block(model="latest_model")
        ctx, _ = _make_context_with_audio(sep)
        result = proc.execute(sep.id, ctx)

        assert is_ok(result)
        assert captured["model_name"] == "htdemucs_ft"


# ---------------------------------------------------------------------------
# Tests: Validation errors
# ---------------------------------------------------------------------------


class TestSeparateAudioValidation:
    """Settings validation."""

    def test_no_audio_input(self):
        """Missing audio input returns error."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block()
        # Build context WITHOUT pre-loaded audio
        graph = _build_graph([sep], [])
        ctx = ExecutionContext(
            execution_id="test", graph=graph, progress_bus=RuntimeBus()
        )

        result = proc.execute(sep.id, ctx)
        assert is_err(result)
        assert "no audio input" in str(result.error).lower()

    def test_unknown_model(self):
        """Invalid model name returns validation error."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block(model="nonexistent_model")
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_err(result)
        assert isinstance(result.error, ValidationError)

    def test_unknown_device(self):
        """Invalid device returns validation error."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block(device="tpu")
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_err(result)
        assert isinstance(result.error, ValidationError)

    def test_invalid_two_stems(self):
        """Invalid two_stems value returns validation error."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block(two_stems="guitar")  # not valid for 4-stem models
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_err(result)
        assert isinstance(result.error, ValidationError)

    def test_negative_shifts(self):
        """Negative shifts returns validation error."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block(shifts=-1)
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_err(result)
        assert isinstance(result.error, ValidationError)

    def test_invalid_output_format(self):
        """Invalid output format returns validation error."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block(output_format="flac")
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_err(result)

    def test_invalid_mp3_bitrate(self):
        """Invalid MP3 bitrate returns validation error."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block(mp3_bitrate=256)
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_err(result)

    def test_block_not_in_graph(self):
        """Block ID not in graph returns error."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block()
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute("nonexistent_block", ctx)
        assert is_err(result)


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestSeparateAudioErrors:
    """Separation function failures."""

    def test_separation_failure(self):
        """Separation function raising returns error result."""
        proc = SeparateAudioProcessor(separate_fn=_failing_separate_fn)
        sep = _make_separator_block()
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_err(result)
        assert "GPU exploded" in str(result.error)

    def test_empty_stems(self):
        """Separation returning no stems returns error."""
        proc = SeparateAudioProcessor(separate_fn=_empty_separate_fn)
        sep = _make_separator_block()
        ctx, _ = _make_context_with_audio(sep)

        result = proc.execute(sep.id, ctx)
        assert is_err(result)
        assert "no stems" in str(result.error).lower()


# ---------------------------------------------------------------------------
# Tests: Progress reporting
# ---------------------------------------------------------------------------


class TestSeparateAudioProgress:
    """Progress events are published."""

    def test_progress_reports_published(self):
        """Separation publishes start, mid, and complete progress reports."""
        proc = SeparateAudioProcessor(separate_fn=_mock_separate_fn)
        sep = _make_separator_block()
        ctx, bus = _make_context_with_audio(sep)

        reports: list[Any] = []
        bus.subscribe(lambda r: reports.append(r))

        proc.execute(sep.id, ctx)

        progress_reports = [r for r in reports if isinstance(r, ProgressReport)]
        assert len(progress_reports) >= 3  # start, mid, complete

        # First is 0%, last is 100%
        assert progress_reports[0].percent == 0.0
        assert progress_reports[-1].percent == 1.0
        assert "complete" in progress_reports[-1].message.lower()


# ---------------------------------------------------------------------------
# Tests: Engine integration
# ---------------------------------------------------------------------------


class TestSeparateAudioEngineIntegration:
    """Full pipeline: LoadAudio → SeparateAudio through the engine."""

    def test_load_then_separate(self):
        """Full pipeline execution with mock processors."""
        # Build graph: LoadAudio → SeparateAudio
        load_block = _make_load_block(file_path="test.wav")
        sep_block = _make_separator_block()
        connection = Connection(
            source_block_id=load_block.id,
            source_output_name="audio_out",
            target_block_id=sep_block.id,
            target_input_name="audio_in",
        )
        graph = _build_graph([load_block, sep_block], [connection])

        bus = RuntimeBus()
        engine = ExecutionEngine(graph, bus)

        # Mock LoadAudio executor
        class MockLoadAudio:
            def execute(self, block_id, context):
                return Ok(AudioData(
                    sample_rate=44100, duration=180.0,
                    file_path="test.wav", channel_count=2,
                ))

        engine.register_executor("LoadAudio", MockLoadAudio())
        engine.register_executor("SeparateAudio", SeparateAudioProcessor(
            separate_fn=_mock_separate_fn
        ))

        planner = GraphPlanner()
        plan = planner.plan(graph)

        result = engine.run(plan)
        assert is_ok(result)

        outputs = result.value
        assert "sep" in outputs
        sep_outputs = outputs["sep"]
        assert isinstance(sep_outputs, dict)
        assert "drums_out" in sep_outputs
        assert "vocals_out" in sep_outputs


# ---------------------------------------------------------------------------
# Tests: Model catalog
# ---------------------------------------------------------------------------


class TestDemucsModels:
    """Model catalog metadata."""

    def test_all_models_have_required_fields(self):
        for name, info in DEMUCS_MODELS.items():
            assert "description" in info, f"{name} missing description"
            assert "quality" in info, f"{name} missing quality"
            assert "speed" in info, f"{name} missing speed"
            assert "stems" in info, f"{name} missing stems"
            assert len(info["stems"]) >= 4, f"{name} has fewer than 4 stems"

    def test_htdemucs_6s_has_6_stems(self):
        assert len(DEMUCS_MODELS["htdemucs_6s"]["stems"]) == 6
        assert "guitar" in DEMUCS_MODELS["htdemucs_6s"]["stems"]
        assert "piano" in DEMUCS_MODELS["htdemucs_6s"]["stems"]

    def test_default_model_exists(self):
        assert "htdemucs" in DEMUCS_MODELS

    def test_latest_model_alias_points_to_valid_model(self):
        assert resolve_demucs_model_name("latest_model") in DEMUCS_MODELS


# ---------------------------------------------------------------------------
# Tests: Device detection
# ---------------------------------------------------------------------------


class TestDeviceDetection:
    """Device auto-detection."""

    def test_explicit_cpu(self):
        assert _detect_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        assert _detect_device("cuda") == "cuda"

    def test_auto_returns_string(self):
        # auto resolves to either cuda or cpu
        result = _detect_device("auto")
        assert result in ("cpu", "cuda")
