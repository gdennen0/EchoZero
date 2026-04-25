from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path
from uuid import uuid4

import pytest
import numpy as np

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import AudioData, Block, BlockSettings, Connection, Event, EventData, Layer, Port
from echozero.errors import ValidationError
from echozero.execution import ExecutionContext
from echozero.inference_eval.runtime_preflight import checkpoint_contract_fingerprint, run_runtime_preflight
from echozero.processors.pytorch_audio_classify import PyTorchAudioClassifyProcessor, _default_classify
from echozero.progress import RuntimeBus
from echozero.result import Err
from echozero.runtime_models import loader as runtime_loader
from echozero.runtime_models.architectures import SimpleCnnRuntimeModel


_TEST_TMP_ROOT = Path(__file__).resolve().parents[2] / ".processor-test-tmp"


@pytest.fixture
def local_tmp_path() -> Path:
    _TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = _TEST_TMP_ROOT / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _event_in() -> Port:
    return Port(name="events_in", port_type=PortType.EVENT, direction=Direction.INPUT)


def _event_out() -> Port:
    return Port(name="events_out", port_type=PortType.EVENT, direction=Direction.OUTPUT)


def _audio_in() -> Port:
    return Port(name="audio_in", port_type=PortType.AUDIO, direction=Direction.INPUT)


def _audio_out() -> Port:
    return Port(name="audio_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT)


def _make_graph(model_path: str, *, include_audio: bool = False) -> Graph:
    graph = Graph()
    onset_block = Block(
        id="onset1",
        name="Detect Onsets",
        block_type="DetectOnsets",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(_event_out(),),
    )
    classify_block = Block(
        id="classify1",
        name="PyTorch Audio Classify",
        block_type="PyTorchAudioClassify",
        category=BlockCategory.PROCESSOR,
        input_ports=(_event_in(), _audio_in()),
        output_ports=(_event_out(),),
        settings=BlockSettings({"model_path": model_path}),
    )
    graph.add_block(onset_block)
    graph.add_block(classify_block)
    if include_audio:
        graph.add_block(
            Block(
                id="load1",
                name="Load Audio",
                block_type="LoadAudio",
                category=BlockCategory.PROCESSOR,
                input_ports=(),
                output_ports=(_audio_out(),),
            )
        )
    graph.add_connection(
        Connection(
            source_block_id="onset1",
            source_output_name="events_out",
            target_block_id="classify1",
            target_input_name="events_in",
        )
    )
    if include_audio:
        graph.add_connection(
            Connection(
                source_block_id="load1",
                source_output_name="audio_out",
                target_block_id="classify1",
                target_input_name="audio_in",
            )
        )
    return graph


def _make_context(graph: Graph) -> ExecutionContext:
    return ExecutionContext(
        execution_id="test-run",
        graph=graph,
        progress_bus=RuntimeBus(),
    )


def _events() -> EventData:
    return EventData(
        layers=(
            Layer(
                id="onsets",
                name="Detected Onsets",
                events=(
                    Event(id="event_1", time=0.5, duration=0.1, classifications={}, metadata={}, origin="onset1"),
                    Event(id="event_2", time=2.0, duration=0.1, classifications={}, metadata={}, origin="onset1"),
                    Event(id="event_3", time=4.0, duration=0.1, classifications={}, metadata={}, origin="onset1"),
                ),
            ),
        )
    )


def _checkpoint() -> dict[str, object]:
    return {
        "model_state_dict": {},
        "classes": ["kick", "snare"],
        "classification_mode": "multiclass",
        "preprocessing": {
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "trainer": "cnn_melspec_v1",
    }


def _manifest(model_name: str, *, fingerprint: str | None = None) -> dict[str, object]:
    if fingerprint is None:
        fingerprint = checkpoint_contract_fingerprint(_checkpoint())
    return {
        "schema": "foundry.artifact_manifest.v1",
        "weightsPath": model_name,
        "sharedContractFingerprint": fingerprint,
        "classes": ["kick", "snare"],
        "classificationMode": "multiclass",
        "runtime": {"consumer": "PyTorchAudioClassify"},
        "inferencePreprocessing": {
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
    }


def _write_manifest(directory: Path, payload: dict[str, object]) -> Path:
    path = directory / "art_test.manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_wav(path: Path, *, seconds: float = 0.25, sample_rate: int = 22050) -> Path:
    frames = max(1, int(round(seconds * sample_rate)))
    timeline = np.linspace(0.0, seconds, frames, endpoint=False)
    samples = (0.3 * np.sin(2.0 * np.pi * 220.0 * timeline) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return path


def _write_runtime_model(local_tmp_path: Path) -> Path:
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    model_path = local_tmp_path / "model.pth"
    classes = ["kick", "snare"]
    model = SimpleCnnRuntimeModel(num_classes=len(classes))
    state_dict = model.state_dict()
    for tensor in state_dict.values():
        tensor.zero_()
    state_dict["classifier.3.bias"] = torch.tensor([0.1, 1.5], dtype=torch.float32)
    checkpoint = {
        "model_state_dict": state_dict,
        "classes": classes,
        "classification_mode": "multiclass",
        "preprocessing": {
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "trainer": "cnn_melspec_v1",
    }
    torch.save(checkpoint, model_path)
    _write_manifest(
        local_tmp_path,
        _manifest(model_path.name, fingerprint=checkpoint_contract_fingerprint(checkpoint)),
    )
    return model_path


def test_runtime_preflight_rejects_missing_manifest(local_tmp_path: Path) -> None:
    model_path = local_tmp_path / "model.pth"
    model_path.write_bytes(b"weights")

    with pytest.raises(ValidationError) as exc_info:
        run_runtime_preflight(model_path, _checkpoint())

    message = str(exc_info.value)
    assert (
        "artifact manifest is required for runtime preflight; no *.manifest.json files were found"
        in message
    )
    assert f"requested model path: {model_path.resolve()}" in message


def test_runtime_preflight_rejects_invalid_runtime_bundle(local_tmp_path: Path) -> None:
    model_path = local_tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    payload = _manifest(model_path.name)
    payload["runtime"] = {"consumer": "OtherProcessor"}
    payload["inferencePreprocessing"] = {
        "sampleRate": 22050,
        "maxLength": 22050,
        "nFft": 2048,
        "nMels": 128,
        "fmax": 8000,
    }
    _write_manifest(local_tmp_path, payload)

    with pytest.raises(
        ValidationError,
        match=(
            "Runtime bundle preflight failed for model\\.pth: "
            "manifest\\.inferencePreprocessing missing keys: hopLength; "
            "manifest\\.runtime\\.consumer must match the validated consumer"
        ),
    ):
        run_runtime_preflight(model_path, _checkpoint())


def test_runtime_preflight_rejects_fingerprint_mismatch(local_tmp_path: Path) -> None:
    model_path = local_tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    _write_manifest(local_tmp_path, _manifest(model_path.name, fingerprint="bad-fingerprint"))

    with pytest.raises(
        ValidationError,
        match=(
            "Runtime bundle preflight failed for model\\.pth: "
            "manifest\\.sharedContractFingerprint must match the checkpoint-derived shared contract fingerprint"
        ),
    ):
        run_runtime_preflight(model_path, _checkpoint())


def test_runtime_preflight_rejects_class_order_mismatch(local_tmp_path: Path) -> None:
    model_path = local_tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    payload = _manifest(model_path.name)
    payload["classes"] = ["snare", "kick"]
    _write_manifest(local_tmp_path, payload)

    with pytest.raises(
        ValidationError,
        match=(
            "Runtime bundle preflight failed for model\\.pth: "
            "manifest\\.classes must match checkpoint class map order"
        ),
    ):
        run_runtime_preflight(model_path, _checkpoint())


def test_runtime_preflight_rejects_preprocessing_mismatch_against_checkpoint(local_tmp_path: Path) -> None:
    model_path = local_tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    payload = _manifest(model_path.name)
    payload["inferencePreprocessing"]["hopLength"] = 256
    _write_manifest(local_tmp_path, payload)

    with pytest.raises(
        ValidationError,
        match=(
            "Runtime bundle preflight failed for model\\.pth: "
            "manifest\\.inferencePreprocessing\\.hopLength must match checkpoint preprocessing"
        ),
    ):
        run_runtime_preflight(model_path, _checkpoint())


def test_runtime_preflight_rejects_legacy_checkpoint_missing_manifest_required_metadata(local_tmp_path: Path) -> None:
    model_path = local_tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    legacy_checkpoint = {
        "model_state_dict": {},
        "preprocessing": {
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
        },
        "trainer": "cnn_melspec_v1",
    }
    _write_manifest(
        local_tmp_path,
        _manifest(model_path.name, fingerprint=checkpoint_contract_fingerprint(legacy_checkpoint)),
    )

    with pytest.raises(ValidationError) as exc_info:
        run_runtime_preflight(model_path, legacy_checkpoint)

    message = str(exc_info.value)
    assert "checkpoint.classes must be present when validating against an artifact manifest" in message
    assert "checkpoint preprocessing missing keys required for manifest verification: fmax" in message
    assert "checkpoint classification mode must be present when validating against an artifact manifest" in message


def test_runtime_preflight_rejects_missing_shared_contract_fingerprint(local_tmp_path: Path) -> None:
    model_path = local_tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    payload = _manifest(model_path.name)
    payload.pop("sharedContractFingerprint", None)
    _write_manifest(local_tmp_path, payload)

    with pytest.raises(
        ValidationError,
        match=(
            "Runtime bundle preflight failed for model\\.pth: "
            "manifest\\.sharedContractFingerprint is required"
        ),
    ):
        run_runtime_preflight(model_path, _checkpoint())


def test_default_classify_runs_preflight_once_per_model_load(
    local_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _write_runtime_model(local_tmp_path)
    audio_path = _write_wav(local_tmp_path / "audio.wav")

    calls: list[tuple[Path, dict[str, object]]] = []

    def spy_preflight(model: str | Path, loaded_checkpoint: dict[str, object], *, consumer: str = "PyTorchAudioClassify") -> None:
        assert consumer == "PyTorchAudioClassify"
        calls.append((Path(model), loaded_checkpoint))

    monkeypatch.setattr(runtime_loader, "run_runtime_preflight", spy_preflight)

    classified = _default_classify(
        list(_events().layers[0].events),
        audio_file=str(audio_path),
        model_path=str(model_path),
        device="cpu",
        batch_size=2,
    )

    assert len(classified) == 3
    assert len(calls) == 1
    assert calls[0][0] == model_path
    assert all(event.classifications["class"] == "snare" for event in classified)
    assert all(event.metadata["source_model"] == "model.pth" for event in classified)
    assert all(event.metadata["model_artifact"]["schema"] == "echozero.model_artifact_ref.v1" for event in classified)
    assert all(
        event.metadata["model_artifact"]["artifactIdentity"]["sharedContractFingerprint"]
        == checkpoint_contract_fingerprint(_checkpoint())
        for event in classified
    )
    assert all(
        event.classifications["model_artifact"]["displayIdentity"]["weightsFile"] == "model.pth"
        for event in classified
    )


def test_processor_returns_validation_error_when_preflight_fails(
    local_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _write_runtime_model(local_tmp_path)
    audio_path = _write_wav(local_tmp_path / "audio.wav")

    monkeypatch.setattr(
        runtime_loader,
        "run_runtime_preflight",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValidationError("bundle mismatch")),
    )

    graph = _make_graph(str(model_path), include_audio=True)
    context = _make_context(graph)
    context.set_output("onset1", "events_out", _events())
    context.set_output(
        "load1",
        "audio_out",
        AudioData(sample_rate=22050, duration=0.25, file_path=str(audio_path), channel_count=1),
    )

    result = PyTorchAudioClassifyProcessor().execute("classify1", context)

    assert isinstance(result, Err)
    assert str(result.error) == "bundle mismatch"
