from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path
from uuid import uuid4

import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import Block, BlockSettings, Connection, Event, EventData, Layer, Port
from echozero.errors import ValidationError
from echozero.execution import ExecutionContext
from echozero.inference_eval.runtime_preflight import checkpoint_contract_fingerprint, run_runtime_preflight
from echozero.processors import pytorch_audio_classify as classify_module
from echozero.processors.pytorch_audio_classify import PyTorchAudioClassifyProcessor, _default_classify
from echozero.progress import RuntimeBus
from echozero.result import Err


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


def _make_graph(model_path: str) -> Graph:
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
        input_ports=(_event_in(),),
        output_ports=(_event_out(),),
        settings=BlockSettings({"model_path": model_path}),
    )
    graph.add_block(onset_block)
    graph.add_block(classify_block)
    graph.add_connection(
        Connection(
            source_block_id="onset1",
            source_output_name="events_out",
            target_block_id="classify1",
            target_input_name="events_in",
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
    manifest = {
        "schema": "foundry.artifact_manifest.v1",
        "weightsPath": model_name,
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
    if fingerprint is not None:
        manifest["sharedContractFingerprint"] = fingerprint
    return manifest


def _write_manifest(directory: Path, payload: dict[str, object]) -> Path:
    path = directory / "art_test.manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, checkpoint: dict[str, object]) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.load = lambda *args, **kwargs: checkpoint
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_runtime_preflight_keeps_manifest_absent_behavior(local_tmp_path: Path) -> None:
    model_path = local_tmp_path / "model.pth"
    model_path.write_bytes(b"weights")

    run_runtime_preflight(model_path, _checkpoint())


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


def test_default_classify_runs_preflight_once_per_model_load(
    local_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = local_tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    checkpoint = _checkpoint()
    _write_manifest(
        local_tmp_path,
        _manifest(model_path.name, fingerprint=checkpoint_contract_fingerprint(checkpoint)),
    )

    calls: list[tuple[Path, dict[str, object]]] = []

    def spy_preflight(model: str | Path, loaded_checkpoint: dict[str, object], *, consumer: str = "PyTorchAudioClassify") -> None:
        assert consumer == "PyTorchAudioClassify"
        calls.append((Path(model), loaded_checkpoint))

    class _FakeModel:
        def load_state_dict(self, state: object) -> None:
            assert state == {}

        def eval(self) -> None:
            return None

        def to(self, device: str) -> "_FakeModel":
            assert device == "cpu"
            return self

    monkeypatch.setattr(classify_module, "run_runtime_preflight", spy_preflight)
    monkeypatch.setattr(classify_module, "_create_model_from_config", lambda config, device: _FakeModel())
    monkeypatch.setattr(classify_module, "_predict_event_class", lambda *args, **kwargs: "kick")
    _install_fake_torch(monkeypatch, checkpoint)

    classified = _default_classify(
        list(_events().layers[0].events),
        audio_file=None,
        model_path=str(model_path),
        device="cpu",
        batch_size=32,
    )

    assert len(classified) == 3
    assert len(calls) == 1
    assert calls[0][0] == model_path
    assert calls[0][1] is checkpoint


def test_processor_returns_validation_error_when_preflight_fails(
    local_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = local_tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    checkpoint = _checkpoint()

    monkeypatch.setattr(
        classify_module,
        "run_runtime_preflight",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValidationError("bundle mismatch")),
    )
    monkeypatch.setattr(classify_module, "_create_model_from_config", lambda config, device: object())
    _install_fake_torch(monkeypatch, checkpoint)

    graph = _make_graph(str(model_path))
    context = _make_context(graph)
    context.set_output("onset1", "events_out", _events())

    result = PyTorchAudioClassifyProcessor().execute("classify1", context)

    assert isinstance(result, Err)
    assert str(result.error) == "bundle mismatch"
