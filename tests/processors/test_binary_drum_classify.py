"""
BinaryDrumClassifyProcessor tests: verify per-class layer output from drum onset classification.
Exists because the app-facing classified-drum pipeline must produce separate kick/snare layers.
Tests assert on layer splitting, validation, and audio/event input requirements.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import AudioData, Block, BlockSettings, Connection, Event, EventData, Layer, Port
from echozero.execution import ExecutionContext
from echozero.processors.binary_drum_classify import (
    BinaryAssignmentConfig,
    BinaryDrumClassifyProcessor,
    DrumLabelInferenceInput,
    _default_binary_classify,
    _apply_assignment_config,
)
from echozero.progress import RuntimeBus
from echozero.result import Err, Ok
from echozero.runtime_models.loader import LoadedRuntimeModel


def _event_in(name: str = "events_in") -> Port:
    return Port(name=name, port_type=PortType.EVENT, direction=Direction.INPUT)


def _audio_in(name: str = "audio_in") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.INPUT)


def _event_out(name: str = "events_out") -> Port:
    return Port(name=name, port_type=PortType.EVENT, direction=Direction.OUTPUT)


def _make_graph(
    *,
    include_label_specific_inputs: bool = False,
    binary_settings: dict[str, Any] | None = None,
) -> Graph:
    input_ports = [_audio_in(), _event_in()]
    if include_label_specific_inputs:
        input_ports.extend(
            (
                _audio_in("kick_audio_in"),
                _audio_in("snare_audio_in"),
                _event_in("kick_events_in"),
                _event_in("snare_events_in"),
            )
        )
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
            input_ports=tuple(input_ports),
            output_ports=(_event_out(),),
            settings=BlockSettings(
                {
                    "kick_model_path": "/models/kick.manifest.json",
                    "snare_model_path": "/models/snare.manifest.json",
                    **(binary_settings or {}),
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
    inputs: tuple[DrumLabelInferenceInput, ...],
    device: str,
    assignment: BinaryAssignmentConfig,
) -> dict[str, list[tuple[Event, float]]]:
    del device, assignment
    by_label = {label_input.label: label_input for label_input in inputs}
    return {
        "kick": [
            (
                Event(
                    id=f"{by_label['kick'].events[0].id}_kick",
                    time=by_label["kick"].events[0].time,
                    duration=by_label["kick"].events[0].duration,
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
                    id=f"{by_label['snare'].events[0].id}_snare",
                    time=by_label["snare"].events[0].time,
                    duration=by_label["snare"].events[0].duration,
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


def test_binary_drum_classify_processor_supports_label_specific_inputs() -> None:
    graph = _make_graph(
        include_label_specific_inputs=True,
        binary_settings={
            "kick_positive_threshold": 0.35,
            "snare_positive_threshold": 0.7,
            "assignment_mode": "exclusive_max",
            "winner_margin": 0.08,
            "event_match_window_ms": 25.0,
            "min_event_peak": 0.004,
            "min_event_rms": 0.002,
        },
    )
    graph.add_block(
        Block(
            id="kick_audio",
            name="Kick Audio",
            block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
        )
    )
    graph.add_block(
        Block(
            id="snare_audio",
            name="Snare Audio",
            block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
        )
    )
    graph.add_block(
        Block(
            id="kick_events",
            name="Kick Events",
            block_type="DetectOnsets",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(_event_out(),),
        )
    )
    graph.add_block(
        Block(
            id="snare_events",
            name="Snare Events",
            block_type="DetectOnsets",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(_event_out(),),
        )
    )
    graph.add_connection(Connection("kick_audio", "audio_out", "binary", "kick_audio_in"))
    graph.add_connection(Connection("snare_audio", "audio_out", "binary", "snare_audio_in"))
    graph.add_connection(Connection("kick_events", "events_out", "binary", "kick_events_in"))
    graph.add_connection(Connection("snare_events", "events_out", "binary", "snare_events_in"))
    context = _make_context(graph)
    context.set_output(
        "kick_audio",
        "audio_out",
        AudioData(sample_rate=44100, duration=1.0, file_path="/tmp/kick.wav", channel_count=1),
    )
    context.set_output(
        "snare_audio",
        "audio_out",
        AudioData(sample_rate=44100, duration=1.0, file_path="/tmp/snare.wav", channel_count=1),
    )
    context.set_output(
        "kick_events",
        "events_out",
        EventData(
            layers=(
                Layer(
                    id="kick_onsets",
                    name="Kick Onsets",
                    events=(
                        Event(id="kick_evt", time=0.10, duration=0.05, classifications={}, metadata={}, origin="kick"),
                    ),
                ),
            )
        ),
    )
    context.set_output(
        "snare_events",
        "events_out",
        EventData(
            layers=(
                Layer(
                    id="snare_onsets",
                    name="Snare Onsets",
                    events=(
                        Event(id="snare_evt", time=0.20, duration=0.05, classifications={}, metadata={}, origin="snare"),
                    ),
                ),
            )
        ),
    )

    captured: dict[str, object] = {}

    def _capture_binary_classify(
        inputs: tuple[DrumLabelInferenceInput, ...],
        device: str,
        assignment: BinaryAssignmentConfig,
    ) -> dict[str, list[tuple[Event, float]]]:
        captured["inputs"] = inputs
        captured["device"] = device
        captured["assignment"] = assignment
        return {"kick": [], "snare": []}

    processor = BinaryDrumClassifyProcessor(classify_fn=_capture_binary_classify)
    result = processor.execute("binary", context)

    assert isinstance(result, Ok)
    inputs = captured["inputs"]
    assert isinstance(inputs, tuple)
    kick_input = next(label_input for label_input in inputs if label_input.label == "kick")
    snare_input = next(label_input for label_input in inputs if label_input.label == "snare")
    assert kick_input.audio_file == "/tmp/kick.wav"
    assert snare_input.audio_file == "/tmp/snare.wav"
    assert kick_input.positive_threshold == 0.35
    assert snare_input.positive_threshold == 0.7
    assert [event.id for event in kick_input.events] == ["kick_evt"]
    assert [event.id for event in snare_input.events] == ["snare_evt"]
    assert captured["device"] == "cpu"
    assert captured["assignment"] == BinaryAssignmentConfig(
        assignment_mode="exclusive_max",
        winner_margin=0.08,
        event_match_window_seconds=0.025,
        min_event_peak=0.004,
        min_event_rms=0.002,
    )


def test_apply_assignment_config_prefers_stronger_candidate_within_window() -> None:
    resolved = _apply_assignment_config(
        {
            "kick": [
                (
                    Event(
                        id="evt_kick",
                        time=0.25,
                        duration=0.05,
                        classifications={"class": "kick"},
                        metadata={
                            "detection": {"threshold_passed": True},
                        },
                        origin="kick",
                    ),
                    0.91,
                )
            ],
            "snare": [
                (
                    Event(
                        id="evt_snare",
                        time=0.27,
                        duration=0.05,
                        classifications={"class": "snare"},
                        metadata={
                            "detection": {"threshold_passed": True},
                        },
                        origin="snare",
                    ),
                    0.72,
                )
            ],
        },
        BinaryAssignmentConfig(
            assignment_mode="exclusive_max",
            winner_margin=0.05,
            event_match_window_seconds=0.04,
        ),
    )

    assert [event.id for event, _score in resolved["kick"]] == ["evt_kick"]
    assert resolved["kick"][0][0].metadata["detection"]["promotion_state"] == "promoted"
    assert [event.id for event, _score in resolved["snare"]] == ["evt_snare"]
    assert resolved["snare"][0][0].metadata["detection"]["promotion_state"] == "demoted"


def test_apply_assignment_config_drops_ambiguous_candidates_when_margin_fails() -> None:
    resolved = _apply_assignment_config(
        {
            "kick": [
                (
                    Event(
                        id="evt_kick",
                        time=0.25,
                        duration=0.05,
                        classifications={"class": "kick"},
                        metadata={
                            "detection": {"threshold_passed": True},
                        },
                        origin="kick",
                    ),
                    0.82,
                )
            ],
            "snare": [
                (
                    Event(
                        id="evt_snare",
                        time=0.27,
                        duration=0.05,
                        classifications={"class": "snare"},
                        metadata={
                            "detection": {"threshold_passed": True},
                        },
                        origin="snare",
                    ),
                    0.80,
                )
            ],
        },
        BinaryAssignmentConfig(
            assignment_mode="exclusive_max",
            winner_margin=0.05,
            event_match_window_seconds=0.04,
        ),
    )

    assert [event.id for event, _score in resolved["kick"]] == ["evt_kick"]
    assert resolved["kick"][0][0].metadata["detection"]["promotion_state"] == "demoted"
    assert [event.id for event, _score in resolved["snare"]] == ["evt_snare"]
    assert resolved["snare"][0][0].metadata["detection"]["promotion_state"] == "demoted"


def test_default_binary_classify_stamps_structured_model_artifact_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        SimpleNamespace(resample=lambda audio, **_: audio),
    )
    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(read=lambda *args, **kwargs: (np.full(64, 0.25, dtype=np.float32), 22050)),
    )

    def _fake_load_runtime_model(model_path: str, *, device: str) -> LoadedRuntimeModel:
        label = "kick" if "kick" in model_path else "snare"
        return LoadedRuntimeModel(
            model=object(),
            classes=(label, "other"),
            sample_rate=22050,
            max_length=32,
            n_fft=8,
            hop_length=4,
            n_mels=16,
            fmax=8000,
            device=device,
            source_path=Path(f"/tmp/{label}.pth"),
            manifest_path=None,
            artifact_manifest={
                "artifactId": f"art_{label}",
                "runId": f"run_{label}",
                "datasetVersionId": "dv_123",
                "specHash": f"spec_{label}",
                "sharedContractFingerprint": f"fp_{label}",
                "weightsPath": f"{label}.pth",
                "classes": [label, "other"],
                "classificationMode": "binary",
                "runtime": {"consumer": "PyTorchAudioClassify"},
            },
        )

    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.resolve_device",
        lambda device: device,
    )
    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.load_runtime_model",
        _fake_load_runtime_model,
    )
    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.build_feature_tensor",
        lambda **kwargs: np.zeros((1, 1, 2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.predict_probabilities",
        lambda runtime_model, feature: np.array([0.95, 0.05], dtype=np.float32),
    )

    classified = _default_binary_classify(
        (
            DrumLabelInferenceInput(
                label="kick",
                audio_file="/tmp/kick.wav",
                events=(
                    Event(id="kick_evt", time=0.0, duration=0.01, classifications={}, metadata={}, origin="src"),
                ),
                model_path="/tmp/kick_model.pth",
                positive_threshold=0.5,
            ),
            DrumLabelInferenceInput(
                label="snare",
                audio_file="/tmp/snare.wav",
                events=(
                    Event(id="snare_evt", time=0.0, duration=0.01, classifications={}, metadata={}, origin="src"),
                ),
                model_path="/tmp/snare_model.pth",
                positive_threshold=0.5,
            ),
        ),
        device="cpu",
        assignment=BinaryAssignmentConfig(),
    )

    kick_event = classified["kick"][0][0]
    snare_event = classified["snare"][0][0]
    assert kick_event.metadata["source_model"] == "kick_model.pth"
    assert kick_event.metadata["model_artifact"] == kick_event.classifications["model_artifact"]
    assert kick_event.metadata["model_artifact"]["schema"] == "echozero.model_artifact_ref.v1"
    assert kick_event.metadata["model_artifact"]["artifactIdentity"]["artifactId"] == "art_kick"
    assert kick_event.metadata["model_artifact"]["displayIdentity"]["weightsFile"] == "kick.pth"
    assert kick_event.metadata["detection"]["threshold_passed"] is True
    assert kick_event.metadata["detection"]["promotion_state"] == "promoted"
    assert "review" not in kick_event.metadata
    assert snare_event.metadata["model_artifact"]["artifactIdentity"]["artifactId"] == "art_snare"


def test_default_binary_classify_strips_inherited_review_semantics_from_source_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        SimpleNamespace(resample=lambda audio, **_: audio),
    )
    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(read=lambda *args, **kwargs: (np.full(64, 0.25, dtype=np.float32), 22050)),
    )

    def _fake_load_runtime_model(model_path: str, *, device: str) -> LoadedRuntimeModel:
        label = "kick" if "kick" in model_path else "snare"
        return LoadedRuntimeModel(
            model=object(),
            classes=(label, "other"),
            sample_rate=22050,
            max_length=32,
            n_fft=8,
            hop_length=4,
            n_mels=16,
            fmax=8000,
            device=device,
            source_path=Path(f"/tmp/{label}.pth"),
        )

    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.resolve_device",
        lambda device: device,
    )
    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.load_runtime_model",
        _fake_load_runtime_model,
    )
    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.build_feature_tensor",
        lambda **kwargs: np.zeros((1, 1, 2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.predict_probabilities",
        lambda runtime_model, feature: np.array([0.95, 0.05], dtype=np.float32),
    )

    inherited_review_metadata = {
        "review": {
            "promotion_state": "demoted",
            "review_state": "corrected",
            "review_outcome": "incorrect",
        },
        "promotion_state": "demoted",
        "review_state": "corrected",
        "review_outcome": "incorrect",
        "review_decision_kind": "rejected",
    }
    classified = _default_binary_classify(
        (
            DrumLabelInferenceInput(
                label="kick",
                audio_file="/tmp/kick.wav",
                events=(
                    Event(
                        id="kick_evt",
                        time=0.0,
                        duration=0.01,
                        classifications={},
                        metadata=inherited_review_metadata,
                        origin="src",
                    ),
                ),
                model_path="/tmp/kick_model.pth",
                positive_threshold=0.5,
            ),
            DrumLabelInferenceInput(
                label="snare",
                audio_file="/tmp/snare.wav",
                events=(
                    Event(
                        id="snare_evt",
                        time=0.0,
                        duration=0.01,
                        classifications={},
                        metadata=inherited_review_metadata,
                        origin="src",
                    ),
                ),
                model_path="/tmp/snare_model.pth",
                positive_threshold=0.5,
            ),
        ),
        device="cpu",
        assignment=BinaryAssignmentConfig(),
    )

    kick_event = classified["kick"][0][0]
    assert "review" not in kick_event.metadata
    assert "promotion_state" not in kick_event.metadata
    assert "review_state" not in kick_event.metadata
    assert "review_outcome" not in kick_event.metadata
    assert "review_decision_kind" not in kick_event.metadata
    assert kick_event.metadata["detection"]["promotion_state"] == "promoted"


def test_apply_assignment_config_requires_detection_threshold_signal() -> None:
    resolved = _apply_assignment_config(
        {
            "kick": [
                (
                    Event(
                        id="evt_kick",
                        time=0.25,
                        duration=0.05,
                        classifications={"class": "kick"},
                        metadata={
                            "classifier_score": 0.99,
                            "positive_threshold": 0.5,
                        },
                        origin="kick",
                    ),
                    0.91,
                )
            ],
            "snare": [],
        },
        BinaryAssignmentConfig(
            assignment_mode="exclusive_max",
            winner_margin=0.0,
            event_match_window_seconds=0.04,
        ),
    )

    assert resolved["kick"][0][0].metadata["detection"]["promotion_state"] == "demoted"


def test_default_binary_classify_skips_near_silent_events_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        SimpleNamespace(resample=lambda audio, **_: audio),
    )
    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(read=lambda *args, **kwargs: (np.zeros(64, dtype=np.float32), 22050)),
    )

    def _fake_load_runtime_model(model_path: str, *, device: str) -> LoadedRuntimeModel:
        label = "kick" if "kick" in model_path else "snare"
        return LoadedRuntimeModel(
            model=object(),
            classes=(label, "other"),
            sample_rate=22050,
            max_length=32,
            n_fft=8,
            hop_length=4,
            n_mels=16,
            fmax=8000,
            device=device,
            source_path=Path(f"/tmp/{label}.pth"),
        )

    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.resolve_device",
        lambda device: device,
    )
    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.load_runtime_model",
        _fake_load_runtime_model,
    )
    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.build_feature_tensor",
        lambda **kwargs: np.zeros((1, 1, 2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.predict_probabilities",
        lambda runtime_model, feature: np.array([0.95, 0.05], dtype=np.float32),
    )

    classified = _default_binary_classify(
        (
            DrumLabelInferenceInput(
                label="kick",
                audio_file="/tmp/kick.wav",
                events=(
                    Event(id="kick_evt", time=0.0, duration=0.01, classifications={}, metadata={}, origin="src"),
                ),
                model_path="/tmp/kick_model.pth",
                positive_threshold=0.5,
            ),
            DrumLabelInferenceInput(
                label="snare",
                audio_file="/tmp/snare.wav",
                events=(
                    Event(id="snare_evt", time=0.0, duration=0.01, classifications={}, metadata={}, origin="src"),
                ),
                model_path="/tmp/snare_model.pth",
                positive_threshold=0.5,
            ),
        ),
        device="cpu",
        assignment=BinaryAssignmentConfig(),
    )

    assert classified["kick"] == []
    assert classified["snare"] == []


def test_default_binary_classify_demotes_scores_below_threshold_instead_of_dropping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        SimpleNamespace(resample=lambda audio, **_: audio),
    )
    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(read=lambda *args, **kwargs: (np.full(64, 0.25, dtype=np.float32), 22050)),
    )

    def _fake_load_runtime_model(model_path: str, *, device: str) -> LoadedRuntimeModel:
        label = "kick" if "kick" in model_path else "snare"
        return LoadedRuntimeModel(
            model=object(),
            classes=(label, "other"),
            sample_rate=22050,
            max_length=32,
            n_fft=8,
            hop_length=4,
            n_mels=16,
            fmax=8000,
            device=device,
            source_path=Path(f"/tmp/{label}.pth"),
        )

    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.resolve_device",
        lambda device: device,
    )
    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.load_runtime_model",
        _fake_load_runtime_model,
    )
    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.build_feature_tensor",
        lambda **kwargs: np.zeros((1, 1, 2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "echozero.processors.binary_drum_classify.predict_probabilities",
        lambda runtime_model, feature: np.array([0.31, 0.69], dtype=np.float32),
    )

    classified = _default_binary_classify(
        (
            DrumLabelInferenceInput(
                label="kick",
                audio_file="/tmp/kick.wav",
                events=(
                    Event(id="kick_evt", time=0.0, duration=0.01, classifications={}, metadata={}, origin="src"),
                ),
                model_path="/tmp/kick_model.pth",
                positive_threshold=0.5,
            ),
            DrumLabelInferenceInput(
                label="snare",
                audio_file="/tmp/snare.wav",
                events=(
                    Event(id="snare_evt", time=0.0, duration=0.01, classifications={}, metadata={}, origin="src"),
                ),
                model_path="/tmp/snare_model.pth",
                positive_threshold=0.5,
            ),
        ),
        device="cpu",
        assignment=BinaryAssignmentConfig(),
    )

    kick_event = classified["kick"][0][0]
    snare_event = classified["snare"][0][0]
    assert kick_event.metadata["detection"]["promotion_state"] == "demoted"
    assert kick_event.metadata["detection"]["threshold_passed"] is False
    assert kick_event.classifications["label"] == "Kick"
    assert snare_event.metadata["detection"]["promotion_state"] == "demoted"
    assert snare_event.metadata["detection"]["threshold_passed"] is False
