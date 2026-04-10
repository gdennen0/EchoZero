from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Mapping, Protocol


JsonDict = dict[str, Any]


def _freeze_mapping(payload: Mapping[str, Any] | None) -> JsonDict:
    if payload is None:
        return {}
    return dict(payload)


@dataclass(frozen=True, slots=True)
class InferenceContract:
    """Shared, app-agnostic inference contract."""

    schema: str = "echozero.shared.inference_contract.v1"
    preprocessing: JsonDict = field(default_factory=dict)
    class_map: tuple[str, ...] = field(default_factory=tuple)
    model_signature: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "preprocessing", _freeze_mapping(self.preprocessing))
        object.__setattr__(self, "class_map", tuple(self.class_map))


@dataclass(frozen=True, slots=True)
class EvalContract:
    """Shared, app-agnostic eval contract."""

    schema: str = "echozero.shared.eval_contract.v1"
    classification_mode: str = "multiclass"
    metric_keys: tuple[str, ...] = ("accuracy", "macro_f1")
    split_name: str = "test"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_keys", tuple(self.metric_keys))


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    contract: InferenceContract
    inputs: tuple[JsonDict, ...]
    context: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(dict(item) for item in self.inputs))
        object.__setattr__(self, "context", _freeze_mapping(self.context))


@dataclass(frozen=True, slots=True)
class InferenceResult:
    predictions: tuple[JsonDict, ...]
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "predictions", tuple(dict(item) for item in self.predictions))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class EvalRequest:
    contract: EvalContract
    run_id: str
    dataset_version_id: str | None = None
    summary: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _freeze_mapping(self.summary))


@dataclass(frozen=True, slots=True)
class EvalResult:
    metrics: JsonDict
    aggregate_metrics: JsonDict = field(default_factory=dict)
    per_class_metrics: JsonDict = field(default_factory=dict)
    baseline: JsonDict = field(default_factory=dict)
    threshold_policy: JsonDict | None = None
    confusion: JsonDict | None = None
    summary: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _freeze_mapping(self.metrics))
        object.__setattr__(self, "aggregate_metrics", _freeze_mapping(self.aggregate_metrics))
        object.__setattr__(self, "per_class_metrics", _freeze_mapping(self.per_class_metrics))
        object.__setattr__(self, "baseline", _freeze_mapping(self.baseline))
        object.__setattr__(self, "summary", _freeze_mapping(self.summary))
        if self.threshold_policy is not None:
            object.__setattr__(self, "threshold_policy", _freeze_mapping(self.threshold_policy))
        if self.confusion is not None:
            object.__setattr__(self, "confusion", _freeze_mapping(self.confusion))


class InferenceCore(Protocol):
    def infer(self, request: InferenceRequest) -> InferenceResult: ...


class EvalCore(Protocol):
    def evaluate(self, request: EvalRequest) -> EvalResult: ...


def canonical_contract_payload(inference_contract: InferenceContract, eval_contract: EvalContract) -> JsonDict:
    return {
        "inference": {
            "schema": inference_contract.schema,
            "preprocessing": inference_contract.preprocessing,
            "class_map": list(inference_contract.class_map),
            "model_signature": inference_contract.model_signature,
        },
        "eval": {
            "schema": eval_contract.schema,
            "classification_mode": eval_contract.classification_mode,
            "metric_keys": list(eval_contract.metric_keys),
            "split_name": eval_contract.split_name,
        },
    }


def contract_fingerprint(inference_contract: InferenceContract, eval_contract: EvalContract) -> str:
    payload = canonical_contract_payload(inference_contract, eval_contract)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()
